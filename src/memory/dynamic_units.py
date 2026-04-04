from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import torch

from src.core.config import MemoryConfig
from src.core.time_utils import within_session
from src.memory.metrics import compute_unit_outcome_dispersion, compute_unit_utility
from src.memory.path_stats import aggregate_path_statistics, aggregate_risk_statistics
from src.memory.schemas import MemoryItem, MemoryUnit, UnitStatus
from src.memory.similarity import pairwise_similarity


@dataclass
class DynamicUnitReport:
    created_units: int = 0
    updated_units: int = 0
    promoted_units: int = 0
    novelty_threshold: float = 0.0
    assignment_similarity_mean: float = 0.0
    candidate_units: int = 0
    mature_units: int = 0
    quarantined_units: int = 0
    conflict_rejections: int = 0
    saturation_attenuated_updates: int = 0


@dataclass
class _UnitAccumulator:
    unit_id: str
    status: str
    prototype: torch.Tensor
    member_indices: list[int]
    assignment_similarities: list[float]
    outcome_signs: list[float]
    created_at: object
    updated_at: object
    created_step: int
    updated_step: int
    cumulative_updates: int = 0
    conflict_history: list[float] = field(default_factory=list)

    def add_member(
        self,
        index: int,
        item: MemoryItem,
        similarity: float,
        *,
        plasticity: float = 1.0,
    ) -> None:
        support_count = len(self.member_indices)
        # SatSOM-inspired: plasticity-attenuated prototype update
        # High plasticity (new units) -> full update. Low plasticity (mature) -> smaller shift.
        effective_lr = plasticity / float(support_count + 1)
        delta = item.latent.detach().cpu() - self.prototype
        self.prototype = self.prototype + effective_lr * delta
        self.member_indices.append(index)
        self.assignment_similarities.append(float(similarity))
        self.outcome_signs.append(
            1.0 if item.outcome.primary_return >= 0.0 else -1.0
        )
        self.updated_at = item.timestamp
        self.updated_step = index
        self.cumulative_updates += 1

    def record_conflict(self, conflict_value: float) -> None:
        self.conflict_history.append(float(conflict_value))

    @property
    def recent_conflict_rate(self) -> float:
        window = self.conflict_history[-20:] if self.conflict_history else []
        if not window:
            return 0.0
        return sum(1.0 for v in window if v > 0.5) / len(window)


def build_dynamic_units(
    items: list[MemoryItem],
    latent_matrix: torch.Tensor,
    config: MemoryConfig,
) -> tuple[list[MemoryUnit], DynamicUnitReport]:
    if not items:
        return [], DynamicUnitReport()
    assignment_history: list[float] = []
    accumulators: list[_UnitAccumulator] = []
    report = DynamicUnitReport()
    for index, item in enumerate(items):
        novelty_threshold = _novelty_threshold(assignment_history, config)
        strong_threshold = max(config.min_similarity, novelty_threshold)
        best_mature = _best_unit(item, accumulators, status="mature", metric=config.similarity_metric, temperature=config.attention_temperature)
        best_candidate = _best_unit(item, accumulators, status="candidate", metric=config.similarity_metric, temperature=config.attention_temperature)

        # --- Outcome-aware assignment: check if outcome is compatible ---
        if best_mature is not None and best_mature[1] >= strong_threshold:
            accumulator = accumulators[best_mature[0]]
            outcome_compatible = _check_outcome_compatibility(accumulator, item, config)
            if outcome_compatible:
                plasticity = _compute_plasticity(accumulator, config)
                accumulator.add_member(index, item, best_mature[1], plasticity=plasticity)
                if plasticity < 1.0:
                    report.saturation_attenuated_updates += 1
                assignment_history.append(best_mature[1])
                report.updated_units += 1
                # Track conflict for the unit
                conflict_val = _item_conflict_with_unit(accumulator, item)
                accumulator.record_conflict(conflict_val)
                continue
            else:
                report.conflict_rejections += 1
                # Fall through to create new unit for conflicting outcome

        if best_candidate is not None and best_candidate[1] >= novelty_threshold:
            accumulator = accumulators[best_candidate[0]]
            outcome_compatible = _check_outcome_compatibility(accumulator, item, config)
            if outcome_compatible:
                plasticity = _compute_plasticity(accumulator, config)
                accumulator.add_member(index, item, best_candidate[1], plasticity=plasticity)
                assignment_history.append(best_candidate[1])
                report.updated_units += 1
                conflict_val = _item_conflict_with_unit(accumulator, item)
                accumulator.record_conflict(conflict_val)
                if _should_promote(accumulator, assignment_history, config):
                    accumulator.status = "mature"
                    report.promoted_units += 1
                continue
            else:
                report.conflict_rejections += 1

        # Create new candidate unit
        unit_id = f"unit_{len(accumulators):04d}"
        accumulators.append(
            _UnitAccumulator(
                unit_id=unit_id,
                status="candidate",
                prototype=item.latent.detach().cpu().clone(),
                member_indices=[index],
                assignment_similarities=[1.0],
                outcome_signs=[1.0 if item.outcome.primary_return >= 0.0 else -1.0],
                created_at=item.timestamp,
                updated_at=item.timestamp,
                created_step=index,
                updated_step=index,
            )
        )
        assignment_history.append(0.0 if not accumulators[:-1] else min(best_mature[1] if best_mature else 0.0, best_candidate[1] if best_candidate else 0.0))
        report.created_units += 1
    report.novelty_threshold = _novelty_threshold(assignment_history, config)
    report.assignment_similarity_mean = (
        float(torch.tensor(assignment_history, dtype=torch.float32).mean().item()) if assignment_history else 0.0
    )
    units = [_finalize_unit(accumulator, items, config, latest_step=len(items) - 1) for accumulator in accumulators]
    report.candidate_units = sum(1 for unit in units if unit.status == UnitStatus.CANDIDATE.value)
    report.mature_units = sum(1 for unit in units if unit.status == UnitStatus.MATURE.value)
    report.quarantined_units = sum(1 for unit in units if unit.status == UnitStatus.QUARANTINED.value)
    return units, report


def _check_outcome_compatibility(accumulator: _UnitAccumulator, item: MemoryItem, config: MemoryConfig) -> bool:
    """Check if a new item's outcome is compatible with the unit's established direction.

    Returns True if compatible (should assign), False if conflicting (should create new unit).
    Only enforced when the unit has enough members to have an established direction.
    """
    if not getattr(config, "require_outcome_alignment_for_strong_update", False):
        return True
    if len(accumulator.outcome_signs) < max(config.min_candidate_support, 3):
        return True
    existing_direction = sum(accumulator.outcome_signs) / len(accumulator.outcome_signs)
    item_sign = 1.0 if item.outcome.primary_return >= 0.0 else -1.0
    # If the unit has a clear direction and the item contradicts it, reject
    if abs(existing_direction) > 0.5 and existing_direction * item_sign < 0:
        return False
    return True


def _item_conflict_with_unit(accumulator: _UnitAccumulator, item: MemoryItem) -> float:
    """Compute how much a new item conflicts with the unit's established direction."""
    if not accumulator.outcome_signs:
        return 0.0
    existing_direction = sum(accumulator.outcome_signs) / len(accumulator.outcome_signs)
    item_sign = 1.0 if item.outcome.primary_return >= 0.0 else -1.0
    # 0.0 = fully aligned, 1.0 = fully conflicting
    return max(0.0, -existing_direction * item_sign)


def _compute_plasticity(accumulator: _UnitAccumulator, config: MemoryConfig) -> float:
    """SatSOM-inspired: compute plasticity factor for a unit.

    Mature, well-established units should be more resistant to update (lower plasticity).
    Fresh candidate units should remain fully plastic.
    """
    if not getattr(config, "saturation_enabled", False):
        return 1.0
    if accumulator.status != "mature":
        return 1.0
    support = len(accumulator.member_indices)
    attenuation = getattr(config, "maturity_update_attenuation", 0.95)
    # Plasticity decays exponentially with cumulative updates, floored by (1 - attenuation)
    plasticity = max(
        attenuation ** accumulator.cumulative_updates,
        1.0 - attenuation,
    )
    return float(plasticity)


def _compute_saturation(accumulator: _UnitAccumulator, config: MemoryConfig) -> float:
    """SatSOM-inspired: compute saturation score for a unit.

    Saturation approaches 1.0 as the unit becomes well-established with consistent outcomes.
    """
    support = len(accumulator.member_indices)
    min_for_saturation = max(config.min_candidate_support * 2, 6)
    support_factor = min(support / min_for_saturation, 1.0)
    # Consistency factor: how consistent are the outcomes?
    if not accumulator.outcome_signs:
        consistency = 0.0
    else:
        consistency = abs(sum(accumulator.outcome_signs)) / len(accumulator.outcome_signs)
    # Similarity factor: how tight is the cluster?
    if len(accumulator.assignment_similarities) < 2:
        sim_factor = 0.0
    else:
        sim_tensor = torch.tensor(accumulator.assignment_similarities, dtype=torch.float32)
        sim_factor = float(sim_tensor.mean().item())
    saturation = support_factor * consistency * sim_factor
    return min(max(saturation, 0.0), 1.0)


def _best_unit(
    item: MemoryItem,
    accumulators: list[_UnitAccumulator],
    *,
    status: str,
    metric: str,
    temperature: float,
) -> tuple[int, float] | None:
    candidates = [(index, accumulator) for index, accumulator in enumerate(accumulators) if accumulator.status == status]
    if not candidates:
        return None
    centroids = torch.stack([accumulator.prototype for _, accumulator in candidates], dim=0)
    _, normalized_scores = pairwise_similarity(
        item.latent.detach().cpu(),
        centroids,
        metric=metric,
        temperature=temperature,
    )
    best_index = int(torch.argmax(normalized_scores).item())
    return candidates[best_index][0], float(normalized_scores[best_index].item())


def _novelty_threshold(assignment_history: list[float], config: MemoryConfig) -> float:
    if len(assignment_history) < max(config.min_candidate_support, 4):
        return max(config.min_similarity * 0.5, 0.1)
    tensor = torch.tensor(assignment_history, dtype=torch.float32)
    quantile_value = float(torch.quantile(tensor, config.novelty_quantile).item())
    return max(min(quantile_value, config.min_similarity), 0.0)


def _should_promote(accumulator: _UnitAccumulator, assignment_history: list[float], config: MemoryConfig) -> bool:
    support_ready = len(accumulator.member_indices) >= config.min_candidate_support
    if not support_ready:
        return False
    if config.candidate_promotion_mode == "support_only":
        return True
    threshold = _novelty_threshold(assignment_history, config)
    mean_similarity = float(torch.tensor(accumulator.assignment_similarities, dtype=torch.float32).mean().item())
    return mean_similarity >= threshold


def _finalize_unit(
    accumulator: _UnitAccumulator,
    items: list[MemoryItem],
    config: MemoryConfig,
    *,
    latest_step: int,
) -> MemoryUnit:
    member_items = [items[index] for index in accumulator.member_indices]
    outcomes = [item.outcome for item in member_items]
    regime_counts = Counter()
    symbol_counts = Counter(item.symbol for item in member_items)
    session_counts = Counter(_session_label(item.timestamp) for item in member_items)
    for item in member_items:
        if item.regime_weights is None or item.regime_weights.numel() == 0:
            continue
        regime_counts[f"regime_{int(torch.argmax(item.regime_weights).item())}"] += 1
    assignment_tensor = torch.tensor(accumulator.assignment_similarities, dtype=torch.float32)
    member_latents = torch.stack([item.latent.detach().cpu() for item in member_items], dim=0)
    latent_dispersion = float(torch.mean(torch.norm(member_latents - accumulator.prototype, dim=-1)).item())
    path_stats = aggregate_path_statistics(outcomes)
    risk_stats = aggregate_risk_statistics(outcomes)
    conflict_rate = 1.0 - max(path_stats.direction_probabilities.values(), default=0.0)

    # Compute saturation and plasticity
    saturation = _compute_saturation(accumulator, config)
    plasticity = _compute_plasticity(accumulator, config)

    # Quarantine units with persistently high conflict
    resolved_status = accumulator.status
    if (
        accumulator.recent_conflict_rate > 0.6
        and len(accumulator.member_indices) >= config.min_candidate_support * 2
        and getattr(config, "conflict_tracking_enabled", False)
    ):
        resolved_status = UnitStatus.QUARANTINED.value

    outcome_dispersion = compute_unit_outcome_dispersion(
        MemoryUnit(
            unit_id=accumulator.unit_id,
            status=resolved_status,
            prototype=accumulator.prototype,
            support_count=len(member_items),
            member_indices=list(accumulator.member_indices),
            average_similarity=float(assignment_tensor.mean().item()),
            min_similarity=float(assignment_tensor.min().item()),
            max_similarity=float(assignment_tensor.max().item()),
            similarity_std=float(assignment_tensor.std(unbiased=False).item()) if assignment_tensor.numel() > 1 else 0.0,
            path_stats=path_stats,
            risk_stats=risk_stats,
            regime_composition={},
            symbol_composition={},
            session_composition={},
            utility_score=0.0,
            maturity_score=0.0,
            latent_dispersion=latent_dispersion,
            outcome_dispersion=0.0,
            created_at=accumulator.created_at,
            updated_at=accumulator.updated_at,
            created_step=accumulator.created_step,
            updated_step=accumulator.updated_step,
        )
    )
    maturity_score = min(len(member_items) / max(config.min_candidate_support, 1), 1.0) * float(assignment_tensor.mean().item())

    # GNG-inspired: local error = mean distance of assigned items to prototype
    local_error = latent_dispersion

    utility_score = compute_unit_utility(
        support_count=len(member_items),
        maturity_score=maturity_score,
        outcome_dispersion=outcome_dispersion,
        staleness_steps=latest_step - accumulator.updated_step,
        utility_decay=config.utility_decay,
    )
    return MemoryUnit(
        unit_id=accumulator.unit_id,
        status=resolved_status,
        prototype=accumulator.prototype.detach().cpu(),
        support_count=len(member_items),
        member_indices=list(accumulator.member_indices),
        average_similarity=float(assignment_tensor.mean().item()),
        min_similarity=float(assignment_tensor.min().item()),
        max_similarity=float(assignment_tensor.max().item()),
        similarity_std=float(assignment_tensor.std(unbiased=False).item()) if assignment_tensor.numel() > 1 else 0.0,
        path_stats=path_stats,
        risk_stats=risk_stats,
        regime_composition=_normalize_counts(regime_counts),
        symbol_composition=_normalize_counts(symbol_counts),
        session_composition=_normalize_counts(session_counts),
        utility_score=utility_score,
        maturity_score=maturity_score,
        latent_dispersion=latent_dispersion,
        outcome_dispersion=outcome_dispersion,
        created_at=accumulator.created_at,
        updated_at=accumulator.updated_at,
        created_step=accumulator.created_step,
        updated_step=accumulator.updated_step,
        calibration_confidence=0.0,
        conflict_rate=conflict_rate,
        saturation_score=saturation,
        plasticity_factor=plasticity,
        cumulative_updates=accumulator.cumulative_updates,
        conflict_history=list(accumulator.conflict_history[-50:]),
        recent_conflict_rate=accumulator.recent_conflict_rate,
        edge_error=0.0,
        local_error_accumulator=local_error,
        insertion_candidate_score=max(0.0, local_error * (1.0 - saturation)),
        metadata={},
    )


def _normalize_counts(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {str(name): count / total for name, count in sorted(counter.items())}


def _session_label(timestamp: object) -> str:
    if within_session(timestamp, "regular"):
        return "regular"
    if within_session(timestamp, "premarket"):
        return "premarket"
    if within_session(timestamp, "afterhours"):
        return "afterhours"
    return "other"
