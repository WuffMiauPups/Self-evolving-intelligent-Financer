from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from src.core.config import MemoryConfig
from src.memory.metrics import compute_unit_outcome_dispersion, compute_unit_utility
from src.memory.path_stats import aggregate_path_statistics, aggregate_risk_statistics
from src.memory.schemas import MemoryItem, MemoryUnit, UnitStatus
from src.memory.similarity import pairwise_similarity


@dataclass
class MaintenanceReport:
    merged_units: int = 0
    split_units: int = 0
    pruned_units: int = 0
    archived_units: list[MemoryUnit] = field(default_factory=list)


def maintain_memory_units(
    units: list[MemoryUnit],
    items: list[MemoryItem],
    config: MemoryConfig,
    *,
    latest_step: int,
    dry_run: bool = False,
) -> tuple[list[MemoryUnit], MaintenanceReport]:
    working_units = [replace(unit) for unit in units]
    report = MaintenanceReport()
    if config.merge_enabled:
        working_units, merged = _merge_units(working_units, items, config)
        report.merged_units = merged
    if config.split_enabled:
        working_units, split_count = _split_units(working_units, items, config)
        report.split_units = split_count
    if config.prune_enabled:
        working_units, pruned, archived = _prune_units(working_units, config, latest_step=latest_step)
        report.pruned_units = pruned
        report.archived_units = archived
    if dry_run:
        return units, report
    return working_units, report


def _merge_units(units: list[MemoryUnit], items: list[MemoryItem], config: MemoryConfig) -> tuple[list[MemoryUnit], int]:
    mature_units = [unit for unit in units if unit.status == "mature"]
    if len(mature_units) < 2:
        return units, 0
    centroids = torch.stack([unit.prototype for unit in mature_units], dim=0)
    _, normalized = pairwise_similarity(
        centroids,
        centroids,
        metric=config.similarity_metric,
        temperature=config.attention_temperature,
    )
    flat_scores = normalized[~torch.eye(normalized.shape[0], dtype=torch.bool)]
    adaptive_threshold = (
        float(torch.quantile(flat_scores, max(0.5, 1.0 - config.novelty_quantile)).item())
        if flat_scores.numel() > 0
        else config.min_similarity
    )
    merged_indices: set[int] = set()
    merged_units: list[MemoryUnit] = []
    merge_count = 0
    for index, unit in enumerate(units):
        if unit.status != "mature" or index in merged_indices:
            if index not in merged_indices:
                merged_units.append(unit)
            continue
        merge_target = None
        mature_index = mature_units.index(unit)
        for other_index, other_unit in enumerate(mature_units):
            if other_unit.unit_id == unit.unit_id:
                continue
            similarity = float(normalized[mature_index, other_index].item())
            if similarity < adaptive_threshold:
                continue
            consistency = _outcome_consistency(unit, other_unit)
            if consistency >= config.merge_outcome_consistency_threshold:
                merge_target = other_unit
                break
        if merge_target is None:
            merged_units.append(unit)
            continue
        merged = _rebuild_unit_from_indices(
            unit_id=unit.unit_id,
            status="mature",
            member_indices=sorted(set(unit.member_indices + merge_target.member_indices)),
            items=items,
            latest_step=max(unit.updated_step, merge_target.updated_step),
            config=config,
        )
        merged_indices.add(units.index(merge_target))
        merged_units.append(merged)
        merge_count += 1
    return merged_units, merge_count


def _split_units(units: list[MemoryUnit], items: list[MemoryItem], config: MemoryConfig) -> tuple[list[MemoryUnit], int]:
    split_units: list[MemoryUnit] = []
    split_count = 0
    for unit in units:
        if unit.status != "mature" or unit.support_count < 2:
            split_units.append(unit)
            continue
        if max(unit.outcome_dispersion, unit.latent_dispersion) < config.split_variance_trigger:
            split_units.append(unit)
            continue
        positive = [
            index for index in unit.member_indices
            if items[index].outcome.primary_return >= 0.0
        ]
        negative = [
            index for index in unit.member_indices
            if items[index].outcome.primary_return < 0.0
        ]
        adaptive_split_support = max(1, min(config.min_cluster_size, unit.support_count // 3))
        if min(len(positive), len(negative)) < adaptive_split_support:
            split_units.append(unit)
            continue
        split_units.append(
            _rebuild_unit_from_indices(
                unit_id=f"{unit.unit_id}_up",
                status="mature" if len(positive) >= config.min_candidate_support else "candidate",
                member_indices=positive,
                items=items,
                latest_step=unit.updated_step,
                config=config,
            )
        )
        split_units.append(
            _rebuild_unit_from_indices(
                unit_id=f"{unit.unit_id}_down",
                status="mature" if len(negative) >= config.min_candidate_support else "candidate",
                member_indices=negative,
                items=items,
                latest_step=unit.updated_step,
                config=config,
            )
        )
        split_count += 1
    return split_units, split_count


def _prune_units(
    units: list[MemoryUnit],
    config: MemoryConfig,
    *,
    latest_step: int,
) -> tuple[list[MemoryUnit], int, list[MemoryUnit]]:
    refreshed: list[MemoryUnit] = []
    archived: list[MemoryUnit] = []
    for unit in units:
        utility = compute_unit_utility(
            support_count=unit.support_count,
            maturity_score=unit.maturity_score,
            outcome_dispersion=unit.outcome_dispersion,
            staleness_steps=latest_step - unit.updated_step,
            utility_decay=config.utility_decay,
        )
        refreshed.append(replace(unit, utility_score=utility))
    refreshed.sort(key=lambda unit: (unit.status != UnitStatus.MATURE.value, -unit.utility_score))
    kept: list[MemoryUnit] = []
    for unit in refreshed:
        stale_candidate = (
            unit.status == UnitStatus.CANDIDATE.value
            and unit.support_count < config.min_candidate_support
            and (latest_step - unit.updated_step) > config.max_candidate_age
        )
        stale_quarantined = (
            unit.status == UnitStatus.QUARANTINED.value
            and (latest_step - unit.updated_step) > config.max_candidate_age * 2
        )
        low_utility = unit.utility_score < config.min_utility
        overflow = len(kept) >= config.max_units
        if stale_candidate or stale_quarantined or low_utility or overflow:
            if config.archive_pruned:
                archived.append(replace(unit, status="archived"))
            continue
        kept.append(unit)
    return kept, len(refreshed) - len(kept), archived


def _outcome_consistency(left: MemoryUnit, right: MemoryUnit) -> float:
    horizon = max(left.path_stats.horizon_return_mean)
    left_return = left.path_stats.horizon_return_mean[horizon]
    right_return = right.path_stats.horizon_return_mean.get(horizon, 0.0)
    return_diff = abs(left_return - right_return)
    risk_diff = abs(left.risk_stats.max_adverse_mean - right.risk_stats.max_adverse_mean)
    return 1.0 / (1.0 + return_diff + risk_diff)


def _rebuild_unit_from_indices(
    *,
    unit_id: str,
    status: str,
    member_indices: list[int],
    items: list[MemoryItem],
    latest_step: int,
    config: MemoryConfig,
) -> MemoryUnit:
    member_items = [items[index] for index in member_indices]
    latents = torch.stack([item.latent.detach().cpu() for item in member_items], dim=0)
    prototype = latents.mean(dim=0)
    distances = torch.norm(latents - prototype, dim=-1)
    similarities = 1.0 / (1.0 + distances)
    path_stats = aggregate_path_statistics([item.outcome for item in member_items])
    risk_stats = aggregate_risk_statistics([item.outcome for item in member_items])
    conflict_rate = 1.0 - max(path_stats.direction_probabilities.values(), default=0.0)
    latent_dispersion = float(distances.mean().item())
    temp_unit = MemoryUnit(
        unit_id=unit_id,
        status=status,
        prototype=prototype,
        support_count=len(member_items),
        member_indices=list(member_indices),
        average_similarity=float(similarities.mean().item()),
        min_similarity=float(similarities.min().item()),
        max_similarity=float(similarities.max().item()),
        similarity_std=float(similarities.std(unbiased=False).item()) if similarities.numel() > 1 else 0.0,
        path_stats=path_stats,
        risk_stats=risk_stats,
        regime_composition={},
        symbol_composition={},
        session_composition={},
        utility_score=0.0,
        maturity_score=min(len(member_items) / max(config.min_candidate_support, 1), 1.0),
        latent_dispersion=latent_dispersion,
        outcome_dispersion=0.0,
        created_at=min(item.timestamp for item in member_items),
        updated_at=max(item.timestamp for item in member_items),
        created_step=min(member_indices),
        updated_step=max(member_indices),
        conflict_rate=conflict_rate,
    )
    outcome_dispersion = compute_unit_outcome_dispersion(temp_unit)
    utility = compute_unit_utility(
        support_count=temp_unit.support_count,
        maturity_score=temp_unit.maturity_score,
        outcome_dispersion=outcome_dispersion,
        staleness_steps=latest_step - temp_unit.updated_step,
        utility_decay=config.utility_decay,
    )
    return replace(temp_unit, outcome_dispersion=outcome_dispersion, utility_score=utility)
