from __future__ import annotations

import math

from src.memory.schemas import MemoryHealthReport, MemoryUnit, UnitStatus


def compute_unit_outcome_dispersion(unit: MemoryUnit) -> float:
    primary_horizon = max(unit.path_stats.horizon_return_p75)
    return abs(unit.path_stats.horizon_return_p75[primary_horizon] - unit.path_stats.horizon_return_p25[primary_horizon])


def compute_unit_utility(
    *,
    support_count: int,
    maturity_score: float,
    outcome_dispersion: float,
    staleness_steps: int,
    utility_decay: float,
) -> float:
    stability = 1.0 / (1.0 + max(outcome_dispersion, 0.0))
    support_term = math.log1p(max(support_count, 0))
    decay = utility_decay ** max(staleness_steps, 0)
    return float(support_term * max(maturity_score, 0.0) * stability * decay)


def compute_memory_health(
    units: list[MemoryUnit],
    *,
    archived_count: int,
    latest_step: int,
    weak_match_rate: float = 0.0,
    unknown_state_rate: float = 0.0,
    topology_density: float = 0.0,
    graph_edge_count: int = 0,
    average_edge_age: float = 0.0,
    created_units: int = 0,
    updated_units: int = 0,
) -> MemoryHealthReport:
    if not units:
        return MemoryHealthReport(
            total_units=0,
            mature_units=0,
            candidate_units=0,
            archived_units=archived_count,
            average_support_count=0.0,
            average_utility=0.0,
            stale_unit_ratio=0.0,
            average_compactness=0.0,
            weak_match_rate=weak_match_rate,
            unknown_state_rate=unknown_state_rate,
            mature_candidate_ratio=0.0,
            average_retrieval_hits=0.0,
            topology_density=topology_density,
        )
    mature_units = [unit for unit in units if unit.status == UnitStatus.MATURE.value]
    candidate_units = [unit for unit in units if unit.status == UnitStatus.CANDIDATE.value]
    quarantined_units = [unit for unit in units if unit.status == UnitStatus.QUARANTINED.value]
    stale_count = sum(1 for unit in units if latest_step - unit.updated_step > max(1, unit.support_count))
    average_support = sum(unit.support_count for unit in units) / len(units)
    average_utility = sum(unit.utility_score for unit in units) / len(units)
    average_compactness = sum(max(0.0, 1.0 - unit.latent_dispersion) for unit in units) / len(units)
    average_retrieval_hits = sum(unit.retrieval_hits for unit in units) / len(units)
    mature_candidate_ratio = len(mature_units) / max(len(candidate_units), 1)

    # SatSOM / GNG diagnostics
    average_saturation = sum(unit.saturation_score for unit in units) / len(units)
    average_plasticity = sum(unit.plasticity_factor for unit in units) / len(units)
    saturated_unit_count = sum(1 for unit in units if unit.saturation_score > 0.8)
    high_plasticity_unit_count = sum(1 for unit in units if unit.plasticity_factor > 0.8)
    conflict_heavy_unit_count = sum(1 for unit in units if unit.recent_conflict_rate > 0.4)
    average_conflict_rate = sum(unit.conflict_rate for unit in units) / len(units)
    create_vs_update_ratio = created_units / max(updated_units, 1) if (created_units + updated_units) > 0 else 0.0

    return MemoryHealthReport(
        total_units=len(units),
        mature_units=len(mature_units),
        candidate_units=len(candidate_units),
        archived_units=archived_count,
        average_support_count=average_support,
        average_utility=average_utility,
        stale_unit_ratio=stale_count / len(units),
        average_compactness=average_compactness,
        weak_match_rate=weak_match_rate,
        unknown_state_rate=unknown_state_rate,
        mature_candidate_ratio=mature_candidate_ratio,
        average_retrieval_hits=average_retrieval_hits,
        topology_density=topology_density,
        average_saturation=average_saturation,
        average_plasticity=average_plasticity,
        saturated_unit_count=saturated_unit_count,
        high_plasticity_unit_count=high_plasticity_unit_count,
        conflict_heavy_unit_count=conflict_heavy_unit_count,
        average_conflict_rate=average_conflict_rate,
        graph_edge_count=graph_edge_count,
        average_edge_age=average_edge_age,
        create_vs_update_ratio=create_vs_update_ratio,
        quarantined_units=len(quarantined_units),
    )
