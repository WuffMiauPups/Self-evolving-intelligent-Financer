from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ModelDecisionEvent:
    timestamp: datetime
    symbol: str
    confidence: float
    target_fraction: float
    soma_activation: float


@dataclass(frozen=True)
class RiskEvent:
    timestamp: datetime
    symbol: str
    approved: bool
    reasons: list[str]


@dataclass(frozen=True)
class OrderEvent:
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    status: str
    reason_code: str | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StartupCheckEvent:
    timestamp: datetime
    name: str
    passed: bool
    reason_code: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionGuardEvent:
    timestamp: datetime
    symbol: str
    allowed: bool
    reason_code: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconciliationEvent:
    timestamp: datetime
    safe_to_trade: bool
    reason_code: str
    severity: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryBuildEvent:
    timestamp: datetime
    item_count: int
    cluster_count: int
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRetrievalEvent:
    timestamp: datetime
    symbol: str
    weak_match: bool
    no_trade_suggested: bool
    reason_code: str
    match_strength: float
    average_similarity: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryUpdateEvent:
    timestamp: datetime
    symbol: str
    finalized_count: int
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionFusionEvent:
    """Legacy event schema — the decision-fusion layer was removed in the
    Great Simplification; nothing emits this anymore. Kept so old event
    logs remain parseable."""
    timestamp: datetime
    symbol: str
    support_level: str
    reason_codes: list[str]
    size_multiplier: float
    confidence_adjustment: float
    unknown_state_flag: bool
    topology_warning_flag: bool
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionEvaluationEvent:
    """Legacy event schema — offline fusion evaluation was removed with the
    decision-fusion layer; nothing emits this anymore. Kept so old event
    logs remain parseable."""
    timestamp: datetime
    evaluated_samples: int
    support_count: int
    neutral_count: int
    dampen_count: int
    veto_count: int
    avg_future_return_all: float
    directional_hit_rate_all: float
    support_vs_outcome_correlation: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyGrowthEvent:
    """Structured event emitted after topology growth runs."""
    timestamp: datetime
    insertions_applied: int
    insertions_skipped: int
    split_candidates_found: int
    max_units_reached: bool
    growth_rate_capped: bool
    total_units_after: int
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UnitInsertionEvent:
    """Structured event emitted when a new unit is inserted via topology growth."""
    timestamp: datetime
    unit_id: str
    source_reason: str
    novelty_score: float
    topology_sparsity: float
    sample_count: int
    score: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitCandidateEvent:
    """Structured event emitted when a unit is flagged as split candidate."""
    timestamp: datetime
    unit_id: str
    reason: str
    conflict_rate: float
    outcome_dispersion: float
    support_count: int
    score: float
    details: dict[str, object] = field(default_factory=dict)
