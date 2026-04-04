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
