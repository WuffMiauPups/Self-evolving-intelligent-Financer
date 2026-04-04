from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    timestamp: datetime
    action: int
    confidence: float
    target_fraction: float
    price: float
    spread_bps: float
    request_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: list[str]
    quantity: float = 0.0
    side: str | None = None
    target_position: float = 0.0
    client_order_id: str | None = None
