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
    # Expected-value signals from the model's risk head (in basis points).
    # Default 0.0 means "unknown"; the EV gate in RiskManager treats zero-gain
    # proposals as negative-EV and rejects them.
    expected_gain_bps: float = 0.0
    expected_loss_bps: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: list[str]
    quantity: float = 0.0
    side: str | None = None
    target_position: float = 0.0
    client_order_id: str | None = None
