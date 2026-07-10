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


def action_annotation(proposal: "TradeProposal", decision: RiskDecision) -> str:
    """Compact executed-action record for memory items and event logs.

    Outcome-driven learning must credit the EXECUTED (risk-clamped)
    action, never the model's raw intent (MASTER_PLAN 4.5.2).
    """
    reasons = [str(r) for r in decision.reasons]
    approved = bool(decision.approved)
    return (
        f"action={int(proposal.action)}"
        f" approved={int(approved)}"
        f" side={(decision.side or '-') if approved else '-'}"
        f" qty={float(decision.quantity) if approved else 0.0:.4f}"
        f" resized={int('resized_to_cap' in reasons)}"
    )
