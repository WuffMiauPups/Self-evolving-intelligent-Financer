"""Deterministic P&L position exits — the position guard.

Every synced bar, held positions are checked against unrealized-P&L
thresholds (``risk.position_stop_loss_pct`` / ``position_take_profit_pct``);
a breach originates a FLATTEN proposal (action=0, target 0) that rides the
NORMAL shared risk path:

- risk-reducing → exempt from kill-switch and daily-loss halts;
- full flatten → exempt from the min-notional deadband;
- action=0 → the EV gate does not apply;
- ``request_id`` embeds the bar bucket, so a blocked submission (e.g.
  spread_too_wide) retries at most once per bar, not per poll;
- after the exit fill, ``cooldown_sec`` blocks an instant model re-entry.

The guard lives in the shared risk layer and is wired into BOTH the
backtest engine and the live tick processor (§10.4): a stop the live
path would honor must cost the same in the backtest, or stop tuning is
fiction. Off by default (both thresholds 0.0) — pick values from a
backtest sweep before enabling live.

P&L is measured against the position's entry price, sign-adjusted for
shorts: long +2% and short −2% price moves are both "+2% P&L" here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.risk.constraints import TradeProposal

_EPS_QTY = 1.0e-9


@dataclass(frozen=True)
class GuardSnapshot:
    """Minimal snapshot stand-in for dispatching a guard exit through the
    same submission code path as model proposals."""

    symbol: str
    timestamp: datetime
    last_price: float
    spread_bps: float = 0.0


def position_pnl_pct(qty: float, avg_price: float, market_price: float) -> float:
    """Signed unrealized P&L fraction vs entry, positive = in profit."""
    if avg_price <= 0.0 or market_price <= 0.0:
        return 0.0
    move = (market_price - avg_price) / avg_price
    return move if qty > 0 else -move


def check_position_exits(
    portfolio,
    config,
    *,
    timestamp: datetime,
    spread_bps_by_symbol: dict[str, float] | None = None,
) -> list[tuple[GuardSnapshot, TradeProposal, str]]:
    """(snapshot, flatten proposal, reason) per position breaching a threshold.

    ``config`` is a RiskConfig; both thresholds at 0.0 make this a no-op.
    """
    stop = float(getattr(config, "position_stop_loss_pct", 0.0))
    take = float(getattr(config, "position_take_profit_pct", 0.0))
    if stop <= 0.0 and take <= 0.0:
        return []
    spreads = spread_bps_by_symbol or {}
    exits: list[tuple[GuardSnapshot, TradeProposal, str]] = []
    bucket = timestamp.strftime("%Y%m%d%H%M")
    for symbol, position in portfolio.positions.items():
        qty = float(position.qty)
        if abs(qty) < _EPS_QTY:
            continue
        pnl = position_pnl_pct(qty, float(position.avg_price), float(position.market_price))
        if stop > 0.0 and pnl <= -stop:
            reason = "position_stop_loss"
        elif take > 0.0 and pnl >= take:
            reason = "position_take_profit"
        else:
            continue
        snapshot = GuardSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            last_price=float(position.market_price),
            spread_bps=float(spreads.get(symbol, 0.0)),
        )
        proposal = TradeProposal(
            symbol=symbol,
            timestamp=timestamp,
            action=0,
            confidence=1.0,
            target_fraction=0.0,
            price=float(position.market_price),
            spread_bps=snapshot.spread_bps,
            request_id=f"pguard-{symbol}-{bucket}-{reason}",
            metadata={"position_guard": reason, "pnl_pct": round(pnl, 6)},
            strategy_id="position_guard",
        )
        exits.append((snapshot, proposal, reason))
    return exits


GUARD_DIAGNOSTICS: dict[str, object] = {
    "branch_confidences": {},
    "soma_weights": [],
    "soma_activation": 1.0,
    "position_guard": True,
}
