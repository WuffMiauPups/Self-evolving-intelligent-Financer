"""Churn deadband: risk.min_trade_notional (2026-07-15 paper run).

Re-deciding every tick against an unchanged 5-min bar produced dust
rebalances — the observed case was a 0.008-share NVDA trim (~$1.69) sixty
seconds after entry, paying spread and fees for a sub-dollar adjustment.
Orders below the notional floor are rejected with a named reason; full
flattens are exempt so dust positions always stay closeable.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.config import RiskConfig, load_config
from src.core.exceptions import ConfigurationError
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState, apply_fill
from src.risk.risk_manager import RiskManager

T = datetime(2026, 7, 15, 15, 0, tzinfo=UTC)


def _portfolio(position_qty: float = 0.0, price: float = 211.0) -> PortfolioState:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    if position_qty:
        apply_fill(
            portfolio, symbol="NVDA", signed_delta=position_qty, fill_price=price, timestamp=T
        )
        portfolio.get_position("NVDA").market_price = price
    return portfolio


def _proposal(target_fraction: float, action: int, *, price: float = 211.0):
    return TradeProposal(
        symbol="NVDA",
        timestamp=T,
        action=action,
        confidence=0.9,
        target_fraction=target_fraction,
        price=price,
        spread_bps=1.0,
        request_id="mtn-1",
        expected_gain_bps=100.0,
        expected_loss_bps=10.0,
    )


def _manager(min_trade_notional: float = 100.0) -> RiskManager:
    return RiskManager(
        RiskConfig(
            cooldown_sec=0,
            duplicate_order_window_sec=0,
            min_trade_notional=min_trade_notional,
        )
    )


def test_observed_dust_trim_is_rejected() -> None:
    """The exact 2026-07-15 case: holding 5.534 NVDA, target drifts to
    5.526 on an unchanged bar -> a 0.008-share (~$1.69) trim."""
    manager = _manager()
    portfolio = _portfolio(position_qty=5.534)
    # target fraction whose sized qty is ~5.526: 5.526 * 211 / 100k
    decision = manager.evaluate(
        _proposal(target_fraction=0.01166, action=1), portfolio, evaluation_time=T
    )
    assert decision.approved is False
    assert "below_min_trade_notional" in decision.reasons


def test_full_flatten_is_exempt_even_at_dust_size() -> None:
    """Dust positions must always be closeable (target 0 bypasses the gate)."""
    manager = _manager()
    portfolio = _portfolio(position_qty=0.008)
    decision = manager.evaluate(
        _proposal(target_fraction=0.0, action=0), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.side == "sell"
    assert decision.quantity == pytest.approx(0.008)


def test_normal_sized_order_passes() -> None:
    manager = _manager()
    decision = manager.evaluate(
        _proposal(target_fraction=0.01, action=1), _portfolio(), evaluation_time=T
    )
    assert decision.approved is True
    assert decision.quantity * 211.0 > 100.0


def test_zero_disables_the_gate() -> None:
    manager = _manager(min_trade_notional=0.0)
    portfolio = _portfolio(position_qty=5.534)
    decision = manager.evaluate(
        _proposal(target_fraction=0.01166, action=1), portfolio, evaluation_time=T
    )
    assert "below_min_trade_notional" not in decision.reasons


def test_negative_threshold_refused_at_load() -> None:
    with pytest.raises(ConfigurationError, match="min_trade_notional"):
        RiskConfig(min_trade_notional=-1.0)


def test_default_stack_carries_the_deadband() -> None:
    assert load_config("configs/default.yaml").risk.min_trade_notional == pytest.approx(100.0)


def test_failed_submit_pops_attribution_maps_source_contract() -> None:
    """A failed submission never reaches the fill-capture path that consumes
    strategy/clamped map entries — the not-accepted branch must pop them
    (the 2026-07-15 fractional-short retry loop leaked 8+ in 10 minutes)."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "execution" / "tick_processor.py"
    ).read_text(encoding="utf-8")
    not_accepted_branch = source.split("if not result.accepted:", 1)[1].split("continue", 1)[0]
    assert "strategy_by_client_order_id.pop(client_order_id" in not_accepted_branch
    assert "clamped_by_client_order_id.pop(client_order_id" in not_accepted_branch
