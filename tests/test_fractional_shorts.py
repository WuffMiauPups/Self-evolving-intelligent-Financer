"""Whole-share short constraint + staleness-vs-bar-period sanity.

Both found by the 2026-07-15 Alpaca paper run (MASTER_PLAN §3 ⟲1 step 5,
"paper-validate"):

1. The bot targeted a sub-share QQQ short (−0.412 shares). Alpaca rejects
   with 42210000 "fractional orders cannot be sold short"; the risk layer
   then re-approved the identical order every duplicate_order_window, so it
   resubmitted forever and the short never executed. Worse, the BACKTEST
   simulated those fills happily — every scorecard row banked short trades
   the live broker would refuse. The constraint therefore belongs in the
   shared risk path, not the live adapter.
2. ``risk.stale_data_sec`` below the bar period blocks the back half of
   every bar: 56% of decisions died on stale_data, which the ⟲1 runbook
   names as a failing verify criterion.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.config import RiskConfig, load_config
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager

T = datetime(2026, 7, 15, 14, 30, tzinfo=UTC)


def _portfolio(equity: float = 100_000.0) -> PortfolioState:
    return PortfolioState(cash=equity, equity=equity)


def _with_position(portfolio: PortfolioState, symbol: str, qty: float, price: float):
    from src.risk.portfolio_state import apply_fill

    apply_fill(portfolio, symbol=symbol, signed_delta=qty, fill_price=price, timestamp=T)
    position = portfolio.get_position(symbol)
    if position is not None:
        position.market_price = price
    return portfolio


def _proposal(target_fraction: float, action: int, *, price: float = 719.0, request_id: str = "r1"):
    return TradeProposal(
        symbol="QQQ",
        timestamp=T,
        action=action,
        confidence=0.9,
        target_fraction=target_fraction,
        price=price,
        spread_bps=1.0,
        request_id=request_id,
        expected_gain_bps=100.0,
        expected_loss_bps=10.0,
    )


def _manager() -> RiskManager:
    # Cooldown/dup windows off: this suite is about sizing, not throttling.
    config = RiskConfig(cooldown_sec=0, duplicate_order_window_sec=0)
    return RiskManager(config)


# ── 1. Sub-share shorts are unexecutable, not retryable ─────────────────


def test_sub_share_short_target_is_refused_with_named_reason() -> None:
    """The exact 2026-07-15 QQQ case: −0.3% of 100k at $719 = −0.417 shares."""
    manager = _manager()
    decision = manager.evaluate(
        _proposal(target_fraction=0.003, action=-1), _portfolio(), evaluation_time=T
    )
    assert decision.approved is False
    assert "fractional_short_unsupported" in decision.reasons
    assert decision.quantity == pytest.approx(0.0)


def test_whole_share_short_target_is_approved_and_floored() -> None:
    # 2% of 100k at $719 = 2.78 shares -> short exactly 2 (never 2.78).
    manager = _manager()
    decision = manager.evaluate(
        _proposal(target_fraction=0.02, action=-1), _portfolio(), evaluation_time=T
    )
    assert decision.approved is True
    assert decision.target_position == pytest.approx(-2.0)
    assert decision.quantity == pytest.approx(2.0)
    assert float(decision.quantity).is_integer()


def test_crossing_from_fractional_long_flattens_first() -> None:
    """A fractional long -> short would be ONE fractional sell-short order.
    Flatten instead; the short opens from flat on a later tick."""
    manager = _manager()
    portfolio = _with_position(_portfolio(), "QQQ", qty=0.4, price=719.0)
    decision = manager.evaluate(
        _proposal(target_fraction=0.02, action=-1), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.target_position == pytest.approx(0.0), "must flatten, not cross"
    assert decision.side == "sell"
    assert decision.quantity == pytest.approx(0.4)
    # Approved-with-tag, so the log says WHY this is a flatten and not the
    # requested short.
    assert "flatten_before_short" in decision.reasons


def test_whole_share_long_to_short_flip_stays_a_flip() -> None:
    """+50 -> -20 submits a sell of 70.0 — a whole order the broker accepts.
    It must NOT be softened into a flatten: the halt gates (daily loss,
    kill switch) must still see it as the fresh-exposure flip it is, which
    is what tests/test_risk_correctness.py::test_tripped_daily_loss_blocks_
    side_flip depends on."""
    manager = _manager()
    portfolio = _with_position(_portfolio(), "QQQ", qty=50.0, price=100.0)
    decision = manager.evaluate(
        _proposal(target_fraction=0.02, action=-1, price=100.0), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.target_position == pytest.approx(-20.0), "flip must survive as a flip"
    assert decision.quantity == pytest.approx(70.0)
    assert "flatten_before_short" not in decision.reasons


def test_short_from_flat_after_flatten_is_whole_share() -> None:
    manager = _manager()
    decision = manager.evaluate(
        _proposal(target_fraction=0.02, action=-1), _portfolio(), evaluation_time=T
    )
    assert decision.target_position == pytest.approx(-2.0)


def test_fractional_buy_crossing_short_to_long_covers_first() -> None:
    """The live 2026-07-15 17:03 loop: short 3 NVDA, model wants long 3.933
    -> a 6.933-share fractional buy crossing zero; Alpaca rejects fractional
    side-flips (40310000) and the intent retried every duplicate window.
    Must cover to flat (whole) instead; the long opens from flat later."""
    manager = _manager()
    portfolio = _with_position(_portfolio(), "QQQ", qty=-3.0, price=100.0)
    decision = manager.evaluate(
        _proposal(target_fraction=0.003933, action=1, price=100.0), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.target_position == pytest.approx(0.0), "must cover to flat, not cross"
    assert decision.side == "buy"
    assert decision.quantity == pytest.approx(3.0)
    assert "flatten_before_long" in decision.reasons


def test_whole_share_short_to_long_flip_stays_a_flip() -> None:
    """Whole-share side flips are broker-valid (only FRACTIONAL orders
    cannot flip side): short 3 -> long 5 = one buy of 8.0."""
    manager = _manager()
    portfolio = _with_position(_portfolio(), "QQQ", qty=-3.0, price=100.0)
    decision = manager.evaluate(
        _proposal(target_fraction=0.005, action=1, price=100.0), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.target_position == pytest.approx(5.0)
    assert decision.quantity == pytest.approx(8.0)
    assert "flatten_before_long" not in decision.reasons


def test_fractional_long_targets_are_untouched() -> None:
    """Longs may be fractional — only shorts carry the constraint."""
    manager = _manager()
    decision = manager.evaluate(
        _proposal(target_fraction=0.003, action=1), _portfolio(), evaluation_time=T
    )
    assert decision.approved is True
    assert decision.quantity == pytest.approx(0.417, abs=1e-3)
    assert not float(decision.quantity).is_integer()


def test_fractional_sell_closing_a_long_is_untouched() -> None:
    """Selling to flat is allowed fractionally (it never goes short)."""
    manager = _manager()
    portfolio = _with_position(_portfolio(), "QQQ", qty=0.582, price=719.0)
    decision = manager.evaluate(
        _proposal(target_fraction=0.0, action=0), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.side == "sell"
    assert decision.quantity == pytest.approx(0.582)


def test_opt_in_flag_restores_fractional_shorts() -> None:
    config = RiskConfig(
        cooldown_sec=0, duplicate_order_window_sec=0, fractional_shorts_supported=True
    )
    decision = RiskManager(config).evaluate(
        _proposal(target_fraction=0.003, action=-1), _portfolio(), evaluation_time=T
    )
    assert decision.approved is True
    assert decision.target_position == pytest.approx(-0.417, abs=1e-3)


def test_default_is_alpaca_reality() -> None:
    assert RiskConfig().fractional_shorts_supported is False
    assert load_config("configs/default.yaml").risk.fractional_shorts_supported is False


# ── 2. Staleness must not fire on normal in-bar age ─────────────────────


def _bar_seconds(timeframe: str) -> int:
    from src.data.bar_aggregator import timeframe_delta

    return int(timeframe_delta(timeframe).total_seconds())


def test_paper_stale_threshold_exceeds_bar_period() -> None:
    """A bar timestamp ages to one full bar period by bucket close; a
    threshold below that blocks the back half of every bar."""
    config = load_config("configs/default.yaml", "configs/paper_trade.yaml")
    bar_seconds = _bar_seconds(config.runtime.timeframe)
    assert config.risk.stale_data_sec > bar_seconds, (
        f"risk.stale_data_sec={config.risk.stale_data_sec} <= bar period "
        f"{bar_seconds}s: the runtime would block ~"
        f"{100 * (1 - config.risk.stale_data_sec / bar_seconds):.0f}% of decisions "
        f"on stale_data (the ⟲1 runbook's failing verify criterion)."
    )


def test_default_stack_stale_threshold_exceeds_bar_period() -> None:
    config = load_config("configs/default.yaml")
    assert config.risk.stale_data_sec > _bar_seconds(config.runtime.timeframe)
