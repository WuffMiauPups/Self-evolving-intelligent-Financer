"""Risk-layer correctness tests (master plan workstream 1E).

Covers: daily-PnL day-boundary reset, risk-reducing exemption from halt
reasons, post-trade cap evaluation, resize-to-cap semantics, EV spread cost,
sizer/portfolio hardening, gate breadth, and the learned-proposes /
risk-disposes safety invariant.
"""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from src.core.config import RiskConfig
from src.execution.reason_codes import ReasonCode
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState, PositionState
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager

# Tuesday 2026-07-07 11:00 ET — inside the "regular" session.
T = datetime(2026, 7, 7, 15, 0, tzinfo=UTC)


def _proposal(**overrides) -> TradeProposal:
    """EV comfortably positive: 0.9*30 - 0.1*10 - (1+2+1) = 22 bps > 0."""
    base = dict(
        symbol="AAPL",
        timestamp=T,
        action=1,
        confidence=0.9,
        target_fraction=0.02,
        price=100.0,
        spread_bps=1.0,
        request_id="rc-default",
        expected_gain_bps=30.0,
        expected_loss_bps=10.0,
    )
    base.update(overrides)
    return TradeProposal(**base)


def _portfolio(**overrides) -> PortfolioState:
    base = dict(cash=100_000.0, equity=100_000.0)
    base.update(overrides)
    return PortfolioState(**base)


def _with_position(portfolio: PortfolioState, symbol: str, qty: float, price: float = 100.0) -> PortfolioState:
    portfolio.positions[symbol] = PositionState(
        symbol=symbol,
        qty=qty,
        avg_price=price,
        market_price=price,
        entry_time=T - timedelta(hours=1),
    )
    return portfolio


# ---------------------------------------------------------------------------
# Task 1: daily PnL resets at US/Eastern trading-day boundaries
# ---------------------------------------------------------------------------


def test_daily_pnl_persists_within_same_eastern_day() -> None:
    portfolio = _portfolio()
    portfolio.mark_to_market({}, datetime(2026, 7, 6, 18, 0, tzinfo=UTC))  # 14:00 ET Jul 6
    portfolio.daily_pnl = -400.0
    portfolio.mark_to_market({}, datetime(2026, 7, 6, 20, 0, tzinfo=UTC))  # 16:00 ET Jul 6
    assert portfolio.daily_pnl == pytest.approx(-400.0)


def test_daily_pnl_resets_on_eastern_day_rollover() -> None:
    portfolio = _portfolio()
    portfolio.mark_to_market({}, datetime(2026, 7, 6, 18, 0, tzinfo=UTC))
    portfolio.daily_pnl = -400.0
    portfolio.mark_to_market({}, datetime(2026, 7, 7, 13, 30, tzinfo=UTC))  # 09:30 ET Jul 7
    assert portfolio.daily_pnl == 0.0


def test_daily_pnl_does_not_reset_at_utc_midnight() -> None:
    """02:00 UTC Jul 7 is still 22:00 ET Jul 6 — no reset until the ET date turns."""
    portfolio = _portfolio()
    portfolio.mark_to_market({}, datetime(2026, 7, 6, 18, 0, tzinfo=UTC))
    portfolio.daily_pnl = -400.0
    portfolio.mark_to_market({}, datetime(2026, 7, 7, 2, 0, tzinfo=UTC))
    assert portfolio.daily_pnl == pytest.approx(-400.0)


def test_daily_loss_limit_releases_after_day_rollover() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    portfolio = _portfolio()
    portfolio.mark_to_market({}, datetime(2026, 7, 6, 18, 0, tzinfo=UTC))
    portfolio.daily_pnl = -(config.max_daily_loss + 1.0)
    blocked = manager.evaluate(
        _proposal(timestamp=datetime(2026, 7, 6, 18, 0, tzinfo=UTC), request_id="rollover-1"),
        portfolio,
        evaluation_time=datetime(2026, 7, 6, 18, 0, tzinfo=UTC),
    )
    assert "daily_loss_limit" in blocked.reasons
    portfolio.mark_to_market({}, T)  # next ET trading day
    released = manager.evaluate(
        _proposal(request_id="rollover-2"), portfolio, evaluation_time=T
    )
    assert released.approved is True


# ---------------------------------------------------------------------------
# Task 2: risk-reducing orders are exempt from halt reasons
# ---------------------------------------------------------------------------


def _tripped_daily_loss(config: RiskConfig) -> tuple[RiskManager, PortfolioState]:
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)
    portfolio.daily_pnl = -(config.max_daily_loss + 100.0)
    return manager, portfolio


def test_tripped_daily_loss_blocks_new_entry() -> None:
    manager, portfolio = _tripped_daily_loss(RiskConfig())
    decision = manager.evaluate(
        _proposal(symbol="MSFT", request_id="dl-entry"), portfolio, evaluation_time=T
    )
    assert decision.approved is False
    assert "daily_loss_limit" in decision.reasons


def test_tripped_daily_loss_approves_flatten() -> None:
    manager, portfolio = _tripped_daily_loss(RiskConfig())
    decision = manager.evaluate(
        _proposal(action=0, target_fraction=0.0, request_id="dl-flatten"),
        portfolio,
        evaluation_time=T,
    )
    assert decision.approved is True
    assert decision.side == "sell"
    assert decision.quantity == pytest.approx(50.0)
    assert decision.target_position == pytest.approx(0.0)


def test_tripped_daily_loss_approves_partial_reduce() -> None:
    manager, portfolio = _tripped_daily_loss(RiskConfig())
    # target 20 shares long < current 50 long: same sign, strictly smaller.
    decision = manager.evaluate(
        _proposal(request_id="dl-reduce"), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.side == "sell"
    assert decision.quantity == pytest.approx(30.0)
    assert decision.target_position == pytest.approx(20.0)


def test_tripped_daily_loss_blocks_side_flip() -> None:
    manager, portfolio = _tripped_daily_loss(RiskConfig())
    # Long 50 -> short 20 reduces gross qty but opens fresh exposure: blocked.
    decision = manager.evaluate(
        _proposal(action=-1, request_id="dl-flip"), portfolio, evaluation_time=T
    )
    assert decision.approved is False
    assert "daily_loss_limit" in decision.reasons


def test_kill_switch_blocks_entry_but_not_flatten() -> None:
    manager = RiskManager(RiskConfig())
    manager.emergency_stop()
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)
    entry = manager.evaluate(
        _proposal(symbol="MSFT", request_id="ks-entry"), portfolio, evaluation_time=T
    )
    assert entry.approved is False
    assert "kill_switch" in entry.reasons
    flatten = manager.evaluate(
        _proposal(action=0, target_fraction=0.0, request_id="ks-flatten"),
        portfolio,
        evaluation_time=T,
    )
    assert flatten.approved is True


def test_rolling_drawdown_blocks_entry_but_not_reduce() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)
    portfolio.pnl_history.extend([100_000.0, 85_000.0])  # -15% < -10% limit
    entry = manager.evaluate(
        _proposal(symbol="MSFT", request_id="dd-entry"), portfolio, evaluation_time=T
    )
    assert entry.approved is False
    assert "rolling_drawdown_limit" in entry.reasons
    reduce = manager.evaluate(
        _proposal(request_id="dd-reduce"), portfolio, evaluation_time=T
    )
    assert reduce.approved is True


def test_aggregate_cap_breach_still_allows_reduce() -> None:
    # per-position cap lowered too: aggregate >= position is now validated.
    config = RiskConfig(max_aggregate_notional=1_000.0, max_position_notional=1_000.0)
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)  # 5000 notional > cap
    entry = manager.evaluate(
        _proposal(symbol="MSFT", request_id="agg-entry"), portfolio, evaluation_time=T
    )
    assert entry.approved is False
    assert "max_aggregate_notional" in entry.reasons
    reduce = manager.evaluate(
        _proposal(request_id="agg-reduce"), portfolio, evaluation_time=T
    )
    assert reduce.approved is True


def test_flatten_still_subject_to_staleness_and_session() -> None:
    config = RiskConfig(allowed_sessions=["regular"])
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)
    stale = manager.evaluate(
        _proposal(action=0, target_fraction=0.0, request_id="fl-stale"),
        portfolio,
        evaluation_time=T + timedelta(seconds=config.stale_data_sec + 1),
    )
    assert stale.approved is False
    assert ReasonCode.STALE_DATA in stale.reasons
    premarket = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)  # 08:00 ET
    off_session = manager.evaluate(
        _proposal(action=0, target_fraction=0.0, timestamp=premarket, request_id="fl-session"),
        portfolio,
        evaluation_time=premarket,
    )
    assert off_session.approved is False
    assert ReasonCode.SESSION_DISALLOWED in off_session.reasons


# ---------------------------------------------------------------------------
# Task 3: post-trade cap evaluation (aggregate notional / leverage)
# ---------------------------------------------------------------------------


def test_aggregate_notional_includes_proposed_order() -> None:
    config = RiskConfig(max_aggregate_notional=6_000.0)
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "MSFT", qty=40.0)  # 4000 notional
    # +3000 proposed -> prospective 7000 > 6000: rejected pre-trade would pass.
    decision = manager.evaluate(
        _proposal(target_fraction=0.03, request_id="agg-over"), portfolio, evaluation_time=T
    )
    assert decision.approved is False
    assert "max_aggregate_notional" in decision.reasons


def test_aggregate_notional_boundary_exactly_at_cap_approves() -> None:
    config = RiskConfig(max_aggregate_notional=6_000.0)
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "MSFT", qty=40.0)
    # +2000 proposed -> prospective exactly 6000: not a breach (strict >).
    decision = manager.evaluate(
        _proposal(target_fraction=0.02, request_id="agg-at"), portfolio, evaluation_time=T
    )
    assert decision.approved is True


def test_prospective_notional_replaces_current_position_contribution() -> None:
    # per-position cap lowered too (>= the 4000 target so no resize occurs):
    # aggregate >= position is now validated at config load.
    config = RiskConfig(max_aggregate_notional=4_500.0, max_position_notional=4_500.0)
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "AAPL", qty=30.0)  # 3000 notional
    # Target 4000 REPLACES the 3000 (prospective 4000, not 3000+4000=7000).
    decision = manager.evaluate(
        _proposal(target_fraction=0.04, request_id="agg-replace"), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert decision.quantity == pytest.approx(10.0)


def test_leverage_ratio_includes_proposed_order() -> None:
    config = RiskConfig(max_leverage_ratio=0.05)
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "MSFT", qty=40.0)  # 4000/100k = 0.04
    over = manager.evaluate(
        _proposal(target_fraction=0.02, request_id="lev-over"), portfolio, evaluation_time=T
    )  # prospective 6000/100k = 0.06 > 0.05
    assert over.approved is False
    assert "max_leverage_ratio" in over.reasons
    at_cap = manager.evaluate(
        _proposal(target_fraction=0.01, request_id="lev-at"), portfolio, evaluation_time=T
    )  # prospective 5000/100k = 0.05, not a breach
    assert at_cap.approved is True


# ---------------------------------------------------------------------------
# Task 4: resize path (soft position-notional cap)
# ---------------------------------------------------------------------------


def test_oversized_proposal_is_resized_to_cap_and_approved() -> None:
    config = RiskConfig()  # max_position_notional = 5000
    manager = RiskManager(config)
    portfolio = _portfolio()
    # Requested 20000 notional; cap shrinks it to 5000 -> 50 shares @ 100.
    decision = manager.evaluate(
        _proposal(target_fraction=0.2, request_id="resize-1"), portfolio, evaluation_time=T
    )
    assert decision.approved is True
    assert "resized_to_cap" in decision.reasons
    assert decision.quantity == pytest.approx(50.0)
    assert decision.target_position == pytest.approx(50.0)


def test_resized_proposal_failing_other_gates_still_rejects() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    portfolio = _portfolio()
    decision = manager.evaluate(
        _proposal(target_fraction=0.2, spread_bps=config.max_spread_bps, request_id="resize-2"),
        portfolio,
        evaluation_time=T,
    )
    assert decision.approved is False
    assert "spread_too_wide" in decision.reasons
    assert "resized_to_cap" not in decision.reasons


def test_within_cap_approval_carries_empty_reasons() -> None:
    manager = RiskManager(RiskConfig())
    decision = manager.evaluate(
        _proposal(request_id="resize-3"), _portfolio(), evaluation_time=T
    )
    assert decision.approved is True
    assert decision.reasons == []


# ---------------------------------------------------------------------------
# Task 5: EV gate includes spread cost
# ---------------------------------------------------------------------------


def test_ev_gate_charges_full_spread_round_trip() -> None:
    config = RiskConfig()  # fee 1 + slippage 2
    manager = RiskManager(config)
    portfolio = _portfolio()
    # confidence 1.0, gain 10: EV = 10 - (1+2+spread).
    rejected = manager.evaluate(
        _proposal(confidence=1.0, expected_gain_bps=10.0, expected_loss_bps=0.0,
                  spread_bps=8.0, request_id="ev-spread-reject"),
        portfolio,
        evaluation_time=T,
    )  # 10 - 11 = -1
    assert rejected.approved is False
    assert "negative_ev" in rejected.reasons
    approved = manager.evaluate(
        _proposal(confidence=1.0, expected_gain_bps=10.0, expected_loss_bps=0.0,
                  spread_bps=5.0, request_id="ev-spread-approve"),
        portfolio,
        evaluation_time=T,
    )  # 10 - 8 = 2
    assert approved.approved is True


# ---------------------------------------------------------------------------
# Task 6a: sizer hardening
# ---------------------------------------------------------------------------


def test_sizer_returns_zero_for_non_positive_equity() -> None:
    sizer = PositionSizer(quantity_precision=0.001)
    for equity in (0.0, -50_000.0):
        qty = sizer.calculate_target_quantity(
            equity=equity, price=100.0, target_fraction=0.05, max_notional=5_000.0
        )
        assert qty == 0.0


def test_sizer_floors_to_quantity_precision() -> None:
    whole_share = PositionSizer(quantity_precision=1.0)
    assert whole_share.calculate_target_quantity(
        equity=100_000.0, price=100.0, target_fraction=0.0155, max_notional=50_000.0
    ) == pytest.approx(15.0)
    fractional = PositionSizer(quantity_precision=0.001)
    assert fractional.calculate_target_quantity(
        equity=100_000.0, price=100.0, target_fraction=0.0155, max_notional=50_000.0
    ) == pytest.approx(15.5)


def test_negative_equity_never_flips_order_direction() -> None:
    manager = RiskManager(RiskConfig())
    portfolio = _portfolio(cash=-10_000.0, equity=-10_000.0)
    decision = manager.evaluate(
        _proposal(request_id="neg-equity"), portfolio, evaluation_time=T
    )
    assert decision.approved is False
    assert "no_position_change" in decision.reasons


# ---------------------------------------------------------------------------
# Task 6b: one pnl_history sample per unique timestamp
# ---------------------------------------------------------------------------


def test_mark_to_market_same_timestamp_replaces_last_sample() -> None:
    portfolio = _with_position(_portfolio(), "AAPL", qty=10.0)
    portfolio.mark_to_market({"AAPL": 100.0}, T)
    portfolio.mark_to_market({"AAPL": 101.0}, T)
    assert len(portfolio.pnl_history) == 1
    assert portfolio.pnl_history[-1] == pytest.approx(portfolio.cash + 10.0 * 101.0)
    portfolio.mark_to_market({"AAPL": 102.0}, T + timedelta(minutes=1))
    assert len(portfolio.pnl_history) == 2


# ---------------------------------------------------------------------------
# Task 7: gate breadth — cooldown, duplicate windows, boundaries
# ---------------------------------------------------------------------------


def test_cooldown_blocks_reentry_until_window_elapses() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    portfolio = _portfolio()
    manager.register_fill("AAPL", 0.0, timestamp=T)  # position exited at T
    blocked = manager.evaluate(
        _proposal(timestamp=T + timedelta(seconds=30), request_id="cd-1"),
        portfolio,
        evaluation_time=T + timedelta(seconds=30),
    )
    assert blocked.approved is False
    assert "cooldown_active" in blocked.reasons
    later = T + timedelta(seconds=config.cooldown_sec + 1)
    released = manager.evaluate(
        _proposal(timestamp=later, request_id="cd-2"), portfolio, evaluation_time=later
    )
    assert released.approved is True


def test_duplicate_order_window_blocks_rapid_repeat_and_side_flip() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    portfolio = _portfolio()
    first = manager.evaluate(_proposal(request_id="dw-1"), portfolio, evaluation_time=T)
    assert first.approved is True
    in_window = T + timedelta(seconds=config.duplicate_order_window_sec - 1)
    repeat = manager.evaluate(
        _proposal(timestamp=in_window, request_id="dw-2"), portfolio, evaluation_time=in_window
    )
    assert repeat.approved is False
    assert "duplicate_order_window" in repeat.reasons
    # The symbol-wide "_any" key catches immediate side flips too.
    flip = manager.evaluate(
        _proposal(action=-1, timestamp=in_window, request_id="dw-3"),
        portfolio,
        evaluation_time=in_window,
    )
    assert flip.approved is False
    assert "duplicate_order_window" in flip.reasons
    after = T + timedelta(seconds=config.duplicate_order_window_sec + 1)
    released = manager.evaluate(
        _proposal(timestamp=after, request_id="dw-4"), portfolio, evaluation_time=after
    )
    assert released.approved is True


def test_duplicate_request_id_rejected() -> None:
    manager = RiskManager(RiskConfig())
    portfolio = _portfolio()
    first = manager.evaluate(_proposal(request_id="same-id"), portfolio, evaluation_time=T)
    assert first.approved is True
    later = T + timedelta(seconds=120)
    second = manager.evaluate(
        _proposal(timestamp=later, request_id="same-id"), portfolio, evaluation_time=later
    )
    assert second.approved is False
    assert "duplicate_request_id" in second.reasons


def test_seed_existing_positions_applies_duplicate_window() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    manager.seed_existing_positions({"AAPL": 10.0}, timestamp=T)
    portfolio = _with_position(_portfolio(), "AAPL", qty=10.0)
    # Buying more AAPL right after restart is blocked by the seeded window...
    add = manager.evaluate(
        _proposal(target_fraction=0.03, request_id="seed-add"), portfolio, evaluation_time=T
    )
    assert add.approved is False
    assert "duplicate_order_window" in add.reasons
    # ...but reducing the seeded position is not (only the held side is seeded).
    reduce = manager.evaluate(
        _proposal(target_fraction=0.005, request_id="seed-reduce"), portfolio, evaluation_time=T
    )
    assert reduce.approved is True
    assert reduce.side == "sell"


def test_daily_loss_boundary_trips_at_exact_limit() -> None:
    config = RiskConfig()
    manager = RiskManager(config)
    portfolio = _portfolio()
    portfolio.daily_pnl = -config.max_daily_loss  # exactly at the limit
    at_limit = manager.evaluate(_proposal(request_id="dlb-1"), portfolio, evaluation_time=T)
    assert "daily_loss_limit" in at_limit.reasons
    portfolio.daily_pnl = -config.max_daily_loss + 0.01
    inside = manager.evaluate(_proposal(request_id="dlb-2"), portfolio, evaluation_time=T)
    assert inside.approved is True


def test_daily_loss_pct_limit_boundary() -> None:
    config = RiskConfig(max_daily_loss=1_000_000.0, max_daily_loss_pct=0.01)
    manager = RiskManager(config)
    portfolio = _portfolio()
    portfolio.daily_pnl = -1_000.0  # exactly 1% of 100k equity
    decision = manager.evaluate(_proposal(request_id="dlp-1"), portfolio, evaluation_time=T)
    assert "daily_loss_pct_limit" in decision.reasons


def test_max_concurrent_positions_boundary() -> None:
    config = RiskConfig(max_concurrent_positions=2)
    manager = RiskManager(config)
    portfolio = _portfolio()
    _with_position(portfolio, "MSFT", qty=10.0)
    _with_position(portfolio, "NVDA", qty=10.0)
    blocked = manager.evaluate(
        _proposal(symbol="AAPL", request_id="cc-1"), portfolio, evaluation_time=T
    )
    assert blocked.approved is False
    assert "max_concurrent_positions" in blocked.reasons
    # Growing an EXISTING position is not "opening new": allowed at the cap.
    grow = manager.evaluate(
        _proposal(symbol="MSFT", target_fraction=0.02, request_id="cc-2"),
        portfolio,
        evaluation_time=T,
    )
    assert grow.approved is True


def test_rolling_drawdown_boundary() -> None:
    config = RiskConfig()  # max_rolling_drawdown = -0.10
    manager = RiskManager(config)
    portfolio = _portfolio()
    portfolio.pnl_history.extend([100_000.0, 90_000.0])  # exactly -10%: trips (<=)
    at_limit = manager.evaluate(_proposal(request_id="ddb-1"), portfolio, evaluation_time=T)
    assert "rolling_drawdown_limit" in at_limit.reasons
    portfolio.pnl_history.clear()
    portfolio.pnl_history.extend([100_000.0, 90_001.0])
    inside = manager.evaluate(_proposal(request_id="ddb-2"), portfolio, evaluation_time=T)
    assert inside.approved is True


def test_flat_proposal_with_no_position_rejected_as_no_change() -> None:
    manager = RiskManager(RiskConfig())
    decision = manager.evaluate(
        _proposal(action=0, target_fraction=0.0, request_id="flat-none"),
        _portfolio(),
        evaluation_time=T,
    )
    assert decision.approved is False
    assert "no_position_change" in decision.reasons
    assert "negative_ev" not in decision.reasons  # flatten bypasses the EV gate


# ---------------------------------------------------------------------------
# Task 8: safety invariants — learned proposes, deterministic risk disposes
# ---------------------------------------------------------------------------


def test_risk_manager_never_mutates_shared_config() -> None:
    config = RiskConfig(max_daily_loss=-350.0)  # negative on purpose (normalized to +350 at validation)
    snapshot = config.model_dump()
    manager = RiskManager(config)
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)
    portfolio.daily_pnl = -400.0
    manager.evaluate(_proposal(symbol="MSFT", request_id="inv-1"), portfolio, evaluation_time=T)
    manager.evaluate(_proposal(target_fraction=0.2, request_id="inv-2"), portfolio, evaluation_time=T)
    manager.emergency_stop()
    manager.evaluate(_proposal(symbol="NVDA", request_id="inv-3"), portfolio, evaluation_time=T)
    assert config.model_dump() == snapshot
    assert config.kill_switch is False  # escalation is manager state, not config


def test_negative_max_daily_loss_config_still_enforced() -> None:
    # Backward-compat contract: negative input is accepted and normalized to
    # its magnitude at validation, so the limit still enforces at -350 PnL.
    config = RiskConfig(max_daily_loss=-350.0)
    assert config.max_daily_loss == 350.0
    manager = RiskManager(config)
    portfolio = _portfolio()
    portfolio.daily_pnl = -400.0
    decision = manager.evaluate(_proposal(request_id="neg-limit"), portfolio, evaluation_time=T)
    assert decision.approved is False
    assert "daily_loss_limit" in decision.reasons


def test_evaluate_does_not_mutate_portfolio() -> None:
    manager = RiskManager(RiskConfig())
    portfolio = _with_position(_portfolio(), "AAPL", qty=50.0)
    before = (
        portfolio.cash,
        portfolio.equity,
        portfolio.daily_pnl,
        portfolio.realized_pnl,
        {s: p.qty for s, p in portfolio.positions.items()},
        len(portfolio.pnl_history),
    )
    manager.evaluate(_proposal(request_id="pf-1"), portfolio, evaluation_time=T)
    manager.evaluate(_proposal(symbol="MSFT", request_id="pf-2"), portfolio, evaluation_time=T)
    after = (
        portfolio.cash,
        portfolio.equity,
        portfolio.daily_pnl,
        portfolio.realized_pnl,
        {s: p.qty for s, p in portfolio.positions.items()},
        len(portfolio.pnl_history),
    )
    assert after == before


def test_trade_proposal_is_frozen() -> None:
    proposal = _proposal(request_id="frozen")
    with pytest.raises(FrozenInstanceError):
        proposal.action = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        proposal.target_fraction = 1.0  # type: ignore[misc]
