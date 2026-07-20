"""Regression tests for Phase 0 live-path fixes.

Covers C1 (duplicate/wrong-timeframe bar ingestion), C3 (broker-mode
accounting + fill capture), and C4 (broker-sync failure resilience).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.core.logging import get_logger
from src.data.bar_aggregator import TimeframeAggregator, bucket_start, timeframe_delta
from src.data.buffers import BarBuffer
from src.data.schemas import MarketBar
from src.execution.order_router import OrderRouter
from src.execution.schemas import OrderState
from src.execution.tick_processor import (
    _SYNC_FAILURE_LIMIT,
    _apply_broker_fill,
    _backfill_gap,
    _capture_broker_fills,
    _sync_account_state,
    _sync_live_state,
)
from src.execution.reconciliation import hydrate_portfolio_positions
from src.monitoring.health import HealthState
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.core.config import RiskConfig


T0 = datetime(2026, 7, 6, 14, 0, tzinfo=UTC)  # 10:00 ET, bucket-aligned
LOGGER = get_logger("test_live_ingestion")


def _bar(ts: datetime, close: float = 100.0, volume: float = 1000.0, symbol: str = "AAPL") -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=ts,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        spread=0.01,
    )


def _ctx(**overrides) -> SimpleNamespace:
    """Minimal RuntimeContext stand-in for the tick helpers."""
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    ctx = SimpleNamespace(
        client=None,
        portfolio=portfolio,
        ledger=Ledger(),
        risk_manager=RiskManager(RiskConfig()),
        router=OrderRouter(paper_executor=None),
        health=HealthState(started_at=T0, last_tick_at=T0),
        order_journal=None,
        bar_aggregator=None,
        sync_failure_count=0,
        processed_fill_ids=set(),
        strategy_by_client_order_id={},
        policy=SimpleNamespace(),  # no record_fill_outcome: arbitration off
        daily_anchor_equity=0.0,
        daily_anchor_date=None,
        buffers={},
        config=SimpleNamespace(runtime=SimpleNamespace(timeframe="5Min")),
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


# ---------------------------------------------------------------------------
# C1: timeframe aggregation + buffer dedup
# ---------------------------------------------------------------------------


def test_same_forming_bar_polled_twice_grows_buffer_by_one() -> None:
    agg = TimeframeAggregator("5Min")
    buf = BarBuffer(maxlen=16)
    bar = _bar(T0, close=100.0, volume=500.0)
    for _ in range(30):  # ~one minute of 2s polls of the same forming bar
        bucket = agg.update("AAPL", bar)
        assert bucket is not None
        buf.upsert(bucket)
    assert len(buf) == 1
    assert buf.last_bar().timestamp == T0


def test_one_minute_polls_aggregate_into_five_minute_buckets() -> None:
    agg = TimeframeAggregator("5Min")
    buf = BarBuffer(maxlen=16)
    closes = [100.0, 101.0, 99.0, 102.0, 98.0]
    for i, close in enumerate(closes):
        bucket = agg.update("AAPL", _bar(T0 + timedelta(minutes=i), close=close, volume=100.0))
        buf.upsert(bucket)
    # All five minutes belong to one bucket.
    assert len(buf) == 1
    bucket_bar = buf.last_bar()
    assert bucket_bar.timestamp == T0
    assert bucket_bar.open == closes[0] - 0.5
    assert bucket_bar.close == closes[-1]
    assert bucket_bar.high == max(c + 1.0 for c in closes)
    assert bucket_bar.low == min(c - 1.0 for c in closes)
    assert bucket_bar.volume == pytest.approx(500.0)
    # The next minute starts a new bucket.
    bucket = agg.update("AAPL", _bar(T0 + timedelta(minutes=5), close=103.0))
    buf.upsert(bucket)
    assert len(buf) == 2
    assert buf.last_bar().timestamp == T0 + timedelta(minutes=5)


def test_forming_minute_revision_updates_volume_not_duplicates() -> None:
    agg = TimeframeAggregator("5Min")
    first = agg.update("AAPL", _bar(T0, volume=100.0))
    revised = agg.update("AAPL", _bar(T0, volume=250.0))
    assert first.volume == pytest.approx(100.0)
    assert revised.volume == pytest.approx(250.0)  # revision, not accumulation


def test_upsert_skips_backwards_bars() -> None:
    buf = BarBuffer(maxlen=8)
    buf.append(_bar(T0 + timedelta(minutes=5)))
    assert buf.upsert(_bar(T0)) == "skipped"
    assert len(buf) == 1


def test_upsert_replace_updates_tensor_row() -> None:
    buf = BarBuffer(maxlen=8)
    buf.append(_bar(T0, close=100.0))
    buf.upsert(_bar(T0, close=111.0))
    assert len(buf) == 1
    assert buf.snapshot_tensor()[-1, 3].item() == pytest.approx(111.0)
    assert buf.last_bar().close == pytest.approx(111.0)


def test_bucket_start_flooring() -> None:
    delta = timeframe_delta("5Min")
    assert bucket_start(T0 + timedelta(minutes=3, seconds=17), delta) == T0
    assert bucket_start(T0 + timedelta(minutes=5), delta) == T0 + timedelta(minutes=5)


def test_backfill_fetches_missed_buckets() -> None:
    fetched: dict[str, object] = {}

    class _Client:
        def get_historical_bars_range(self, *, symbol, timeframe, start, end):
            fetched["range"] = (start, end)
            return [
                _bar(T0 + timedelta(minutes=5), close=101.0),
                _bar(T0 + timedelta(minutes=10), close=102.0),
            ]

    ctx = _ctx(client=_Client())
    ctx.bar_aggregator = TimeframeAggregator("5Min")
    buf = BarBuffer(maxlen=16)
    buf.append(_bar(T0))
    live_bucket = _bar(T0 + timedelta(minutes=15), close=103.0)
    _backfill_gap(ctx, "AAPL", buf, live_bucket, logger=LOGGER)
    assert fetched["range"] == (T0 + timedelta(minutes=5), T0 + timedelta(minutes=15))
    assert len(buf) == 3  # original + two backfilled, live bucket upserted by caller
    assert [b.timestamp for b in buf.snapshot()] == [
        T0,
        T0 + timedelta(minutes=5),
        T0 + timedelta(minutes=10),
    ]


# ---------------------------------------------------------------------------
# C4: broker-sync failure resilience
# ---------------------------------------------------------------------------


class _FailingClient:
    def get_account(self):
        raise ConnectionError("boom")

    def get_positions(self):
        return []

    def get_open_orders(self):
        return []


def test_sync_failure_skips_tick_without_raising() -> None:
    ctx = _ctx(client=_FailingClient())
    result = _sync_live_state(ctx, logger=LOGGER)
    assert result is None
    assert ctx.sync_failure_count == 1
    assert ctx.health.safe_to_trade is True  # one blip never blocks


def test_sync_failure_budget_blocks_trading() -> None:
    ctx = _ctx(client=_FailingClient())
    for _ in range(_SYNC_FAILURE_LIMIT):
        assert _sync_live_state(ctx, logger=LOGGER) is None
    assert ctx.health.safe_to_trade is False


def test_sync_success_resets_failure_count() -> None:
    class _OkClient:
        def get_account(self):
            return {"cash": 100_000.0, "equity": 100_000.0, "last_equity": 100_000.0}

        def get_positions(self):
            return []

        def get_open_orders(self):
            return []

    ctx = _ctx(client=_OkClient(), sync_failure_count=_SYNC_FAILURE_LIMIT - 1)
    assert _sync_live_state(ctx, logger=LOGGER) == []
    assert ctx.sync_failure_count == 0


# ---------------------------------------------------------------------------
# C3: broker-truth accounting + fill capture
# ---------------------------------------------------------------------------


def test_account_sync_anchors_daily_pnl_to_last_equity() -> None:
    ctx = _ctx()
    _sync_account_state(ctx, account={"cash": 90_000.0, "equity": 99_500.0, "last_equity": 100_000.0})
    assert ctx.portfolio.cash == pytest.approx(90_000.0)
    assert ctx.portfolio.daily_pnl == pytest.approx(-500.0)


def test_daily_loss_limit_fires_from_broker_equity() -> None:
    config = RiskConfig()
    config.max_daily_loss = 350.0
    risk = RiskManager(config)
    ctx = _ctx(risk_manager=risk)
    _sync_account_state(ctx, account={"cash": 95_000.0, "equity": 99_000.0, "last_equity": 100_000.0})
    from src.risk.constraints import TradeProposal

    proposal = TradeProposal(
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        action=1,
        confidence=0.9,
        target_fraction=0.05,
        price=100.0,
        spread_bps=1.0,
        request_id="req-1",
        expected_gain_bps=50.0,
        expected_loss_bps=10.0,
    )
    decision = risk.evaluate(proposal, ctx.portfolio)
    assert decision.approved is False
    assert "daily_loss_limit" in decision.reasons


def test_disappeared_order_fill_is_captured() -> None:
    filled_order = {
        "id": "abc",
        "client_order_id": "seif-x-AAPL-b-202607061400-0001",
        "symbol": "AAPL",
        "qty": 10.0,
        "filled_qty": 10.0,
        "remaining_qty": 0.0,
        "filled_avg_price": 100.0,
        "side": "buy",
        "status": "filled",
        "updated_at": T0,
        "terminal": True,
    }

    class _Client:
        def get_order_by_client_order_id(self, client_order_id):
            assert client_order_id == filled_order["client_order_id"]
            return dict(filled_order)

    ctx = _ctx(client=_Client())
    ctx.router.open_orders_by_client_id[filled_order["client_order_id"]] = OrderState(
        client_order_id=filled_order["client_order_id"],
        symbol="AAPL",
        side="buy",
        quantity=10.0,
        filled_qty=0.0,
        remaining_qty=10.0,
        status="accepted",
        updated_at=T0,
        order_id="abc",
        terminal=False,
    )
    transitions, disappeared = ctx.router.sync_open_orders([])
    assert disappeared == [filled_order["client_order_id"]]
    _capture_broker_fills(ctx, transitions=transitions, disappeared=disappeared, logger=LOGGER)

    assert len(ctx.ledger.fills) == 1
    fill = ctx.ledger.fills[0]
    assert fill.symbol == "AAPL"
    assert fill.quantity == pytest.approx(10.0)
    assert fill.price == pytest.approx(100.0)
    assert ctx.portfolio.cash == pytest.approx(100_000.0 - 1_000.0)
    position = ctx.portfolio.get_position("AAPL")
    assert position is not None and position.qty == pytest.approx(10.0)
    # Idempotent: re-processing the same order id does nothing.
    _capture_broker_fills(ctx, transitions=[], disappeared=[filled_order["client_order_id"]], logger=LOGGER)
    assert len(ctx.ledger.fills) == 1


def test_closing_fill_registers_exit_cooldown_and_realized_pnl() -> None:
    ctx = _ctx()
    open_order = {
        "client_order_id": "cid-open",
        "symbol": "AAPL",
        "filled_qty": 10.0,
        "filled_avg_price": 100.0,
        "side": "buy",
        "status": "filled",
        "updated_at": T0,
    }
    close_order = {
        "client_order_id": "cid-close",
        "symbol": "AAPL",
        "filled_qty": 10.0,
        "filled_avg_price": 105.0,
        "side": "sell",
        "status": "filled",
        "updated_at": T0 + timedelta(minutes=5),
    }
    _apply_broker_fill(ctx, order=open_order, logger=LOGGER)
    _apply_broker_fill(ctx, order=close_order, logger=LOGGER)
    assert ctx.portfolio.realized_pnl == pytest.approx(50.0)
    assert ctx.portfolio.get_position("AAPL") is None
    assert "AAPL" in ctx.risk_manager._last_exit_times  # exit cooldown registered
    assert ctx.portfolio.cash == pytest.approx(100_000.0 - 1_000.0 + 1_050.0)


def test_hydration_preserves_entry_time_for_same_side_position() -> None:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    broker_position = {
        "symbol": "AAPL",
        "qty": 10.0,
        "avg_entry_price": 100.0,
        "current_price": 101.0,
        "side": "long",
    }
    hydrate_portfolio_positions(portfolio, [broker_position])
    first_entry = portfolio.positions["AAPL"].entry_time
    hydrate_portfolio_positions(portfolio, [broker_position])
    assert portfolio.positions["AAPL"].entry_time == first_entry


# ---------------------------------------------------------------------------
# Fill-capture vs hydration race (observed live 2026-07-20): a closing
# fill's PnL was booked 0.0 because hydration consumed the fill's position
# effect before the order reached a terminal state.
# ---------------------------------------------------------------------------


AAPL_ENTRY = 325.825404
AAPL_EXIT = 326.500809
AAPL_QTY = 5.481
_SELL_ID = "seif-x-AAPL-s-202607201835-0001"


def _filled_sell_order() -> dict:
    """What Alpaca's closed-order lookup serves once the sell has filled."""
    return {
        "id": "oid-1",
        "client_order_id": _SELL_ID,
        "symbol": "AAPL",
        "filled_qty": AAPL_QTY,
        "filled_avg_price": AAPL_EXIT,
        "side": "sell",
        "status": "filled",
        "updated_at": T0 + timedelta(seconds=6),
    }


def _run_scripted_sync(ctx, script: list[tuple[list[dict], list[dict]]]) -> None:
    """Drive the REAL _sync_live_state once per scripted (positions, orders) poll.

    Models a broker whose positions endpoint runs ahead of the orders
    endpoint: each step is what the two endpoints return on that poll.
    """
    for positions, orders in script:
        ctx.client = SimpleNamespace(
            get_account=lambda: {"cash": 100_000.0, "equity": 100_000.0, "last_equity": 100_000.0},
            get_positions=lambda p=positions: [dict(x) for x in p],
            get_open_orders=lambda o=orders: list(o),
            get_order_by_client_order_id=lambda cid: dict(_filled_sell_order()),
        )
        assert _sync_live_state(ctx, logger=LOGGER) is not None


def _race_ctx(tmp_path=None):
    """Ctx holding a long AAPL book with the sell order tracked in flight."""
    ctx = _ctx()
    from src.risk.portfolio_state import apply_fill

    apply_fill(
        ctx.portfolio, symbol="AAPL", signed_delta=AAPL_QTY, fill_price=AAPL_ENTRY, timestamp=T0
    )
    ctx.router.open_orders_by_client_id[_SELL_ID] = OrderState(
        client_order_id=_SELL_ID,
        symbol="AAPL",
        side="sell",
        quantity=AAPL_QTY,
        filled_qty=0.0,
        remaining_qty=AAPL_QTY,
        status="accepted",
        updated_at=T0,
        order_id="oid-1",
        terminal=False,
    )
    if tmp_path is not None:
        ctx.config.execution = SimpleNamespace(
            observation_stream_enabled=True,
            observation_stream_path=str(tmp_path / "observations.jsonl"),
        )
    return ctx


def test_partial_fill_hydration_race_books_full_close_pnl(tmp_path) -> None:
    """THE 2026-07-20 regression: partial fill seen by the positions endpoint
    while the order is still open -> hydration must NOT consume the position;
    the terminal capture one tick later books the FULL close PnL."""
    partial = {
        "id": "oid-1",
        "client_order_id": _SELL_ID,
        "symbol": "AAPL",
        "side": "sell",
        "qty": AAPL_QTY,
        "filled_qty": 4.0,
        "remaining_qty": AAPL_QTY - 4.0,
        "status": "partially_filled",
        "updated_at": T0 + timedelta(seconds=3),
        "terminal": False,
    }
    ctx = _race_ctx(tmp_path=tmp_path)
    # Tick 1: broker positions ALREADY FLAT (fill executed), order still
    # open as partially_filled -> non-terminal, nothing capturable yet.
    # Tick 2: order gone from the open list -> disappeared -> captured.
    _run_scripted_sync(ctx, [([], [partial]), ([], [])])

    expected_pnl = (AAPL_EXIT - AAPL_ENTRY) * AAPL_QTY
    assert len(ctx.ledger.fills) == 1
    assert ctx.ledger.fills[0].realized_pnl == pytest.approx(expected_pnl, abs=1e-6)
    assert ctx.portfolio.realized_pnl == pytest.approx(expected_pnl, abs=1e-6)
    # Book converged to broker truth (flat) the same tick the fill captured.
    assert ctx.portfolio.get_position("AAPL") is None
    # The arm-1 outcome stream got a RESOLVED row, not an excluded 0.0.
    from src.training.replay_buffer import load_observations

    observations = load_observations(ctx.config.execution.observation_stream_path)
    assert len(observations) == 1
    assert observations[0].realized_pnl == pytest.approx(expected_pnl, abs=1e-6)


def test_endpoint_lag_without_partial_also_preserved(tmp_path) -> None:
    """Variant: no partial fill — the positions endpoint just updates a poll
    before the orders endpoint drops the order."""
    still_open = {
        "id": "oid-1",
        "client_order_id": _SELL_ID,
        "symbol": "AAPL",
        "side": "sell",
        "qty": AAPL_QTY,
        "filled_qty": 0.0,
        "remaining_qty": AAPL_QTY,
        "status": "accepted",
        "updated_at": T0,
        "terminal": False,
    }
    ctx = _race_ctx(tmp_path=tmp_path)
    _run_scripted_sync(ctx, [([], [still_open]), ([], [])])
    expected_pnl = (AAPL_EXIT - AAPL_ENTRY) * AAPL_QTY
    assert ctx.portfolio.realized_pnl == pytest.approx(expected_pnl, abs=1e-6)
    assert ctx.portfolio.get_position("AAPL") is None


def test_hydration_skip_only_applies_while_order_in_flight() -> None:
    """No in-flight order -> hydration overwrites exactly as before."""
    ctx = _ctx()
    from src.risk.portfolio_state import apply_fill

    apply_fill(ctx.portfolio, symbol="AAPL", signed_delta=10.0, fill_price=100.0, timestamp=T0)
    assert ctx.router.symbols_with_open_orders() == set()
    hydrate_portfolio_positions(
        ctx.portfolio, [], skip_symbols=ctx.router.symbols_with_open_orders()
    )
    assert ctx.portfolio.get_position("AAPL") is None  # broker flat wins


def test_router_symbols_with_open_orders_lifecycle() -> None:
    router = OrderRouter(paper_executor=None)
    state = OrderState(
        client_order_id="cid-1",
        symbol="NVDA",
        side="buy",
        quantity=1.0,
        filled_qty=0.0,
        remaining_qty=1.0,
        status="accepted",
        updated_at=T0,
        terminal=False,
    )
    router.open_orders_by_client_id["cid-1"] = state
    assert router.symbols_with_open_orders() == {"NVDA"}
    transitions, disappeared = router.sync_open_orders([])  # order vanished
    assert disappeared == ["cid-1"]
    assert router.symbols_with_open_orders() == set()
