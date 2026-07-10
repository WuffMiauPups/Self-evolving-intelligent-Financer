"""Graceful shutdown + restart-safe session ids (MASTER_PLAN 4.7).

1. A KeyboardInterrupt reaching the trade loop must cancel tracked open
   orders at the broker, persist memory state, write a final runtime-status
   snapshot with safe_to_trade=False, and re-raise.
2. runtime_session_id carries a pid nonce so a same-day restart cannot
   regenerate colliding client_order_ids; the order-router id grammar must
   accept the longer session token.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import src.execution.runtime_bootstrap as runtime_bootstrap
import src.execution.trade_engine as trade_engine
from src.execution.order_router import CLIENT_ORDER_ID_PARSE, OrderRouter
from src.execution.runtime_bootstrap import make_runtime_session_id
from src.execution.tick_processor import TickResult


# ---------------------------------------------------------------------------
# stub runtime context
# ---------------------------------------------------------------------------


class _InterruptingStream:
    """Yields one tick, then raises KeyboardInterrupt (operator Ctrl+C)."""

    def __init__(self) -> None:
        self.circuit_breaker_threshold = 0
        self.consecutive_empty_polls = 0

    async def stream(self):
        yield {"AAPL": SimpleNamespace(close=100.0)}
        raise KeyboardInterrupt


class _StatusWriterSpy:
    def __init__(self) -> None:
        self.snapshots = []

    def write(self, snap) -> None:
        self.snapshots.append(snap)


def _make_stub_ctx() -> tuple[SimpleNamespace, list[str], _StatusWriterSpy]:
    cancelled: list[str] = []
    status_writer = _StatusWriterSpy()
    open_orders = {
        "seif-20260710p0001-AAPL-b-202607101430-0001": SimpleNamespace(
            order_id="broker-123", symbol="AAPL", terminal=False
        ),
        # Never acknowledged by the broker: no order_id, nothing to cancel.
        "seif-20260710p0001-MSFT-b-202607101430-0001": SimpleNamespace(
            order_id=None, symbol="MSFT", terminal=False
        ),
    }
    ctx = SimpleNamespace(
        stream=_InterruptingStream(),
        health=SimpleNamespace(
            safe_to_trade=True,
            blocked_reason_code=None,
            blocked_details=None,
            error_count=0,
            mark_error=lambda: None,
        ),
        router=SimpleNamespace(open_orders_by_client_id=open_orders),
        client=SimpleNamespace(cancel_order=lambda order_id: cancelled.append(order_id) or True),
        status_writer=status_writer,
        portfolio=SimpleNamespace(equity=100_000.0),
        ledger=SimpleNamespace(fills=[]),
        memory_bank=None,
        policy=SimpleNamespace(updater=None),
        config=SimpleNamespace(
            memory=SimpleNamespace(
                enabled=False,
                persist_pending_observations=False,
                persist_path="artifacts/market_memory.pt",
                pending_store_path="artifacts/pending_observations.pt",
            ),
            runtime=SimpleNamespace(allow_online_model_updates=False),
        ),
        runtime_session_id="20260710p0001",
    )
    return ctx, cancelled, status_writer


def _fake_tick_result() -> TickResult:
    return TickResult(
        session_peak_equity=100_000.0,
        last_decision_payload={"action": 1},
        last_risk_payload={"approved": False},
        last_symbol="AAPL",
        should_break=False,
    )


def test_keyboard_interrupt_cancels_orders_and_writes_final_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, cancelled, status_writer = _make_stub_ctx()

    async def fake_build_runtime_context(config, *, confirm_live=False):
        return ctx

    monkeypatch.setattr(trade_engine, "build_runtime_context", fake_build_runtime_context)
    monkeypatch.setattr(
        trade_engine, "process_tick", lambda *args, **kwargs: _fake_tick_result()
    )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(trade_engine._run_async_trade_loop(SimpleNamespace()))

    # Broker-acknowledged order cancelled; unacknowledged one skipped.
    assert cancelled == ["broker-123"]
    # Final runtime-status snapshot: safe_to_trade=False, session carried.
    assert len(status_writer.snapshots) == 1
    snap = status_writer.snapshots[0]
    assert snap.safe_to_trade is False
    assert snap.session_id == "20260710p0001"
    assert snap.step == 1  # one tick processed before the interrupt
    assert snap.symbol == "AAPL"


def test_shutdown_survives_cancel_failures_and_still_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, _, status_writer = _make_stub_ctx()

    def broken_cancel(order_id: str) -> bool:
        raise RuntimeError("broker down")

    ctx.client = SimpleNamespace(cancel_order=broken_cancel)

    async def fake_build_runtime_context(config, *, confirm_live=False):
        return ctx

    monkeypatch.setattr(trade_engine, "build_runtime_context", fake_build_runtime_context)
    monkeypatch.setattr(
        trade_engine, "process_tick", lambda *args, **kwargs: _fake_tick_result()
    )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(trade_engine._run_async_trade_loop(SimpleNamespace()))
    # Cleanup failure must not prevent the final snapshot.
    assert len(status_writer.snapshots) == 1
    assert status_writer.snapshots[0].safe_to_trade is False


def test_graceful_shutdown_persists_memory_and_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.memory.storage as memory_storage

    ctx, _, status_writer = _make_stub_ctx()

    class _Bank:
        def __len__(self) -> int:
            return 7

    saved: list[tuple[str, object]] = []
    pending_saved: list[str] = []
    ctx.memory_bank = _Bank()
    ctx.config.memory.enabled = True
    ctx.config.memory.persist_pending_observations = True
    ctx.policy = SimpleNamespace(
        updater=SimpleNamespace(save_pending=lambda path: pending_saved.append(path) or 1)
    )
    monkeypatch.setattr(
        memory_storage, "save_memory_bank", lambda path, bank: saved.append((path, bank))
    )

    logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    trade_engine._graceful_shutdown(ctx, logger, step=3, last_symbol="AAPL")

    assert saved == [("artifacts/market_memory.pt", ctx.memory_bank)]
    assert pending_saved == ["artifacts/pending_observations.pt"]
    assert len(status_writer.snapshots) == 1
    assert status_writer.snapshots[0].safe_to_trade is False


# ---------------------------------------------------------------------------
# client_order_id restart-collision nonce (MASTER_PLAN 4.7 ops nits)
# ---------------------------------------------------------------------------


def test_runtime_session_id_carries_pid_nonce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_bootstrap.os, "getpid", lambda: 43219)
    started_at = datetime(2026, 7, 10, 14, 30, tzinfo=UTC)
    assert make_runtime_session_id(started_at) == "20260710p3219"


def test_same_day_restart_gets_distinct_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    started_at = datetime(2026, 7, 10, 14, 30, tzinfo=UTC)
    monkeypatch.setattr(runtime_bootstrap.os, "getpid", lambda: 1111)
    first = make_runtime_session_id(started_at)
    monkeypatch.setattr(runtime_bootstrap.os, "getpid", lambda: 2222)
    second = make_runtime_session_id(started_at)
    assert first != second


def test_client_order_id_grammar_accepts_nonce_session_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_bootstrap.os, "getpid", lambda: 43219)
    session_id = make_runtime_session_id(datetime(2026, 7, 10, 14, 30, tzinfo=UTC))
    router = OrderRouter(paper_executor=None)
    client_order_id, sequence = router.next_client_order_id(
        session_id=session_id,
        symbol="AAPL",
        side="buy",
        timestamp=datetime(2026, 7, 10, 14, 30, 12, tzinfo=UTC),
    )
    assert sequence == 1
    match = CLIENT_ORDER_ID_PARSE.match(client_order_id)
    assert match is not None
    assert match.group("session") == session_id
    # Router must be able to re-seed sequences from ids it emitted.
    router.sequence_by_key.clear()
    router._seed_sequence(client_order_id)
    assert router.sequence_by_key == {(session_id, "AAPL", "buy", "202607101430"): 1}
    # Broker client_order_id budget (Alpaca: 48 chars) still holds for
    # the longest configured symbols.
    assert len(client_order_id) <= 48
