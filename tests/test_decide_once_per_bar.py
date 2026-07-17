"""Once-per-bar decision gate: runtime.decide_once_per_bar.

The model is trained on completed bars and its output is constant within
a forming bucket — per-poll re-deciding on the same bar manufactured
~150 identical no-ops per bar per symbol (64% of all rejections in the
2026-07-17 paper session). At most one proposal per (symbol, bucket);
the poll cadence for broker sync, fill capture, halts, and the data lane
is untouched. A bucket only counts as decided when its proposal reached
Phase C dispatch, so interrupted ticks retry.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.config import RuntimeConfig, load_config
from src.execution.tick_processor import _bucket_already_decided
from src.monitoring.metrics_logger import MetricsLogger

T = datetime(2026, 7, 17, 15, 40, tzinfo=UTC)


def _ctx(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        last_decision_bucket={},
        metrics=MetricsLogger(),
        config=SimpleNamespace(runtime=RuntimeConfig(decide_once_per_bar=enabled)),
    )


def test_new_bucket_is_not_gated_and_same_bucket_is() -> None:
    ctx = _ctx()
    assert _bucket_already_decided(ctx, symbol="META", bucket_ts=T) is False
    ctx.last_decision_bucket["META"] = T  # the Phase C dispatch mark
    assert _bucket_already_decided(ctx, symbol="META", bucket_ts=T) is True
    assert ctx.metrics.snapshot().get("bar_gate_skips") == 1
    # The next bucket opens the gate again.
    assert (
        _bucket_already_decided(ctx, symbol="META", bucket_ts=T + timedelta(minutes=5))
        is False
    )


def test_other_symbols_are_independent() -> None:
    ctx = _ctx()
    ctx.last_decision_bucket["META"] = T
    assert _bucket_already_decided(ctx, symbol="AAPL", bucket_ts=T) is False


def test_flag_off_restores_per_poll_deciding() -> None:
    ctx = _ctx(enabled=False)
    ctx.last_decision_bucket["META"] = T
    assert _bucket_already_decided(ctx, symbol="META", bucket_ts=T) is False
    assert "bar_gate_skips" not in ctx.metrics.snapshot()


def test_default_stack_carries_the_gate() -> None:
    config = load_config("configs/default.yaml")
    assert config.runtime.decide_once_per_bar is True


def _tick_processor_source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "src" / "execution" / "tick_processor.py"
    ).read_text(encoding="utf-8")


def test_gate_checked_in_phase_a_before_memory_and_snapshot_source_contract() -> None:
    """The gate must skip the whole proposal path (memory prep, guards,
    snapshot build) for an already-decided bucket."""
    source = _tick_processor_source()
    phase_a = source.split("warmup_logged.discard(symbol)", 1)[1].split(
        "prepare_symbol_history", 1
    )[0]
    assert "_bucket_already_decided(" in phase_a


def test_bucket_marked_only_at_phase_c_dispatch_source_contract() -> None:
    """Marking lives in Phase C (post safe_to_trade), so a tick that broke
    before dispatch retries the bucket on the next poll."""
    source = _tick_processor_source()
    phase_c = source.split("for (symbol, snapshot, bar), (proposal, diagnostics) in zip(", 1)[1]
    dispatch_head = phase_c.split("risk_manager.evaluate", 1)[0]
    assert "ctx.last_decision_bucket[symbol]" in dispatch_head
    # And nowhere in Phase A: the only assignment is the Phase C mark.
    phase_a = source.split("for symbol in trading_symbols:", 1)[1].split(
        "# --- Phase B", 1
    )[0]
    assert "ctx.last_decision_bucket[symbol] =" not in phase_a
