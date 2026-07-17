"""Failed-submit backoff: execution.submit_failure_cooldown_sec (§7.1.1).

Both 2026-07-15 live retry loops (42210000 fractional short, 40310000
fractional side-flip) shared one PATTERN: a terminal ``submit_failed``
intent re-submitted every duplicate_order_window forever.  The root causes
were fixed same-day, but the next broker rule we haven't met (e.g.
insufficient buying power) must degrade to a bounded cadence instead of a
hot loop: a terminal submit_failed puts its (symbol, side) on a time-based
cooldown.  The OTHER side is never blocked, and fills clear nothing — the
map expires by time only.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.config import ExecutionConfig, load_config
from src.execution.tick_processor import _note_submit_failure, _submit_cooldown_active
from src.monitoring.metrics_logger import MetricsLogger

T = datetime(2026, 7, 17, 14, 0, tzinfo=UTC)


class _CaptureLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **kwargs: object) -> None:
        self.events.append((event, kwargs))

    def warning(self, event: str, **kwargs: object) -> None:
        self.events.append((event, kwargs))

    def names(self) -> list[str]:
        return [event for event, _ in self.events]


def _ctx(cooldown_sec: float = 300.0) -> SimpleNamespace:
    return SimpleNamespace(
        submit_failed_until={},
        metrics=MetricsLogger(),
        config=SimpleNamespace(
            execution=ExecutionConfig(submit_failure_cooldown_sec=cooldown_sec)
        ),
    )


def _failed_result(symbol: str = "PLTR", side: str = "sell") -> SimpleNamespace:
    return SimpleNamespace(
        status="submit_failed",
        symbol=symbol,
        side=side,
        reason_code="order_submit_retry_exhausted",
    )


def test_submit_failed_arms_cooldown_and_skips_same_side() -> None:
    ctx = _ctx()
    logger = _CaptureLogger()
    _note_submit_failure(ctx, result=_failed_result(), now=T, logger=logger)
    assert ("PLTR", "sell") in ctx.submit_failed_until
    assert "submit_failure_cooldown_set" in logger.names()

    skipped = _submit_cooldown_active(
        ctx, symbol="PLTR", side="sell", now=T + timedelta(seconds=1), logger=logger
    )
    assert skipped is True
    assert "submit_failed_cooldown" in logger.names()
    assert ctx.metrics.snapshot().get("submit_failed_cooldown_skips") == 1


def test_other_side_and_other_symbol_are_not_blocked() -> None:
    ctx = _ctx()
    logger = _CaptureLogger()
    _note_submit_failure(ctx, result=_failed_result(), now=T, logger=logger)
    later = T + timedelta(seconds=1)
    assert _submit_cooldown_active(ctx, symbol="PLTR", side="buy", now=later, logger=logger) is False
    assert _submit_cooldown_active(ctx, symbol="META", side="sell", now=later, logger=logger) is False


def test_cooldown_expires_by_time_and_prunes_the_entry() -> None:
    ctx = _ctx(cooldown_sec=300.0)
    logger = _CaptureLogger()
    _note_submit_failure(ctx, result=_failed_result(), now=T, logger=logger)
    expired = T + timedelta(seconds=300)
    assert _submit_cooldown_active(ctx, symbol="PLTR", side="sell", now=expired, logger=logger) is False
    assert ("PLTR", "sell") not in ctx.submit_failed_until


def test_non_submit_failed_status_does_not_arm() -> None:
    ctx = _ctx()
    logger = _CaptureLogger()
    rejected = SimpleNamespace(status="rejected", symbol="PLTR", side="sell", reason_code="order_rejected")
    _note_submit_failure(ctx, result=rejected, now=T, logger=logger)
    assert ctx.submit_failed_until == {}


def test_zero_cooldown_disables_the_mechanism() -> None:
    ctx = _ctx(cooldown_sec=0.0)
    logger = _CaptureLogger()
    _note_submit_failure(ctx, result=_failed_result(), now=T, logger=logger)
    assert ctx.submit_failed_until == {}


def test_default_stack_carries_the_cooldown() -> None:
    config = load_config("configs/default.yaml")
    assert config.execution.submit_failure_cooldown_sec == pytest.approx(300.0)


def _tick_processor_source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "src" / "execution" / "tick_processor.py"
    ).read_text(encoding="utf-8")


def test_call_site_checks_before_order_construction_source_contract() -> None:
    """The cooldown check must run BEFORE client_order_id construction (§7.1.1
    placement decision) so a known-refused shape never re-enters the router."""
    source = _tick_processor_source()
    submit_region = source.split('side = decision.side or "buy"', 1)[1].split(
        "next_client_order_id(", 1
    )[0]
    assert "_submit_cooldown_active(" in submit_region


def test_not_accepted_branch_arms_the_cooldown_source_contract() -> None:
    source = _tick_processor_source()
    not_accepted_branch = source.split("if not result.accepted:", 1)[1].split("continue", 1)[0]
    assert "_note_submit_failure(" in not_accepted_branch


def test_fill_capture_path_never_touches_the_map_source_contract() -> None:
    """Fills clear NOTHING — the map expires by time only. The broker fill
    capture path must not reference submit_failed_until."""
    source = _tick_processor_source()
    fill_capture_region = source.split("def _capture_broker_fills", 1)[1].split(
        "def _sync_account_state", 1
    )[0]
    assert "submit_failed_until" not in fill_capture_region
