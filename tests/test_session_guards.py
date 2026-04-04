from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.core.config import load_config
from src.data.schemas import MarketBar
from src.execution.reason_codes import ReasonCode
from src.execution.session_guards import SessionGuard


def _bar(timestamp: datetime) -> MarketBar:
    return MarketBar(
        symbol="AAPL",
        timestamp=timestamp,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        spread=0.01,
    )


def test_closed_session_prevents_trading() -> None:
    config = load_config("configs/default.yaml")
    guard = SessionGuard(config.runtime)
    now = datetime(2026, 4, 2, 2, 0, tzinfo=UTC)
    result = guard.evaluate(
        symbol="AAPL",
        bar=_bar(now),
        now=now,
        broker_clock={"is_open": False},
    )
    assert result.allowed is False
    assert result.reason_code == ReasonCode.SESSION_DISALLOWED


def test_stale_data_prevents_trading() -> None:
    config = load_config("configs/default.yaml")
    config.runtime.allowed_sessions = []
    config.runtime.require_market_open = False
    guard = SessionGuard(config.runtime)
    now = datetime(2026, 4, 2, 15, 0, tzinfo=UTC)
    result = guard.evaluate(
        symbol="AAPL",
        bar=_bar(now - timedelta(seconds=config.runtime.max_data_staleness_sec + 10)),
        now=now,
        broker_clock=None,
    )
    assert result.allowed is False
    assert result.reason_code == ReasonCode.STALE_DATA


def test_backwards_timestamps_prevent_trading() -> None:
    config = load_config("configs/default.yaml")
    config.runtime.allowed_sessions = []
    config.runtime.require_market_open = False
    guard = SessionGuard(config.runtime)
    now = datetime(2026, 4, 2, 15, 0, tzinfo=UTC)
    first = _bar(now)
    second = _bar(now - timedelta(minutes=1))
    assert guard.evaluate(symbol="AAPL", bar=first, now=now, broker_clock=None).allowed is True
    result = guard.evaluate(symbol="AAPL", bar=second, now=now, broker_clock=None)
    assert result.allowed is False
    assert result.reason_code == ReasonCode.BACKWARDS_TIMESTAMP


def test_large_data_gap_blocks_until_reset() -> None:
    config = load_config("configs/default.yaml")
    config.runtime.allowed_sessions = []
    config.runtime.require_market_open = False
    config.runtime.state_reset_gap_sec = 60
    guard = SessionGuard(config.runtime)
    now = datetime(2026, 4, 2, 15, 0, tzinfo=UTC)
    first = _bar(now)
    second = _bar(now + timedelta(seconds=120))
    assert guard.evaluate(symbol="AAPL", bar=first, now=now, broker_clock=None).allowed is True
    result = guard.evaluate(symbol="AAPL", bar=second, now=now + timedelta(seconds=120), broker_clock=None)
    assert result.allowed is False
    assert result.reason_code == ReasonCode.DATA_GAP_BLOCK
