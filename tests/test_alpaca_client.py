from __future__ import annotations

from datetime import timedelta

import pytest

from src.data.alpaca_client import AlpacaClient, _historical_lookback_delta


def test_historical_lookback_for_minute_bars_is_longer_than_intraday_window() -> None:
    lookback = _historical_lookback_delta("1Min", 64)
    assert lookback >= timedelta(days=2)


def test_historical_lookback_scales_for_daily_bars() -> None:
    lookback = _historical_lookback_delta("1Day", 20)
    assert lookback >= timedelta(days=30)


def test_historical_lookback_scales_with_minute_multiplier() -> None:
    # 3900 five-minute bars span 5x the minutes of 3900 one-minute bars:
    # 3900 * 5 / 390 = 50 trading days minimum.
    lookback_5min = _historical_lookback_delta("5Min", 3900)
    lookback_1min = _historical_lookback_delta("1Min", 3900)
    assert lookback_5min >= timedelta(days=50)
    assert lookback_5min > lookback_1min


def test_historical_lookback_scales_with_hour_multiplier() -> None:
    lookback_1hour = _historical_lookback_delta("1Hour", 140)
    lookback_2hour = _historical_lookback_delta("2Hour", 140)
    # 140 hourly bars at ~7 per session need ~20 days of lookback.
    assert lookback_1hour >= timedelta(days=20)
    assert lookback_2hour > lookback_1hour


def test_client_defaults_to_fully_adjusted_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    client = AlpacaClient()
    assert client.bar_adjustment == "all"


def test_client_rejects_unknown_bar_adjustment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    with pytest.raises(ValueError, match="bar_adjustment"):
        AlpacaClient(bar_adjustment="bogus")
