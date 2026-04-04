from __future__ import annotations

from datetime import timedelta

from src.data.alpaca_client import _historical_lookback_delta


def test_historical_lookback_for_minute_bars_is_longer_than_intraday_window() -> None:
    lookback = _historical_lookback_delta("1Min", 64)
    assert lookback >= timedelta(days=2)


def test_historical_lookback_scales_for_daily_bars() -> None:
    lookback = _historical_lookback_delta("1Day", 20)
    assert lookback >= timedelta(days=30)
