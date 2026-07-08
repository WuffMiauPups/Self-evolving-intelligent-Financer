"""Timeframe-aware aggregation of polled latest bars.

The polling stream fetches the broker's *latest* bar every
``polling_interval_sec``.  Alpaca's latest-bar endpoint always returns
the newest (possibly still-forming) 1-minute bar regardless of the
configured runtime timeframe, so feeding polls straight into
``BarBuffer`` duplicated the same bar dozens of times per period and
mixed 1-minute rows into buffers cold-started with N-minute bars.

``TimeframeAggregator`` folds each polled bar into its enclosing
timeframe bucket.  Re-polls of the same forming minute *revise* that
minute (latest poll wins) rather than accumulate, and the bucket bar is
rebuilt from its constituent minutes, so the output stream is exactly
one bar per bucket, stamped at the bucket start — the same convention
the historical cache and cold-start bars use.

Consumers pair this with ``BarBuffer.upsert``: a bar whose timestamp
equals the newest buffered row replaces it (the forming bucket being
revised in place), a newer timestamp appends, an older one is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.data.schemas import MarketBar


def timeframe_delta(timeframe: str) -> timedelta:
    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        return timedelta(minutes=max(int(normalized[:-3]), 1))
    if normalized.endswith("hour"):
        return timedelta(hours=max(int(normalized[:-4]), 1))
    if normalized.endswith("day"):
        return timedelta(days=max(int(normalized[:-3]), 1))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def bucket_start(timestamp: datetime, delta: timedelta) -> datetime:
    epoch = timestamp.timestamp()
    step = delta.total_seconds()
    floored = (epoch // step) * step
    return datetime.fromtimestamp(floored, tz=timestamp.tzinfo or UTC)


@dataclass
class _BucketState:
    start: datetime
    minutes: dict[datetime, MarketBar] = field(default_factory=dict)


class TimeframeAggregator:
    """Per-symbol aggregation of polled bars into timeframe buckets."""

    def __init__(self, timeframe: str) -> None:
        self.timeframe = timeframe
        self.delta = timeframe_delta(timeframe)
        self._buckets: dict[str, _BucketState] = {}

    def update(self, symbol: str, bar: MarketBar) -> MarketBar | None:
        """Fold one polled bar into its bucket and return the bucket bar.

        The returned bar is stamped at the bucket start and reflects all
        constituent minutes seen so far (it is the forming bucket until a
        later bucket's first bar arrives).  Returns ``None`` for
        out-of-order input older than the bucket currently being formed.
        """
        start = bucket_start(bar.timestamp, self.delta)
        state = self._buckets.get(symbol)
        if state is None or start > state.start:
            state = _BucketState(start=start)
            self._buckets[symbol] = state
        elif start < state.start:
            return None
        state.minutes[bar.timestamp] = bar
        return self._aggregate(symbol, state)

    def _aggregate(self, symbol: str, state: _BucketState) -> MarketBar:
        ordered = [state.minutes[ts] for ts in sorted(state.minutes)]
        return MarketBar(
            symbol=symbol,
            timestamp=state.start,
            open=ordered[0].open,
            high=max(b.high for b in ordered),
            low=min(b.low for b in ordered),
            close=ordered[-1].close,
            volume=float(sum(b.volume for b in ordered)),
            spread=ordered[-1].spread,
        )
