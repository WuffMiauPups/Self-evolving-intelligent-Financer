from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.core.logging import get_logger
from src.data.alpaca_client import BrokerDataClient
from src.data.cache import HistoricalBarCache, HistoricalCacheMetadata, HistoricalCacheRecord
from src.data.schemas import MarketBar


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def parse_date_or_datetime(value: str | date | datetime | None, *, default: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return _ensure_utc(datetime(value.year, value.month, value.day, tzinfo=UTC))
    if value is None:
        if default is None:
            raise ValueError("A date/datetime value is required")
        return _ensure_utc(default)
    parsed = datetime.fromisoformat(value)
    return _ensure_utc(parsed)


def generate_synthetic_bars(
    symbol: str,
    count: int,
    *,
    start_price: float = 100.0,
) -> list[MarketBar]:
    bars: list[MarketBar] = []
    timestamp = datetime.now(UTC) - timedelta(minutes=count)
    price = start_price
    for index in range(count):
        seasonal = 0.35 * ((index % 24) / 24.0 - 0.5)
        drift = 0.03 if (index // 25) % 2 == 0 else -0.02
        shock = ((index % 11) - 5) * 0.01
        close = max(price * (1.0 + seasonal * 0.001 + drift * 0.001 + shock * 0.001), 1.0)
        open_price = price
        high = max(open_price, close) * 1.0015
        low = min(open_price, close) * 0.9985
        volume = 10_000 + (index % 13) * 350
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=timestamp + timedelta(minutes=index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                spread=0.01 + (index % 5) * 0.001,
            )
        )
        price = close
    return bars


@dataclass(frozen=True)
class HistoricalFetchReport:
    symbol: str
    timeframe: str
    requested_start: datetime
    requested_end: datetime
    cache_hit: bool
    fetched_ranges: list[tuple[datetime, datetime]]
    row_count: int
    cache_path: str
    source: str
    gap_count: int = 0
    max_gap_seconds: float = 0.0


@dataclass
class HistoricalDataService:
    client: BrokerDataClient | None = None
    cache: HistoricalBarCache = field(default_factory=lambda: HistoricalBarCache("artifacts/historical"))
    timezone_normalization: str = "UTC"
    max_missing_gap_handling: str = "warn"

    def load_recent_bars(self, symbol: str, timeframe: str, limit: int) -> list[MarketBar]:
        if self.client is None:
            raise RuntimeError("HistoricalDataService has no broker client configured for remote recent-bar fetches")
        return self.client.get_historical_bars(symbol=symbol, timeframe=timeframe, limit=limit)

    def load_bars_range(
        self,
        symbol: str,
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
        force_refresh: bool = False,
    ) -> tuple[list[MarketBar], HistoricalFetchReport]:
        logger = get_logger("historical_data")
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)
        if end_utc <= start_utc:
            raise ValueError("Historical range end must be after start")

        record = None if force_refresh else self.cache.load(symbol, timeframe)
        fetched_ranges: list[tuple[datetime, datetime]] = []
        # Tolerate a cached range one timeframe-delta short at each edge:
        # the cached end is the last bar's timestamp, so a request ending
        # "now" is almost always a hair past it.  Without the tolerance,
        # offline loads demand sliver refetches they can never satisfy.
        edge_tolerance = _timeframe_delta(timeframe)
        if record is None:
            if self.client is None:
                raise RuntimeError(f"No cached historical coverage for {symbol} {timeframe} and no broker client available")
            bars = self.client.get_historical_bars_range(
                symbol=symbol,
                timeframe=timeframe,
                start=start_utc,
                end=end_utc,
            )
            merged_bars = HistoricalBarCache.merge_bars(list(bars))
            fetched_ranges.append((start_utc, end_utc))
            if not merged_bars:
                # Weekends, holidays, or symbols absent from the feed
                # legitimately return zero bars — report the gap instead
                # of crashing on an empty record.
                logger.warning(
                    "historical_range_empty",
                    symbol=symbol,
                    timeframe=timeframe,
                    requested_start=start_utc.isoformat(),
                    requested_end=end_utc.isoformat(),
                )
                return [], HistoricalFetchReport(
                    symbol=symbol,
                    timeframe=timeframe,
                    requested_start=start_utc,
                    requested_end=end_utc,
                    cache_hit=False,
                    fetched_ranges=fetched_ranges,
                    row_count=0,
                    cache_path=str(self.cache.path_for(symbol, timeframe)),
                    source=type(self.client).__name__,
                    gap_count=1,
                    max_gap_seconds=(end_utc - start_utc).total_seconds(),
                )
            record = self._build_record(symbol, timeframe, merged_bars, source=type(self.client).__name__)
            self.cache.save(record)
            cache_hit = False
        else:
            cache_hit = (
                record.metadata.start <= start_utc + edge_tolerance
                and record.metadata.end >= end_utc - edge_tolerance
            )
            merged_bars = list(record.bars)
            if not cache_hit:
                if start_utc < record.metadata.start:
                    prefetch_end = min(record.metadata.start, end_utc)
                    if prefetch_end > start_utc:
                        if self.client is None:
                            raise RuntimeError(
                                f"Historical cache for {symbol} {timeframe} is missing leading coverage and no broker client is available"
                            )
                        fetched_ranges.append((start_utc, prefetch_end))
                        merged_bars = HistoricalBarCache.merge_bars(
                            merged_bars,
                            self.client.get_historical_bars_range(
                                symbol=symbol,
                                timeframe=timeframe,
                                start=start_utc,
                                end=prefetch_end,
                            ),
                        )
                if end_utc > record.metadata.end:
                    postfetch_start = max(record.metadata.end, start_utc)
                    if end_utc > postfetch_start:
                        if self.client is None:
                            raise RuntimeError(
                                f"Historical cache for {symbol} {timeframe} is missing trailing coverage and no broker client is available"
                            )
                        fetched_ranges.append((postfetch_start, end_utc))
                        merged_bars = HistoricalBarCache.merge_bars(
                            merged_bars,
                            self.client.get_historical_bars_range(
                                symbol=symbol,
                                timeframe=timeframe,
                                start=postfetch_start,
                                end=end_utc,
                            ),
                        )
                if merged_bars:
                    record = self._build_record(
                        symbol,
                        timeframe,
                        merged_bars,
                        generated_at=record.metadata.generated_at,
                        source=type(self.client).__name__,
                    )
                    self.cache.save(record)
            else:
                merged_bars = list(record.bars)

        filtered_bars = self.cache.filter_range(record, start=start_utc, end=end_utc)
        gap_count, max_gap_seconds = _detect_gaps(filtered_bars, timeframe=timeframe)
        if gap_count > 0 and self.max_missing_gap_handling == "error":
            raise RuntimeError(
                f"Historical bars for {symbol} {timeframe} contain {gap_count} gaps; max gap seconds={max_gap_seconds:.1f}"
            )
        logger.info(
            "historical_range_loaded",
            symbol=symbol,
            timeframe=timeframe,
            cache_hit=cache_hit,
            requested_start=start_utc.isoformat(),
            requested_end=end_utc.isoformat(),
            fetched_ranges=[(left.isoformat(), right.isoformat()) for left, right in fetched_ranges],
            row_count=len(filtered_bars),
            cache_path=str(self.cache.path_for(symbol, timeframe)),
            gap_count=gap_count,
            max_gap_seconds=max_gap_seconds,
        )
        return filtered_bars, HistoricalFetchReport(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=start_utc,
            requested_end=end_utc,
            cache_hit=cache_hit,
            fetched_ranges=fetched_ranges,
            row_count=len(filtered_bars),
            cache_path=str(self.cache.path_for(symbol, timeframe)),
            source=record.metadata.source,
            gap_count=gap_count,
            max_gap_seconds=max_gap_seconds,
        )

    def load_bars_for_symbols(
        self,
        symbols: list[str],
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
        force_refresh: bool = False,
    ) -> tuple[dict[str, list[MarketBar]], list[HistoricalFetchReport]]:
        bars_by_symbol: dict[str, list[MarketBar]] = {}
        reports: list[HistoricalFetchReport] = []
        for symbol in symbols:
            bars, report = self.load_bars_range(
                symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                force_refresh=force_refresh,
            )
            bars_by_symbol[symbol] = bars
            reports.append(report)
        return bars_by_symbol, reports

    def cache_path_for(self, symbol: str, timeframe: str) -> Path:
        return self.cache.path_for(symbol, timeframe)

    def _build_record(
        self,
        symbol: str,
        timeframe: str,
        bars: list[MarketBar],
        *,
        generated_at: datetime | None = None,
        source: str = "alpaca",
    ) -> HistoricalCacheRecord:
        if not bars:
            raise ValueError("Cannot build a historical cache record from empty bars")
        now = datetime.now(UTC)
        return HistoricalCacheRecord(
            metadata=HistoricalCacheMetadata(
                symbol=symbol,
                timeframe=timeframe,
                start=_ensure_utc(bars[0].timestamp),
                end=_ensure_utc(bars[-1].timestamp),
                source=source,
                timezone=self.timezone_normalization,
                row_count=len(bars),
                generated_at=_ensure_utc(generated_at or now),
                updated_at=now,
            ),
            bars=bars,
        )


def _timeframe_delta(timeframe: str) -> timedelta:
    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        return timedelta(minutes=max(int(normalized[:-3]), 1))
    if normalized.endswith("hour"):
        return timedelta(hours=max(int(normalized[:-4]), 1))
    if normalized.endswith("day"):
        return timedelta(days=max(int(normalized[:-3]), 1))
    return timedelta(minutes=1)


def _detect_gaps(bars: list[MarketBar], *, timeframe: str) -> tuple[int, float]:
    if len(bars) < 2:
        return 0, 0.0
    expected = _timeframe_delta(timeframe)
    expected_seconds = expected.total_seconds()
    gap_count = 0
    max_gap_seconds = 0.0
    for left, right in zip(bars, bars[1:]):
        delta_seconds = (_ensure_utc(right.timestamp) - _ensure_utc(left.timestamp)).total_seconds()
        if delta_seconds > expected_seconds * 1.5:
            gap_count += 1
            max_gap_seconds = max(max_gap_seconds, delta_seconds)
    return gap_count, max_gap_seconds
