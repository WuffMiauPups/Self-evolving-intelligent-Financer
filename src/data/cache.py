from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

from src.data.schemas import MarketBar

# ---------------------------------------------------------------------------
# Fast columnar payload format
# ---------------------------------------------------------------------------
# The legacy cache stored every bar as a Python dict with an ISO-8601
# timestamp string.  Loading 500k 1-minute bars then parses 500k strings
# through ``datetime.fromisoformat`` and constructs 500k ``MarketBar``
# instances — several seconds of pure-Python work even on a warm cache hit.
#
# The fast format stores each numeric column as a single torch tensor
# (float64 for prices/volume/spread, int64 nanoseconds for timestamps) and
# reconstructs ``MarketBar`` objects in one tight loop.  Files written in
# this format carry ``cache_format == "tensor_v1"``.  Legacy files are still
# readable so existing caches don't need to be regenerated; they will be
# rewritten in the new format the next time the cache is saved.
# ---------------------------------------------------------------------------

CACHE_FORMAT_TENSOR_V1 = "tensor_v1"

# Cache schema version.  Bumped from "1.0" to "2.0" when historical
# fetches switched to split/dividend-adjusted prices
# (``adjustment=Adjustment.ALL``): version-1 caches hold raw prices and
# must never be merged with adjusted fetches, so ``load`` treats them as
# missing (forcing a refetch) instead of serving them.
CACHE_VERSION = "2.0"


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def serialize_bar(bar: MarketBar) -> dict[str, Any]:
    return {
        "symbol": bar.symbol,
        "timestamp": _ensure_utc(bar.timestamp).isoformat(),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
        "spread": float(bar.spread),
    }


def deserialize_bar(payload: dict[str, Any]) -> MarketBar:
    return MarketBar(
        symbol=str(payload["symbol"]),
        timestamp=_ensure_utc(datetime.fromisoformat(str(payload["timestamp"]))),
        open=float(payload["open"]),
        high=float(payload["high"]),
        low=float(payload["low"]),
        close=float(payload["close"]),
        volume=float(payload["volume"]),
        spread=float(payload.get("spread", 0.0)),
    )


def _bars_to_columnar_payload(symbol: str, bars: list[MarketBar]) -> dict[str, Any]:
    """Pack a list of bars into a columnar tensor payload.

    Uses Python list comprehensions feeding into ``torch.tensor`` once per
    column.  ``torch.tensor`` accepts plain Python lists and is one of the
    fastest ways to materialize a numeric tensor from a list of attribute
    accesses (slightly slower than going through numpy + ``from_numpy``,
    but avoids the numpy dependency at this layer).
    """
    if not bars:
        return {
            "cache_format": CACHE_FORMAT_TENSOR_V1,
            "symbol": symbol,
            "timestamps_ns": torch.empty(0, dtype=torch.int64),
            "open": torch.empty(0, dtype=torch.float64),
            "high": torch.empty(0, dtype=torch.float64),
            "low": torch.empty(0, dtype=torch.float64),
            "close": torch.empty(0, dtype=torch.float64),
            "volume": torch.empty(0, dtype=torch.float64),
            "spread": torch.empty(0, dtype=torch.float64),
        }
    # Per-column list comprehensions are ~10x faster than per-bar tensor
    # index assignment because they avoid the per-element tensor write
    # path entirely.
    timestamps_ns = [
        int(_ensure_utc(bar.timestamp).timestamp() * 1_000_000_000) for bar in bars
    ]
    return {
        "cache_format": CACHE_FORMAT_TENSOR_V1,
        "symbol": symbol,
        "timestamps_ns": torch.tensor(timestamps_ns, dtype=torch.int64),
        "open": torch.tensor([float(bar.open) for bar in bars], dtype=torch.float64),
        "high": torch.tensor([float(bar.high) for bar in bars], dtype=torch.float64),
        "low": torch.tensor([float(bar.low) for bar in bars], dtype=torch.float64),
        "close": torch.tensor([float(bar.close) for bar in bars], dtype=torch.float64),
        "volume": torch.tensor([float(bar.volume) for bar in bars], dtype=torch.float64),
        "spread": torch.tensor([float(bar.spread) for bar in bars], dtype=torch.float64),
    }


def _columnar_payload_to_bars(payload: dict[str, Any], symbol: str) -> list[MarketBar]:
    """Reconstruct MarketBar objects from a columnar tensor payload."""
    timestamps = payload["timestamps_ns"]
    opens = payload["open"]
    highs = payload["high"]
    lows = payload["low"]
    closes = payload["close"]
    volumes = payload["volume"]
    spreads = payload["spread"]
    if hasattr(timestamps, "tolist"):
        ts_list = timestamps.tolist()
        opens_list = opens.tolist()
        highs_list = highs.tolist()
        lows_list = lows.tolist()
        closes_list = closes.tolist()
        volumes_list = volumes.tolist()
        spreads_list = spreads.tolist()
    else:
        ts_list = list(timestamps)
        opens_list = list(opens)
        highs_list = list(highs)
        lows_list = list(lows)
        closes_list = list(closes)
        volumes_list = list(volumes)
        spreads_list = list(spreads)
    bars: list[MarketBar] = []
    for i, ts_ns in enumerate(ts_list):
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC),
                open=float(opens_list[i]),
                high=float(highs_list[i]),
                low=float(lows_list[i]),
                close=float(closes_list[i]),
                volume=float(volumes_list[i]),
                spread=float(spreads_list[i]),
            )
        )
    return bars


@dataclass(frozen=True)
class HistoricalCacheMetadata:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    source: str
    timezone: str
    row_count: int
    generated_at: datetime
    updated_at: datetime
    cache_version: str = CACHE_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": _ensure_utc(self.start).isoformat(),
            "end": _ensure_utc(self.end).isoformat(),
            "source": self.source,
            "timezone": self.timezone,
            "row_count": self.row_count,
            "generated_at": _ensure_utc(self.generated_at).isoformat(),
            "updated_at": _ensure_utc(self.updated_at).isoformat(),
            "cache_version": self.cache_version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HistoricalCacheMetadata":
        return cls(
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            start=_ensure_utc(datetime.fromisoformat(str(payload["start"]))),
            end=_ensure_utc(datetime.fromisoformat(str(payload["end"]))),
            source=str(payload["source"]),
            timezone=str(payload.get("timezone", "UTC")),
            row_count=int(payload["row_count"]),
            generated_at=_ensure_utc(datetime.fromisoformat(str(payload["generated_at"]))),
            updated_at=_ensure_utc(datetime.fromisoformat(str(payload["updated_at"]))),
            cache_version=str(payload.get("cache_version", "1.0")),
        )


@dataclass(frozen=True)
class HistoricalCacheRecord:
    metadata: HistoricalCacheMetadata
    bars: list[MarketBar]

    def to_payload(self) -> dict[str, Any]:
        """Serialize the record using the fast columnar tensor format.

        New writes always use ``cache_format == "tensor_v1"``.  Existing
        legacy files (a list of per-bar dicts) are still readable via
        ``from_payload`` so users do not need to regenerate caches.
        """
        return {
            "metadata": self.metadata.to_payload(),
            "bars_columnar": _bars_to_columnar_payload(self.metadata.symbol, self.bars),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HistoricalCacheRecord":
        metadata = HistoricalCacheMetadata.from_payload(payload["metadata"])
        # Fast path: columnar tensor format.
        columnar = payload.get("bars_columnar")
        if isinstance(columnar, dict) and columnar.get("cache_format") == CACHE_FORMAT_TENSOR_V1:
            bars = _columnar_payload_to_bars(columnar, symbol=metadata.symbol)
            return cls(metadata=metadata, bars=bars)
        # Legacy fallback: list of per-bar dicts.
        legacy_bars = payload.get("bars", [])
        bars = [deserialize_bar(bar_payload) for bar_payload in legacy_bars]
        return cls(metadata=metadata, bars=bars)


class HistoricalBarCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        normalized_symbol = symbol.upper().replace("/", "_")
        normalized_timeframe = timeframe.replace(" ", "")
        return self.root / f"{normalized_symbol}_{normalized_timeframe}.pt"

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self.path_for(symbol, timeframe).exists()

    def load(self, symbol: str, timeframe: str) -> HistoricalCacheRecord | None:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("cache_load_legacy_format path=%s", path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        record = HistoricalCacheRecord.from_payload(payload)
        if record.metadata.cache_version != CACHE_VERSION:
            # Pre-adjustment (raw-price) caches must not be served or
            # merged with adjusted fetches; treat them as missing so the
            # caller refetches the range.
            logger.warning(
                "cache_version_stale path=%s found=%s expected=%s",
                path, record.metadata.cache_version, CACHE_VERSION,
            )
            return None
        return record

    def summarize(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        record = self.load(symbol, timeframe)
        if record is None:
            return None
        return {
            "symbol": record.metadata.symbol,
            "timeframe": record.metadata.timeframe,
            "start": record.metadata.start.isoformat(),
            "end": record.metadata.end.isoformat(),
            "row_count": record.metadata.row_count,
            "source": record.metadata.source,
            "timezone": record.metadata.timezone,
            "generated_at": record.metadata.generated_at.isoformat(),
            "updated_at": record.metadata.updated_at.isoformat(),
            "cache_path": str(self.path_for(symbol, timeframe)),
        }

    def save(self, record: HistoricalCacheRecord) -> Path:
        path = self.path_for(record.metadata.symbol, record.metadata.timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a crash mid-save must never leave a truncated
        # cache file behind.  The tmp name includes pid + uuid so
        # concurrent writers (threads or processes) can never collide on
        # the same tmp path — os.replace on Windows fails if the source
        # is open in another writer.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(record.to_payload(), tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return path

    def filter_range(
        self,
        record: HistoricalCacheRecord | None,
        *,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        if record is None:
            return []
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)
        # Reconstruction by ``_columnar_payload_to_bars`` already returns
        # tz-aware UTC timestamps, so the redundant per-bar
        # ``_ensure_utc`` round-trip can be skipped on the fast path.  We
        # still tolerate naive bars from older legacy paths just in case.
        result: list[MarketBar] = []
        for bar in record.bars:
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if start_utc <= ts <= end_utc:
                result.append(bar)
        return result

    @staticmethod
    def merge_bars(*bar_lists: list[MarketBar]) -> list[MarketBar]:
        merged: dict[datetime, MarketBar] = {}
        for bars in bar_lists:
            for bar in bars:
                merged[_ensure_utc(bar.timestamp)] = MarketBar(
                    symbol=bar.symbol,
                    timestamp=_ensure_utc(bar.timestamp),
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    spread=float(bar.spread),
                )
        return [merged[timestamp] for timestamp in sorted(merged)]
