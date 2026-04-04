from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from src.data.schemas import MarketBar


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
    cache_version: str = "1.0"

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
        return {
            "metadata": self.metadata.to_payload(),
            "bars": [serialize_bar(bar) for bar in self.bars],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HistoricalCacheRecord":
        return cls(
            metadata=HistoricalCacheMetadata.from_payload(payload["metadata"]),
            bars=[deserialize_bar(bar_payload) for bar_payload in payload.get("bars", [])],
        )


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
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return HistoricalCacheRecord.from_payload(payload)

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
        torch.save(record.to_payload(), path)
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
        return [
            bar
            for bar in record.bars
            if start_utc <= _ensure_utc(bar.timestamp) <= end_utc
        ]

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
