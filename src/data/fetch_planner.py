"""Chunked historical fetch planner, cache manifest, and merge/validate pipeline.

Splits large multi-year date range requests into manageable chunks,
tracks coverage in a persistent JSON manifest, and provides merge/dedupe/validate
utilities for assembling final bar sequences.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.schemas import MarketBar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchChunk:
    """A single fetch unit: one symbol, one date range."""

    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    chunk_id: str

    @staticmethod
    def make_id(symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
        s = start.strftime("%Y%m%d")
        e = end.strftime("%Y%m%d")
        return f"{symbol}_{timeframe}_{s}_{e}"


def _month_starts(start: datetime, end: datetime) -> list[datetime]:
    """Return first-of-month boundaries covering [start, end)."""
    boundaries: list[datetime] = []
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    while cursor < end:
        boundaries.append(cursor)
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC)
    return boundaries


def _quarter_starts(start: datetime, end: datetime) -> list[datetime]:
    boundaries: list[datetime] = []
    quarter_month = ((start.month - 1) // 3) * 3 + 1
    cursor = datetime(start.year, quarter_month, 1, tzinfo=UTC)
    while cursor < end:
        boundaries.append(cursor)
        if cursor.month >= 10:
            cursor = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        else:
            cursor = datetime(cursor.year, cursor.month + 3, 1, tzinfo=UTC)
    return boundaries


def _year_starts(start: datetime, end: datetime) -> list[datetime]:
    boundaries: list[datetime] = []
    cursor = datetime(start.year, 1, 1, tzinfo=UTC)
    while cursor < end:
        boundaries.append(cursor)
        cursor = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
    return boundaries


def plan_chunks(
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    chunk_size: str = "monthly",
) -> list[FetchChunk]:
    """Split a multi-symbol, multi-year request into deterministic chunks.

    Args:
        symbols: Symbols to fetch.
        timeframe: Bar timeframe string (e.g. "1Min", "1Day").
        start: Inclusive start (UTC).
        end: Exclusive end (UTC).
        chunk_size: "monthly", "quarterly", or "yearly".

    Returns:
        Deterministically ordered list of FetchChunk objects.
    """
    if end <= start:
        return []
    if chunk_size == "monthly":
        boundaries = _month_starts(start, end)
    elif chunk_size == "quarterly":
        boundaries = _quarter_starts(start, end)
    elif chunk_size == "yearly":
        boundaries = _year_starts(start, end)
    else:
        raise ValueError(f"Unsupported chunk_size: {chunk_size}")

    chunks: list[FetchChunk] = []
    for symbol in sorted(symbols):
        for i, boundary in enumerate(boundaries):
            chunk_start = max(boundary, start)
            if i + 1 < len(boundaries):
                chunk_end = min(boundaries[i + 1], end)
            else:
                chunk_end = end
            if chunk_end <= chunk_start:
                continue
            chunk_id = FetchChunk.make_id(symbol, timeframe, chunk_start, chunk_end)
            chunks.append(
                FetchChunk(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=chunk_start,
                    end=chunk_end,
                    chunk_id=chunk_id,
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Cache manifest / coverage index
# ---------------------------------------------------------------------------


@dataclass
class ChunkCoverage:
    """Tracks a single fetched chunk's metadata."""

    chunk_id: str
    symbol: str
    timeframe: str
    start: str  # ISO format
    end: str  # ISO format
    row_count: int
    fetched_at: str  # ISO format
    validated: bool = False
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "row_count": self.row_count,
            "fetched_at": self.fetched_at,
            "validated": self.validated,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkCoverage:
        return cls(
            chunk_id=str(data["chunk_id"]),
            symbol=str(data["symbol"]),
            timeframe=str(data["timeframe"]),
            start=str(data["start"]),
            end=str(data["end"]),
            row_count=int(data["row_count"]),
            fetched_at=str(data["fetched_at"]),
            validated=bool(data.get("validated", False)),
            file_path=str(data.get("file_path", "")),
        )


@dataclass
class CacheManifest:
    """Persistent JSON manifest tracking all fetched chunks.

    Provides exact coverage knowledge, fast inspection, and
    safe recovery after interrupted fetches.
    """

    path: Path
    chunks: dict[str, ChunkCoverage] = field(default_factory=dict)
    version: str = "1.0"

    @classmethod
    def load(cls, path: str | Path) -> CacheManifest:
        path = Path(path)
        manifest = cls(path=path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            manifest.version = str(data.get("version", "1.0"))
            for chunk_data in data.get("chunks", []):
                coverage = ChunkCoverage.from_dict(chunk_data)
                manifest.chunks[coverage.chunk_id] = coverage
        return manifest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "chunks": [c.to_dict() for c in self.chunks.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.path)

    def record_chunk(
        self,
        chunk: FetchChunk,
        row_count: int,
        file_path: str = "",
        validated: bool = False,
    ) -> None:
        self.chunks[chunk.chunk_id] = ChunkCoverage(
            chunk_id=chunk.chunk_id,
            symbol=chunk.symbol,
            timeframe=chunk.timeframe,
            start=chunk.start.isoformat(),
            end=chunk.end.isoformat(),
            row_count=row_count,
            fetched_at=datetime.now(UTC).isoformat(),
            validated=validated,
            file_path=file_path,
        )

    def is_covered(self, chunk_id: str) -> bool:
        return chunk_id in self.chunks

    def coverage_for_symbol(self, symbol: str, timeframe: str) -> list[ChunkCoverage]:
        return sorted(
            [c for c in self.chunks.values() if c.symbol == symbol and c.timeframe == timeframe],
            key=lambda c: c.start,
        )

    def total_rows_for_symbol(self, symbol: str, timeframe: str) -> int:
        return sum(c.row_count for c in self.coverage_for_symbol(symbol, timeframe))

    def filter_uncovered(self, chunks: list[FetchChunk]) -> list[FetchChunk]:
        """Return only chunks not yet in the manifest."""
        return [c for c in chunks if not self.is_covered(c.chunk_id)]

    def symbols(self) -> list[str]:
        return sorted({c.symbol for c in self.chunks.values()})

    def summary(self) -> dict[str, Any]:
        """Return a concise human-readable summary."""
        by_symbol: dict[str, dict[str, Any]] = {}
        for coverage in self.chunks.values():
            key = f"{coverage.symbol}_{coverage.timeframe}"
            if key not in by_symbol:
                by_symbol[key] = {
                    "symbol": coverage.symbol,
                    "timeframe": coverage.timeframe,
                    "chunk_count": 0,
                    "total_rows": 0,
                    "earliest": coverage.start,
                    "latest": coverage.end,
                    "last_updated": coverage.fetched_at,
                    "all_validated": True,
                }
            entry = by_symbol[key]
            entry["chunk_count"] += 1
            entry["total_rows"] += coverage.row_count
            if coverage.start < entry["earliest"]:
                entry["earliest"] = coverage.start
            if coverage.end > entry["latest"]:
                entry["latest"] = coverage.end
            if coverage.fetched_at > entry["last_updated"]:
                entry["last_updated"] = coverage.fetched_at
            if not coverage.validated:
                entry["all_validated"] = False
        return {
            "version": self.version,
            "total_chunks": len(self.chunks),
            "symbols": list(by_symbol.values()),
        }

    def missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Identify date ranges in [start, end) not covered by any chunk."""
        coverages = self.coverage_for_symbol(symbol, timeframe)
        if not coverages:
            return [(start, end)]

        covered: list[tuple[datetime, datetime]] = []
        for c in coverages:
            c_start = datetime.fromisoformat(c.start)
            c_end = datetime.fromisoformat(c.end)
            if c_start.tzinfo is None:
                c_start = c_start.replace(tzinfo=UTC)
            if c_end.tzinfo is None:
                c_end = c_end.replace(tzinfo=UTC)
            covered.append((c_start, c_end))
        covered.sort()

        # Merge overlapping ranges
        merged: list[tuple[datetime, datetime]] = []
        for cs, ce in covered:
            if merged and cs <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ce))
            else:
                merged.append((cs, ce))

        # Find gaps within [start, end)
        gaps: list[tuple[datetime, datetime]] = []
        cursor = start
        for ms, me in merged:
            if ms > cursor:
                gap_end = min(ms, end)
                if gap_end > cursor:
                    gaps.append((cursor, gap_end))
            cursor = max(cursor, me)
        if cursor < end:
            gaps.append((cursor, end))
        return gaps


# ---------------------------------------------------------------------------
# Merge / dedupe / validate pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Result of merge/validate pass."""

    symbol: str
    timeframe: str
    total_rows: int
    duplicates_removed: int
    is_monotonic: bool
    gap_count: int
    max_gap_seconds: float
    gaps: list[tuple[str, str, float]]  # (from_iso, to_iso, gap_seconds)


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _timeframe_expected_seconds(timeframe: str) -> float:
    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        return max(int(normalized[:-3]), 1) * 60.0
    if normalized.endswith("hour"):
        return max(int(normalized[:-4]), 1) * 3600.0
    if normalized.endswith("day"):
        return max(int(normalized[:-3]), 1) * 86400.0
    return 60.0


def merge_dedupe_validate(
    bar_lists: list[list[MarketBar]],
    *,
    symbol: str,
    timeframe: str,
) -> tuple[list[MarketBar], ValidationReport]:
    """Merge multiple chunk bar lists into a single deduplicated, sorted, validated sequence.

    Returns:
        (merged_bars, validation_report)
    """
    # Merge and dedupe by timestamp
    seen: dict[datetime, MarketBar] = {}
    total_input = 0
    for bars in bar_lists:
        total_input += len(bars)
        for bar in bars:
            ts = _ensure_utc(bar.timestamp)
            seen[ts] = MarketBar(
                symbol=bar.symbol,
                timestamp=ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                spread=float(bar.spread),
            )
    sorted_timestamps = sorted(seen.keys())
    merged = [seen[ts] for ts in sorted_timestamps]
    duplicates_removed = total_input - len(merged)

    # Validate monotonicity (guaranteed by sort, but check anyway)
    is_monotonic = True
    for i in range(1, len(merged)):
        if _ensure_utc(merged[i].timestamp) <= _ensure_utc(merged[i - 1].timestamp):
            is_monotonic = False
            break

    # Detect gaps
    expected_seconds = _timeframe_expected_seconds(timeframe)
    gap_threshold = expected_seconds * 1.5
    gaps: list[tuple[str, str, float]] = []
    max_gap_seconds = 0.0
    for i in range(1, len(merged)):
        delta = (_ensure_utc(merged[i].timestamp) - _ensure_utc(merged[i - 1].timestamp)).total_seconds()
        if delta > gap_threshold:
            gaps.append((
                merged[i - 1].timestamp.isoformat(),
                merged[i].timestamp.isoformat(),
                delta,
            ))
            max_gap_seconds = max(max_gap_seconds, delta)

    return merged, ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        total_rows=len(merged),
        duplicates_removed=duplicates_removed,
        is_monotonic=is_monotonic,
        gap_count=len(gaps),
        max_gap_seconds=max_gap_seconds,
        gaps=gaps,
    )


# ---------------------------------------------------------------------------
# Inspect / report helpers
# ---------------------------------------------------------------------------


def inspect_cache_coverage(
    manifest: CacheManifest,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Generate a per-symbol inspection report."""
    reports: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        coverages = manifest.coverage_for_symbol(symbol, timeframe)
        total_rows = sum(c.row_count for c in coverages)
        # If the trailing gap between the last fetched chunk end and 'end'
        # is very small (< 1 day), cap effective_end to avoid false
        # incompleteness caused by datetime.now() moving forward between
        # the fetch run and the coverage check.
        effective_end = end
        if coverages:
            max_chunk_end = max(
                datetime.fromisoformat(c.end).replace(tzinfo=UTC)
                if datetime.fromisoformat(c.end).tzinfo is None
                else datetime.fromisoformat(c.end)
                for c in coverages
            )
            trailing_gap = (end - max_chunk_end).total_seconds()
            if 0 < trailing_gap < 86400:  # less than 1 day
                effective_end = max_chunk_end
        missing = manifest.missing_ranges(symbol, timeframe, start, effective_end)
        total_missing_days = sum((e - s).total_seconds() / 86400 for s, e in missing)
        reports.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "chunk_count": len(coverages),
            "total_rows": total_rows,
            "fully_covered": len(missing) == 0,
            "missing_ranges": [(s.isoformat(), e.isoformat()) for s, e in missing],
            "missing_days": round(total_missing_days, 1),
            "last_updated": max((c.fetched_at for c in coverages), default="never"),
            "all_validated": all(c.validated for c in coverages) if coverages else False,
        })
    return reports
