"""Parallel fetch executor for chunked historical data retrieval.

Uses bounded thread pool for I/O-bound Alpaca API calls, with
retry/backoff, rate-limit awareness, and resume-safe execution.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.data.alpaca_client import BrokerDataClient
from src.data.cache import HistoricalBarCache, HistoricalCacheMetadata, HistoricalCacheRecord
from src.data.fetch_planner import CacheManifest, FetchChunk, merge_dedupe_validate
from src.data.schemas import MarketBar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkFetchResult:
    """Result of fetching a single chunk."""

    chunk: FetchChunk
    success: bool
    row_count: int = 0
    error: str = ""
    retries_used: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class ParallelFetchExecutor:
    """Parallel, resume-safe fetch executor for historical bar chunks.

    Features:
    - Configurable worker count with bounded concurrency
    - Retry with exponential backoff
    - Skips chunks already in manifest (resume-safe)
    - Per-chunk result logging
    - Saves each chunk to cache immediately after fetch
    """

    client: BrokerDataClient
    cache: HistoricalBarCache
    manifest: CacheManifest
    max_workers: int = 4
    retry_count: int = 3
    retry_backoff_sec: float = 1.0
    timezone_normalization: str = "UTC"

    def __post_init__(self) -> None:
        # Per-symbol locks to prevent concurrent cache read-modify-write corruption
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        with self._locks_lock:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    def _fetch_single_chunk(self, chunk: FetchChunk) -> ChunkFetchResult:
        """Fetch a single chunk with retry logic. Thread-safe."""
        if self.manifest.is_covered(chunk.chunk_id):
            logger.debug("chunk_already_covered chunk_id=%s", chunk.chunk_id)
            coverage = self.manifest.chunks[chunk.chunk_id]
            return ChunkFetchResult(
                chunk=chunk,
                success=True,
                row_count=coverage.row_count,
            )

        start_time = time.monotonic()
        last_error = ""
        for attempt in range(self.retry_count + 1):
            try:
                bars = self.client.get_historical_bars_range(
                    symbol=chunk.symbol,
                    timeframe=chunk.timeframe,
                    start=chunk.start,
                    end=chunk.end,
                )
                elapsed = time.monotonic() - start_time

                # Serialize cache read-modify-write per symbol to prevent corruption
                with self._get_symbol_lock(chunk.symbol):
                    existing = self.cache.load(chunk.symbol, chunk.timeframe)
                    if existing is not None:
                        merged = HistoricalBarCache.merge_bars(existing.bars, bars)
                    else:
                        merged = HistoricalBarCache.merge_bars(bars)

                    now = datetime.now(UTC)
                    record = HistoricalCacheRecord(
                        metadata=HistoricalCacheMetadata(
                            symbol=chunk.symbol,
                            timeframe=chunk.timeframe,
                            start=merged[0].timestamp if merged else chunk.start,
                            end=merged[-1].timestamp if merged else chunk.end,
                            source=type(self.client).__name__,
                            timezone=self.timezone_normalization,
                            row_count=len(merged),
                            generated_at=existing.metadata.generated_at if existing else now,
                            updated_at=now,
                        ),
                        bars=merged,
                    )
                    cache_path = self.cache.save(record)

                    # Record in manifest (also serialized to avoid torn writes)
                    self.manifest.record_chunk(
                        chunk,
                        row_count=len(bars),
                        file_path=str(cache_path),
                        validated=False,
                    )
                    self.manifest.save()

                logger.info(
                    "chunk_fetched chunk_id=%s rows=%d elapsed=%.1fs retries=%d",
                    chunk.chunk_id, len(bars), elapsed, attempt,
                )
                return ChunkFetchResult(
                    chunk=chunk,
                    success=True,
                    row_count=len(bars),
                    retries_used=attempt,
                    elapsed_seconds=elapsed,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.retry_count:
                    backoff = self.retry_backoff_sec * (2 ** attempt)
                    logger.warning(
                        "chunk_fetch_retry chunk_id=%s attempt=%d/%d backoff=%.1fs error=%s",
                        chunk.chunk_id, attempt + 1, self.retry_count, backoff, last_error,
                    )
                    # time.sleep is safe here: runs inside a ThreadPoolExecutor
                    # worker thread, so it blocks only this worker, not the
                    # main thread or any async event loop.
                    time.sleep(backoff)

        elapsed = time.monotonic() - start_time
        logger.error(
            "chunk_fetch_failed chunk_id=%s retries=%d elapsed=%.1fs error=%s",
            chunk.chunk_id, self.retry_count, elapsed, last_error,
        )
        return ChunkFetchResult(
            chunk=chunk,
            success=False,
            error=last_error,
            retries_used=self.retry_count,
            elapsed_seconds=elapsed,
        )

    def execute(
        self,
        chunks: list[FetchChunk],
        *,
        skip_covered: bool = True,
    ) -> list[ChunkFetchResult]:
        """Execute parallel fetch of all chunks.

        Args:
            chunks: Chunks to fetch.
            skip_covered: If True, skip chunks already in manifest.

        Returns:
            List of results, one per input chunk.
        """
        if skip_covered:
            to_fetch = [c for c in chunks if not self.manifest.is_covered(c.chunk_id)]
            skipped = len(chunks) - len(to_fetch)
            if skipped > 0:
                logger.info("parallel_fetch skipping=%d already_covered", skipped)
        else:
            to_fetch = list(chunks)

        if not to_fetch:
            logger.info("parallel_fetch nothing_to_fetch")
            return []

        logger.info(
            "parallel_fetch_start total_chunks=%d workers=%d",
            len(to_fetch), self.max_workers,
        )

        results: list[ChunkFetchResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._fetch_single_chunk, chunk): chunk
                for chunk in to_fetch
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_rows = sum(r.row_count for r in results if r.success)
        logger.info(
            "parallel_fetch_complete succeeded=%d failed=%d total_rows=%d",
            succeeded, failed, total_rows,
        )
        return results


def build_merged_bars_from_cache(
    cache: HistoricalBarCache,
    manifest: CacheManifest,
    symbol: str,
    timeframe: str,
) -> list[MarketBar]:
    """Load the full merged bar sequence for a symbol from cache.

    This reads the single cache file (which has been incrementally
    updated by the parallel fetcher) and returns the sorted bars.
    """
    record = cache.load(symbol, timeframe)
    if record is None:
        return []
    return list(record.bars)
