"""Fetch and cache historical Alpaca bars with parallel chunked pipeline.

Supports:
- Chunked multi-year fetches with configurable granularity
- Parallel fetch with bounded concurrency and retry/backoff
- Resume-safe execution from persistent manifest
- Rich inspect-cache diagnostics
- Merge/dedupe/validate after fetch
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.fetch_planner import (
    CacheManifest,
    inspect_cache_coverage,
    merge_dedupe_validate,
    plan_chunks,
)
from src.data.historical_data import HistoricalDataService, parse_date_or_datetime
from src.data.parallel_fetch import ParallelFetchExecutor, build_merged_bars_from_cache


def _print_inspect_report(reports: list[dict], manifest_summary: dict) -> None:
    """Print a human-readable cache inspection report."""
    print(f"\n{'='*60}")
    print(f"  CACHE COVERAGE REPORT")
    print(f"  manifest_version={manifest_summary.get('version', '?')}"
          f"  total_chunks={manifest_summary.get('total_chunks', 0)}")
    print(f"{'='*60}\n")

    all_covered = True
    for report in reports:
        status = "COMPLETE" if report["fully_covered"] else "INCOMPLETE"
        if not report["fully_covered"]:
            all_covered = False
        print(
            f"  {report['symbol']:8s} {report['timeframe']:6s}"
            f"  chunks={report['chunk_count']:4d}"
            f"  rows={report['total_rows']:>10,}"
            f"  [{status}]"
            f"  last_updated={report['last_updated']}"
        )
        if not report["fully_covered"]:
            print(f"           missing_days={report['missing_days']}")
            for ms, me in report["missing_ranges"][:5]:
                print(f"           gap: {ms} → {me}")
            if len(report["missing_ranges"]) > 5:
                print(f"           ... and {len(report['missing_ranges']) - 5} more gaps")
    print()
    if all_covered and reports:
        print("  All symbols fully covered. Ready for offline training.")
    elif reports:
        print("  Some symbols have missing coverage. Run fetch to complete.")
    else:
        print("  No symbols configured or no data cached.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache historical Alpaca bars")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Config file(s), later files override earlier files. "
            "Default: configs/default.yaml + configs/training.yaml. "
            "Passing --config REPLACES the default stack (it does not append to it)."
        ),
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and refetch all chunks.")
    parser.add_argument("--allow-mock", action="store_true", help="Generate synthetic data when credentials are missing.")
    parser.add_argument("--inspect-cache", action="store_true", help="Only inspect cache coverage without fetching.")
    parser.add_argument("--validate", action="store_true", help="Run merge/dedupe/validate on cached data after fetch.")
    parser.add_argument("--workers", type=int, default=None, help="Override max_fetch_workers.")
    parser.add_argument("--chunk-size", type=str, default=None, choices=["monthly", "quarterly", "yearly"],
                        help="Override chunk size.")
    parser.add_argument("--cache-only", action="store_true", help="Use only cached data, do not fetch.")
    args = parser.parse_args()
    # argparse action='append' appends to a non-None default instead of
    # replacing it, so the default stack is applied post-parse.
    if args.config is None:
        args.config = ["configs/default.yaml", "configs/training.yaml"]

    config = load_config(*args.config)
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("fetch_historical")

    pipeline_config = config.data.pipeline
    cache = HistoricalBarCache(config.data.cache_dir)
    manifest = CacheManifest.load(pipeline_config.manifest_path)

    start = parse_date_or_datetime(
        config.data.start_date,
        default=datetime.now(UTC) - timedelta(days=365 * 2),
    )
    end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))
    symbols = config.data.symbols
    timeframe = config.data.timeframe

    # --- Inspect mode ---
    if args.inspect_cache:
        reports = inspect_cache_coverage(manifest, symbols, timeframe, start, end)
        _print_inspect_report(reports, manifest.summary())

        # Also show legacy single-file cache info
        for symbol in symbols:
            summary = cache.summarize(symbol, timeframe)
            if summary:
                logger.info("legacy_cache_summary", symbol=symbol, summary=summary)
        return 0

    # --- Cache-only mode ---
    if args.cache_only:
        reports = inspect_cache_coverage(manifest, symbols, timeframe, start, end)
        incomplete = [r for r in reports if not r["fully_covered"]]
        if incomplete:
            for r in incomplete:
                logger.error("cache_only_incomplete symbol=%s missing_days=%.1f", r["symbol"], r["missing_days"])
            return 1
        logger.info("cache_only_complete all_symbols_covered")
        return 0

    # --- Plan chunks ---
    chunk_size = args.chunk_size or pipeline_config.chunk_size
    all_chunks = plan_chunks(symbols, timeframe, start, end, chunk_size=chunk_size)
    logger.info(
        "fetch_plan total_chunks=%d symbols=%d chunk_size=%s range=%s→%s",
        len(all_chunks), len(symbols), chunk_size,
        start.isoformat(), end.isoformat(),
    )

    if args.force_refresh:
        to_fetch = all_chunks
    else:
        to_fetch = manifest.filter_uncovered(all_chunks)

    if not to_fetch:
        logger.info("fetch_plan_complete all_chunks_covered=%d", len(all_chunks))
        if args.validate:
            _run_validation(cache, manifest, symbols, timeframe, logger)
        return 0

    logger.info("fetch_plan_pending chunks_to_fetch=%d already_covered=%d",
                len(to_fetch), len(all_chunks) - len(to_fetch))

    # --- Build client ---
    if credentials_available():
        client = AlpacaClient(
            paper=config.broker.paper,
            data_feed=config.broker.data_feed,
            bar_adjustment=config.data.bar_adjustment,
        )
    elif args.allow_mock:
        client = MockBrokerClient(paper=True, data_feed="mock")
    else:
        logger.error("no_credentials_and_no_mock set ALPACA_API_KEY/ALPACA_API_SECRET or use --allow-mock")
        return 1

    # --- Execute parallel fetch ---
    max_workers = args.workers or pipeline_config.max_fetch_workers
    executor = ParallelFetchExecutor(
        client=client,
        cache=cache,
        manifest=manifest,
        max_workers=max_workers,
        retry_count=pipeline_config.retry_count,
        retry_backoff_sec=pipeline_config.retry_backoff_sec,
        timezone_normalization=config.data.timezone_normalization,
    )
    results = executor.execute(to_fetch, skip_covered=not args.force_refresh)

    # Report results
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_rows = sum(r.row_count for r in results if r.success)
    print(
        f"\nfetch_complete"
        f" succeeded={succeeded}"
        f" failed={failed}"
        f" total_rows={total_rows:,}"
    )

    for result in results:
        if not result.success:
            logger.error(
                "chunk_failed chunk_id=%s error=%s",
                result.chunk.chunk_id, result.error,
            )

    # --- Optional validation ---
    if args.validate:
        _run_validation(cache, manifest, symbols, timeframe, logger)

    return 1 if failed > 0 else 0


def _run_validation(cache, manifest, symbols, timeframe, logger):
    """Run merge/dedupe/validate on all cached data per symbol."""
    for symbol in symbols:
        record = cache.load(symbol, timeframe)
        if record is None:
            logger.warning("validate_skip symbol=%s reason=no_cache", symbol)
            continue
        _, report = merge_dedupe_validate(
            [record.bars],
            symbol=symbol,
            timeframe=timeframe,
        )
        logger.info(
            "validate_complete symbol=%s rows=%d dupes=%d monotonic=%s gaps=%d max_gap=%.0fs",
            symbol, report.total_rows, report.duplicates_removed,
            report.is_monotonic, report.gap_count, report.max_gap_seconds,
        )
        if report.gaps:
            for gap_from, gap_to, gap_sec in report.gaps[:5]:
                logger.info("  gap: %s → %s (%.0fs)", gap_from, gap_to, gap_sec)

        # Mark all chunks for this symbol as validated
        for chunk_id, coverage in manifest.chunks.items():
            if coverage.symbol == symbol and coverage.timeframe == timeframe:
                coverage.validated = True
        manifest.save()


if __name__ == "__main__":
    raise SystemExit(main())
