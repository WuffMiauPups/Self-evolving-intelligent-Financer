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
from src.data.historical_data import HistoricalDataService, parse_date_or_datetime


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache historical Alpaca bars")
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/default.yaml", "configs/training.yaml"],
        help="Config file(s), later files override earlier files.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache coverage and refetch requested ranges.")
    parser.add_argument("--allow-mock", action="store_true", help="Allow explicit mock historical generation when credentials are missing.")
    parser.add_argument("--inspect-cache", action="store_true", help="Only inspect local cache coverage without fetching missing data.")
    args = parser.parse_args()

    config = load_config(*args.config)
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("fetch_historical")
    cache = HistoricalBarCache(config.data.cache_dir)
    start = parse_date_or_datetime(
        config.data.start_date,
        default=datetime.now(UTC) - timedelta(days=365 * 2),
    )
    end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))

    if args.inspect_cache:
        for symbol in config.data.symbols:
            summary = cache.summarize(symbol, config.data.timeframe)
            logger.info("historical_cache_summary", symbol=symbol, summary=summary)
        return 0

    if credentials_available():
        client = AlpacaClient(paper=config.broker.paper, data_feed=config.broker.data_feed)
        using_mock = False
    elif args.allow_mock:
        client = MockBrokerClient(paper=True, data_feed="mock")
        using_mock = True
    else:
        client = None
        using_mock = False

    service = HistoricalDataService(
        client=client,
        cache=cache,
        timezone_normalization=config.data.timezone_normalization,
        max_missing_gap_handling=config.data.max_missing_gap_handling,
    )
    _, reports = service.load_bars_for_symbols(
        config.data.symbols,
        timeframe=config.data.timeframe,
        start=start,
        end=end,
        force_refresh=args.force_refresh or config.data.force_refresh,
    )
    for report in reports:
        logger.info(
            "historical_fetch_complete",
            symbol=report.symbol,
            timeframe=report.timeframe,
            cache_hit=report.cache_hit,
            row_count=report.row_count,
            cache_path=report.cache_path,
            gap_count=report.gap_count,
            max_gap_seconds=report.max_gap_seconds,
            using_mock=using_mock,
            source=report.source,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
