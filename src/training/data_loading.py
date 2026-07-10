from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import HistoricalDataService, generate_synthetic_bars, parse_date_or_datetime
from src.data.schemas import MarketBar


def _advisory_stylized_facts_check(bars_by_symbol: dict[str, list[MarketBar]], logger: object) -> None:
    """Advisory data-quality gate: log suspected unadjusted splits / data errors.

    Never raises — training proceeds regardless; the warning tells the
    operator the cache likely needs an adjusted refetch.
    """
    try:
        from src.evaluation.stylized_facts import compute_stylized_facts

        report = compute_stylized_facts(bars_by_symbol)
        if report.suspicious:
            logger.warning(
                "stylized_facts_suspicious",
                flagged=[
                    (symbol, ts.isoformat() if ts is not None else None, round(ret, 6))
                    for symbol, ts, ret in report.flagged],
                note="Single-bar |return| > 20% usually means an unadjusted split or bad data; refetch with adjusted bars.",
            )
    except Exception as exc:  # noqa: BLE001 - advisory only, never block training
        logger.warning("stylized_facts_check_failed", error=str(exc))


def _build_training_client(config: AppConfig) -> tuple[object | None, bool]:
    if credentials_available():
        return (
            AlpacaClient(
                paper=config.broker.paper,
                data_feed=config.broker.data_feed,
                bar_adjustment=config.data.bar_adjustment,
            ),
            False,
        )
    if config.training.use_real_historical_data:
        return None, False
    return MockBrokerClient(paper=True, data_feed="mock"), True


def _resolve_bars_by_symbol(config: AppConfig) -> tuple[dict[str, list[MarketBar]], dict[str, object]]:
    logger = get_logger("training")
    fast = config.training.fast_mode

    # Fast-mode: determine effective symbol list
    effective_symbols = list(config.data.symbols)
    if fast.enabled and fast.symbols:
        effective_symbols = [s for s in fast.symbols if s in set(config.data.symbols)]
        skipped = [s for s in fast.symbols if s not in set(config.data.symbols)]
        if skipped:
            logger.warning(
                "fast_mode_symbol_not_in_config",
                skipped=skipped,
                configured=list(config.data.symbols),
                note="These fast_mode.symbols are not in data.symbols and will be ignored.",
            )
        if not effective_symbols:
            raise RuntimeError(
                f"fast_mode.symbols {list(fast.symbols)!r} has no overlap with "
                f"configured data.symbols {list(config.data.symbols)!r}"
            )

    if config.training.use_real_historical_data:
        client, using_mock = _build_training_client(config)
        start = parse_date_or_datetime(
            config.data.start_date,
            default=datetime.now(UTC) - timedelta(days=365 * 2),
        )
        end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))

        # Fast-mode: override date range
        if fast.enabled:
            if fast.fast_months is not None:
                start = datetime.now(UTC) - timedelta(days=30 * fast.fast_months)
                logger.info(
                    "fast_mode_date_range_override",
                    fast_months=fast.fast_months,
                    effective_start=start.isoformat(),
                )
            elif fast.fast_start_date is not None:
                start = parse_date_or_datetime(fast.fast_start_date, default=start)
                logger.info("fast_mode_date_range_override", fast_start_date=fast.fast_start_date)
            if fast.fast_end_date is not None:
                end = parse_date_or_datetime(fast.fast_end_date, default=end)

        service = HistoricalDataService(
            client=client,
            cache=HistoricalBarCache(config.data.cache_dir),
            timezone_normalization=config.data.timezone_normalization,
            max_missing_gap_handling=config.data.max_missing_gap_handling,
        )
        bars_by_symbol, reports = service.load_bars_for_symbols(
            effective_symbols,
            timeframe=config.data.timeframe,
            start=start,
            end=end,
            force_refresh=config.data.force_refresh,
        )
        for report in reports:
            logger.info(
                "historical_fetch_report",
                symbol=report.symbol,
                timeframe=report.timeframe,
                cache_hit=report.cache_hit,
                row_count=report.row_count,
                requested_start=report.requested_start.isoformat(),
                requested_end=report.requested_end.isoformat(),
                fetched_ranges=[(left.isoformat(), right.isoformat()) for left, right in report.fetched_ranges],
                cache_path=report.cache_path,
                source=report.source,
                using_mock=using_mock,
                gap_count=report.gap_count,
                max_gap_seconds=report.max_gap_seconds,
            )
        if client is None and any(not report.cache_hit for report in reports):
            raise RuntimeError("Real historical training requested bars outside cached coverage and no Alpaca credentials are available")
        _advisory_stylized_facts_check(bars_by_symbol, logger)
        return bars_by_symbol, {
            "using_mock": using_mock,
            "source": "historical_cache",
            "timeframe": config.data.timeframe,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "symbols": effective_symbols,
        }

    synthetic_count = max(
        config.training.sequence_length * 8,
        config.runtime.warmup_bars + max(config.training.forward_horizons) + 128,
    )
    bars_by_symbol = {
        symbol: generate_synthetic_bars(symbol, synthetic_count)
        for symbol in effective_symbols
    }
    logger.info(
        "synthetic_training_data",
        symbols=effective_symbols,
        bar_count=synthetic_count,
    )
    return bars_by_symbol, {
        "using_mock": True,
        "source": "synthetic",
        "timeframe": config.data.timeframe,
        "symbols": effective_symbols,
    }


_FAST_SMALL_MODEL_OVERRIDES = {
    "cnn_channels": [8, 16],
    "temporal_blocks": 2,
    "branch_hidden_dim": 16,
    "branch_output_dim": 12,
    "soma_hidden_dim": 16,
}
