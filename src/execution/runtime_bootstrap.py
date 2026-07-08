from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from src.core.config import AppConfig
from src.core.device import DeviceContext, build_device_context
from src.core.logging import configure_logging, get_logger
from src.core.time_utils import utc_now
from src.data.alpaca_client import AlpacaClient, BrokerDataClient, MockBrokerClient, credentials_available
from src.data.buffers import BarBuffer
from src.data.historical_data import HistoricalDataService
from src.data.stream_manager import PollingStreamManager
from src.execution.order_router import OrderRouter
from src.execution.paper_executor import PaperExecutor
from src.execution.reason_codes import ReasonCode
from src.execution.reconciliation import run_startup_reconciliation
from src.execution.session_guards import SessionGuard
from src.execution.startup_checks import StartupReport, run_startup_checks
from src.execution.symbol_validation import SymbolEligibilityResult, SymbolValidationService
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.storage import load_memory_bank, save_memory_bank
from src.memory.updater import AssociativeMemoryUpdater, build_memory_bank_from_bars
from src.models.factory import resolve_inference_model
from src.models.state_encoder import ModelStateEncoder
from src.monitoring.equity_log import EquityLogWriter
from src.monitoring.health import HealthState
from src.monitoring.metrics_logger import MetricsLogger
from src.monitoring.positions_status import PositionsStatusWriter
from src.monitoring.runtime_status import RuntimeStatusWriter
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.strategies.base import TradingPolicy
from src.strategies.brain_policy import BrainPolicy
from src.strategies.memory_policy import MemoryAwarePolicy

# Performance constants — kept as fallback for call sites that run before
# config is available. The canonical value lives in RuntimeConfig.max_parallel_workers.
COLD_START_MAX_WORKERS = 8


@dataclass
class RuntimeContext:
    config: AppConfig
    device_ctx: DeviceContext
    health: HealthState
    client: BrokerDataClient
    using_mock: bool
    portfolio: PortfolioState
    ledger: Ledger
    risk_manager: RiskManager
    router: OrderRouter
    policy: object  # BrainPolicy or MemoryAwarePolicy
    memory_bank: MemoryBank | None
    pipeline: FeaturePipeline
    buffers: dict[str, BarBuffer]
    eligible_symbols: list[str]
    session_guard: SessionGuard
    stream: PollingStreamManager
    metrics: MetricsLogger
    equity_writer: EquityLogWriter
    status_writer: RuntimeStatusWriter
    positions_writer: PositionsStatusWriter
    runtime_session_id: str
    started_at: datetime
    order_journal: object | None = None  # OrderJournal
    alerter: object | None = None  # WebhookAlerter


def resolve_runtime_symbols(
    symbol_results: dict[str, SymbolEligibilityResult],
    *,
    block_on_symbol_ineligible: bool,
) -> tuple[list[str], dict[str, SymbolEligibilityResult]]:
    eligible_symbols = [symbol for symbol, result in symbol_results.items() if result.eligible]
    ineligible_results = {symbol: result for symbol, result in symbol_results.items() if not result.eligible}
    if ineligible_results and block_on_symbol_ineligible:
        raise RuntimeError(f"Ineligible symbols blocked startup: {list(ineligible_results)}")
    if not eligible_symbols:
        raise RuntimeError("No eligible symbols remain after startup validation")
    return eligible_symbols, ineligible_results


def _log_startup_report(logger: object, report: StartupReport) -> None:
    for check in report.checks:
        logger.info(
            "startup_check",
            name=check.name,
            passed=check.passed,
            reason_code=check.reason_code,
            severity=check.severity,
            details=check.details,
        )
    logger.info("startup_summary", **report.summary())


def _log_symbol_results(logger: object, results: dict[str, SymbolEligibilityResult]) -> None:
    for symbol, result in results.items():
        logger.info(
            "symbol_eligibility",
            symbol=symbol,
            eligible=result.eligible,
            reason_code=result.reason_code,
            checked_at=result.checked_at.isoformat(),
            quarantined=result.quarantined,
            release_mode=result.release_mode,
            details=result.details,
        )


def _build_client(config: AppConfig) -> tuple[BrokerDataClient, bool]:
    if credentials_available():
        return (
            AlpacaClient(
                paper=config.broker.paper,
                data_feed=config.broker.data_feed,
            ),
            False,
        )
    if config.broker.live_enabled or not config.broker.paper:
        raise RuntimeError("Live mode requires Alpaca credentials and cannot fall back to mock mode")
    if config.broker.mock_data_on_missing_credentials:
        return MockBrokerClient(paper=True, data_feed="mock"), True
    raise RuntimeError("No Alpaca credentials available and mock data disabled")


def _compute_max_window(config: AppConfig) -> int:
    return max(
        config.runtime.warmup_bars,
        max(config.features.windows.values()),
        max(config.features.volatility_windows) + 1,
        max(config.features.momentum_windows) + 1,
        (max(config.memory.forward_horizons) + 1) if config.memory.enabled else 0,
    )


def _initialize_portfolio(config: AppConfig, startup_report: StartupReport | None) -> PortfolioState:
    cash = config.backtest.initial_cash
    equity = config.backtest.initial_cash
    if startup_report is not None and startup_report.account is not None:
        cash = float(startup_report.account.get("cash", cash) or cash)
        equity = float(startup_report.account.get("portfolio_value", startup_report.account.get("equity", equity)) or equity)
    return PortfolioState(cash=cash, equity=equity)


def _empty_memory_bank(
    *,
    config: AppConfig,
    feature_contract: object,
    latent_dim: int,
    encoder_fingerprint: str | None = None,
) -> MemoryBank:
    return MemoryBank(
        feature_contract=feature_contract,
        latent_dim=latent_dim,
        similarity_metric=config.memory.similarity_metric,
        max_items=config.memory.max_items,
        encoder_fingerprint=encoder_fingerprint,
        require_encoder_match=config.memory.require_encoder_match,
        meta={
            "source": "empty_runtime_memory",
            "encoder_fingerprint": encoder_fingerprint,
        },
    )


def _build_or_load_memory_bank(
    *,
    config: AppConfig,
    history_service: HistoricalDataService,
    pipeline: FeaturePipeline,
    encoder: ModelStateEncoder,
    eligible_symbols: list[str],
    feature_contract: object,
    latent_dim: int,
    logger: object,
) -> MemoryBank:
    persist_path = Path(config.memory.persist_path)
    if persist_path.exists() and not config.memory.rebuild_on_startup:
        try:
            bank = load_memory_bank(persist_path, map_location="cpu")
            bank.feature_contract.assert_matches(feature_contract, context=f"memory_bank[{persist_path.name}]")
            if bank.latent_dim != latent_dim:
                raise ValueError(f"Expected latent_dim {latent_dim}, got {bank.latent_dim}")
            # Encoder compatibility: if the persisted bank carries a
            # fingerprint, it must match the current encoder's fingerprint.
            # Otherwise the bank's latent vectors are from a different
            # encoder generation and any retrieval against them is unsafe.
            if (
                bank.encoder_fingerprint is not None
                and encoder.encoder_fingerprint is not None
                and bank.encoder_fingerprint != encoder.encoder_fingerprint
            ):
                raise ValueError(
                    "memory_bank encoder fingerprint mismatch: "
                    f"bank={bank.encoder_fingerprint}, encoder={encoder.encoder_fingerprint}. "
                    "Rebuild the bank against the current checkpoint."
                )
            # Stamp the bank if it was built before fingerprint stamping was
            # introduced. This makes future appends gate on the encoder.
            if bank.encoder_fingerprint is None and encoder.encoder_fingerprint is not None:
                bank.encoder_fingerprint = encoder.encoder_fingerprint
            logger.info("memory_bank_loaded", path=str(persist_path), **bank.summary())
            return bank
        except Exception as exc:
            logger.info("memory_bank_load_failed", path=str(persist_path), error=str(exc))

    history_limit = max(
        config.memory.build_history_bars,
        config.runtime.warmup_bars + max(config.memory.forward_horizons) + 1,
    )
    bars_by_symbol: dict[str, list[object]] = {}
    # Memory-bank cold start: parallelize the per-symbol historical bar
    # load just like the main warmup buffer load above.  Each call hits
    # a distinct cache key and the broker SDK is thread-safe; results
    # are stitched back together on the main thread.
    if eligible_symbols:
        max_workers = min(len(eligible_symbols), config.runtime.max_parallel_workers)
        timeframe = config.runtime.timeframe

        def _load_history(symbol: str) -> tuple[str, object]:
            try:
                return symbol, history_service.load_recent_bars(symbol, timeframe, history_limit)
            except Exception as exc:  # noqa: BLE001 - we want to log the per-symbol failure
                return symbol, exc

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_load_history, eligible_symbols))

        forward_horizon_max = max(config.memory.forward_horizons)
        for symbol, payload in results:
            if isinstance(payload, Exception):
                logger.info("memory_history_load_failed", symbol=symbol, error=str(payload))
                continue
            if len(payload) <= config.runtime.warmup_bars + forward_horizon_max:
                logger.info("memory_history_insufficient", symbol=symbol, bar_count=len(payload))
                continue
            bars_by_symbol[symbol] = payload
    if not bars_by_symbol:
        empty_bank = _empty_memory_bank(
            config=config,
            feature_contract=feature_contract,
            latent_dim=latent_dim,
            encoder_fingerprint=encoder.encoder_fingerprint,
        )
        logger.info("memory_bank_empty", **empty_bank.summary())
        return empty_bank
    try:
        bank = build_memory_bank_from_bars(
            bars_by_symbol=bars_by_symbol,
            pipeline=pipeline,
            encoder=encoder,
            memory_config=config.memory,
            warmup_bars=config.runtime.warmup_bars,
        )
        save_memory_bank(persist_path, bank)
        logger.info("memory_bank_built", path=str(persist_path), **bank.summary())
        return bank
    except Exception as exc:
        logger.info("memory_bank_build_failed", path=str(persist_path), error=str(exc))
        return _empty_memory_bank(
            config=config,
            feature_contract=feature_contract,
            latent_dim=latent_dim,
            encoder_fingerprint=encoder.encoder_fingerprint,
        )


async def build_runtime_context(config: AppConfig, *, confirm_live: bool = False) -> RuntimeContext:
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("trade_engine")
    device_ctx = build_device_context(config.device.preferred, config.device.use_amp)
    torch.set_float32_matmul_precision(config.device.matmul_precision)
    logger.info("device_info", device=str(device_ctx.device), use_amp=device_ctx.use_amp)

    started_at = utc_now()
    health = HealthState(started_at=started_at, last_tick_at=started_at)
    client, using_mock = _build_client(config)
    logger.info("runtime_mode", paper=config.broker.paper, live_enabled=config.broker.live_enabled, using_mock=using_mock)

    probe_symbol = config.runtime.symbols[0] if config.runtime.symbols else None
    startup_report = run_startup_checks(
        config=config,
        client=client,
        using_mock=using_mock,
        confirm_live_cli=confirm_live,
        probe_symbol=probe_symbol,
    )
    _log_startup_report(logger, startup_report)
    if not startup_report.safe_to_trade:
        health.block_trading(
            ReasonCode.SAFE_TO_TRADE_FALSE,
            timestamp=started_at,
            details={"stage": "startup", "failures": startup_report.summary()["failures"]},
        )
        raise RuntimeError(f"Unsafe startup conditions: {startup_report.summary()['failures']}")

    symbol_validator = SymbolValidationService(
        client=client,
        timeframe=config.runtime.timeframe,
        using_mock=using_mock,
    )
    symbol_results = symbol_validator.validate_all(config.runtime.symbols)
    _log_symbol_results(logger, symbol_results)
    try:
        eligible_symbols, ineligible_results = resolve_runtime_symbols(
            symbol_results,
            block_on_symbol_ineligible=config.runtime.block_on_symbol_ineligible,
        )
    except RuntimeError as exc:
        health.block_trading(
            ReasonCode.SAFE_TO_TRADE_FALSE,
            timestamp=started_at,
            details={"stage": "symbol_validation", "error": str(exc)},
        )
        raise

    portfolio = _initialize_portfolio(config, startup_report)
    ledger = Ledger()
    risk_manager = RiskManager(config.risk)
    risk_manager.reset_state()
    # Route orders through the real broker client whenever we have one.
    #
    # History / terminology note: an earlier iteration of the router treated
    # ``broker.paper=true`` as "do everything in-process via PaperExecutor"
    # and discarded the AlpacaClient entirely, even when Alpaca credentials
    # were available.  That meant "paper mode" never actually reached
    # Alpaca's paper-trading endpoint — orders were simulated locally and
    # the Alpaca dashboard stayed empty.  Users (reasonably) expect
    # ``broker.paper=true`` to mean "hit Alpaca's paper endpoint", which is
    # exactly what the AlpacaClient's own ``paper`` flag does (see
    # ``_build_client`` above: ``AlpacaClient(paper=config.broker.paper)``).
    #
    # So we now always pass the client through and always allow the router
    # to submit via it.  The ``paper`` vs ``live`` distinction is handled
    # by the AlpacaClient at construction time (paper.alpaca.markets vs
    # api.alpaca.markets), not at the router layer.  PaperExecutor remains
    # available as a safety fallback for callers that explicitly build a
    # router with no broker client (e.g. unit tests, or a future offline
    # simulation mode).
    router = OrderRouter(
        paper_executor=PaperExecutor(
            portfolio=portfolio,
            ledger=ledger,
            fee_bps=config.backtest.fee_bps,
            slippage_bps=config.backtest.slippage_bps,
        ),
        broker_client=client,
        allow_live=client is not None,
        max_submit_retries=config.execution.max_submit_retries,
        retry_backoff_sec=config.execution.retry_backoff_sec,
    )
    router.reset_runtime_state()
    for symbol, result in ineligible_results.items():
        router.quarantine_symbol(
            symbol,
            reason_code=result.reason_code,
            release_mode=result.release_mode or "until_restart",
            release_condition="manual runtime restart required after startup symbol validation",
            details={
                "eligible": result.eligible,
                "quarantined_from": "startup_validation",
                "checked_at": result.checked_at.isoformat(),
            },
        )

    if config.runtime.require_startup_reconciliation:
        reconciliation = run_startup_reconciliation(
            client=client,
            portfolio=portfolio,
            using_mock=using_mock,
            block_on_mismatch=config.runtime.block_on_reconciliation_mismatch,
        )
        router.seed_open_orders(reconciliation.hydrated_open_orders)
        # Seed the risk manager with existing positions so cooldowns
        # and duplicate-order windows apply from startup.
        risk_manager.seed_existing_positions(
            {sym: pos.qty for sym, pos in reconciliation.hydrated_positions.items()},
            timestamp=started_at,
        )
        for issue in reconciliation.issues:
            logger.info(
                "reconciliation_issue",
                severity=issue.severity,
                reason_code=issue.reason_code,
                details=issue.details,
            )
        logger.info("reconciliation_summary", **reconciliation.summary())
        if not reconciliation.safe_to_trade:
            health.block_trading(
                ReasonCode.SAFE_TO_TRADE_FALSE,
                timestamp=started_at,
                details={"stage": "reconciliation", **reconciliation.summary()},
            )
            raise RuntimeError(f"Unsafe reconciliation result: {reconciliation.summary()['issue_codes']}")

    history_service = HistoricalDataService(client=client)
    max_window = _compute_max_window(config)
    buffers = {symbol: BarBuffer(maxlen=max_window * 2) for symbol in eligible_symbols}
    # ------------------------------------------------------------------
    # Cold-start: parallelize the per-symbol historical bar load.
    #
    # The previous implementation walked ``eligible_symbols`` serially
    # and called ``history_service.load_recent_bars`` once per symbol.
    # In live/paper mode each call is a blocking broker request
    # (~500ms-2s); the cache-hit path is faster but still pays a
    # per-file deserialization cost.  At 10-20 symbols this added
    # ~10-30s to every cold start before the first tick could fire.
    #
    # We now use a small ``ThreadPoolExecutor`` (capped at the symbol
    # count or 8 workers, whichever is smaller) so the per-symbol
    # latency overlaps.  ``HistoricalDataService.load_recent_bars`` is
    # safe to call concurrently because each call hits a distinct
    # cache key/file and the broker SDK is thread-safe.  We then
    # serialize the buffer ``extend`` calls back on the main thread to
    # avoid touching ``BarBuffer`` from worker threads.
    # ------------------------------------------------------------------
    if eligible_symbols:
        max_workers = min(len(eligible_symbols), config.runtime.max_parallel_workers)
        timeframe = config.runtime.timeframe

        def _load_one(symbol: str) -> tuple[str, list]:
            return symbol, history_service.load_recent_bars(symbol, timeframe, max_window)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for symbol, bars in pool.map(_load_one, eligible_symbols):
                buffers[symbol].extend(bars)

    pipeline = FeaturePipeline(config.features, device=device_ctx.device)
    sample_symbol = eligible_symbols[0]
    sample_snapshot = pipeline.build(buffers[sample_symbol].snapshot(), symbol=sample_symbol)
    runtime_contract = sample_snapshot.contract
    artifacts = resolve_inference_model(
        model_config=config.model,
        feature_config=config.features,
        feature_contract=runtime_contract,
        device=device_ctx.device,
        default_model_kind="brain",
        # Paper/live runtime must NEVER fall back to a randomly-initialized
        # brain. Reject the run before it sends an order based on noise.
        allow_untrained=False,
        device_config=config.device,
    )
    logger.info(
        "model_loaded",
        model_kind=artifacts.model_kind,
        checkpoint=config.model.checkpoint_path,
        feature_dim=artifacts.feature_contract.feature_dim,
        regime_dim=artifacts.feature_contract.regime_dim,
    )
    # --- Frozen-brain enforcement ---
    if not config.runtime.allow_online_model_updates:
        artifacts.model.eval()
        for param in artifacts.model.parameters():
            param.requires_grad = False
        logger.info(
            "frozen_brain_policy",
            allow_online_model_updates=False,
            allow_online_memory_updates=config.runtime.allow_online_memory_updates,
            status="model_weights_frozen",
        )
    else:
        logger.info(
            "frozen_brain_policy",
            allow_online_model_updates=True,
            allow_online_memory_updates=config.runtime.allow_online_memory_updates,
            status="model_weights_mutable_WARNING",
        )
    base_policy = BrainPolicy(
        model=artifacts.model,
        device=device_ctx.device,
        feature_contract=artifacts.feature_contract,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
        default_target_fraction=config.runtime.default_target_fraction,
    )
    policy: object = base_policy
    memory_bank: MemoryBank | None = None
    if config.memory.enabled:
        encoder = ModelStateEncoder(
            model=artifacts.model,
            device=device_ctx.device,
            feature_contract=artifacts.feature_contract,
            expected_latent_dim=config.memory.latent_dim,
            encoder_fingerprint=artifacts.encoder_fingerprint,
        )
        sample_encoded = encoder.encode(sample_snapshot)
        memory_bank = _build_or_load_memory_bank(
            config=config,
            history_service=history_service,
            pipeline=pipeline,
            encoder=encoder,
            eligible_symbols=eligible_symbols,
            feature_contract=artifacts.feature_contract,
            latent_dim=sample_encoded.latent.shape[0],
            logger=logger,
        )
        # Enforce frozen-brain policy on memory updates
        effective_memory_config = config.memory
        if not config.runtime.allow_online_memory_updates:
            effective_memory_config = config.memory.model_copy(
                update={"runtime_update_mode": "disabled"}
            )
            logger.info(
                "memory_updates_disabled",
                reason="allow_online_memory_updates=false",
                original_mode=config.memory.runtime_update_mode,
            )
        memory_updater = AssociativeMemoryUpdater(memory_bank, effective_memory_config)
        # Restore pending observations from disk if configured
        if effective_memory_config.persist_pending_observations:
            restored = memory_updater.load_pending(effective_memory_config.pending_store_path)
            if restored > 0:
                logger.info("pending_observations_loaded", count=restored, path=effective_memory_config.pending_store_path)
        # Post-Great-Simplification: the minimal bank has no
        # organization layer (no prototypes, no ESOM map, no dynamic
        # units), so there is nothing for a background refresher to
        # do. ``async_memory_refresh`` is now a no-op even when True.
        # Platt calibrator (load from disk if available)
        _platt_calibrator = None
        if config.training.platt_calibration_enabled:
            from src.training.platt_calibration import PlattCalibrator
            _platt_calibrator = PlattCalibrator()
            _platt_calibrator.load(Path("runs/platt_calibration.json"))

        policy = MemoryAwarePolicy(
            base_policy=base_policy,
            encoder=encoder,
            bank=memory_bank,
            updater=memory_updater,
            platt_calibrator=_platt_calibrator,
        )
        logger.info(
            "memory_runtime_ready",
            enabled=True,
            runtime_update_mode=effective_memory_config.runtime_update_mode,
            allow_online_memory_updates=config.runtime.allow_online_memory_updates,
            **memory_bank.summary(),
        )
    if isinstance(policy, TradingPolicy):
        policy.reset_all_state(reason="runtime_start")
    session_guard = SessionGuard(config.runtime)
    stream = PollingStreamManager(
        client=client,
        symbols=eligible_symbols,
        timeframe=config.runtime.timeframe,
        poll_interval_sec=config.broker.polling_interval_sec,
        circuit_breaker_threshold=config.runtime.stream_circuit_breaker_limit,
        empty_poll_warn_every=config.runtime.stream_empty_poll_warn_frequency,
    )
    metrics = MetricsLogger()
    runtime_session_id = started_at.strftime("%Y%m%d")
    equity_writer = EquityLogWriter(
        max_bytes=config.runtime.log_rotation_max_bytes,
        max_backups=config.runtime.log_rotation_max_backups,
    )
    status_writer = RuntimeStatusWriter(
        max_bytes=config.runtime.log_rotation_max_bytes,
        max_backups=config.runtime.log_rotation_max_backups,
    )
    positions_writer = PositionsStatusWriter()

    # Order journal
    order_journal = None
    if config.execution.order_journal_enabled:
        from src.execution.order_journal import OrderJournal
        order_journal = OrderJournal(
            path=config.execution.order_journal_path,
            max_bytes=config.execution.order_journal_max_bytes,
            max_backups=config.execution.order_journal_max_backups,
        )

    # Webhook alerting
    alerter = None
    if config.runtime.alert_webhook_url:
        from src.monitoring.alerting import WebhookAlerter
        alerter = WebhookAlerter(
            webhook_url=config.runtime.alert_webhook_url,
            timeout_sec=config.runtime.alert_timeout_sec,
        )

    return RuntimeContext(
        config=config,
        device_ctx=device_ctx,
        health=health,
        client=client,
        using_mock=using_mock,
        portfolio=portfolio,
        ledger=ledger,
        risk_manager=risk_manager,
        router=router,
        policy=policy,
        memory_bank=memory_bank,
        pipeline=pipeline,
        buffers=buffers,
        eligible_symbols=eligible_symbols,
        session_guard=session_guard,
        stream=stream,
        metrics=metrics,
        equity_writer=equity_writer,
        status_writer=status_writer,
        positions_writer=positions_writer,
        runtime_session_id=runtime_session_id,
        started_at=started_at,
        order_journal=order_journal,
        alerter=alerter,
    )
