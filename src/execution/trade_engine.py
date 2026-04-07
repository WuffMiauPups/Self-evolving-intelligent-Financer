from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from src.core.config import AppConfig
from src.core.device import build_device_context
from src.core.logging import configure_logging, get_logger
from src.core.time_utils import utc_now
from src.data.alpaca_client import AlpacaClient, BrokerDataClient, MockBrokerClient, credentials_available
from src.data.buffers import BarBuffer
from src.data.historical_data import HistoricalDataService
from src.data.stream_manager import PollingStreamManager
from src.execution.order_router import OrderRouter
from src.execution.paper_executor import PaperExecutor
from src.execution.reason_codes import ReasonCode
from src.execution.reconciliation import build_order_states, hydrate_portfolio_positions, run_startup_reconciliation
from src.execution.schemas import OrderIntent
from src.execution.session_guards import SessionGuard
from src.execution.startup_checks import StartupReport, run_startup_checks
from src.execution.symbol_validation import SymbolEligibilityResult, SymbolValidationService
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.retrieval import AssociativeRetriever
from src.memory.storage import load_memory_bank, save_memory_bank
from src.memory.updater import AssociativeMemoryUpdater, build_memory_bank_from_bars
from src.models.factory import resolve_inference_model
from src.models.state_encoder import ModelStateEncoder
from src.monitoring.equity_log import EquityLogWriter, EquitySnapshot
from src.monitoring.health import HealthState
from src.monitoring.metrics_logger import MetricsLogger
from src.monitoring.runtime_status import RuntimeStatusSnapshot, RuntimeStatusWriter
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.strategies.brain_policy import BrainPolicy
from src.strategies.memory_policy import MemoryAwarePolicy


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


def _sync_live_state(
    *,
    client: BrokerDataClient,
    portfolio: PortfolioState,
    router: OrderRouter,
) -> list[object]:
    positions = client.get_positions()
    hydrate_portfolio_positions(portfolio, positions)
    open_orders = build_order_states(client.get_open_orders())
    return router.sync_open_orders(open_orders)


def _empty_memory_bank(
    *,
    config: AppConfig,
    feature_contract: object,
    latent_dim: int,
) -> MemoryBank:
    return MemoryBank(
        feature_contract=feature_contract,
        latent_dim=latent_dim,
        similarity_metric=config.memory.similarity_metric,
        max_items=config.memory.max_items,
        require_encoder_match=config.memory.require_encoder_match,
        meta={"source": "empty_runtime_memory"},
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
            if config.memory.dynamic_growth_enabled and not bank.memory_units:
                bank.refresh_organization(config)
                save_memory_bank(persist_path, bank)
            logger.info("memory_bank_loaded", path=str(persist_path), **bank.summary())
            return bank
        except Exception as exc:
            logger.info("memory_bank_load_failed", path=str(persist_path), error=str(exc))

    history_limit = max(
        config.memory.build_history_bars,
        config.runtime.warmup_bars + max(config.memory.forward_horizons) + 1,
    )
    bars_by_symbol: dict[str, list[object]] = {}
    for symbol in eligible_symbols:
        try:
            bars = history_service.load_recent_bars(symbol, config.runtime.timeframe, history_limit)
        except Exception as exc:
            logger.info("memory_history_load_failed", symbol=symbol, error=str(exc))
            continue
        if len(bars) <= config.runtime.warmup_bars + max(config.memory.forward_horizons):
            logger.info("memory_history_insufficient", symbol=symbol, bar_count=len(bars))
            continue
        bars_by_symbol[symbol] = bars
    if not bars_by_symbol:
        empty_bank = _empty_memory_bank(
            config=config,
            feature_contract=feature_contract,
            latent_dim=latent_dim,
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
        )


async def _run_async_trade_loop(config: AppConfig, *, confirm_live: bool = False) -> None:
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
    router = OrderRouter(
        paper_executor=PaperExecutor(
            portfolio=portfolio,
            ledger=ledger,
            fee_bps=config.backtest.fee_bps,
            slippage_bps=config.backtest.slippage_bps,
        ),
        broker_client=client if not config.broker.paper else None,
        allow_live=config.broker.live_enabled and not config.broker.paper,
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
    for symbol in eligible_symbols:
        buffers[symbol].extend(
            history_service.load_recent_bars(symbol, config.runtime.timeframe, max_window)
        )

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
    )
    policy: object = base_policy
    memory_bank: MemoryBank | None = None
    if config.memory.enabled:
        encoder = ModelStateEncoder(
            model=artifacts.model,
            device=device_ctx.device,
            feature_contract=artifacts.feature_contract,
            expected_latent_dim=config.memory.latent_dim,
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
        policy = MemoryAwarePolicy(
            base_policy=base_policy,
            encoder=encoder,
            retriever=AssociativeRetriever(memory_bank, config.memory),
            updater=memory_updater,
            require_strong_match_for_trade=config.memory.require_strong_match_for_trade,
            fusion_config=config.memory.decision_fusion,
        )
        logger.info(
            "memory_runtime_ready",
            enabled=True,
            runtime_update_mode=effective_memory_config.runtime_update_mode,
            allow_online_memory_updates=config.runtime.allow_online_memory_updates,
            require_strong_match_for_trade=config.memory.require_strong_match_for_trade,
            decision_fusion_enabled=config.memory.decision_fusion.enabled,
            topology_growth_enabled=config.memory.topology_growth.enabled,
            **memory_bank.summary(),
        )
    if hasattr(policy, "reset_all_state"):
        policy.reset_all_state(reason="runtime_start")
    session_guard = SessionGuard(config.runtime)
    stream = PollingStreamManager(
        client=client,
        symbols=eligible_symbols,
        timeframe=config.runtime.timeframe,
        poll_interval_sec=config.broker.polling_interval_sec,
    )
    metrics = MetricsLogger()
    runtime_session_id = started_at.strftime("%Y%m%d")
    equity_writer = EquityLogWriter()
    status_writer = RuntimeStatusWriter()
    last_decision_payload: dict | None = None
    last_risk_payload: dict | None = None
    last_symbol: str = ""
    warmup_logged: set[str] = set()
    step = 0
    async for latest_bars in stream.stream():
        if not health.safe_to_trade:
            logger.info(
                "safe_to_trade_blocked",
                reason_code=health.blocked_reason_code,
                details=health.blocked_details,
            )
            break
        try:
            if stream.last_errors:
                logger.info("stream_symbol_errors", errors=stream.last_errors)
                for symbol in stream.last_errors:
                    metrics.increment_reason("stream_error", ReasonCode.SYMBOL_NO_MARKET_DATA)
            if not config.broker.paper and router.broker_client is not None:
                transitions = _sync_live_state(client=client, portfolio=portfolio, router=router)
                for transition in transitions:
                    logger.info(
                        "order_transition",
                        symbol=transition.symbol,
                        client_order_id=transition.client_order_id,
                        status=transition.status,
                        filled_qty=transition.filled_qty,
                        remaining_qty=transition.remaining_qty,
                        terminal=transition.terminal,
                    )

            latest_prices: dict[str, float] = {}
            for symbol, bar in latest_bars.items():
                buffers[symbol].append(bar)
                latest_prices[symbol] = bar.close
            if not latest_prices:
                continue
            tick_time = max(bar.timestamp for bar in latest_bars.values())
            portfolio.mark_to_market(latest_prices, tick_time)
            equity_writer.write(
                EquitySnapshot(
                    ts=tick_time,
                    equity=float(portfolio.equity),
                    realized_pnl=float(getattr(portfolio, "realized_pnl", 0.0)),
                    unrealized_pnl=float(portfolio.equity - portfolio.cash
                                          - getattr(portfolio, "realized_pnl", 0.0)),
                    fills=len(ledger.fills),
                    session_id=runtime_session_id,
                )
            )
            broker_clock = None
            try:
                broker_clock = client.get_clock()
            except Exception as exc:
                logger.info("broker_clock_unavailable", error=str(exc))

            for symbol in eligible_symbols:
                if not health.safe_to_trade:
                    break
                quarantine = router.is_symbol_quarantined(symbol)
                if quarantine is not None and quarantine.release_mode == "until_restart":
                    logger.info(
                        "symbol_quarantined",
                        symbol=symbol,
                        reason_code=quarantine.reason_code,
                        quarantined_at=quarantine.quarantined_at.isoformat(),
                        release_mode=quarantine.release_mode,
                        release_condition=quarantine.release_condition,
                        details=quarantine.details,
                    )
                    metrics.increment_reason("symbol_quarantine", quarantine.reason_code)
                    continue
                if symbol not in latest_bars:
                    continue
                if len(buffers[symbol]) < config.runtime.warmup_bars:
                    if symbol not in warmup_logged:
                        logger.info(
                            "warmup_pending",
                            symbol=symbol,
                            buffer_size=len(buffers[symbol]),
                            required_warmup_bars=config.runtime.warmup_bars,
                        )
                        warmup_logged.add(symbol)
                    continue
                warmup_logged.discard(symbol)
                if hasattr(policy, "prepare_symbol_history"):
                    finalized_items = policy.prepare_symbol_history(symbol, buffers[symbol].snapshot())
                    if finalized_items:
                        logger.info(
                            "memory_update",
                            symbol=symbol,
                            finalized_count=len(finalized_items),
                            bank_items=0 if memory_bank is None else len(memory_bank),
                            bank_clusters=0 if memory_bank is None else len(memory_bank.prototypes),
                        )
                        if memory_bank is not None:
                            save_memory_bank(config.memory.persist_path, memory_bank)
                    # Persist pending observations for restart safety
                    if (
                        hasattr(policy, "updater")
                        and policy.updater is not None
                        and config.memory.persist_pending_observations
                    ):
                        policy.updater.save_pending(config.memory.pending_store_path)

                bar = latest_bars[symbol]
                guard_result = session_guard.evaluate(
                    symbol=symbol,
                    bar=bar,
                    now=utc_now(),
                    broker_clock=broker_clock,
                )
                logger.info(
                    "session_guard",
                    symbol=symbol,
                    allowed=guard_result.allowed,
                    reason_code=guard_result.reason_code,
                    details=guard_result.details,
                )
                if not guard_result.allowed:
                    metrics.increment_reason("session_guard", guard_result.reason_code)
                    if guard_result.reason_code == ReasonCode.DATA_GAP_BLOCK:
                        policy.reset_symbol_state(symbol, reason="data_gap_block")
                    continue

                snapshot = pipeline.build(buffers[symbol].snapshot(), symbol=symbol)
                proposal, diagnostics = policy.propose(snapshot)
                memory_support = diagnostics.get("memory_support")
                if memory_support is not None:
                    logger.info("memory_retrieval", symbol=symbol, **memory_support)
                if diagnostics.get("state_reset_reason") is not None:
                    logger.info(
                        "state_reset",
                        symbol=symbol,
                        reason=diagnostics["state_reset_reason"],
                    )
                logger.info(
                    "model_decision",
                    symbol=symbol,
                    confidence=proposal.confidence,
                    target_fraction=proposal.target_fraction,
                    action=proposal.action,
                    branch_confidences=diagnostics["branch_confidences"],
                    soma_weights=diagnostics["soma_weights"],
                    soma_activation=diagnostics["soma_activation"],
                )
                # Capture for structured runtime status snapshot.
                _fusion_diag = diagnostics.get("decision_fusion") or {}
                last_decision_payload = {
                    "action": int(proposal.action),
                    "confidence": float(proposal.confidence),
                    "target_fraction": float(proposal.target_fraction),
                    "fusion_level": _fusion_diag.get("support_level") if isinstance(_fusion_diag, dict) else None,
                }
                last_symbol = symbol
                if hasattr(policy, "record_snapshot"):
                    policy.record_snapshot(symbol)

                decision = risk_manager.evaluate(proposal, portfolio)
                logger.info(
                    "risk_decision",
                    symbol=symbol,
                    approved=decision.approved,
                    reasons=decision.reasons,
                )
                last_risk_payload = {
                    "approved": bool(decision.approved),
                    "reasons": [str(r) for r in decision.reasons],
                    "side": decision.side,
                    "quantity": float(decision.quantity),
                }
                if not decision.approved:
                    metrics.increment("risk_rejections")
                    for reason in decision.reasons:
                        metrics.increment_reason("risk_rejection", str(reason))
                    continue

                submission_guard = session_guard.evaluate(
                    symbol=symbol,
                    bar=bar,
                    now=utc_now(),
                    broker_clock=broker_clock,
                )
                if not submission_guard.allowed:
                    logger.info(
                        "order_submission_blocked",
                        symbol=symbol,
                        reason_code=submission_guard.reason_code,
                        details=submission_guard.details,
                    )
                    metrics.increment_reason("submission_guard", submission_guard.reason_code)
                    continue

                side = decision.side or "buy"
                client_order_id, sequence = router.next_client_order_id(
                    session_id=runtime_session_id,
                    symbol=symbol,
                    side=side,
                    timestamp=snapshot.timestamp,
                )
                intent = OrderIntent(
                    symbol=symbol,
                    side=side,
                    quantity=decision.quantity,
                    client_order_id=client_order_id,
                    reference_price=snapshot.last_price,
                    timestamp=snapshot.timestamp,
                    paper=config.broker.paper,
                    risk_approved=True,
                    session_id=runtime_session_id,
                    sequence=sequence,
                )
                logger.info(
                    "order_submit_attempt",
                    symbol=symbol,
                    client_order_id=client_order_id,
                    side=side,
                    quantity=decision.quantity,
                    sequence=sequence,
                )
                result = router.route(intent)
                logger.info(
                    "order_transition",
                    symbol=result.symbol,
                    side=result.side,
                    quantity=result.quantity,
                    client_order_id=result.client_order_id,
                    order_id=result.order_id,
                    status=result.status,
                    broker_status=result.broker_status,
                    filled_qty=result.filled_qty,
                    remaining_qty=result.remaining_qty,
                    terminal=result.terminal,
                    reason_code=result.reason_code,
                    retry_count=result.retry_count,
                    details=result.details,
                )
                if not result.accepted:
                    metrics.increment_reason("order_rejection", result.reason_code or "unknown")
                    continue
                if config.broker.paper and result.terminal:
                    position = portfolio.get_position(symbol)
                    risk_manager.register_fill(
                        symbol,
                        position.qty if position else 0.0,
                        timestamp=snapshot.timestamp,
                    )
                metrics.increment("orders_submitted")

            health.mark_tick(tick_time)
            step += 1
            # Structured runtime status snapshot — one line per tick. The GUI
            # reads the latest line via RuntimeStatusWriter.read_latest() to
            # display real operator state without parsing the log file.
            status_writer.write(
                RuntimeStatusSnapshot(
                    ts=tick_time,
                    step=step,
                    session_id=runtime_session_id,
                    symbol=last_symbol,
                    equity=float(portfolio.equity),
                    fills=len(ledger.fills),
                    safe_to_trade=bool(health.safe_to_trade),
                    last_decision=last_decision_payload,
                    last_risk=last_risk_payload,
                    fusion_enabled=bool(config.memory.decision_fusion.enabled),
                    memory_enabled=bool(config.memory.enabled),
                    frozen_brain=not bool(config.runtime.allow_online_model_updates),
                )
            )
            if config.runtime.max_runtime_steps > 0 and step >= config.runtime.max_runtime_steps:
                runtime_metrics = metrics.snapshot()
                if hasattr(policy, "metrics_snapshot"):
                    runtime_metrics.update(policy.metrics_snapshot())
                logger.info(
                    "runtime_complete",
                    steps=step,
                    fills=len(ledger.fills),
                    metrics=runtime_metrics,
                    equity=portfolio.equity,
                    safe_to_trade=health.safe_to_trade,
                )
                break
        except Exception as exc:
            health.mark_error()
            health.block_trading(
                ReasonCode.SAFE_TO_TRADE_FALSE,
                timestamp=utc_now(),
                details={"stage": "runtime_exception", "error": str(exc)},
            )
            logger.exception("runtime_error", step=step, errors=health.error_count)
            raise


def run_trade_loop(config: AppConfig, *, confirm_live: bool = False) -> None:
    asyncio.run(_run_async_trade_loop(config, confirm_live=confirm_live))
