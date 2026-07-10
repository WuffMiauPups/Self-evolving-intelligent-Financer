"""Per-tick trading loop logic extracted from trade_engine.py.

The ``process_tick`` function contains the body of the
``async for latest_bars in stream.stream():`` loop, factored out so
the trade engine's main loop stays thin and the per-tick logic is
independently testable.

The function is synchronous — the caller fetches ``latest_bars`` from
the stream and hands them in already resolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.core.time_utils import MARKET_TZ, utc_now
from src.data.alpaca_client import BrokerDataClient
from src.data.bar_aggregator import TimeframeAggregator
from src.data.buffers import BarBuffer
from src.data.schemas import MarketBar
from src.execution.order_router import OrderRouter
from src.execution.reason_codes import ReasonCode
from src.execution.reconciliation import build_order_states, hydrate_portfolio_positions
from src.execution.schemas import OrderIntent
from src.execution.session_guards import SessionGuard
from src.execution.runtime_bootstrap import RuntimeContext
from src.features.feature_pipeline import FeaturePipeline, FeatureSnapshot
from src.memory.memory_bank import MemoryBank
from src.memory.storage import save_memory_bank
from src.monitoring.equity_log import EquityLogWriter, EquitySnapshot
from src.monitoring.health import HealthState
from src.monitoring.metrics_logger import MetricsLogger
from src.monitoring.positions_status import (
    AccountSnapshot,
    PositionPeakTracker,
    PositionSnapshot,
    PositionsStatusSnapshot,
    PositionsStatusWriter,
)
from src.monitoring.runtime_status import RuntimeStatusSnapshot, RuntimeStatusWriter
from src.risk.constraints import action_annotation
from src.risk.ledger import FillRecord, Ledger
from src.risk.portfolio_state import PortfolioState, apply_fill
from src.risk.risk_manager import RiskManager
from src.strategies.base import BatchTradingPolicy, MemoryCapablePolicy, TradingPolicy


@dataclass
class TickResult:
    session_peak_equity: float
    last_decision_payload: dict | None
    last_risk_payload: dict | None
    last_symbol: str
    should_break: bool  # True if max_runtime_steps reached


# Consecutive broker-sync failures tolerated before trading is blocked.
# At the default 2 s poll cadence this is roughly a one-minute outage;
# each failed tick skips trading but keeps the loop (and data lane)
# alive, mirroring the stream layer's empty-poll policy.
_SYNC_FAILURE_LIMIT = 30


def _sync_live_state(ctx: RuntimeContext, *, logger: object) -> list[object] | None:
    """Reconcile broker truth into local state.  Never raises.

    Order matters: fills are captured against the *pre-hydration*
    position book (so close-out PnL is computed from the entry prices we
    tracked), then positions are overwritten from the broker, then cash/
    equity/daily-PnL sync from the account snapshot.

    Returns the order transitions, or ``None`` when the broker was
    unreachable — the caller must skip trading for the tick but keep the
    loop alive.  After ``_SYNC_FAILURE_LIMIT`` consecutive failures,
    trading is blocked via ``ctx.health`` (surfaced as a runtime error
    by the trade engine) instead of crashing on the first blip.
    """
    try:
        account = ctx.client.get_account()
        positions = ctx.client.get_positions()
        open_orders = build_order_states(ctx.client.get_open_orders())
    except Exception as exc:  # noqa: BLE001 - any broker/network error is a skip-tick
        ctx.sync_failure_count += 1
        logger.warning(
            "broker_sync_failed",
            error=str(exc),
            consecutive_failures=ctx.sync_failure_count,
            failure_limit=_SYNC_FAILURE_LIMIT,
        )
        if ctx.sync_failure_count >= _SYNC_FAILURE_LIMIT and ctx.health.safe_to_trade:
            ctx.health.block_trading(
                ReasonCode.SAFE_TO_TRADE_FALSE,
                timestamp=utc_now(),
                details={
                    "stage": "broker_sync",
                    "consecutive_failures": ctx.sync_failure_count,
                    "error": str(exc),
                },
            )
        return None
    ctx.sync_failure_count = 0
    transitions, disappeared = ctx.router.sync_open_orders(open_orders)
    _capture_broker_fills(ctx, transitions=transitions, disappeared=disappeared, logger=logger)
    hydrate_portfolio_positions(ctx.portfolio, positions)
    _sync_account_state(ctx, account=account)
    return transitions


def _capture_broker_fills(
    ctx: RuntimeContext,
    *,
    transitions: list,
    disappeared: list[str],
    logger: object,
) -> None:
    """Convert broker order events into portfolio/ledger/risk fill records.

    Broker-routed fills used to vanish here: only the (unreachable)
    PaperExecutor ever wrote realized PnL, the Ledger, or exit cooldowns.
    Every terminal filled order — whether it transitioned in the open
    list or disappeared between polls — is resolved via the closed-order
    lookup so its fill price and quantity land in local accounting.
    """
    candidate_ids: list[str] = []
    for state in transitions:
        if state.terminal and state.status == "filled":
            candidate_ids.append(state.client_order_id)
    candidate_ids.extend(disappeared)
    for client_order_id in candidate_ids:
        if client_order_id in ctx.processed_fill_ids:
            continue
        try:
            order = ctx.client.get_order_by_client_order_id(client_order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("order_fill_lookup_failed", client_order_id=client_order_id, error=str(exc))
            continue
        if order is None:
            logger.warning("order_disappeared_unresolved", client_order_id=client_order_id)
            continue
        if str(order.get("status", "")).lower() != "filled":
            continue
        _apply_broker_fill(ctx, order=order, logger=logger)


def _apply_broker_fill(ctx: RuntimeContext, *, order: dict, logger: object) -> None:
    client_order_id = str(order.get("client_order_id", ""))
    filled_qty = float(order.get("filled_qty", 0.0))
    price = float(order.get("filled_avg_price", 0.0))
    symbol = str(order.get("symbol", ""))
    if not client_order_id or not symbol or filled_qty <= 0 or price <= 0:
        logger.warning(
            "broker_fill_unusable",
            client_order_id=client_order_id,
            symbol=symbol,
            filled_qty=filled_qty,
            filled_avg_price=price,
        )
        return
    ctx.processed_fill_ids.add(client_order_id)
    side = "buy" if "buy" in str(order.get("side", "")).lower() else "sell"
    signed_delta = filled_qty if side == "buy" else -filled_qty
    fill_time = order.get("updated_at")
    if not isinstance(fill_time, datetime):
        fill_time = utc_now()
    realized = apply_fill(
        ctx.portfolio,
        symbol=symbol,
        signed_delta=signed_delta,
        fill_price=price,
        timestamp=fill_time,
    )
    ctx.portfolio.cash -= signed_delta * price
    ctx.portfolio.realized_pnl += realized
    ctx.portfolio.daily_pnl += realized
    position = ctx.portfolio.get_position(symbol)
    ctx.risk_manager.register_fill(
        symbol,
        position.qty if position else 0.0,
        timestamp=fill_time,
    )
    ctx.ledger.record(
        FillRecord(
            timestamp=fill_time,
            symbol=symbol,
            side=side,
            quantity=filled_qty,
            price=price,
            fee=0.0,
            realized_pnl=realized,
            client_order_id=client_order_id,
        )
    )
    if ctx.order_journal is not None:
        try:
            from src.execution.order_journal import OrderJournalEntry
            ctx.order_journal.record(OrderJournalEntry(
                timestamp=fill_time.isoformat(),
                symbol=symbol,
                side=side,
                quantity=filled_qty,
                price=price,
                status="filled",
                client_order_id=client_order_id,
                broker_order_id=order.get("id"),
                fee=0.0,
            ))
        except Exception:
            pass  # journal must never raise into trade loop
    logger.info(
        "broker_fill_captured",
        symbol=symbol,
        side=side,
        quantity=filled_qty,
        price=price,
        realized_pnl=realized,
        client_order_id=client_order_id,
    )


def _sync_account_state(ctx: RuntimeContext, *, account: dict) -> None:
    """Overwrite cash/equity from broker truth and anchor daily PnL.

    ``daily_pnl`` is equity-based (includes unrealized swings), anchored
    to the broker's ``last_equity`` (previous trading-day close) when
    available, else to the first equity seen for the current US/Eastern
    trading date.  This is what the risk manager's daily-loss limits
    compare against — previously frozen at 0.0 in broker mode.
    """
    cash = account.get("cash")
    equity = account.get("equity")
    if cash is None or equity is None:
        return
    ctx.portfolio.cash = float(cash)
    ctx.portfolio.equity = float(equity)
    today = utc_now().astimezone(MARKET_TZ).date()
    if ctx.daily_anchor_date != today:
        last_equity = float(account.get("last_equity", 0.0) or 0.0)
        ctx.daily_anchor_date = today
        ctx.daily_anchor_equity = last_equity if last_equity > 0 else float(equity)
    if ctx.daily_anchor_equity > 0:
        ctx.portfolio.daily_pnl = float(equity) - ctx.daily_anchor_equity


def _ingest_bar(ctx: RuntimeContext, symbol: str, bar: MarketBar, *, logger: object) -> None:
    """Fold a polled bar into its timeframe bucket and upsert the buffer.

    Replaces the old unconditional append: re-polls of the same forming
    bar revise the buffer's last row in place, so duplicates can never
    accumulate, and every buffered row is a bucket-start-stamped bar of
    the configured runtime timeframe — matching cold-start history and
    training data.
    """
    if ctx.bar_aggregator is None:
        ctx.bar_aggregator = TimeframeAggregator(ctx.config.runtime.timeframe)
    bucket_bar = ctx.bar_aggregator.update(symbol, bar)
    if bucket_bar is None:
        return
    buf = ctx.buffers[symbol]
    _backfill_gap(ctx, symbol, buf, bucket_bar, logger=logger)
    buf.upsert(bucket_bar)


def _backfill_gap(
    ctx: RuntimeContext,
    symbol: str,
    buf: BarBuffer,
    bucket_bar: MarketBar,
    *,
    logger: object,
) -> None:
    """Fetch buckets missed between the buffer tail and the live bucket.

    Fires once per gap (outage, symbol errors, session boundary): after
    a successful backfill the buffer tail advances past the gap, so the
    range fetch does not repeat.  Best-effort — a failed fetch leaves a
    gap for the session guard's gap detection to handle.
    """
    if len(buf) == 0 or ctx.client is None or ctx.bar_aggregator is None:
        return
    last_ts = buf.last_bar().timestamp
    delta = ctx.bar_aggregator.delta
    if bucket_bar.timestamp - last_ts <= delta:
        return
    try:
        missing = ctx.client.get_historical_bars_range(
            symbol=symbol,
            timeframe=ctx.config.runtime.timeframe,
            start=last_ts + delta,
            end=bucket_bar.timestamp,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("bar_backfill_failed", symbol=symbol, error=str(exc))
        return
    appended = 0
    for missing_bar in missing:
        if missing_bar.timestamp >= bucket_bar.timestamp:
            continue
        if buf.upsert(missing_bar) == "appended":
            appended += 1
    if appended:
        logger.info(
            "bar_backfill",
            symbol=symbol,
            bars=appended,
            gap_start=last_ts.isoformat(),
            gap_end=bucket_bar.timestamp.isoformat(),
        )


def _intervention_rate(ctx: RuntimeContext) -> float:
    counters = ctx.metrics.snapshot()
    proposals = float(counters.get("proposals", 0.0))
    if proposals <= 0:
        return 0.0
    return float(counters.get("risk_interventions", 0.0)) / proposals


def _write_decision_event(
    ctx: RuntimeContext,
    *,
    proposal: object,
    decision: object,
    result: object | None,
) -> None:
    """Record what the model proposed vs what actually executed.

    The self-evolution loop must learn from executed (risk-clamped)
    actions, never raw intent — this event stream is where that record
    lives.  Best-effort: never raises into the trade loop.
    """
    if ctx.decision_writer is None:
        return
    try:
        from src.monitoring.decision_log import DecisionEvent

        reasons = [str(r) for r in getattr(decision, "reasons", [])]
        ctx.decision_writer.write(
            DecisionEvent(
                ts=proposal.timestamp,
                symbol=proposal.symbol,
                request_id=proposal.request_id,
                proposed_action=int(proposal.action),
                confidence=float(proposal.confidence),
                target_fraction=float(proposal.target_fraction),
                approved=bool(decision.approved),
                reasons=reasons,
                executed_side=(decision.side if decision.approved else None),
                executed_qty=(float(decision.quantity) if decision.approved else 0.0),
                resized="resized_to_cap" in reasons,
                order_status=(None if result is None else str(getattr(result, "status", None))),
                client_order_id=(
                    None if result is None else getattr(result, "client_order_id", None)
                ),
            )
        )
    except Exception:  # noqa: BLE001 - observability must never block trading
        pass


def _write_positions_snapshot(
    *,
    writer: PositionsStatusWriter,
    tick_time: datetime,
    portfolio: PortfolioState,
    config: AppConfig,
    session_start_equity: float,
    session_peak_equity: float,
    peak_tracker: PositionPeakTracker,
) -> None:
    """Assemble + atomically publish the per-tick positions snapshot.

    Pure observability — wrapped in a broad except so a writer crash
    never takes down the trade loop.  The GUI's ``PositionsTab`` polls
    the produced JSON file at ~5 s cadence via ``PositionsStatusWriter.
    read_latest``.
    """
    try:
        # Session-level account stats.  ``buying_power`` isn't tracked in
        # the local ``PortfolioState`` (it's a broker concept), so we
        # approximate it as ``cash`` for the in-process case and let
        # broker reconciliation override when we have live account data
        # (future enhancement).  ``session_drawdown`` is negative or
        # zero by construction (current_equity - peak).
        session_return = 0.0
        if session_start_equity > 0:
            session_return = (
                float(portfolio.equity) - session_start_equity
            ) / session_start_equity
        session_drawdown = 0.0
        if session_peak_equity > 0:
            session_drawdown = min(
                0.0,
                (float(portfolio.equity) - session_peak_equity) / session_peak_equity,
            )
        account = AccountSnapshot(
            equity=float(portfolio.equity),
            cash=float(portfolio.cash),
            buying_power=float(portfolio.cash),
            portfolio_value=float(portfolio.equity),
            session_start_equity=float(session_start_equity),
            session_peak_equity=float(session_peak_equity),
            session_return=float(session_return),
            session_drawdown=float(session_drawdown),
            max_position_notional=float(config.risk.max_position_notional),
            max_concurrent_positions=int(config.risk.max_concurrent_positions),
            min_trade_confidence=0.0,  # deprecated (EV gate replaced magic threshold)
            kill_switch=bool(config.risk.kill_switch),
        )

        # Per-position snapshots with peak-tracker driven drawdown.
        position_snapshots: list[PositionSnapshot] = []
        active_symbols: set[str] = set()
        for symbol, position in portfolio.positions.items():
            if abs(position.qty) < 1.0e-9:
                continue
            active_symbols.add(symbol)
            qty = float(position.qty)
            avg_price = float(position.avg_price)
            market_price = float(position.market_price)
            notional = abs(qty * market_price)
            entry_notional = abs(qty * avg_price) or 1.0
            unrealized_pnl = (market_price - avg_price) * qty
            unrealized_pnl_pct = unrealized_pnl / entry_notional
            entry_epoch = (
                position.entry_time.timestamp()
                if position.entry_time is not None
                else 0.0
            )
            peak, drawdown = peak_tracker.update(
                symbol=symbol,
                qty=qty,
                unrealized_pnl=unrealized_pnl,
                entry_epoch=entry_epoch,
            )
            drawdown_pct = drawdown / entry_notional if entry_notional else 0.0
            position_snapshots.append(
                PositionSnapshot(
                    symbol=symbol,
                    side="long" if qty > 0 else "short",
                    qty=qty,
                    avg_price=avg_price,
                    market_price=market_price,
                    notional=notional,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    peak_unrealized_pnl=peak,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                    entry_time=position.entry_time,
                )
            )
        # Drop peak-tracker entries for symbols that no longer hold a
        # position so a re-entry starts fresh.
        stale_symbols = [
            s for s in list(peak_tracker._state.keys()) if s not in active_symbols
        ]
        for s in stale_symbols:
            peak_tracker.forget(s)

        writer.write(
            PositionsStatusSnapshot(
                ts=tick_time,
                account=account,
                positions=position_snapshots,
            )
        )
    except Exception:
        # Observability writer must never block trading.
        pass


def process_tick(
    ctx: RuntimeContext,
    latest_bars: dict[str, MarketBar],
    *,
    step: int,
    warmup_logged: set[str],
    session_peak_equity: float,
    session_start_equity: float,
    position_peak_tracker: PositionPeakTracker,
    last_decision_payload: dict | None,
    last_risk_payload: dict | None,
    last_symbol: str,
) -> TickResult:
    logger = get_logger("trade_engine")

    # --- Stream error logging ---
    if ctx.stream.last_errors:
        logger.info("stream_symbol_errors", errors=ctx.stream.last_errors)
        for symbol in ctx.stream.last_errors:
            ctx.metrics.increment_reason("stream_error", ReasonCode.SYMBOL_NO_MARKET_DATA)
    # Reconcile open orders + positions whenever we have a real
    # broker client (Alpaca or mock).  Previously this was gated on
    # ``not config.broker.paper`` because "paper" meant "in-process
    # simulator", but now ``broker.paper=true`` with credentials
    # talks to Alpaca's paper endpoint and still needs reconciliation
    # for fills that arrive after submission.
    sync_ok = True
    if ctx.router.broker_client is not None:
        transitions = _sync_live_state(ctx, logger=logger)
        sync_ok = transitions is not None
        for transition in transitions or []:
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
        _ingest_bar(ctx, symbol, bar, logger=logger)
        latest_prices[symbol] = bar.close
    if not latest_prices:
        return TickResult(
            session_peak_equity=session_peak_equity,
            last_decision_payload=last_decision_payload,
            last_risk_payload=last_risk_payload,
            last_symbol=last_symbol,
            should_break=False,
        )
    tick_time = max(bar.timestamp for bar in latest_bars.values())
    ctx.portfolio.mark_to_market(latest_prices, tick_time)
    ctx.equity_writer.write(
        EquitySnapshot(
            ts=tick_time,
            equity=float(ctx.portfolio.equity),
            realized_pnl=float(getattr(ctx.portfolio, "realized_pnl", 0.0)),
            unrealized_pnl=float(ctx.portfolio.unrealized_pnl),
            fills=len(ctx.ledger.fills),
            session_id=ctx.runtime_session_id,
        )
    )
    broker_clock = None
    try:
        broker_clock = ctx.client.get_clock()
    except Exception as exc:
        logger.info("broker_clock_unavailable", error=str(exc))

    # ------------------------------------------------------------------
    # Per-tick processing is now split into three phases so the
    # model forward can run once for the whole symbol batch:
    #
    #   Phase A: per-symbol prep — quarantine / warmup / memory
    #            finalization / session guards / feature snapshot
    #            build.  Collects eligible snapshots in order.
    #   Phase B: single batched model forward via
    #            ``policy.propose_batch(snapshots)``.  One GPU
    #            matmul covers all eligible symbols at once.
    #   Phase C: per-symbol decision dispatch — risk manager,
    #            session guard recheck, order routing, logging.
    #            Sequential so portfolio state updates from one
    #            symbol's fill are seen by the next symbol's
    #            risk check.
    #
    # Safety invariants preserved: per-symbol state resets and
    # session guards fire *before* we enter the batch, so a
    # symbol that needs to be excluded never makes it into the
    # forward.  The flat-base-never-promoted check still runs
    # per symbol inside ``MemoryAwarePolicy.propose_batch``.
    # ------------------------------------------------------------------
    eligible_batch: list[tuple[str, FeatureSnapshot, object]] = []
    # A failed broker sync means portfolio/order state is stale: skip
    # proposal generation for the tick (the data lane above still ran)
    # and let the next poll retry the sync.
    trading_symbols = ctx.eligible_symbols if sync_ok else []
    if not sync_ok:
        logger.info("tick_trading_skipped", reason="broker_sync_failed")
    for symbol in trading_symbols:
        if not ctx.health.safe_to_trade:
            break
        quarantine = ctx.router.is_symbol_quarantined(symbol)
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
            ctx.metrics.increment_reason("symbol_quarantine", quarantine.reason_code)
            continue
        if symbol not in latest_bars:
            continue
        if len(ctx.buffers[symbol]) < ctx.config.runtime.warmup_bars:
            if symbol not in warmup_logged:
                logger.info(
                    "warmup_pending",
                    symbol=symbol,
                    buffer_size=len(ctx.buffers[symbol]),
                    required_warmup_bars=ctx.config.runtime.warmup_bars,
                )
                warmup_logged.add(symbol)
            continue
        warmup_logged.discard(symbol)
        if isinstance(ctx.policy, MemoryCapablePolicy):
            finalized_items = ctx.policy.prepare_symbol_history(symbol, ctx.buffers[symbol].snapshot())
            if finalized_items:
                logger.info(
                    "memory_update",
                    symbol=symbol,
                    finalized_count=len(finalized_items),
                    bank_items=0 if ctx.memory_bank is None else len(ctx.memory_bank),
                )
                if ctx.memory_bank is not None:
                    try:
                        save_memory_bank(ctx.config.memory.persist_path, ctx.memory_bank)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("memory_bank_save_failed", symbol=symbol, error=str(exc))
            # Persist pending observations for restart safety
            if (
                ctx.policy.updater is not None
                and ctx.config.memory.persist_pending_observations
            ):
                try:
                    ctx.policy.updater.save_pending(ctx.config.memory.pending_store_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pending_save_failed", symbol=symbol, error=str(exc))

        bar = latest_bars[symbol]
        guard_result = ctx.session_guard.evaluate(
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
            ctx.metrics.increment_reason("session_guard", guard_result.reason_code)
            if guard_result.reason_code == ReasonCode.DATA_GAP_BLOCK:
                ctx.policy.reset_symbol_state(symbol, reason="data_gap_block")
            continue

        # Hot path: pull the buffer's preallocated tensor view
        # straight into the feature pipeline.  No copy, no
        # Python loop — just a view into the backing tensor.
        buf = ctx.buffers[symbol]
        snapshot = ctx.pipeline.build_from_tensor(
            buf.snapshot_tensor(),
            symbol=symbol,
            last_bar=buf.last_bar(),
        )
        eligible_batch.append((symbol, snapshot, bar))

    # --- Phase B: single batched model forward ---------------------
    if not eligible_batch:
        batched_results: list[tuple[object, dict[str, object]]] = []
    elif isinstance(ctx.policy, BatchTradingPolicy):
        batched_results = ctx.policy.propose_batch(
            [entry[1] for entry in eligible_batch]
        )
    else:
        batched_results = [
            ctx.policy.propose(entry[1]) for entry in eligible_batch
        ]

    # --- Phase C: per-symbol decision dispatch ---------------------
    for (symbol, snapshot, bar), (proposal, diagnostics) in zip(
        eligible_batch, batched_results
    ):
        if not ctx.health.safe_to_trade:
            break
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

        decision = ctx.risk_manager.evaluate(proposal, ctx.portfolio)
        # Memory records the EXECUTED (risk-clamped) action, not raw
        # intent — outcome learning must credit what actually happened.
        if isinstance(ctx.policy, MemoryCapablePolicy):
            ctx.policy.record_snapshot(symbol, action_annotation=action_annotation(proposal, decision))
        ctx.metrics.increment("proposals")
        if not decision.approved or "resized_to_cap" in decision.reasons:
            ctx.metrics.increment("risk_interventions")
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
            _write_decision_event(ctx, proposal=proposal, decision=decision, result=None)
            ctx.metrics.increment("risk_rejections")
            for reason in decision.reasons:
                ctx.metrics.increment_reason("risk_rejection", str(reason))
            if ctx.alerter is not None:
                _critical = {"kill_switch", "daily_loss_limit", "daily_loss_pct_limit", "rolling_drawdown_limit"}
                if _critical & set(decision.reasons):
                    try:
                        from src.monitoring.alerting import AlertEvent
                        ctx.alerter.send(AlertEvent(
                            severity="critical",
                            event_type=decision.reasons[0],
                            message=f"Risk rejection: {', '.join(decision.reasons)} for {symbol}",
                        ))
                    except Exception:
                        pass  # alerter must never raise into trade loop
            continue

        submission_guard = ctx.session_guard.evaluate(
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
            ctx.metrics.increment_reason("submission_guard", submission_guard.reason_code)
            continue

        side = decision.side or "buy"
        client_order_id, sequence = ctx.router.next_client_order_id(
            session_id=ctx.runtime_session_id,
            symbol=symbol,
            side=side,
            timestamp=snapshot.timestamp,
        )
        # ``intent.paper`` is a legacy field that, when True, forces
        # OrderRouter to short-circuit through the in-process
        # PaperExecutor and never call the broker client.  We now
        # only set it when we truly have no broker client to talk
        # to (the safety fallback path), so that ``broker.paper=
        # true`` + Alpaca credentials actually hits Alpaca's paper
        # endpoint via AlpacaClient.  See the router construction
        # above for the full reasoning.
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            quantity=decision.quantity,
            client_order_id=client_order_id,
            reference_price=snapshot.last_price,
            timestamp=snapshot.timestamp,
            paper=(ctx.client is None),
            risk_approved=True,
            session_id=ctx.runtime_session_id,
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
        result = ctx.router.route(intent)
        _write_decision_event(ctx, proposal=proposal, decision=decision, result=result)
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
        if ctx.order_journal is not None:
            try:
                from src.execution.order_journal import OrderJournalEntry
                ctx.order_journal.record(OrderJournalEntry(
                    timestamp=snapshot.timestamp.isoformat(),
                    symbol=result.symbol,
                    side=result.side,
                    quantity=result.quantity,
                    price=result.fill_price,
                    status=result.status,
                    client_order_id=result.client_order_id,
                    broker_order_id=result.order_id,
                    fee=result.fee,
                ))
            except Exception:
                pass  # journal must never raise into trade loop
        if not result.accepted:
            ctx.metrics.increment_reason("order_rejection", result.reason_code or "unknown")
            continue
        # PaperExecutor updates the portfolio synchronously inside
        # ``execute()``, so its fills only need the risk manager told.
        # Broker-routed orders that fill synchronously at submit (the
        # normal market-order case) are never tracked as open, so the
        # sync-time transition/disappearance capture would miss them —
        # capture the fill here; ``processed_fill_ids`` keeps later
        # sync events for the same order idempotent.
        if ctx.client is None and result.terminal:
            position = ctx.portfolio.get_position(symbol)
            ctx.risk_manager.register_fill(
                symbol,
                position.qty if position else 0.0,
                timestamp=snapshot.timestamp,
            )
        elif ctx.client is not None and result.terminal and result.status == "filled":
            _apply_broker_fill(
                ctx,
                order={
                    "client_order_id": result.client_order_id,
                    "symbol": result.symbol,
                    "side": result.side,
                    "filled_qty": result.filled_qty,
                    "filled_avg_price": result.fill_price,
                    "id": result.order_id,
                    "updated_at": utc_now(),
                },
                logger=logger,
            )
        ctx.metrics.increment("orders_submitted")

    ctx.health.mark_tick(tick_time)
    step += 1
    # Structured runtime status snapshot — one line per tick. The GUI
    # reads the latest line via RuntimeStatusWriter.read_latest() to
    # display real operator state without parsing the log file.
    ctx.status_writer.write(
        RuntimeStatusSnapshot(
            ts=tick_time,
            step=step,
            session_id=ctx.runtime_session_id,
            symbol=last_symbol,
            equity=float(ctx.portfolio.equity),
            fills=len(ctx.ledger.fills),
            safe_to_trade=bool(ctx.health.safe_to_trade),
            last_decision=last_decision_payload,
            last_risk=last_risk_payload,
            intervention_rate=_intervention_rate(ctx),
            memory_enabled=bool(ctx.config.memory.enabled),
            frozen_brain=not bool(ctx.config.runtime.allow_online_model_updates),
        )
    )

    # --- Positions snapshot for the GUI's POSITIONS tab ---------
    # Update session peak + per-position peak trackers, build an
    # atomic-overwrite JSON file that the GUI reads every ~5s.
    # This is a pure observability write — failures inside the
    # writer never raise back into the trade loop.
    session_peak_equity = max(session_peak_equity, float(ctx.portfolio.equity))
    _write_positions_snapshot(
        writer=ctx.positions_writer,
        tick_time=tick_time,
        portfolio=ctx.portfolio,
        config=ctx.config,
        session_start_equity=session_start_equity,
        session_peak_equity=session_peak_equity,
        peak_tracker=position_peak_tracker,
    )
    should_break = False
    if ctx.config.runtime.max_runtime_steps > 0 and step >= ctx.config.runtime.max_runtime_steps:
        runtime_metrics = ctx.metrics.snapshot()
        if isinstance(ctx.policy, MemoryCapablePolicy):
            runtime_metrics.update(ctx.policy.metrics_snapshot())
        logger.info(
            "runtime_complete",
            steps=step,
            fills=len(ctx.ledger.fills),
            metrics=runtime_metrics,
            equity=ctx.portfolio.equity,
            safe_to_trade=ctx.health.safe_to_trade,
        )
        should_break = True
    return TickResult(
        session_peak_equity=session_peak_equity,
        last_decision_payload=last_decision_payload,
        last_risk_payload=last_risk_payload,
        last_symbol=last_symbol,
        should_break=should_break,
    )
