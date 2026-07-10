"""Trade engine — thin orchestrator.

Delegates initialization to ``runtime_bootstrap.build_runtime_context``
and per-tick processing to ``tick_processor.process_tick``.  The only
logic that lives here is the async-for-tick loop shell and the
error/shutdown handling.
"""
from __future__ import annotations

import asyncio

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.core.time_utils import utc_now
from src.execution.reason_codes import ReasonCode
from src.execution.runtime_bootstrap import build_runtime_context
from src.execution.tick_processor import TickResult, process_tick
from src.monitoring.positions_status import PositionPeakTracker
from src.monitoring.runtime_status import RuntimeStatusSnapshot


# Consecutive process_tick exceptions tolerated before the loop gives up.
# A single transient failure (broker hiccup mid-phase, disk blip) skips
# the tick; only a persistent error streak blocks trading and raises.
_TICK_ERROR_LIMIT = 5


def _graceful_shutdown(
    ctx: object,
    logger: object,
    *,
    step: int,
    last_symbol: str = "",
    last_decision_payload: dict | None = None,
    last_risk_payload: dict | None = None,
    reason: str = "KeyboardInterrupt",
) -> None:
    """Best-effort cleanup when the trade loop is interrupted.

    Cancels every tracked open order at the broker, persists the memory
    bank + pending observations (when enabled), and writes a final
    runtime-status snapshot with ``safe_to_trade=False`` so the GUI/ops
    tooling sees an intentional stop instead of a stale "running" line.

    Every step is individually guarded: shutdown cleanup must never mask
    the original interrupt with a secondary failure.
    """
    logger.info("graceful_shutdown_start", reason=reason, step=step)

    # 1) Cancel tracked open orders via the broker client.
    cancelled: list[str] = []
    failed: list[str] = []
    try:
        open_orders = dict(ctx.router.open_orders_by_client_id)
    except Exception:  # noqa: BLE001 - router state must not break shutdown
        open_orders = {}
    for client_order_id, order in open_orders.items():
        order_id = getattr(order, "order_id", None)
        if not order_id:
            # Never acknowledged by the broker — nothing to cancel there.
            continue
        try:
            ctx.client.cancel_order(order_id)
            cancelled.append(client_order_id)
        except Exception as exc:  # noqa: BLE001 - keep cancelling the rest
            failed.append(client_order_id)
            logger.warning(
                "shutdown_order_cancel_failed",
                client_order_id=client_order_id,
                order_id=order_id,
                error=str(exc),
            )
    if cancelled or failed:
        logger.info("shutdown_open_orders_cancelled", cancelled=cancelled, failed=failed)

    # 2) Persist memory bank + pending observations for restart safety.
    try:
        if ctx.memory_bank is not None and ctx.config.memory.enabled:
            from src.memory.storage import save_memory_bank

            save_memory_bank(ctx.config.memory.persist_path, ctx.memory_bank)
            logger.info(
                "shutdown_memory_bank_saved",
                path=str(ctx.config.memory.persist_path),
                items=len(ctx.memory_bank),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("shutdown_memory_bank_save_failed", error=str(exc))
    try:
        updater = getattr(ctx.policy, "updater", None)
        if updater is not None and ctx.config.memory.persist_pending_observations:
            updater.save_pending(ctx.config.memory.pending_store_path)
            logger.info(
                "shutdown_pending_observations_saved",
                path=str(ctx.config.memory.pending_store_path),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("shutdown_pending_save_failed", error=str(exc))

    # 3) Final runtime status snapshot: safe_to_trade=False.
    try:
        ctx.status_writer.write(
            RuntimeStatusSnapshot(
                ts=utc_now(),
                step=step,
                session_id=ctx.runtime_session_id,
                symbol=last_symbol,
                equity=float(ctx.portfolio.equity),
                fills=len(ctx.ledger.fills),
                safe_to_trade=False,
                last_decision=last_decision_payload,
                last_risk=last_risk_payload,
                memory_enabled=bool(ctx.config.memory.enabled),
                frozen_brain=not bool(ctx.config.runtime.allow_online_model_updates),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("shutdown_status_snapshot_failed", error=str(exc))
    logger.info("graceful_shutdown_complete", reason=reason, step=step)


async def _run_async_trade_loop(config: AppConfig, *, confirm_live: bool = False) -> None:
    ctx = await build_runtime_context(config, confirm_live=confirm_live)
    logger = get_logger("trade_engine")

    # Mutable per-tick state carried across iterations.
    position_peak_tracker = PositionPeakTracker()
    session_start_equity = float(ctx.portfolio.equity)
    session_peak_equity = session_start_equity
    last_decision_payload: dict | None = None
    last_risk_payload: dict | None = None
    last_symbol: str = ""
    warmup_logged: set[str] = set()
    step = 0
    consecutive_errors = 0
    blocked_mid_session = False

    try:
        async for latest_bars in ctx.stream.stream():
            if not ctx.health.safe_to_trade:
                logger.info(
                    "safe_to_trade_blocked",
                    reason_code=ctx.health.blocked_reason_code,
                    details=ctx.health.blocked_details,
                )
                blocked_mid_session = True
                break
            try:
                result: TickResult = process_tick(
                    ctx,
                    latest_bars,
                    step=step,
                    warmup_logged=warmup_logged,
                    session_peak_equity=session_peak_equity,
                    session_start_equity=session_start_equity,
                    position_peak_tracker=position_peak_tracker,
                    last_decision_payload=last_decision_payload,
                    last_risk_payload=last_risk_payload,
                    last_symbol=last_symbol,
                )
                consecutive_errors = 0
                session_peak_equity = result.session_peak_equity
                last_decision_payload = result.last_decision_payload
                last_risk_payload = result.last_risk_payload
                last_symbol = result.last_symbol
                step += 1
                if result.should_break:
                    return
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as exc:
                consecutive_errors += 1
                ctx.health.mark_error()
                logger.exception(
                    "runtime_error",
                    step=step,
                    errors=ctx.health.error_count,
                    consecutive_errors=consecutive_errors,
                    error_limit=_TICK_ERROR_LIMIT,
                )
                if consecutive_errors >= _TICK_ERROR_LIMIT:
                    ctx.health.block_trading(
                        ReasonCode.SAFE_TO_TRADE_FALSE,
                        timestamp=utc_now(),
                        details={"stage": "runtime_exception", "error": str(exc)},
                    )
                    raise
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        # Operator-initiated stop (Ctrl+C / CTRL_BREAK via the GUI) or task
        # cancellation: cancel live open orders, persist memory state, and
        # leave a final safe_to_trade=False status line — then re-raise so
        # the caller still observes the interrupt.
        _graceful_shutdown(
            ctx,
            logger,
            step=step,
            last_symbol=last_symbol,
            last_decision_payload=last_decision_payload,
            last_risk_payload=last_risk_payload,
            reason=type(exc).__name__,
        )
        raise

    # The stream generator ending means its circuit breaker tripped
    # (persistent all-symbol failures).  Exiting 0 here made the GUI
    # report STOPPED after a broker outage; surface it as an error.
    if (
        ctx.stream.circuit_breaker_threshold > 0
        and ctx.stream.consecutive_empty_polls >= ctx.stream.circuit_breaker_threshold
    ):
        raise RuntimeError(
            "stream circuit breaker tripped after "
            f"{ctx.stream.consecutive_empty_polls} consecutive empty polls"
        )
    if blocked_mid_session:
        raise RuntimeError(
            f"trading blocked mid-session: {ctx.health.blocked_reason_code} "
            f"{ctx.health.blocked_details}"
        )


def run_trade_loop(config: AppConfig, *, confirm_live: bool = False) -> None:
    asyncio.run(_run_async_trade_loop(config, confirm_live=confirm_live))
