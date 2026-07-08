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

    async for latest_bars in ctx.stream.stream():
        if not ctx.health.safe_to_trade:
            logger.info(
                "safe_to_trade_blocked",
                reason_code=ctx.health.blocked_reason_code,
                details=ctx.health.blocked_details,
            )
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
            session_peak_equity = result.session_peak_equity
            last_decision_payload = result.last_decision_payload
            last_risk_payload = result.last_risk_payload
            last_symbol = result.last_symbol
            step += 1
            if result.should_break:
                break
        except Exception as exc:
            ctx.health.mark_error()
            ctx.health.block_trading(
                ReasonCode.SAFE_TO_TRADE_FALSE,
                timestamp=utc_now(),
                details={"stage": "runtime_exception", "error": str(exc)},
            )
            logger.exception("runtime_error", step=step, errors=ctx.health.error_count)
            raise


def run_trade_loop(config: AppConfig, *, confirm_live: bool = False) -> None:
    asyncio.run(_run_async_trade_loop(config, confirm_live=confirm_live))
