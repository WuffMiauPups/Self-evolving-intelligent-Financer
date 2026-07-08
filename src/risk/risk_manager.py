from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta

from src.core.config import RiskConfig
from src.core.time_utils import utc_now, within_session
from src.execution.reason_codes import ReasonCode
from src.risk.constraints import RiskDecision, TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.position_sizer import PositionSizer


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        # Normalize max_daily_loss to always be positive so the
        # comparison ``daily_pnl <= -max_daily_loss`` is unambiguous.
        if self.config.max_daily_loss < 0:
            self.config.max_daily_loss = abs(self.config.max_daily_loss)
        self.position_sizer = PositionSizer(
            quantity_precision=getattr(config, "quantity_precision", 0.001),
        )
        self._processed_request_ids: OrderedDict[str, None] = OrderedDict()
        self._recent_orders: dict[tuple[str, str], object] = {}
        self._last_exit_times: dict[str, object] = {}

    def reset_state(self) -> None:
        self._processed_request_ids.clear()
        self._recent_orders.clear()
        self._last_exit_times.clear()

    def emergency_stop(self) -> None:
        self.config.kill_switch = True

    def evaluate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState,
        *,
        evaluation_time: datetime | None = None,
    ) -> RiskDecision:
        reasons: list[str] = []
        now = evaluation_time or utc_now()

        if self.config.kill_switch:
            reasons.append("kill_switch")
        if proposal.request_id in self._processed_request_ids:
            reasons.append("duplicate_request_id")
        age = now - proposal.timestamp
        if age > timedelta(seconds=self.config.stale_data_sec):
            reasons.append(ReasonCode.STALE_DATA)
        # At-the-limit spreads reject (defensive default): "wider-or-equal than max".
        if proposal.spread_bps >= self.config.max_spread_bps:
            reasons.append("spread_too_wide")
        if portfolio.daily_pnl <= -self.config.max_daily_loss:
            reasons.append("daily_loss_limit")
        if self.config.max_daily_loss_pct is not None and portfolio.equity > 0:
            pct_limit = portfolio.equity * self.config.max_daily_loss_pct
            if portfolio.daily_pnl <= -pct_limit:
                reasons.append("daily_loss_pct_limit")
        # --- Expected-value gate (replaces the old min_trade_confidence /
        # min_trade_fraction magic thresholds).  No fixed decision number:
        # inputs are calibrated model confidence, model-predicted MFE/MAE in
        # bps, and round-trip trading costs (physics).
        if proposal.action != 0:
            cost_bps = self.config.fee_bps + self.config.slippage_bps
            ev_bps = (
                proposal.confidence * proposal.expected_gain_bps
                - (1.0 - proposal.confidence) * proposal.expected_loss_bps
                - cost_bps
            )
            if ev_bps <= 0:
                reasons.append("negative_ev")
        if self.config.allowed_sessions and not any(
            within_session(proposal.timestamp, session) for session in self.config.allowed_sessions
        ):
            reasons.append(ReasonCode.SESSION_DISALLOWED)

        current_position = portfolio.get_position(proposal.symbol)
        current_qty = current_position.qty if current_position else 0.0
        if proposal.symbol in self._last_exit_times:
            since_exit = now - self._last_exit_times[proposal.symbol]
            if since_exit < timedelta(seconds=self.config.cooldown_sec):
                reasons.append("cooldown_active")

        target_qty_abs = self.position_sizer.calculate_target_quantity(
            equity=portfolio.equity,
            price=proposal.price,
            target_fraction=proposal.target_fraction,
            max_notional=self.config.max_position_notional,
        )
        target_qty = float(proposal.action) * target_qty_abs
        delta_qty = target_qty - current_qty
        side = "buy" if delta_qty > 0 else "sell"
        order_qty = abs(delta_qty)

        recent_key = (proposal.symbol, side)
        if recent_key in self._recent_orders:
            since_order = now - self._recent_orders[recent_key]
            if since_order < timedelta(seconds=self.config.duplicate_order_window_sec):
                reasons.append("duplicate_order_window")
        # Also check symbol-only key to prevent rapid side-flips from
        # bypassing the duplicate window.
        symbol_key = (proposal.symbol, "_any")
        if symbol_key in self._recent_orders:
            since_any = now - self._recent_orders[symbol_key]
            if since_any < timedelta(seconds=self.config.duplicate_order_window_sec):
                reasons.append("duplicate_order_window")

        opening_new = abs(current_qty) < 1.0e-9 and abs(target_qty) > 1.0e-9
        if opening_new and portfolio.concurrent_positions() >= self.config.max_concurrent_positions:
            reasons.append("max_concurrent_positions")
        if abs(target_qty) > 0.0 and abs(target_qty * proposal.price) > self.config.max_position_notional:
            reasons.append("max_position_notional")
        if portfolio.aggregate_notional > self.config.max_aggregate_notional:
            reasons.append("max_aggregate_notional")
        if portfolio.leverage_ratio > self.config.max_leverage_ratio:
            reasons.append("max_leverage_ratio")
        if portfolio.rolling_drawdown(self.config.rolling_drawdown_window) <= self.config.max_rolling_drawdown:
            reasons.append("rolling_drawdown_limit")

        if order_qty < 1.0e-6:
            reasons.append("no_position_change")

        if reasons:
            return RiskDecision(approved=False, reasons=reasons)

        self._processed_request_ids[proposal.request_id] = None
        if len(self._processed_request_ids) > self.config.max_processed_request_ids:
            to_remove = len(self._processed_request_ids) // 2
            for _ in range(to_remove):
                self._processed_request_ids.popitem(last=False)
        self._recent_orders[recent_key] = now
        self._recent_orders[(proposal.symbol, "_any")] = now
        return RiskDecision(
            approved=True,
            reasons=[],
            quantity=order_qty,
            side=side,
            target_position=target_qty,
            client_order_id=proposal.request_id,
        )

    def seed_existing_positions(
        self,
        positions: dict[str, float],
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Seed the risk manager with pre-existing broker positions.

        Called after startup reconciliation so that cooldowns and
        duplicate-order windows reflect the broker's actual state.
        Without this, a restart allows immediate re-trading of symbols
        that should be in cooldown.
        """
        now = timestamp or utc_now()
        for symbol, qty in positions.items():
            if abs(qty) > 1.0e-9:
                # Mark both sides as recently ordered so the duplicate
                # window applies.  This prevents immediate re-entry.
                side = "buy" if qty > 0 else "sell"
                self._recent_orders[(symbol, side)] = now

    def register_fill(
        self,
        symbol: str,
        resulting_position_qty: float,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        if abs(resulting_position_qty) < 1.0e-9:
            self._last_exit_times[symbol] = timestamp or utc_now()
