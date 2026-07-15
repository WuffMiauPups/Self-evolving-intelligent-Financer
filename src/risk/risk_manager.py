from __future__ import annotations

import math
from collections import OrderedDict
from datetime import datetime, timedelta

from src.core.config import RiskConfig
from src.core.time_utils import utc_now, within_session
from src.execution.reason_codes import ReasonCode
from src.risk.constraints import RiskDecision, TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.position_sizer import PositionSizer


def _is_whole_shares(quantity: float) -> bool:
    """Whether an order quantity is a whole number of shares.

    Tolerance absorbs float noise from fraction*equity/price sizing (a
    "50.0" position can arrive as 50.000000001).
    """
    return abs(quantity - round(quantity)) < 1.0e-9


def adaptive_budget_multiplier(
    rolling_drawdown: float,
    *,
    full_at_drawdown: float,
    min_multiplier: float,
) -> float:
    """Deterministic size throttle from realized rolling drawdown (§6.4).

    1.0 at zero/positive drawdown, linear down to ``min_multiplier`` when
    the drawdown reaches ``full_at_drawdown`` (a negative fraction), hard
    floor below that. Pure function — no state, no learning.
    """
    if rolling_drawdown >= 0.0:
        return 1.0
    fraction = min(rolling_drawdown / full_at_drawdown, 1.0)
    return 1.0 - fraction * (1.0 - min_multiplier)


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        # The config is shared, learned-module-visible state: the manager
        # never mutates it. Kill-switch escalation lives on the instance.
        self.config = config
        self._kill_switch = False
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
        self._kill_switch = True

    def evaluate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState,
        *,
        evaluation_time: datetime | None = None,
    ) -> RiskDecision:
        reasons: list[str] = []
        now = evaluation_time or utc_now()

        current_position = portfolio.get_position(proposal.symbol)
        current_qty = current_position.qty if current_position else 0.0

        target_qty_abs = self.position_sizer.calculate_target_quantity(
            equity=portfolio.equity,
            price=proposal.price,
            target_fraction=proposal.target_fraction,
            max_notional=self.config.max_position_notional,
        )
        target_qty = float(proposal.action) * target_qty_abs
        # --- Whole-share short constraint (broker physics, not preference) ---
        # Alpaca cannot short fractional shares (42210000 "fractional orders
        # cannot be sold short"). Two unexecutable shapes must be prevented
        # here — in the SHARED risk path, so the backtest never books a fill
        # the live broker would reject:
        #   1. a sub-share short target (|target| < 1) is impossible at all;
        #   2. crossing from a fractional long straight into a short would
        #      submit one fractional sell-short order — so flatten first and
        #      let a later tick open the short from flat, whole-share.
        short_block: str | None = None
        if target_qty < 0.0 and not self.config.fractional_shorts_supported:
            whole_short = math.floor(abs(target_qty))
            if whole_short < 1.0:
                # Sub-share short: unexecutable at any size.
                target_qty = 0.0
                short_block = "fractional_short_unsupported"
            else:
                target_qty = -whole_short
                # The ORDER (not just the target) must be whole when it
                # crosses into a short: selling 2.4 to go from +0.4 to -2 is
                # still a fractional sell-short. A whole-share long -> short
                # flip (e.g. +50 -> -20 = sell 70) is fine and must stay a
                # flip, so that the halt gates below can judge it as one.
                if not _is_whole_shares(abs(target_qty - current_qty)):
                    target_qty = 0.0
                    short_block = "flatten_before_short"
        elif (
            target_qty > 1.0e-9
            and current_qty < -1.0e-9
            and not self.config.fractional_shorts_supported
            and not _is_whole_shares(target_qty - current_qty)
        ):
            # Mirror image, observed live 2026-07-15 17:03: a FRACTIONAL buy
            # crossing from short to long flips side in one fractional order —
            # Alpaca rejects it (40310000 "insufficient qty available") and
            # the unchanged intent then resubmitted every duplicate window.
            # Cover to flat (whole, since shorts are whole); the fractional
            # long opens from flat on a later tick.
            target_qty = 0.0
            short_block = "flatten_before_long"
        delta_qty = target_qty - current_qty
        side = "buy" if delta_qty > 0 else "sell"
        order_qty = abs(delta_qty)
        # Soft cap: the sizer clamps target notional at max_position_notional,
        # so an oversized request proceeds at the shrunken size — the approval
        # is tagged "resized_to_cap" instead of being rejected.
        resized = (
            proposal.action != 0
            and target_qty_abs > 0.0
            and abs(proposal.target_fraction) * portfolio.equity > self.config.max_position_notional
        )
        # Strict exposure reduction: full flatten or same-side shrink. A side
        # flip is NOT risk-reducing (it opens fresh exposure the other way).
        # Halt/exposure limits must never block de-risking the book.
        risk_reducing = abs(current_qty) > 1.0e-9 and (
            abs(target_qty) < 1.0e-9
            or (target_qty * current_qty > 0.0 and abs(target_qty) < abs(current_qty))
        )

        # --- Outcome-adaptive risk budget (§6.4): realized drawdown throttles
        # risk-INCREASING size, bounded [budget_min_multiplier, 1.0]. Never
        # touches risk-reducing trades, and never converts an intended add
        # into a trim — if the throttled target would fall below the current
        # position, it clamps to "no change" (rejected as no_position_change).
        budget_multiplier = 1.0
        if (
            self.config.adaptive_budget_enabled
            and proposal.action != 0
            and not risk_reducing
        ):
            budget_multiplier = adaptive_budget_multiplier(
                portfolio.rolling_drawdown(self.config.rolling_drawdown_window),
                full_at_drawdown=self.config.budget_full_at_drawdown,
                min_multiplier=self.config.budget_min_multiplier,
            )
            if budget_multiplier < 1.0:
                throttled_abs = self.position_sizer.calculate_target_quantity(
                    equity=portfolio.equity,
                    price=proposal.price,
                    target_fraction=proposal.target_fraction * budget_multiplier,
                    max_notional=self.config.max_position_notional,
                )
                throttled_qty = float(proposal.action) * throttled_abs
                if throttled_qty * current_qty > 0.0 and abs(throttled_qty) < abs(current_qty):
                    throttled_qty = current_qty
                target_qty = throttled_qty
                delta_qty = target_qty - current_qty
                side = "buy" if delta_qty > 0 else "sell"
                order_qty = abs(delta_qty)

        if (self.config.kill_switch or self._kill_switch) and not risk_reducing:
            reasons.append("kill_switch")
        if proposal.request_id in self._processed_request_ids:
            reasons.append("duplicate_request_id")
        age = now - proposal.timestamp
        if age > timedelta(seconds=self.config.stale_data_sec):
            reasons.append(ReasonCode.STALE_DATA)
        # At-the-limit spreads reject (defensive default): "wider-or-equal than max".
        if proposal.spread_bps >= self.config.max_spread_bps:
            reasons.append("spread_too_wide")
        if not risk_reducing:
            # abs(): tolerate configs stating the loss limit as a negative number.
            if portfolio.daily_pnl <= -abs(self.config.max_daily_loss):
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
            # spread_bps is the full quoted spread; entering and exiting each
            # cross half of it, so a round trip pays one full spread.
            cost_bps = self.config.fee_bps + self.config.slippage_bps + proposal.spread_bps
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

        if proposal.symbol in self._last_exit_times:
            since_exit = now - self._last_exit_times[proposal.symbol]
            if since_exit < timedelta(seconds=self.config.cooldown_sec):
                reasons.append("cooldown_active")

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

        if not risk_reducing:
            opening_new = abs(current_qty) < 1.0e-9 and abs(target_qty) > 1.0e-9
            if opening_new and portfolio.concurrent_positions() >= self.config.max_concurrent_positions:
                reasons.append("max_concurrent_positions")
            if abs(target_qty) > 0.0 and abs(target_qty * proposal.price) > self.config.max_position_notional:
                reasons.append("max_position_notional")
            # Post-trade caps: evaluate the book as it would look with the
            # proposed target replacing the current position — the pre-trade
            # book alone lets one order overshoot the aggregate caps by up to
            # max_position_notional.
            current_notional = (
                abs(current_qty) * current_position.market_price if current_position else 0.0
            )
            prospective_notional = (
                portfolio.aggregate_notional - current_notional + abs(target_qty) * proposal.price
            )
            if prospective_notional > self.config.max_aggregate_notional:
                reasons.append("max_aggregate_notional")
            if prospective_notional / max(portfolio.equity, 1.0) > self.config.max_leverage_ratio:
                reasons.append("max_leverage_ratio")
            if portfolio.rolling_drawdown(self.config.rolling_drawdown_window) <= self.config.max_rolling_drawdown:
                reasons.append("rolling_drawdown_limit")

        if order_qty < 1.0e-6:
            reasons.append("no_position_change")
            # Name the cause when nothing is left to do BECAUSE the short was
            # impossible — otherwise the decision reads as a bare
            # "no_position_change" and the operator cannot tell a satisfied
            # target from an unexecutable one. (A flatten-before-short still
            # has a real order to place, so it is tagged on approval below.)
            if short_block is not None:
                reasons.append(short_block)
        elif (
            self.config.min_trade_notional > 0.0
            and order_qty * proposal.price < self.config.min_trade_notional
            and abs(target_qty) > 1.0e-9
        ):
            # Churn deadband: dust rebalances pay spread and fees for
            # sub-dollar adjustments (2026-07-15 paper run). Full flattens
            # (target 0) are exempt so dust positions stay closeable.
            reasons.append("below_min_trade_notional")

        if reasons:
            return RiskDecision(
                approved=False, reasons=reasons, budget_multiplier=budget_multiplier
            )

        self._processed_request_ids[proposal.request_id] = None
        if len(self._processed_request_ids) > self.config.max_processed_request_ids:
            to_remove = len(self._processed_request_ids) // 2
            for _ in range(to_remove):
                self._processed_request_ids.popitem(last=False)
        self._recent_orders[recent_key] = now
        self._recent_orders[(proposal.symbol, "_any")] = now
        approval_tags: list[str] = []
        if short_block is not None:
            # The order that IS being placed (e.g. flattening a fractional
            # long so a whole-share short can open from flat next tick).
            approval_tags.append(short_block)
        if resized:
            approval_tags.append("resized_to_cap")
        if budget_multiplier < 1.0:
            approval_tags.append("adaptive_budget")
        return RiskDecision(
            approved=True,
            reasons=approval_tags,
            quantity=order_qty,
            side=side,
            target_position=target_qty,
            client_order_id=proposal.request_id,
            budget_multiplier=budget_multiplier,
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
