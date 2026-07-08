from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

from src.core.time_utils import MARKET_TZ


@dataclass
class PositionState:
    symbol: str
    qty: float
    avg_price: float
    market_price: float
    entry_time: datetime

    @property
    def notional(self) -> float:
        return abs(self.qty * self.market_price)


@dataclass
class PortfolioState:
    cash: float
    equity: float
    positions: dict[str, PositionState] = field(default_factory=dict)
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    last_update: datetime | None = None
    last_trading_date: date | None = None
    pnl_history: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))

    @property
    def aggregate_notional(self) -> float:
        return sum(abs(p.qty * p.market_price) for p in self.positions.values())

    @property
    def leverage_ratio(self) -> float:
        return self.aggregate_notional / max(self.equity, 1.0)

    def rolling_drawdown(self, window: int = 100) -> float:
        if len(self.pnl_history) < 2:
            return 0.0
        recent = list(self.pnl_history)[-window:]
        peak = recent[0]
        max_dd = 0.0
        for equity_val in recent:
            if equity_val > peak:
                peak = equity_val
            dd = (equity_val - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def concurrent_positions(self) -> int:
        return sum(1 for position in self.positions.values() if abs(position.qty) > 1.0e-9)

    def get_position(self, symbol: str) -> PositionState | None:
        return self.positions.get(symbol)

    def mark_to_market(self, prices: dict[str, float], timestamp: datetime) -> None:
        # daily_pnl resets at US/Eastern trading-day boundaries so
        # max_daily_loss is a per-day budget, not a lifetime cumulative cap.
        # (Broker mode re-anchors daily_pnl from broker equity each tick and
        # simply overwrites this; the reset matters in backtest/sim mode.)
        trading_date = timestamp.astimezone(MARKET_TZ).date()
        if self.last_trading_date is not None and trading_date != self.last_trading_date:
            self.daily_pnl = 0.0
        self.last_trading_date = trading_date
        total_notional = 0.0
        for symbol, position in list(self.positions.items()):
            market_price = prices.get(symbol, position.market_price)
            position.market_price = market_price
            total_notional += position.qty * market_price
        self.equity = self.cash + total_notional
        # Re-marks at the same timestamp (backtests mark twice per bar plus
        # once per fill) revise the last sample instead of appending, so the
        # rolling-drawdown window spans distinct bars, not duplicate marks.
        if self.last_update is not None and timestamp == self.last_update and self.pnl_history:
            self.pnl_history[-1] = self.equity
        else:
            self.pnl_history.append(self.equity)
        self.last_update = timestamp

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            (position.market_price - position.avg_price) * position.qty
            for position in self.positions.values()
        )


def apply_fill(
    portfolio: PortfolioState,
    *,
    symbol: str,
    signed_delta: float,
    fill_price: float,
    timestamp: datetime,
) -> float:
    """Apply a signed fill to the position book and return realized PnL.

    Shared close-out math for every fill source: the in-process
    PaperExecutor and broker-reported fills captured during live-state
    sync both route through here so simulated and broker accounting can
    never diverge.  Does not touch cash/realized/daily PnL — callers own
    those.
    """
    position = portfolio.positions.get(symbol)
    if position is None:
        if abs(signed_delta) < 1.0e-9:
            return 0.0
        portfolio.positions[symbol] = PositionState(
            symbol=symbol,
            qty=signed_delta,
            avg_price=fill_price,
            market_price=fill_price,
            entry_time=timestamp,
        )
        return 0.0

    old_qty = position.qty
    new_qty = old_qty + signed_delta
    if old_qty == 0:
        position.qty = new_qty
        position.avg_price = fill_price
        position.market_price = fill_price
        position.entry_time = timestamp
        return 0.0

    same_direction = old_qty * signed_delta > 0
    if same_direction:
        total_qty = abs(old_qty) + abs(signed_delta)
        weighted_avg = ((abs(old_qty) * position.avg_price) + (abs(signed_delta) * fill_price)) / total_qty
        position.qty = new_qty
        position.avg_price = weighted_avg
        position.market_price = fill_price
        return 0.0

    closing_qty = min(abs(old_qty), abs(signed_delta))
    pnl_sign = 1.0 if old_qty > 0 else -1.0
    realized_pnl = (fill_price - position.avg_price) * closing_qty * pnl_sign
    if abs(new_qty) < 1.0e-9:
        del portfolio.positions[symbol]
        return realized_pnl
    if old_qty * new_qty > 0:
        position.qty = new_qty
        position.market_price = fill_price
        return realized_pnl
    position.qty = new_qty
    position.avg_price = fill_price
    position.market_price = fill_price
    position.entry_time = timestamp
    return realized_pnl

