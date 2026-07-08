from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


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
        total_notional = 0.0
        for symbol, position in list(self.positions.items()):
            market_price = prices.get(symbol, position.market_price)
            position.market_price = market_price
            total_notional += position.qty * market_price
        self.equity = self.cash + total_notional
        self.pnl_history.append(self.equity)
        self.last_update = timestamp

