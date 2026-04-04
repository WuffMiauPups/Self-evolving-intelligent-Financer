from __future__ import annotations

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
        self.last_update = timestamp

