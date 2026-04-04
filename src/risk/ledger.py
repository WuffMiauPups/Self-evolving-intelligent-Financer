from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class FillRecord:
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    realized_pnl: float
    client_order_id: str


@dataclass
class Ledger:
    fills: list[FillRecord] = field(default_factory=list)

    def record(self, fill: FillRecord) -> None:
        self.fills.append(fill)

