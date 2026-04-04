from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    client_order_id: str
    reference_price: float
    timestamp: datetime
    paper: bool = True
    risk_approved: bool = False
    session_id: str | None = None
    sequence: int = 0


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    status: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fee: float
    client_order_id: str
    order_id: str | None = None
    broker_status: str | None = None
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    terminal: bool = True
    reason_code: str | None = None
    retry_count: int = 0
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderState:
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    filled_qty: float
    remaining_qty: float
    status: str
    updated_at: datetime
    order_id: str | None = None
    terminal: bool = False
    reason_code: str | None = None


@dataclass(frozen=True)
class SymbolQuarantine:
    symbol: str
    reason_code: str
    quarantined_at: datetime
    release_mode: str
    release_condition: str | None = None
    details: dict[str, object] = field(default_factory=dict)
