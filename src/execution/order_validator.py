from __future__ import annotations

import re

from src.core.exceptions import ExecutionError
from src.execution.schemas import OrderIntent

CLIENT_ORDER_ID_PATTERN = re.compile(r"^seif-[A-Za-z0-9]+-[A-Z0-9.]+-[bs]-\d{12}-\d{4}$")


def validate_order_intent(intent: OrderIntent) -> None:
    if not intent.risk_approved:
        raise ExecutionError("Order intent must have explicit risk approval before routing")
    if intent.side not in {"buy", "sell"}:
        raise ExecutionError(f"Unsupported order side: {intent.side}")
    if intent.quantity <= 0:
        raise ExecutionError("Order quantity must be positive")
    if not intent.client_order_id:
        raise ExecutionError("Order intent must include client_order_id")
    if not CLIENT_ORDER_ID_PATTERN.match(intent.client_order_id):
        raise ExecutionError(f"Order intent has invalid client_order_id schema: {intent.client_order_id}")
    if intent.reference_price <= 0:
        raise ExecutionError("Reference price must be positive")
