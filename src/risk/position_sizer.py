from __future__ import annotations

import math


class PositionSizer:
    def __init__(self, *, quantity_precision: float = 0.001) -> None:
        self.quantity_precision = quantity_precision
        self._precision_inv = 1.0 / quantity_precision

    def calculate_target_quantity(
        self,
        *,
        equity: float,
        price: float,
        target_fraction: float,
        max_notional: float,
    ) -> float:
        if price <= 0:
            return 0.0
        target_notional = min(abs(target_fraction) * equity, max_notional)
        raw_qty = target_notional / price
        return math.floor(raw_qty * self._precision_inv) / self._precision_inv

