from __future__ import annotations


def fill_price_with_slippage(price: float, *, side: str, slippage_bps: float) -> float:
    multiplier = 1.0 + ((slippage_bps / 10_000.0) * (1 if side == "buy" else -1))
    return price * multiplier


def fee_from_bps(notional: float, fee_bps: float) -> float:
    return abs(notional) * (fee_bps / 10_000.0)

