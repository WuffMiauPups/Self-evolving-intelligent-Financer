from __future__ import annotations

from dataclasses import dataclass

from src.backtest.fills import fee_from_bps, fill_price_with_slippage
from src.execution.paper_executor import PaperExecutor
from src.execution.schemas import ExecutionResult, OrderIntent


@dataclass
class BacktestSimulator:
    executor: PaperExecutor
    fee_bps: float
    slippage_bps: float

    def preview_cost(self, intent: OrderIntent) -> dict[str, float]:
        fill_price = fill_price_with_slippage(
            intent.reference_price,
            side=intent.side,
            slippage_bps=self.slippage_bps,
        )
        notional = fill_price * intent.quantity
        fee = fee_from_bps(notional, self.fee_bps)
        return {"fill_price": fill_price, "fee": fee}

    def execute(self, intent: OrderIntent) -> ExecutionResult:
        return self.executor.execute(intent)

