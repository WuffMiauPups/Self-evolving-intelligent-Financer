from __future__ import annotations

from dataclasses import dataclass

from src.execution.paper_executor import PaperExecutor
from src.execution.schemas import ExecutionResult, OrderIntent


@dataclass
class BacktestSimulator:
    executor: PaperExecutor

    def execute(self, intent: OrderIntent) -> ExecutionResult:
        return self.executor.execute(intent)

