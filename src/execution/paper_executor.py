from __future__ import annotations

from dataclasses import dataclass

from src.execution.order_validator import validate_order_intent
from src.execution.schemas import ExecutionResult, OrderIntent
from src.risk.ledger import FillRecord, Ledger
from src.risk.portfolio_state import PortfolioState, apply_fill


@dataclass
class PaperExecutor:
    portfolio: PortfolioState
    ledger: Ledger
    fee_bps: float
    slippage_bps: float

    def execute(self, intent: OrderIntent) -> ExecutionResult:
        validate_order_intent(intent)
        slip_multiplier = 1.0 + ((self.slippage_bps / 10_000.0) * (1 if intent.side == "buy" else -1))
        fill_price = intent.reference_price * slip_multiplier
        notional = fill_price * intent.quantity
        fee = abs(notional) * (self.fee_bps / 10_000.0)
        signed_delta = intent.quantity if intent.side == "buy" else -intent.quantity
        cost = (signed_delta * fill_price) + fee
        if cost > 0 and cost > self.portfolio.cash:
            return ExecutionResult(
                accepted=False,
                status="rejected",
                symbol=intent.symbol,
                side=intent.side,
                quantity=0.0,
                fill_price=fill_price,
                fee=0.0,
                client_order_id=intent.client_order_id,
                order_id=f"paper-{intent.client_order_id}",
                broker_status="insufficient_cash",
                filled_qty=0.0,
                remaining_qty=intent.quantity,
                terminal=True,
            )
        realized_pnl = self._apply_fill(
            symbol=intent.symbol,
            signed_delta=signed_delta,
            fill_price=fill_price,
            fee=fee,
            timestamp=intent.timestamp,
        )
        # Realized PnL is reported NET of the fill's fee — gross figures
        # made every per-trade statistic (win rate, expectancy) optimistic
        # by exactly the costs. Cash already carries the fee via ``cost``.
        realized_net = realized_pnl - fee
        self.portfolio.cash -= cost
        self.portfolio.realized_pnl += realized_net
        self.portfolio.daily_pnl += realized_net
        self.portfolio.mark_to_market({intent.symbol: fill_price}, intent.timestamp)
        self.ledger.record(
            FillRecord(
                timestamp=intent.timestamp,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                price=fill_price,
                fee=fee,
                realized_pnl=realized_net,
                client_order_id=intent.client_order_id,
                strategy_id=intent.strategy_id,
            )
        )
        return ExecutionResult(
            accepted=True,
            status="filled",
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            fill_price=fill_price,
            fee=fee,
            client_order_id=intent.client_order_id,
            order_id=f"paper-{intent.client_order_id}",
            broker_status="filled",
            filled_qty=intent.quantity,
            remaining_qty=0.0,
            terminal=True,
        )

    def _apply_fill(
        self,
        *,
        symbol: str,
        signed_delta: float,
        fill_price: float,
        fee: float,
        timestamp: object,
    ) -> float:
        del fee
        return apply_fill(
            self.portfolio,
            symbol=symbol,
            signed_delta=signed_delta,
            fill_price=fill_price,
            timestamp=timestamp,
        )
