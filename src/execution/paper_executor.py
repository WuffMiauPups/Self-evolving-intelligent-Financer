from __future__ import annotations

from dataclasses import dataclass

from src.execution.order_validator import validate_order_intent
from src.execution.schemas import ExecutionResult, OrderIntent
from src.risk.ledger import FillRecord, Ledger
from src.risk.portfolio_state import PortfolioState, PositionState


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
        self.portfolio.cash -= cost
        self.portfolio.realized_pnl += realized_pnl
        self.portfolio.daily_pnl += realized_pnl
        self.portfolio.mark_to_market({intent.symbol: fill_price}, intent.timestamp)
        self.ledger.record(
            FillRecord(
                timestamp=intent.timestamp,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                price=fill_price,
                fee=fee,
                realized_pnl=realized_pnl,
                client_order_id=intent.client_order_id,
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
        position = self.portfolio.positions.get(symbol)
        if position is None:
            if abs(signed_delta) < 1.0e-9:
                return 0.0
            self.portfolio.positions[symbol] = PositionState(
                symbol=symbol,
                qty=signed_delta,
                avg_price=fill_price,
                market_price=fill_price,
                entry_time=timestamp,
            )
            return 0.0

        old_qty = position.qty
        new_qty = old_qty + signed_delta
        realized_pnl = 0.0
        if old_qty == 0:
            position.qty = new_qty
            position.avg_price = fill_price
            position.market_price = fill_price
            position.entry_time = timestamp
            return 0.0

        same_direction = old_qty * signed_delta > 0
        if same_direction:
            total_qty = abs(old_qty) + abs(signed_delta)
            weighted_avg = ((abs(old_qty) * position.avg_price) + (abs(signed_delta) * fill_price)) / total_qty
            position.qty = new_qty
            position.avg_price = weighted_avg
            position.market_price = fill_price
            return 0.0

        closing_qty = min(abs(old_qty), abs(signed_delta))
        pnl_sign = 1.0 if old_qty > 0 else -1.0
        realized_pnl = (fill_price - position.avg_price) * closing_qty * pnl_sign
        if abs(new_qty) < 1.0e-9:
            del self.portfolio.positions[symbol]
            return realized_pnl
        if old_qty * new_qty > 0:
            position.qty = new_qty
            position.market_price = fill_price
            return realized_pnl
        position.qty = new_qty
        position.avg_price = fill_price
        position.market_price = fill_price
        position.entry_time = timestamp
        return realized_pnl
