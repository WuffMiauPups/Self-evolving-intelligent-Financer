from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from src.core.time_utils import utc_now
from src.data.alpaca_client import BrokerDataClient
from src.execution.reason_codes import ReasonCode
from src.execution.schemas import OrderState
from src.risk.portfolio_state import PortfolioState, PositionState


class ReconciliationSeverity(StrEnum):
    WARN = "warn"
    BLOCK = "block"
    FATAL = "fatal"


@dataclass(frozen=True)
class ReconciliationIssue:
    severity: str
    reason_code: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconciliationResult:
    safe_to_trade: bool
    issues: list[ReconciliationIssue]
    hydrated_positions: dict[str, PositionState] = field(default_factory=dict)
    hydrated_open_orders: list[OrderState] = field(default_factory=list)
    synchronized_at: datetime = field(default_factory=utc_now)

    def summary(self) -> dict[str, Any]:
        return {
            "safe_to_trade": self.safe_to_trade,
            "issue_count": len(self.issues),
            "open_order_count": len(self.hydrated_open_orders),
            "position_count": len(self.hydrated_positions),
            "issue_codes": [issue.reason_code for issue in self.issues],
        }


def run_startup_reconciliation(
    *,
    client: BrokerDataClient,
    portfolio: PortfolioState,
    using_mock: bool,
    block_on_mismatch: bool,
) -> ReconciliationResult:
    if using_mock:
        return ReconciliationResult(
            safe_to_trade=True,
            issues=[
                ReconciliationIssue(
                    severity=ReconciliationSeverity.WARN,
                    reason_code=ReasonCode.MOCK_MODE_ACTIVE,
                    details={"reconciliation": "mock"},
                )
            ],
        )

    issues: list[ReconciliationIssue] = []
    hydrated_positions: dict[str, PositionState] = {}
    hydrated_open_orders: list[OrderState] = []
    try:
        positions = client.get_positions()
        open_orders = client.get_open_orders()
    except Exception as exc:
        return ReconciliationResult(
            safe_to_trade=False,
            issues=[
                ReconciliationIssue(
                    severity=ReconciliationSeverity.FATAL,
                    reason_code=ReasonCode.RECONCILIATION_FETCH_FAILED,
                    details={"error": str(exc)},
                )
            ],
        )

    for position in positions:
        position_state = PositionState(
            symbol=str(position["symbol"]),
            qty=float(position["qty"]),
            avg_price=float(position["avg_entry_price"]),
            market_price=float(position.get("current_price", position["avg_entry_price"])),
            entry_time=utc_now(),
        )
        hydrated_positions[position_state.symbol] = position_state
        portfolio.positions[position_state.symbol] = position_state

    if hydrated_positions:
        issues.append(
            ReconciliationIssue(
                severity=ReconciliationSeverity.WARN,
                reason_code=ReasonCode.RECONCILIATION_POSITIONS_HYDRATED,
                details={"symbols": sorted(hydrated_positions)},
            )
        )

    for order in open_orders:
        hydrated_open_orders.append(
            OrderState(
                client_order_id=str(order["client_order_id"]),
                symbol=str(order["symbol"]),
                side=str(order["side"]),
                quantity=float(order["qty"]),
                filled_qty=float(order.get("filled_qty", 0.0)),
                remaining_qty=float(order.get("remaining_qty", max(float(order["qty"]) - float(order.get("filled_qty", 0.0)), 0.0))),
                status=str(order["status"]),
                updated_at=order.get("updated_at", utc_now()),
                order_id=order.get("id"),
                terminal=bool(order.get("terminal", False)),
                reason_code=ReasonCode.ORDER_PENDING,
            )
        )

    if hydrated_open_orders:
        issues.append(
            ReconciliationIssue(
                severity=ReconciliationSeverity.BLOCK if block_on_mismatch else ReconciliationSeverity.WARN,
                reason_code=ReasonCode.RECONCILIATION_OPEN_ORDERS_PRESENT,
                details={"client_order_ids": [order.client_order_id for order in hydrated_open_orders]},
            )
        )

    if not issues:
        issues.append(
            ReconciliationIssue(
                severity=ReconciliationSeverity.WARN,
                reason_code=ReasonCode.RECONCILIATION_OK,
                details={},
            )
        )

    safe_to_trade = not any(issue.severity in {ReconciliationSeverity.BLOCK, ReconciliationSeverity.FATAL} for issue in issues)
    return ReconciliationResult(
        safe_to_trade=safe_to_trade,
        issues=issues,
        hydrated_positions=hydrated_positions,
        hydrated_open_orders=hydrated_open_orders,
    )


def hydrate_portfolio_positions(
    portfolio: PortfolioState,
    positions: list[dict[str, Any]],
) -> dict[str, PositionState]:
    hydrated_positions: dict[str, PositionState] = {}
    portfolio.positions.clear()
    for position in positions:
        position_state = PositionState(
            symbol=str(position["symbol"]),
            qty=float(position["qty"]),
            avg_price=float(position["avg_entry_price"]),
            market_price=float(position.get("current_price", position["avg_entry_price"])),
            entry_time=utc_now(),
        )
        hydrated_positions[position_state.symbol] = position_state
        portfolio.positions[position_state.symbol] = position_state
    return hydrated_positions


def build_order_states(orders: list[dict[str, Any]]) -> list[OrderState]:
    return [
        OrderState(
            client_order_id=str(order["client_order_id"]),
            symbol=str(order["symbol"]),
            side=str(order["side"]),
            quantity=float(order["qty"]),
            filled_qty=float(order.get("filled_qty", 0.0)),
            remaining_qty=float(order.get("remaining_qty", max(float(order["qty"]) - float(order.get("filled_qty", 0.0)), 0.0))),
            status=str(order["status"]),
            updated_at=order.get("updated_at", utc_now()),
            order_id=order.get("id"),
            terminal=bool(order.get("terminal", False)),
            reason_code=None,
        )
        for order in orders
    ]
