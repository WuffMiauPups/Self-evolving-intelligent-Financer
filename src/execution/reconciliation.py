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
        # Prefer the broker's actual entry timestamp when available
        # (Alpaca returns ``created_at`` on position objects).  Fall
        # back to ``utc_now()`` if the field is missing so position-age
        # tracking degrades gracefully rather than crashing.
        raw_entry = position.get("created_at")
        if raw_entry is not None and not isinstance(raw_entry, datetime):
            try:
                raw_entry = datetime.fromisoformat(str(raw_entry))
            except (ValueError, TypeError):
                raw_entry = None
        entry_time = raw_entry if raw_entry is not None else utc_now()
        position_state = PositionState(
            symbol=str(position["symbol"]),
            qty=float(position["qty"]),
            avg_price=float(position["avg_entry_price"]),
            market_price=float(position.get("current_price", position["avg_entry_price"])),
            entry_time=entry_time,
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
    *,
    skip_symbols: set[str] | None = None,
) -> dict[str, PositionState]:
    """Overwrite local positions from broker truth.

    ``skip_symbols``: symbols whose LOCAL state must be preserved untouched
    this pass — used for symbols with an order still in flight.  The broker
    position already reflects a fill the moment it executes, but the fill is
    only captured (and its PnL realized) when the ORDER reaches a terminal
    state, which can be one or more polls later (partial fills, or the
    positions endpoint updating ahead of the orders endpoint).  Hydrating
    such a symbol in between consumes the fill's position effect without
    realizing anything, so the eventual ``apply_fill`` finds no position and
    returns 0.0 — observed live 2026-07-20 (AAPL close booked
    realized_pnl=0.0 instead of ~+$3.67).  While an order is in flight the
    local book, maintained by ``apply_fill`` at capture time, is the
    accounting truth for that symbol; hydration resumes the same tick the
    order completes.
    """
    previous = dict(portfolio.positions)
    skip = skip_symbols or set()
    hydrated_positions: dict[str, PositionState] = {}
    portfolio.positions.clear()
    for symbol in skip:
        if symbol in previous:
            portfolio.positions[symbol] = previous[symbol]
            hydrated_positions[symbol] = previous[symbol]
    for position in positions:
        if str(position["symbol"]) in skip:
            continue
        raw_entry = position.get("created_at")
        if raw_entry is not None and not isinstance(raw_entry, datetime):
            try:
                raw_entry = datetime.fromisoformat(str(raw_entry))
            except (ValueError, TypeError):
                raw_entry = None
        symbol = str(position["symbol"])
        qty = float(position["qty"])
        if raw_entry is not None:
            entry_time = raw_entry
        else:
            # Alpaca position payloads carry no created_at; without this
            # the entry_time reset every tick, so position age and
            # peak/drawdown tracking never accumulated.  Keep the prior
            # entry_time while the position stays on the same side.
            prior = previous.get(symbol)
            if prior is not None and prior.qty * qty > 0:
                entry_time = prior.entry_time
            else:
                entry_time = utc_now()
        position_state = PositionState(
            symbol=symbol,
            qty=qty,
            avg_price=float(position["avg_entry_price"]),
            market_price=float(position.get("current_price", position["avg_entry_price"])),
            entry_time=entry_time,
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
