from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.core.exceptions import ExecutionError
from src.core.time_utils import truncate_timestamp_to_minute, utc_now
from src.data.alpaca_client import BrokerDataClient
from src.execution.order_validator import validate_order_intent
from src.execution.paper_executor import PaperExecutor
from src.execution.reason_codes import ReasonCode
from src.execution.schemas import ExecutionResult, OrderIntent, OrderState, SymbolQuarantine

CLIENT_ORDER_ID_PARSE = re.compile(
    r"^seif-(?P<session>[A-Za-z0-9]+)-(?P<symbol>[A-Z0-9.]+)-(?P<side>[bs])-(?P<bucket>\d{12})-(?P<seq>\d{4})$"
)


def is_terminal_status(status: str) -> bool:
    return status in {"filled", "canceled", "cancelled", "expired", "rejected"}


def classify_status_reason(status: str) -> str:
    normalized = status.lower()
    if normalized == "filled":
        return ReasonCode.ORDER_FILLED
    if normalized in {"canceled", "cancelled"}:
        return ReasonCode.ORDER_CANCELED
    if normalized == "expired":
        return ReasonCode.ORDER_EXPIRED
    if normalized == "rejected":
        return ReasonCode.ORDER_REJECTED
    return ReasonCode.ORDER_PENDING


@dataclass
class OrderRouter:
    paper_executor: PaperExecutor
    broker_client: BrokerDataClient | None = None
    allow_live: bool = False
    max_submit_retries: int = 0
    retry_backoff_sec: float = 0.0
    open_orders_by_client_id: dict[str, OrderState] = field(default_factory=dict)
    symbol_quarantines: dict[str, SymbolQuarantine] = field(default_factory=dict)
    sequence_by_key: dict[tuple[str, str, str, str], int] = field(default_factory=dict)

    def reset_runtime_state(self) -> None:
        self.open_orders_by_client_id.clear()
        self.symbol_quarantines.clear()
        self.sequence_by_key.clear()

    def next_client_order_id(
        self,
        *,
        session_id: str,
        symbol: str,
        side: str,
        timestamp: datetime,
    ) -> tuple[str, int]:
        bucket = truncate_timestamp_to_minute(timestamp).strftime("%Y%m%d%H%M")
        key = (session_id, symbol.upper(), side.lower(), bucket)
        sequence = self.sequence_by_key.get(key, 0) + 1
        self.sequence_by_key[key] = sequence
        side_token = "b" if side.lower() == "buy" else "s"
        return f"seif-{session_id}-{symbol.upper()}-{side_token}-{bucket}-{sequence:04d}", sequence

    def seed_open_orders(self, orders: list[OrderState]) -> None:
        self.open_orders_by_client_id = {}
        for order in orders:
            if order.terminal:
                continue
            self.open_orders_by_client_id[order.client_order_id] = order
            self._seed_sequence(order.client_order_id)
            self.quarantine_symbol(
                order.symbol,
                reason_code=ReasonCode.RECONCILIATION_OPEN_ORDERS_PRESENT,
                release_mode="until_order_terminal",
                details={"client_order_id": order.client_order_id},
            )

    def sync_open_orders(self, orders: list[OrderState]) -> tuple[list[OrderState], list[str]]:
        """Reconcile tracked orders against the broker's open-order list.

        Returns ``(transitions, disappeared)``: orders whose status or
        fill quantity changed, and client ids of tracked orders that
        vanished from the open list entirely — the normal path for an
        order that filled between polls.  Callers must resolve
        disappeared ids against the broker (closed-order lookup) to
        capture the fill; the router only releases its tracking state.
        """
        transitions: list[OrderState] = []
        disappeared: list[str] = []
        latest_by_client_id = {order.client_order_id: order for order in orders}
        tracked_ids = set(self.open_orders_by_client_id)
        for client_order_id, order in latest_by_client_id.items():
            previous = self.open_orders_by_client_id.get(client_order_id)
            if previous is None or previous.status != order.status or previous.filled_qty != order.filled_qty:
                transitions.append(order)
            if order.terminal:
                self.open_orders_by_client_id.pop(client_order_id, None)
                self._release_symbol_for_order(client_order_id)
            else:
                self.open_orders_by_client_id[client_order_id] = order
        for missing_id in tracked_ids - set(latest_by_client_id):
            self.open_orders_by_client_id.pop(missing_id, None)
            self._release_symbol_for_order(missing_id)
            disappeared.append(missing_id)
        return transitions, disappeared

    def symbols_with_open_orders(self) -> set[str]:
        """Symbols with an order still tracked as in flight.

        Fill capture uses this to defer position hydration: while an order
        is in flight, the local book is the accounting truth for its symbol
        (see ``hydrate_portfolio_positions``).  Terminal and disappeared
        orders are released by ``sync_open_orders`` BEFORE fills are
        captured, so a completed order's symbol re-hydrates the same tick.
        """
        return {state.symbol for state in self.open_orders_by_client_id.values()}

    def quarantine_symbol(
        self,
        symbol: str,
        *,
        reason_code: str,
        release_mode: str,
        release_condition: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self.symbol_quarantines[symbol] = SymbolQuarantine(
            symbol=symbol,
            reason_code=reason_code,
            quarantined_at=utc_now(),
            release_mode=release_mode,
            release_condition=release_condition or self._release_condition_for(release_mode),
            details=details or {},
        )

    def is_symbol_quarantined(self, symbol: str) -> SymbolQuarantine | None:
        return self.symbol_quarantines.get(symbol)

    def route(self, intent: OrderIntent) -> ExecutionResult:
        if not intent.risk_approved:
            raise ExecutionError("Refusing to route order without successful risk approval")
        validate_order_intent(intent)
        existing = self.open_orders_by_client_id.get(intent.client_order_id)
        if existing is not None and not existing.terminal:
            return ExecutionResult(
                accepted=False,
                status="duplicate_blocked",
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                fill_price=intent.reference_price,
                fee=0.0,
                client_order_id=intent.client_order_id,
                order_id=existing.order_id,
                broker_status=existing.status,
                filled_qty=existing.filled_qty,
                remaining_qty=existing.remaining_qty,
                terminal=False,
                reason_code=ReasonCode.ORDER_DUPLICATE_CLIENT_ID,
            )
        quarantine = self.is_symbol_quarantined(intent.symbol)
        if quarantine is not None:
            return ExecutionResult(
                accepted=False,
                status="blocked",
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                fill_price=intent.reference_price,
                fee=0.0,
                client_order_id=intent.client_order_id,
                reason_code=quarantine.reason_code,
                terminal=True,
                details={
                    "quarantined_at": quarantine.quarantined_at.isoformat(),
                    "release_mode": quarantine.release_mode,
                    "release_condition": quarantine.release_condition,
                    **quarantine.details,
                },
            )
        if self._has_open_order_for_symbol(intent.symbol):
            self.quarantine_symbol(
                intent.symbol,
                reason_code=ReasonCode.ORDER_SYMBOL_QUARANTINED,
                release_mode="until_order_terminal",
                release_condition="released when the broker order becomes terminal or disappears during sync",
                details={"symbol": intent.symbol},
            )
            return ExecutionResult(
                accepted=False,
                status="symbol_quarantined",
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                fill_price=intent.reference_price,
                fee=0.0,
                client_order_id=intent.client_order_id,
                terminal=True,
                reason_code=ReasonCode.ORDER_SYMBOL_QUARANTINED,
            )
        if intent.paper or not self.allow_live or self.broker_client is None:
            return self.paper_executor.execute(intent)
        return self._submit_live_order(intent)

    def _submit_live_order(self, intent: OrderIntent) -> ExecutionResult:
        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.max_submit_retries:
            try:
                response = self.broker_client.submit_market_order(
                    symbol=intent.symbol,
                    qty=intent.quantity,
                    side=intent.side,
                    client_order_id=intent.client_order_id,
                )
                result = self._build_execution_result(intent, response, retry_count=attempt)
                self._apply_result_state(result)
                return result
            except Exception as exc:
                last_error = exc
                recovered = self.broker_client.get_order_by_client_order_id(intent.client_order_id)
                if recovered is not None:
                    result = self._build_execution_result(
                        intent,
                        recovered,
                        retry_count=attempt,
                        override_reason=ReasonCode.ORDER_RECOVERED_BY_CLIENT_ID,
                    )
                    self._apply_result_state(result)
                    return result
                if not self._is_retryable_exception(exc) or attempt >= self.max_submit_retries:
                    return ExecutionResult(
                        accepted=False,
                        status="submit_failed",
                        symbol=intent.symbol,
                        side=intent.side,
                        quantity=intent.quantity,
                        fill_price=intent.reference_price,
                        fee=0.0,
                        client_order_id=intent.client_order_id,
                        terminal=True,
                        reason_code=ReasonCode.ORDER_SUBMIT_RETRY_EXHAUSTED
                        if attempt >= self.max_submit_retries
                        else ReasonCode.ORDER_SUBMIT_FAILED,
                        retry_count=attempt,
                        details={"error": str(exc)},
                    )
                attempt += 1
                if self.retry_backoff_sec > 0:
                    time.sleep(self.retry_backoff_sec)
        return ExecutionResult(
            accepted=False,
            status="submit_failed",
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            fill_price=intent.reference_price,
            fee=0.0,
            client_order_id=intent.client_order_id,
            terminal=True,
            reason_code=ReasonCode.ORDER_SUBMIT_FAILED,
            retry_count=self.max_submit_retries,
            details={"error": "" if last_error is None else str(last_error)},
        )

    def _build_execution_result(
        self,
        intent: OrderIntent,
        response: dict[str, object],
        *,
        retry_count: int,
        override_reason: str | None = None,
    ) -> ExecutionResult:
        broker_status = str(response.get("status", "unknown")).lower()
        terminal = bool(response.get("terminal", is_terminal_status(broker_status)))
        reason_code = override_reason or classify_status_reason(broker_status)
        filled_qty = float(response.get("filled_qty", 0.0))
        remaining_qty = float(response.get("remaining_qty", max(intent.quantity - filled_qty, 0.0)))
        accepted = broker_status not in {"rejected", "canceled", "cancelled", "expired"}
        # Prefer the broker's actual average fill price; the reference
        # price is only a fallback for unfilled/rejected responses.
        fill_price = float(response.get("filled_avg_price") or 0.0) or float(
            response.get("fill_price", intent.reference_price)
        )
        return ExecutionResult(
            accepted=accepted,
            status=broker_status,
            symbol=str(response.get("symbol", intent.symbol)),
            side=str(response.get("side", intent.side)),
            quantity=float(response.get("qty", intent.quantity)),
            fill_price=fill_price,
            fee=float(response.get("fee", 0.0)),
            client_order_id=str(response.get("client_order_id", intent.client_order_id)),
            order_id=None if response.get("id") is None else str(response.get("id")),
            broker_status=broker_status,
            filled_qty=filled_qty,
            remaining_qty=remaining_qty,
            terminal=terminal,
            reason_code=reason_code,
            retry_count=retry_count,
        )

    def _apply_result_state(self, result: ExecutionResult) -> None:
        if result.terminal:
            self.open_orders_by_client_id.pop(result.client_order_id, None)
            self._release_symbol_for_order(result.client_order_id)
            if result.reason_code == ReasonCode.ORDER_REJECTED:
                self.quarantine_symbol(
                    result.symbol,
                    reason_code=ReasonCode.ORDER_REJECTED,
                    release_mode="until_restart",
                    release_condition="manual restart required after a rejected live order",
                    details={"client_order_id": result.client_order_id},
                )
            return
        state = OrderState(
            client_order_id=result.client_order_id,
            symbol=result.symbol,
            side=result.side,
            quantity=result.quantity,
            filled_qty=result.filled_qty,
            remaining_qty=result.remaining_qty,
            status=result.status,
            updated_at=utc_now(),
            order_id=result.order_id,
            terminal=result.terminal,
            reason_code=result.reason_code,
        )
        self.open_orders_by_client_id[result.client_order_id] = state
        self._seed_sequence(result.client_order_id)
        self.quarantine_symbol(
            result.symbol,
            reason_code=ReasonCode.ORDER_SYMBOL_QUARANTINED,
            release_mode="until_order_terminal",
            release_condition="released when the broker order becomes terminal or disappears during sync",
            details={"client_order_id": result.client_order_id},
        )

    def _release_symbol_for_order(self, client_order_id: str) -> None:
        symbols_to_release = [
            symbol
            for symbol, quarantine in self.symbol_quarantines.items()
            if quarantine.release_mode == "until_order_terminal"
            and quarantine.details.get("client_order_id") == client_order_id
        ]
        for symbol in symbols_to_release:
            self.symbol_quarantines.pop(symbol, None)

    def _has_open_order_for_symbol(self, symbol: str) -> bool:
        return any(order.symbol == symbol and not order.terminal for order in self.open_orders_by_client_id.values())

    def _seed_sequence(self, client_order_id: str) -> None:
        match = CLIENT_ORDER_ID_PARSE.match(client_order_id)
        if match is None:
            return
        key = (
            match.group("session"),
            match.group("symbol"),
            "buy" if match.group("side") == "b" else "sell",
            match.group("bucket"),
        )
        self.sequence_by_key[key] = max(self.sequence_by_key.get(key, 0), int(match.group("seq")))

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        message = str(exc).lower()
        if any(token in message for token in ("invalid", "rejected", "duplicate", "not tradable", "blocked")):
            return False
        return True

    @staticmethod
    def _release_condition_for(release_mode: str) -> str:
        if release_mode == "until_order_terminal":
            return "released when the tracked broker order reaches a terminal state or disappears during synchronization"
        if release_mode == "until_restart":
            return "manual runtime restart required"
        return "release condition depends on runtime policy"
