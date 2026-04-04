from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.core.exceptions import ExecutionError
from src.core.time_utils import utc_now
from src.execution.order_router import OrderRouter
from src.execution.order_validator import validate_order_intent
from src.execution.paper_executor import PaperExecutor
from src.execution.reason_codes import ReasonCode
from src.execution.schemas import OrderIntent, OrderState
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState


VALID_CLIENT_ORDER_ID = "seif-testsession-AAPL-b-202604021230-0001"


@dataclass
class DummyLiveClient:
    paper: bool = False
    data_feed: str = "iex"
    responses: list[dict[str, object] | Exception] = field(default_factory=list)
    submit_calls: int = 0

    def submit_market_order(self, *, symbol: str, qty: float, side: str, client_order_id: str) -> dict[str, object]:
        del symbol, qty, side, client_order_id
        self.submit_calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, object] | None:
        del client_order_id
        return None

    def get_open_orders(self) -> list[dict[str, object]]:
        return []


def test_order_validation_and_paper_execution() -> None:
    intent = OrderIntent(
        symbol="AAPL",
        side="buy",
        quantity=10,
        client_order_id=VALID_CLIENT_ORDER_ID,
        reference_price=100.0,
        timestamp=utc_now(),
        paper=True,
        risk_approved=True,
        session_id="testsession",
        sequence=1,
    )
    validate_order_intent(intent)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    executor = PaperExecutor(portfolio=portfolio, ledger=Ledger(), fee_bps=1.0, slippage_bps=2.0)
    result = executor.execute(intent)
    assert result.accepted is True
    assert result.terminal is True
    assert result.filled_qty == 10
    assert portfolio.get_position("AAPL") is not None
    assert len(executor.ledger.fills) == 1


def test_order_router_rejects_missing_risk_approval() -> None:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    router = OrderRouter(
        paper_executor=PaperExecutor(portfolio=portfolio, ledger=Ledger(), fee_bps=1.0, slippage_bps=2.0),
    )
    intent = OrderIntent(
        symbol="AAPL",
        side="buy",
        quantity=1,
        client_order_id=VALID_CLIENT_ORDER_ID,
        reference_price=100.0,
        timestamp=utc_now(),
        paper=True,
        risk_approved=False,
        session_id="testsession",
        sequence=1,
    )
    with pytest.raises(ExecutionError):
        router.route(intent)


def test_duplicate_order_protection_blocks_existing_open_client_order_id() -> None:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    router = OrderRouter(
        paper_executor=PaperExecutor(portfolio=portfolio, ledger=Ledger(), fee_bps=1.0, slippage_bps=2.0),
    )
    router.seed_open_orders(
        [
            OrderState(
                client_order_id=VALID_CLIENT_ORDER_ID,
                symbol="AAPL",
                side="buy",
                quantity=1.0,
                filled_qty=0.0,
                remaining_qty=1.0,
                status="new",
                updated_at=utc_now(),
                order_id="broker-1",
                terminal=False,
                reason_code=ReasonCode.ORDER_PENDING,
            )
        ]
    )
    result = router.route(
        OrderIntent(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            client_order_id=VALID_CLIENT_ORDER_ID,
            reference_price=100.0,
            timestamp=utc_now(),
            paper=False,
            risk_approved=True,
            session_id="testsession",
            sequence=1,
        )
    )
    assert result.accepted is False
    assert result.reason_code == ReasonCode.ORDER_DUPLICATE_CLIENT_ID


def test_order_retry_policy_is_bounded() -> None:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    live_client = DummyLiveClient(
        responses=[RuntimeError("temporary timeout"), RuntimeError("temporary timeout"), RuntimeError("temporary timeout")]
    )
    router = OrderRouter(
        paper_executor=PaperExecutor(portfolio=portfolio, ledger=Ledger(), fee_bps=1.0, slippage_bps=2.0),
        broker_client=live_client,
        allow_live=True,
        max_submit_retries=1,
        retry_backoff_sec=0.0,
    )
    result = router.route(
        OrderIntent(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            client_order_id=VALID_CLIENT_ORDER_ID,
            reference_price=100.0,
            timestamp=utc_now(),
            paper=False,
            risk_approved=True,
            session_id="testsession",
            sequence=1,
        )
    )
    assert result.accepted is False
    assert result.reason_code == ReasonCode.ORDER_SUBMIT_RETRY_EXHAUSTED
    assert live_client.submit_calls == 2


def test_rejected_orders_are_handled_safely() -> None:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    live_client = DummyLiveClient(
        responses=[
            {
                "id": "broker-2",
                "client_order_id": VALID_CLIENT_ORDER_ID,
                "symbol": "AAPL",
                "qty": 1.0,
                "filled_qty": 0.0,
                "remaining_qty": 1.0,
                "side": "buy",
                "status": "rejected",
                "terminal": True,
            }
        ]
    )
    router = OrderRouter(
        paper_executor=PaperExecutor(portfolio=portfolio, ledger=Ledger(), fee_bps=1.0, slippage_bps=2.0),
        broker_client=live_client,
        allow_live=True,
    )
    result = router.route(
        OrderIntent(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            client_order_id=VALID_CLIENT_ORDER_ID,
            reference_price=100.0,
            timestamp=utc_now(),
            paper=False,
            risk_approved=True,
            session_id="testsession",
            sequence=1,
        )
    )
    assert result.accepted is False
    assert result.reason_code == ReasonCode.ORDER_REJECTED
    quarantine = router.is_symbol_quarantined("AAPL")
    assert quarantine is not None
    assert quarantine.release_mode == "until_restart"
