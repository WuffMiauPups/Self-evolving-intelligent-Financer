from __future__ import annotations

from src.data.alpaca_client import MockBrokerClient
from src.execution.reason_codes import ReasonCode
from src.execution.reconciliation import run_startup_reconciliation
from src.risk.portfolio_state import PortfolioState


def test_reconciliation_hydrates_positions_and_blocks_on_open_orders() -> None:
    client = MockBrokerClient()
    client.positions["AAPL"] = {
        "symbol": "AAPL",
        "qty": 2.0,
        "avg_entry_price": 100.0,
        "current_price": 101.0,
    }
    client.open_orders["seif-testsession-AAPL-b-202604021230-0001"] = {
        "id": "broker-1",
        "client_order_id": "seif-testsession-AAPL-b-202604021230-0001",
        "symbol": "AAPL",
        "qty": 1.0,
        "filled_qty": 0.0,
        "remaining_qty": 1.0,
        "side": "buy",
        "status": "new",
        "terminal": False,
    }
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    result = run_startup_reconciliation(
        client=client,
        portfolio=portfolio,
        using_mock=False,
        block_on_mismatch=True,
    )
    assert result.safe_to_trade is False
    assert portfolio.get_position("AAPL") is not None
    assert any(issue.reason_code == ReasonCode.RECONCILIATION_OPEN_ORDERS_PRESENT for issue in result.issues)


def test_reconciliation_without_open_orders_is_safe() -> None:
    client = MockBrokerClient()
    client.positions["AAPL"] = {
        "symbol": "AAPL",
        "qty": 2.0,
        "avg_entry_price": 100.0,
        "current_price": 101.0,
    }
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    result = run_startup_reconciliation(
        client=client,
        portfolio=portfolio,
        using_mock=False,
        block_on_mismatch=True,
    )
    assert result.safe_to_trade is True
    assert any(issue.reason_code == ReasonCode.RECONCILIATION_POSITIONS_HYDRATED for issue in result.issues)
