from src.core.config import load_config
from src.core.time_utils import utc_now
from datetime import timedelta
from src.execution.reason_codes import ReasonCode
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager


def test_risk_manager_rejects_wide_spread() -> None:
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = TradeProposal(
        symbol="AAPL",
        timestamp=utc_now(),
        action=1,
        confidence=0.9,
        target_fraction=0.1,
        price=100.0,
        spread_bps=100.0,
        request_id="risk-test-1",
    )
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is False
    assert "spread_too_wide" in decision.reasons


def test_risk_manager_approves_valid_trade() -> None:
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = TradeProposal(
        symbol="AAPL",
        timestamp=utc_now(),
        action=1,
        confidence=0.9,
        target_fraction=0.05,
        price=100.0,
        spread_bps=1.0,
        request_id="risk-test-2",
    )
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is True
    assert decision.quantity > 0


def test_risk_manager_rejects_stale_data_with_reason_code() -> None:
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = TradeProposal(
        symbol="AAPL",
        timestamp=utc_now(),
        action=1,
        confidence=0.9,
        target_fraction=0.05,
        price=100.0,
        spread_bps=1.0,
        request_id="risk-test-3",
    )
    decision = manager.evaluate(
        proposal,
        portfolio,
        evaluation_time=utc_now() + timedelta(seconds=config.risk.stale_data_sec + 1),
    )
    assert decision.approved is False
    assert ReasonCode.STALE_DATA in decision.reasons
