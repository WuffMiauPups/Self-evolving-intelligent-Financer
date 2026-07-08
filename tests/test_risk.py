from src.core.config import load_config
from src.core.time_utils import utc_now
from datetime import timedelta
from src.execution.reason_codes import ReasonCode
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager


def _ev_positive_proposal(**overrides) -> TradeProposal:
    """Build a proposal whose EV easily clears fee+slippage.

    With fee_bps=1.0 + slippage_bps=2.0 → cost=3.0.
    0.9 * 30 - 0.1 * 10 - 3 = 27 - 1 - 3 = 23 bps > 0 → approved.
    """
    base = dict(
        symbol="AAPL",
        timestamp=utc_now(),
        action=1,
        confidence=0.9,
        target_fraction=0.05,
        price=100.0,
        spread_bps=1.0,
        request_id="risk-test",
        expected_gain_bps=30.0,
        expected_loss_bps=10.0,
    )
    base.update(overrides)
    return TradeProposal(**base)


def test_risk_manager_rejects_wide_spread() -> None:
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(spread_bps=100.0, request_id="risk-test-1")
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is False
    assert "spread_too_wide" in decision.reasons


def test_risk_manager_approves_valid_trade() -> None:
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(request_id="risk-test-2")
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is True
    assert decision.quantity > 0


def test_risk_manager_rejects_stale_data_with_reason_code() -> None:
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(request_id="risk-test-3")
    decision = manager.evaluate(
        proposal,
        portfolio,
        evaluation_time=utc_now() + timedelta(seconds=config.risk.stale_data_sec + 1),
    )
    assert decision.approved is False
    assert ReasonCode.STALE_DATA in decision.reasons


# ---- EV gate tests -----------------------------------------------------


def test_ev_gate_rejects_unknown_risk_signals() -> None:
    """Default (0.0) gain/loss bps means unknown → EV is negative → reject."""
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
        request_id="ev-unknown",
        # No expected_gain_bps / expected_loss_bps → defaults 0.0
    )
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is False
    assert "negative_ev" in decision.reasons


def test_ev_gate_rejects_when_cost_exceeds_expected_edge() -> None:
    """High confidence + high gain, but cost dominates → reject."""
    config = load_config("configs/default.yaml")
    # Inflate trading costs so even a high-edge trade loses money.
    config.risk.fee_bps = 50.0
    config.risk.slippage_bps = 50.0  # total cost = 100 bps
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(
        request_id="ev-cost-dominates",
        expected_gain_bps=20.0,  # 0.9 * 20 - 0.1 * 10 - 100 = 18 - 1 - 100 = -83
        expected_loss_bps=10.0,
    )
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is False
    assert "negative_ev" in decision.reasons


def test_ev_gate_rejects_low_confidence_with_symmetric_edge() -> None:
    """Coin-flip confidence + symmetric gain/loss → net zero before cost → reject."""
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(
        request_id="ev-coinflip",
        confidence=0.5,
        expected_gain_bps=20.0,
        expected_loss_bps=20.0,
    )
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is False
    assert "negative_ev" in decision.reasons


def test_ev_gate_approves_positive_ev() -> None:
    """Confidence-weighted edge comfortably exceeds cost → approve."""
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(
        request_id="ev-positive",
        confidence=0.7,
        expected_gain_bps=30.0,
        expected_loss_bps=20.0,
    )
    # 0.7 * 30 - 0.3 * 20 - 3 = 21 - 6 - 3 = 12 > 0 → approve
    decision = manager.evaluate(proposal, portfolio)
    assert decision.approved is True
    assert decision.quantity > 0


def test_ev_gate_does_not_apply_to_flat_actions() -> None:
    """action=0 trades skip the EV gate entirely (they're rejected downstream
    by no_position_change / target_too_small style checks)."""
    config = load_config("configs/default.yaml")
    manager = RiskManager(config.risk)
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    proposal = _ev_positive_proposal(
        request_id="ev-flat",
        action=0,
        target_fraction=0.0,
        expected_gain_bps=0.0,
        expected_loss_bps=0.0,
    )
    decision = manager.evaluate(proposal, portfolio)
    # FLAT actions don't get a negative_ev tag (but will be rejected for
    # no_position_change).
    assert "negative_ev" not in decision.reasons
