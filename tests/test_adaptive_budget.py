"""Phase 2 §6.4-C — outcome-adaptive risk budget.

Contract: realized rolling drawdown throttles risk-INCREASING size,
bounded [budget_min_multiplier, 1.0], deterministic, logged on every
decision. Risk-reducing trades are never throttled and a throttled add
can never silently become a trim.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.config import RiskConfig
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager, adaptive_budget_multiplier

T = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)


def _proposal(**overrides) -> TradeProposal:
    base = dict(
        symbol="AAPL",
        timestamp=T,
        action=1,
        confidence=0.9,
        target_fraction=0.02,
        price=100.0,
        spread_bps=1.0,
        request_id="ab-default",
        expected_gain_bps=30.0,
        expected_loss_bps=10.0,
    )
    base.update(overrides)
    return TradeProposal(**base)


def _portfolio(*, drawdown: float = 0.0) -> PortfolioState:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    if drawdown < 0.0:
        peak = 100_000.0 / (1.0 + drawdown)
        portfolio.pnl_history.extend([peak, 100_000.0])
    else:
        portfolio.pnl_history.extend([100_000.0, 100_000.0])
    return portfolio


def _config(**overrides) -> RiskConfig:
    base = dict(
        adaptive_budget_enabled=True,
        budget_min_multiplier=0.25,
        budget_full_at_drawdown=-0.05,
    )
    base.update(overrides)
    return RiskConfig(**base)


class TestMultiplierMath:
    def test_no_drawdown_no_throttle(self) -> None:
        assert adaptive_budget_multiplier(0.0, full_at_drawdown=-0.05, min_multiplier=0.25) == 1.0
        assert adaptive_budget_multiplier(0.01, full_at_drawdown=-0.05, min_multiplier=0.25) == 1.0

    def test_linear_between_zero_and_floor(self) -> None:
        # Halfway to the floor drawdown → halfway to the floor multiplier.
        assert adaptive_budget_multiplier(
            -0.025, full_at_drawdown=-0.05, min_multiplier=0.25
        ) == pytest.approx(0.625)

    def test_floor_is_hard(self) -> None:
        for drawdown in (-0.05, -0.10, -0.99):
            assert adaptive_budget_multiplier(
                drawdown, full_at_drawdown=-0.05, min_multiplier=0.25
            ) == pytest.approx(0.25)

    def test_bounds_hold_everywhere(self) -> None:
        for i in range(200):
            drawdown = -0.001 * i
            m = adaptive_budget_multiplier(drawdown, full_at_drawdown=-0.05, min_multiplier=0.25)
            assert 0.25 <= m <= 1.0


class TestRiskManagerIntegration:
    def test_drawdown_throttles_new_position_size(self) -> None:
        manager = RiskManager(_config())
        # -2.5% drawdown → multiplier 0.625; fraction 0.02 → 0.0125 → 12 shares.
        decision = manager.evaluate(_proposal(), _portfolio(drawdown=-0.025), evaluation_time=T)
        assert decision.approved is True
        assert decision.budget_multiplier == pytest.approx(0.625)
        assert "adaptive_budget" in decision.reasons
        # fraction 0.02 * 0.625 = 0.0125 -> 1250 notional / 100 = 12.5 shares.
        assert decision.quantity == pytest.approx(12.5)

    def test_no_drawdown_full_size(self) -> None:
        manager = RiskManager(_config())
        decision = manager.evaluate(_proposal(), _portfolio(), evaluation_time=T)
        assert decision.approved is True
        assert decision.budget_multiplier == 1.0
        assert "adaptive_budget" not in decision.reasons
        assert decision.quantity == pytest.approx(20.0)

    def test_disabled_is_bit_identical_to_before(self) -> None:
        enabled_off = RiskManager(_config(adaptive_budget_enabled=False))
        decision = enabled_off.evaluate(
            _proposal(), _portfolio(drawdown=-0.04), evaluation_time=T
        )
        assert decision.budget_multiplier == 1.0
        assert decision.quantity == pytest.approx(20.0)

    def test_risk_reducing_trades_are_never_throttled(self) -> None:
        from src.risk.portfolio_state import apply_fill

        manager = RiskManager(_config())
        portfolio = _portfolio(drawdown=-0.10)  # deep drawdown, floor multiplier
        apply_fill(portfolio, symbol="AAPL", signed_delta=50.0, fill_price=100.0, timestamp=T)
        portfolio.pnl_history.clear()
        peak = 100_000.0 / (1.0 - 0.10)
        portfolio.pnl_history.extend([peak, 100_000.0])
        # Full flatten: action=0 → target 0 → sell 50. Must be untouched.
        decision = manager.evaluate(
            _proposal(action=0, target_fraction=0.0, request_id="ab-flatten"),
            portfolio,
            evaluation_time=T,
        )
        assert decision.approved is True
        assert decision.quantity == pytest.approx(50.0)
        assert decision.budget_multiplier == 1.0

    def test_throttled_add_clamps_to_no_change_not_trim(self) -> None:
        from src.risk.portfolio_state import apply_fill

        manager = RiskManager(_config())
        portfolio = _portfolio(drawdown=-0.10)
        apply_fill(portfolio, symbol="AAPL", signed_delta=18.0, fill_price=100.0, timestamp=T)
        portfolio.pnl_history.clear()
        peak = 100_000.0 / (1.0 - 0.10)
        portfolio.pnl_history.extend([peak, 100_000.0])
        # Raw intent: grow 18 → 20 (an add). Floor multiplier 0.25 → target 5,
        # which is BELOW the current 18 — must clamp to "no change" and be
        # rejected as no_position_change, never emit a surprise sell.
        decision = manager.evaluate(
            _proposal(request_id="ab-clamp"), portfolio, evaluation_time=T
        )
        assert decision.approved is False
        assert "no_position_change" in decision.reasons
        assert decision.budget_multiplier == pytest.approx(0.25)

    def test_multiplier_logged_on_rejections_too(self) -> None:
        manager = RiskManager(_config(kill_switch=True))
        decision = manager.evaluate(
            _proposal(), _portfolio(drawdown=-0.025), evaluation_time=T
        )
        assert decision.approved is False
        assert decision.budget_multiplier == pytest.approx(0.625)


class TestConfigValidation:
    def test_defaults_off(self) -> None:
        assert RiskConfig().adaptive_budget_enabled is False

    def test_invalid_values_raise(self) -> None:
        with pytest.raises(Exception):
            RiskConfig(budget_min_multiplier=0.0)
        with pytest.raises(Exception):
            RiskConfig(budget_min_multiplier=1.5)
        with pytest.raises(Exception):
            RiskConfig(budget_full_at_drawdown=0.05)
