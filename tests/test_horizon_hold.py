"""Horizon-aligned holding period (risk.horizon_hold_sec, §7.1.6).

The model predicts H-bar outcomes; re-deciding every bar pays the round
trip without collecting the predicted move (measured: §9.4 arm 3). While
a directional position is younger than horizon_hold_sec, model-originated
re-decisions (exit, flip, add, trim) are rejected `horizon_hold_active`.
Risk always overrides: position-guard exits pass, and risk-reducing
proposals pass while a halt state (kill switch / daily-loss) is active.
Enforced inside the SHARED RiskManager.evaluate — both engines inherit it.
Off by default (0.0).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.core.config import AppConfig, RiskConfig, load_config
from src.core.exceptions import ConfigurationError
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState, apply_fill
from src.risk.risk_manager import RiskManager

T = datetime(2026, 7, 17, 15, 0, tzinfo=UTC)
HOLD_SEC = 9000.0  # 30 bars x 300 s


def _portfolio(qty: float = 10.0, symbol: str = "NVDA") -> PortfolioState:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    apply_fill(portfolio, symbol=symbol, signed_delta=qty, fill_price=100.0, timestamp=T)
    portfolio.get_position(symbol).market_price = 100.0
    return portfolio


def _config(**overrides) -> RiskConfig:
    return RiskConfig(horizon_hold_sec=HOLD_SEC, cooldown_sec=0, **overrides)


def _proposal(
    action: int,
    fraction: float,
    *,
    at: datetime,
    symbol: str = "NVDA",
    strategy_id: str = "brain",
) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        timestamp=at,
        action=action,
        confidence=1.0,
        target_fraction=fraction,
        price=100.0,
        spread_bps=0.0,
        request_id=f"hold-test-{symbol}-{action}-{fraction}-{at.timestamp()}",
        expected_gain_bps=100.0,
        expected_loss_bps=0.0,
        strategy_id=strategy_id,
    )


def test_hold_blocks_model_exit_flip_and_add_inside_window() -> None:
    inside = T + timedelta(seconds=300)
    for action, fraction in ((0, 0.0), (-1, 0.01), (1, 0.02)):  # exit, flip, add
        decision = RiskManager(_config()).evaluate(
            _proposal(action, fraction, at=inside), _portfolio(), evaluation_time=inside
        )
        assert decision.approved is False, f"action={action} must be held"
        assert "horizon_hold_active" in decision.reasons


def test_hold_releases_once_window_elapses() -> None:
    released = T + timedelta(seconds=HOLD_SEC)
    decision = RiskManager(_config()).evaluate(
        _proposal(0, 0.0, at=released), _portfolio(), evaluation_time=released
    )
    assert decision.approved is True
    assert "horizon_hold_active" not in decision.reasons


def test_position_guard_exit_is_exempt_inside_window() -> None:
    inside = T + timedelta(seconds=300)
    decision = RiskManager(_config()).evaluate(
        _proposal(0, 0.0, at=inside, strategy_id="position_guard"),
        _portfolio(),
        evaluation_time=inside,
    )
    assert decision.approved is True
    assert decision.side == "sell" and decision.quantity == pytest.approx(10.0)


def test_risk_reducing_passes_under_halt_but_adds_stay_blocked() -> None:
    inside = T + timedelta(seconds=300)
    # Kill switch active: a model flatten (risk-reducing) must not be trapped
    # by the hold — de-risking always wins.
    config = _config(kill_switch=True)
    exit_decision = RiskManager(config).evaluate(
        _proposal(0, 0.0, at=inside), _portfolio(), evaluation_time=inside
    )
    assert exit_decision.approved is True
    assert "horizon_hold_active" not in exit_decision.reasons
    # A risk-INCREASING add under the same halt stays blocked by both gates.
    add_decision = RiskManager(config).evaluate(
        _proposal(1, 0.02, at=inside), _portfolio(), evaluation_time=inside
    )
    assert add_decision.approved is False
    assert "kill_switch" in add_decision.reasons
    assert "horizon_hold_active" in add_decision.reasons


def test_daily_loss_breach_also_releases_risk_reducing_exits() -> None:
    inside = T + timedelta(seconds=300)
    portfolio = _portfolio()
    portfolio.daily_pnl = -400.0  # breaches max_daily_loss=350
    decision = RiskManager(_config()).evaluate(
        _proposal(0, 0.0, at=inside), portfolio, evaluation_time=inside
    )
    assert decision.approved is True
    assert "horizon_hold_active" not in decision.reasons


def test_new_entries_and_flat_symbols_are_never_held() -> None:
    inside = T + timedelta(seconds=300)
    # No position at all: entries are not the hold's business.
    empty = PortfolioState(cash=100_000.0, equity=100_000.0)
    decision = RiskManager(_config()).evaluate(
        _proposal(1, 0.01, at=inside), empty, evaluation_time=inside
    )
    assert decision.approved is True
    # Held NVDA must not freeze an unrelated flat symbol.
    other = RiskManager(_config()).evaluate(
        _proposal(1, 0.01, at=inside, symbol="AAPL"), _portfolio(), evaluation_time=inside
    )
    assert other.approved is True


def test_zero_disables_the_hold_entirely() -> None:
    inside = T + timedelta(seconds=1)
    decision = RiskManager(RiskConfig(cooldown_sec=0)).evaluate(
        _proposal(0, 0.0, at=inside), _portfolio(), evaluation_time=inside
    )
    assert decision.approved is True
    assert "horizon_hold_active" not in decision.reasons


def test_entry_time_semantics_pin_the_hold_clock() -> None:
    # Same-side resize keeps the original entry (the hold window does not
    # restart); a flip through zero starts a fresh window.
    portfolio = _portfolio(qty=10.0)
    apply_fill(
        portfolio, symbol="NVDA", signed_delta=5.0, fill_price=101.0, timestamp=T + timedelta(600)
    )
    assert portfolio.get_position("NVDA").entry_time == T
    apply_fill(
        portfolio,
        symbol="NVDA",
        signed_delta=-25.0,
        fill_price=99.0,
        timestamp=T + timedelta(seconds=1200),
    )
    assert portfolio.get_position("NVDA").qty == pytest.approx(-10.0)
    assert portfolio.get_position("NVDA").entry_time == T + timedelta(seconds=1200)


def test_config_pair_defaults_off_and_validator() -> None:
    assert load_config("configs/default.yaml").risk.horizon_hold_sec == 0.0
    assert AppConfig().risk.horizon_hold_sec == 0.0
    with pytest.raises(ConfigurationError, match="horizon_hold_sec"):
        RiskConfig(horizon_hold_sec=-1.0)


def test_backtest_turnover_drops_with_the_hold_on_end_to_end() -> None:
    import torch

    from src.backtest.engine import BacktestEngine
    from src.data.historical_data import generate_synthetic_bars
    from src.features.feature_pipeline import FeaturePipeline

    class ChurnPolicy:
        """Flips direction every bar — the §7.1.6 worst case."""

        def __init__(self) -> None:
            self._flip = 1

        def reset_all_state(self, *, reason: str = "manual_reset") -> str:
            return reason

        def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
            return reason

        def propose(self, snapshot):
            self._flip = -self._flip
            proposal = TradeProposal(
                symbol=snapshot.symbol,
                timestamp=snapshot.timestamp,
                action=self._flip,
                confidence=1.0,
                target_fraction=0.05,
                price=snapshot.last_price,
                spread_bps=snapshot.spread_bps,
                request_id=f"churn-{snapshot.symbol}-{int(snapshot.timestamp.timestamp())}",
                expected_gain_bps=100.0,
                expected_loss_bps=0.0,
            )
            return proposal, proposal.metadata

    bars = generate_synthetic_bars("AAPL", 240)

    def _run(hold_bars: int):
        config = AppConfig()
        config.backtest.fill_timing = "next_open"
        config.risk.cooldown_sec = 0
        config.risk.duplicate_order_window_sec = 0
        config.risk.min_trade_notional = 0.0
        config.risk.horizon_hold_sec = hold_bars * config.backtest.seconds_per_bar
        return BacktestEngine(
            config,
            FeaturePipeline(config.features, device=torch.device("cpu")),
            RiskManager(config.risk.model_copy()),
        ).run({"AAPL": bars}, ChurnPolicy())

    held = _run(hold_bars=6)
    free = _run(hold_bars=0)
    assert free.metrics["fill_count"] > 0
    assert held.metrics["fill_count"] < free.metrics["fill_count"] / 2.0, (
        f"hold must cut churn turnover: {held.metrics['fill_count']} vs "
        f"{free.metrics['fill_count']}"
    )


def test_hold_lives_in_the_shared_path_source_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    risk = (root / "src" / "risk" / "risk_manager.py").read_text(encoding="utf-8")
    engine = (root / "src" / "backtest" / "engine.py").read_text(encoding="utf-8")
    tick = (root / "src" / "execution" / "tick_processor.py").read_text(encoding="utf-8")
    # The gate exists exactly once, inside the shared evaluate().
    assert risk.count("horizon_hold_active") == 1
    # Both engines route decisions through that shared evaluate — no bypass.
    assert "self.risk_manager.evaluate(" in engine
    assert "ctx.risk_manager.evaluate(" in tick
    assert "horizon_hold" not in engine and "horizon_hold" not in tick
