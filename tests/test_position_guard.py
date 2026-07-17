"""Position guard: deterministic P&L exits (risk.position_stop_loss_pct /
position_take_profit_pct).

A breach originates a FLATTEN proposal that rides the normal shared
risk-reducing path: halt-exempt, deadband-exempt, EV-gate-free, cooldown
after the exit fill. Wired into BOTH the backtest engine and the live
tick processor (§10.4). Off by default — 0.0 disables everything.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.core.config import AppConfig, RiskConfig, load_config
from src.core.exceptions import ConfigurationError
from src.risk.portfolio_state import PortfolioState, apply_fill
from src.risk.position_guard import check_position_exits, position_pnl_pct
from src.risk.risk_manager import RiskManager

T = datetime(2026, 7, 17, 15, 0, tzinfo=UTC)


def _portfolio(qty: float, avg: float, market: float, symbol: str = "NVDA") -> PortfolioState:
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    apply_fill(portfolio, symbol=symbol, signed_delta=qty, fill_price=avg, timestamp=T)
    portfolio.get_position(symbol).market_price = market
    return portfolio


def _config(stop: float = 0.05, take: float = 0.0) -> RiskConfig:
    return RiskConfig(position_stop_loss_pct=stop, position_take_profit_pct=take)


def test_pnl_pct_is_sign_adjusted_for_shorts() -> None:
    assert position_pnl_pct(10.0, 100.0, 94.0) == pytest.approx(-0.06)   # long, down
    assert position_pnl_pct(-10.0, 100.0, 94.0) == pytest.approx(+0.06)  # short, down = profit
    assert position_pnl_pct(-10.0, 100.0, 106.0) == pytest.approx(-0.06)  # short, up = loss


def test_stop_loss_triggers_flatten_proposal() -> None:
    exits = check_position_exits(_portfolio(10.0, 100.0, 94.0), _config(stop=0.05), timestamp=T)
    assert len(exits) == 1
    snapshot, proposal, reason = exits[0]
    assert reason == "position_stop_loss"
    assert proposal.action == 0 and proposal.target_fraction == 0.0
    assert proposal.strategy_id == "position_guard"
    assert proposal.metadata["pnl_pct"] == pytest.approx(-0.06)
    assert snapshot.last_price == pytest.approx(94.0)


def test_take_profit_triggers_and_healthy_position_does_not() -> None:
    config = _config(stop=0.05, take=0.08)
    assert check_position_exits(_portfolio(10.0, 100.0, 109.0), config, timestamp=T)[0][2] == (
        "position_take_profit"
    )
    assert check_position_exits(_portfolio(10.0, 100.0, 102.0), config, timestamp=T) == []


def test_zero_thresholds_disable_everything() -> None:
    assert check_position_exits(
        _portfolio(10.0, 100.0, 50.0), _config(stop=0.0, take=0.0), timestamp=T
    ) == []


def test_request_id_is_stable_within_a_bar_bucket() -> None:
    config = _config(stop=0.05)
    first = check_position_exits(_portfolio(10.0, 100.0, 94.0), config, timestamp=T)[0][1]
    again = check_position_exits(
        _portfolio(10.0, 100.0, 94.0), config, timestamp=T + timedelta(seconds=30)
    )[0][1]
    next_bar = check_position_exits(
        _portfolio(10.0, 100.0, 94.0), config, timestamp=T + timedelta(minutes=5)
    )[0][1]
    assert first.request_id == again.request_id      # dedupe within the bar
    assert first.request_id != next_bar.request_id   # fresh attempt next bar


def test_guard_exit_rides_the_risk_reducing_path_even_under_kill_switch() -> None:
    config = _config(stop=0.05)
    config = config.model_copy(update={"kill_switch": True, "cooldown_sec": 0})
    portfolio = _portfolio(10.0, 100.0, 94.0)
    _, proposal, _ = check_position_exits(portfolio, config, timestamp=T)[0]
    decision = RiskManager(config).evaluate(proposal, portfolio, evaluation_time=T)
    assert decision.approved is True          # halt-exempt (risk-reducing)
    assert decision.side == "sell"
    assert decision.quantity == pytest.approx(10.0)
    assert "kill_switch" not in decision.reasons
    assert "negative_ev" not in decision.reasons  # EV gate never sees flattens


def test_backtest_stop_loss_actually_fires_end_to_end() -> None:
    import torch

    from src.backtest.engine import BacktestEngine
    from src.data.historical_data import generate_synthetic_bars
    from src.features.feature_pipeline import FeaturePipeline
    from src.strategies.baselines import BuyAndHoldPolicy

    bars = generate_synthetic_bars("AAPL", 240)
    # Crash the tail hard: -1% per bar for the last 60 bars (~45% cumulative)
    # so ANY long entered earlier breaches a 5% stop with certainty.
    crashed = list(bars[:180])
    price = bars[179].close
    for offset, bar in enumerate(bars[180:]):
        price *= 0.99
        crashed.append(
            type(bar)(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=price * 1.001,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=bar.volume,
                spread=0.0,
            )
        )

    def _run(stop: float):
        config = AppConfig()
        config.backtest.fill_timing = "next_open"
        config.risk.cooldown_sec = 0
        config.risk.duplicate_order_window_sec = 0
        config.risk.min_trade_notional = 0.0
        config.risk.position_stop_loss_pct = stop
        return BacktestEngine(
            config,
            FeaturePipeline(config.features, device=torch.device("cpu")),
            RiskManager(config.risk.model_copy()),
        ).run({"AAPL": crashed}, BuyAndHoldPolicy())

    guarded = _run(stop=0.05)
    unguarded = _run(stop=0.0)
    guarded_sells = [f for f in guarded.fills if f.side == "sell"]
    assert guarded_sells, "the stop must liquidate the crashing long"
    assert any(f.strategy_id == "position_guard" for f in guarded_sells)
    # The guard cut the loss: strictly better terminal equity than riding it.
    assert guarded.metrics["total_return"] > unguarded.metrics["total_return"]


def test_config_pair_defaults_off_and_validators() -> None:
    config = load_config("configs/default.yaml")
    assert config.risk.position_stop_loss_pct == 0.0
    assert config.risk.position_take_profit_pct == 0.0
    with pytest.raises(ConfigurationError, match="position_stop_loss_pct"):
        RiskConfig(position_stop_loss_pct=-0.1)
    with pytest.raises(ConfigurationError, match="fraction"):
        RiskConfig(position_take_profit_pct=5.0)


def test_both_engines_wire_the_guard_source_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = (root / "src" / "backtest" / "engine.py").read_text(encoding="utf-8")
    tick = (root / "src" / "execution" / "tick_processor.py").read_text(encoding="utf-8")
    assert "check_position_exits(" in engine
    assert "check_position_exits(" in tick
    # Live path: exits must be prepended before Phase C dispatch.
    guard_region = tick.split("check_position_exits(", 1)[1].split("# --- Phase C", 1)[0]
    assert "insert(0" in guard_region
