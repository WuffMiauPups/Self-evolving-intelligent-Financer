"""Bootstrap-only synthetic stress (§7.2.6).

Session-block bootstrap conditioned on regime labels: within-session
bars are real, splices are rebased (no jumps), every symbol shares the
sampled session sequence, and the stylized-facts gate refuses splice
artifacts while allowing flags inherited from the source. Adverse-fill
backtest mode worsens fills against the trade, bounded by the bar; the
promotion record carries stress deltas verbatim and the gate NEVER reads
them.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.config import AppConfig, load_config
from src.core.exceptions import ConfigurationError
from src.data.schemas import MarketBar
from src.evaluation.bootstrap_stress import (
    BootstrapStressError,
    generate_bootstrap_paths,
)

T0 = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)


def _session(symbol: str, day_offset: int, *, level: float, volume: float, n_bars: int = 20) -> list[MarketBar]:
    day = T0 + timedelta(days=day_offset)
    bars = []
    for i in range(n_bars):
        close = level * (1.0 + 0.0002 * i)
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=day + timedelta(minutes=5 * i),
                open=close,
                high=close * 1.0005,
                low=close * 0.9995,
                close=close,
                volume=volume,
            )
        )
    return bars


def _bars_and_labels() -> tuple[dict[str, list[MarketBar]], dict[str, int]]:
    """10 sessions, alternating clusters; cluster 1 sessions carry volume 111."""
    bars: dict[str, list[MarketBar]] = {"AAA": [], "BBB": []}
    labels: dict[str, int] = {}
    for day in range(10):
        cluster = day % 2
        volume = 111.0 if cluster == 1 else 999.0
        level = 100.0 + 10.0 * day
        bars["AAA"] += _session("AAA", day, level=level, volume=volume)
        bars["BBB"] += _session("BBB", day, level=level * 3.0, volume=volume)
        labels[(T0 + timedelta(days=day)).date().isoformat()] = cluster
    return bars, labels


def test_paths_condition_on_target_cluster_and_are_deterministic() -> None:
    bars, labels = _bars_and_labels()
    paths_a = generate_bootstrap_paths(
        bars, labels, target_clusters={1}, n_sessions=6, seed=7
    )
    paths_b = generate_bootstrap_paths(
        bars, labels, target_clusters={1}, n_sessions=6, seed=7
    )
    assert all(bar.volume == 111.0 for bar in paths_a["AAA"])  # cluster-1 sessions only
    assert [b.close for b in paths_a["AAA"]] == [b.close for b in paths_b["AAA"]]
    assert len(paths_a["AAA"]) == 6 * 20


def test_splices_are_continuous_and_cross_symbol_coherent() -> None:
    bars, labels = _bars_and_labels()
    paths = generate_bootstrap_paths(
        bars, labels, target_clusters={0, 1}, n_sessions=8, seed=3
    )
    for symbol in ("AAA", "BBB"):
        closes = np.array([bar.close for bar in paths[symbol]])
        returns = np.abs(closes[1:] / closes[:-1] - 1.0)
        assert returns.max() < 0.01  # rebasing: no splice jumps despite 10x level spread
    # Cross-symbol coherence: both symbols sampled the SAME session sequence,
    # so their per-session volume markers align bar for bar.
    assert [bar.volume for bar in paths["AAA"]] == [bar.volume for bar in paths["BBB"]]
    # Synthetic timeline: strictly increasing timestamps on weekdays.
    stamps = [bar.timestamp for bar in paths["AAA"]]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))
    assert all(stamp.weekday() < 5 for stamp in stamps)


def test_inherited_crash_flags_allowed_splice_artifacts_refused() -> None:
    bars, labels = _bars_and_labels()
    # Plant a real -25% crash bar inside day 1's session, and label ONLY
    # day 1 as the target cluster so every sampled session contains it.
    mid = bars["AAA"][30]
    bars["AAA"][30] = MarketBar(
        symbol="AAA",
        timestamp=mid.timestamp,
        open=mid.open,
        high=mid.high,
        low=mid.low * 0.7,
        close=mid.close * 0.75,
        volume=mid.volume,
    )
    crash_only_labels = {
        day: (7 if day == (T0 + timedelta(days=1)).date().isoformat() else 0)
        for day in labels
    }
    paths = generate_bootstrap_paths(
        bars, crash_only_labels, target_clusters={7}, n_sessions=5, seed=11
    )
    assert paths  # inherited flag did not raise
    # A synthetic-only flag must raise: corrupt the generated-path gate by
    # feeding a source WITHOUT the crash while sampling sessions WITH it.
    clean_bars, _ = _bars_and_labels()
    from src.evaluation import bootstrap_stress as module

    with pytest.raises(BootstrapStressError):
        module._stylized_facts_gate(clean_bars, paths, logger=_SilentLogger())


class _SilentLogger:
    def info(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        pass


def test_empty_pool_raises() -> None:
    bars, labels = _bars_and_labels()
    with pytest.raises(BootstrapStressError, match="No sessions"):
        generate_bootstrap_paths(bars, labels, target_clusters={9}, n_sessions=3, seed=1)


def test_adverse_fills_worsen_buy_fill_and_equity() -> None:
    import torch

    from src.backtest.engine import BacktestEngine
    from src.data.historical_data import generate_synthetic_bars
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.strategies.baselines import BuyAndHoldPolicy

    # Generate ONCE: the helper stamps bars from datetime.now, so a second
    # call would produce different timestamps.
    bars = generate_synthetic_bars("AAPL", 240)

    def _run(adverse: bool):
        config = AppConfig()
        config.backtest.fill_timing = "next_open"
        config.backtest.adverse_fills = adverse
        config.backtest.adverse_fill_range_fraction = 0.5
        config.risk.cooldown_sec = 0
        config.risk.duplicate_order_window_sec = 0
        config.risk.min_trade_notional = 0.0
        engine = BacktestEngine(
            config,
            FeaturePipeline(config.features, device=torch.device("cpu")),
            RiskManager(config.risk.model_copy()),
        )
        return engine.run({"AAPL": bars}, BuyAndHoldPolicy())

    baseline = _run(adverse=False)
    stressed = _run(adverse=True)
    assert baseline.fills and stressed.fills
    first_base, first_adverse = baseline.fills[0], stressed.fills[0]
    assert first_adverse.price > first_base.price  # buy pushed against the trade
    # The executor's slippage applies identically to both runs, so the fill
    # DELTA equals the adverse adjustment — bounded by fraction * bar range.
    fill_bar = next(bar for bar in bars if bar.timestamp == first_adverse.timestamp)
    assert first_adverse.price - first_base.price <= 0.5 * (fill_bar.high - fill_bar.low) + 1e-9
    assert stressed.metrics["total_return"] <= baseline.metrics["total_return"]


def test_promotion_record_carries_stress_but_gate_ignores_it() -> None:
    from src.training.checkpoint_promotion import PromotionGate, evaluate_promotion

    kwargs = dict(
        candidate_metrics={"val_report.balanced_accuracy": 0.9, "val_report.macro_f1": 0.9},
        parent_metrics=None,
        gate=PromotionGate(min_val_balanced_accuracy=0.0, min_val_macro_f1=0.0),
    )
    terrible_stress = {"stress_delta_total_return": -0.99, "stress_adverse_sharpe_ratio": -9.0}
    without = evaluate_promotion(**kwargs)
    with_stress = evaluate_promotion(**kwargs, stress_metrics=terrible_stress)
    assert with_stress.stress_metrics == terrible_stress  # recorded verbatim
    assert without.stress_metrics is None
    assert with_stress.eligible == without.eligible  # NEVER a criterion
    assert with_stress.reasons == without.reasons


def test_config_pair_and_validator() -> None:
    config = load_config("configs/default.yaml")
    assert config.backtest.adverse_fills is False  # off by default: zero behavior change
    assert config.backtest.adverse_fill_range_fraction == pytest.approx(0.25)
    with pytest.raises(ConfigurationError, match="adverse_fill_range_fraction"):
        AppConfig.model_validate({"backtest": {"adverse_fill_range_fraction": 1.5}})
