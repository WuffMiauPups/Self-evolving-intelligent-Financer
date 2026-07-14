"""Phase 2 §6.2-C — counterfactual memory credit.

The gate that decides whether memory influence stays enabled: paired
backtests identical in everything except ``memory.influence_mode``.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.core.config import AppConfig
from src.data.schemas import MarketBar
from src.evaluation import memory_credit as mc
from src.evaluation.memory_credit import (
    evaluate_memory_credit,
    sign_test_p_value,
)

_T0 = datetime(2026, 6, 1, 14, 30, tzinfo=UTC)


def _bars(n: int, symbol: str) -> list[MarketBar]:
    return [
        MarketBar(
            symbol=symbol,
            timestamp=_T0 + timedelta(minutes=5 * i),
            open=100.0 + 0.05 * i,
            high=100.5 + 0.05 * i,
            low=99.5 + 0.05 * i,
            close=100.2 + 0.05 * i,
            volume=1000.0,
            spread=0.05,
        )
        for i in range(n)
    ]


class TestSignTest:
    def test_known_values(self) -> None:
        assert sign_test_p_value(4, 4) == pytest.approx(0.0625)
        assert sign_test_p_value(3, 4) == pytest.approx(0.3125)
        assert sign_test_p_value(0, 4) == pytest.approx(1.0)
        assert sign_test_p_value(0, 0) == 1.0


class TestFoldSlicing:
    def test_contiguous_complete_cover(self) -> None:
        bars = {"AAPL": _bars(100, "AAPL"), "MSFT": _bars(100, "MSFT")}
        folds = mc._slice_folds(bars, 4)
        assert len(folds) == 4
        for symbol in bars:
            reassembled = [b for fold in folds for b in fold[symbol]]
            assert reassembled == bars[symbol]


def _scripted_run_arm(script: dict[tuple[str, int], dict[str, float]]):
    """Fake _run_arm keyed on (influence_mode, fold_index)."""
    calls: list[tuple[str, int, int]] = []
    counter = {"fold": -1, "seen": 0}

    def fake(config, fold_bars, policy_builder, *, seed):
        # Two consecutive calls share a fold; the mode distinguishes arms.
        if counter["seen"] % 2 == 0:
            counter["fold"] += 1
        counter["seen"] += 1
        mode = config.memory.influence_mode
        calls.append((mode, counter["fold"], seed))
        return dict(script[(mode, counter["fold"])])

    return fake, calls


class TestCreditAggregation:
    def _config(self) -> AppConfig:
        return AppConfig()

    def test_positive_credit(self, monkeypatch) -> None:
        script = {}
        for fold in range(4):
            script[("off", fold)] = {"total_return": 0.01, "expectancy": 1.0, "sharpe_ratio": 0.5, "fill_count": 50}
            script[("damping", fold)] = {"total_return": 0.02, "expectancy": 1.5, "sharpe_ratio": 0.8, "fill_count": 45}
        fake, calls = _scripted_run_arm(script)
        monkeypatch.setattr(mc, "_run_arm", fake)
        report = evaluate_memory_credit(
            self._config(), {"AAPL": _bars(100, "AAPL")}, lambda *a: None, n_folds=4
        )
        assert report.credit_positive is True
        assert report.positive_folds == 4
        assert report.mean_delta_total_return == pytest.approx(0.01)
        assert report.sign_test_p_value == pytest.approx(0.0625)
        # Paired arms share the seed within each fold.
        seeds_by_fold: dict[int, set[int]] = {}
        for _mode, fold, seed in calls:
            seeds_by_fold.setdefault(fold, set()).add(seed)
        assert all(len(seeds) == 1 for seeds in seeds_by_fold.values())

    def test_negative_credit_refused(self, monkeypatch) -> None:
        script = {}
        for fold in range(4):
            script[("off", fold)] = {"total_return": 0.02, "expectancy": 1.0, "sharpe_ratio": 0.5, "fill_count": 50}
            script[("damping", fold)] = {"total_return": 0.01, "expectancy": 0.5, "sharpe_ratio": 0.3, "fill_count": 45}
        fake, _ = _scripted_run_arm(script)
        monkeypatch.setattr(mc, "_run_arm", fake)
        report = evaluate_memory_credit(
            self._config(), {"AAPL": _bars(100, "AAPL")}, lambda *a: None, n_folds=4
        )
        assert report.credit_positive is False
        assert report.positive_folds == 0

    def test_mixed_folds_require_majority_and_positive_mean(self, monkeypatch) -> None:
        # 2 of 4 positive with net-negative mean → refused.
        script = {}
        for fold in range(4):
            script[("off", fold)] = {"total_return": 0.01, "expectancy": 1.0, "sharpe_ratio": 0.5, "fill_count": 50}
        script[("damping", 0)] = {"total_return": 0.012, "expectancy": 1.1, "sharpe_ratio": 0.6, "fill_count": 45}
        script[("damping", 1)] = {"total_return": 0.011, "expectancy": 1.0, "sharpe_ratio": 0.5, "fill_count": 45}
        script[("damping", 2)] = {"total_return": 0.001, "expectancy": 0.2, "sharpe_ratio": 0.1, "fill_count": 45}
        script[("damping", 3)] = {"total_return": 0.002, "expectancy": 0.3, "sharpe_ratio": 0.2, "fill_count": 45}
        fake, _ = _scripted_run_arm(script)
        monkeypatch.setattr(mc, "_run_arm", fake)
        report = evaluate_memory_credit(
            self._config(), {"AAPL": _bars(100, "AAPL")}, lambda *a: None, n_folds=4
        )
        assert report.positive_folds == 2
        assert report.mean_delta_total_return < 0
        assert report.credit_positive is False

    def test_degenerate_folds_excluded(self, monkeypatch) -> None:
        script = {}
        for fold in range(3):
            script[("off", fold)] = {"total_return": 0.01, "expectancy": 1.0, "sharpe_ratio": 0.5, "fill_count": 50}
            script[("damping", fold)] = {"total_return": 0.02, "expectancy": 1.5, "sharpe_ratio": 0.8, "fill_count": 45}
        # Fold 3: baseline barely traded → excluded from aggregation.
        script[("off", 3)] = {"total_return": -0.5, "expectancy": -9.0, "sharpe_ratio": -3.0, "fill_count": 1}
        script[("damping", 3)] = {"total_return": 0.5, "expectancy": 9.0, "sharpe_ratio": 3.0, "fill_count": 1}
        fake, _ = _scripted_run_arm(script)
        monkeypatch.setattr(mc, "_run_arm", fake)
        report = evaluate_memory_credit(
            self._config(), {"AAPL": _bars(100, "AAPL")}, lambda *a: None, n_folds=4
        )
        assert report.degenerate_folds == 1
        assert report.active_folds == 3
        assert report.mean_delta_total_return == pytest.approx(0.01)
        assert report.credit_positive is True

    def test_all_degenerate_means_no_credit(self, monkeypatch) -> None:
        script = {}
        for fold in range(2):
            script[("off", fold)] = {"total_return": 0.0, "expectancy": 0.0, "sharpe_ratio": 0.0, "fill_count": 0}
            script[("damping", fold)] = {"total_return": 0.1, "expectancy": 1.0, "sharpe_ratio": 1.0, "fill_count": 0}
        fake, _ = _scripted_run_arm(script)
        monkeypatch.setattr(mc, "_run_arm", fake)
        report = evaluate_memory_credit(
            self._config(), {"AAPL": _bars(100, "AAPL")}, lambda *a: None, n_folds=2
        )
        assert report.credit_positive is False
        assert report.active_folds == 0

    def test_report_round_trips_to_json(self, monkeypatch) -> None:
        import json

        script = {
            ("off", 0): {"total_return": 0.01, "expectancy": 1.0, "sharpe_ratio": 0.5, "fill_count": 50},
            ("damping", 0): {"total_return": 0.02, "expectancy": 1.5, "sharpe_ratio": 0.8, "fill_count": 45},
        }
        fake, _ = _scripted_run_arm(script)
        monkeypatch.setattr(mc, "_run_arm", fake)
        report = evaluate_memory_credit(
            self._config(), {"AAPL": _bars(40, "AAPL")}, lambda *a: None, n_folds=1
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["credit_positive"] is True
        assert len(payload["folds"]) == 1


class TestHarnessIntroducesNoSpuriousDelta:
    def test_mode_insensitive_policy_yields_zero_credit(self) -> None:
        """With a policy that ignores influence_mode entirely, every fold
        delta must be exactly zero — proving the paired harness itself
        (seeding, slicing, engine setup) is causally clean."""
        from src.backtest.policy_factory import build_buy_and_hold_policy

        config = AppConfig()
        config.backtest.fill_timing = "next_open"
        bars = {"AAPL": _bars(120, "AAPL")}
        report = evaluate_memory_credit(
            config,
            bars,
            lambda arm_config, fold_bars, pipeline: build_buy_and_hold_policy(),
            n_folds=2,
        )
        for fold in report.folds:
            assert fold.delta_total_return == pytest.approx(0.0, abs=1e-12)
            assert fold.delta_expectancy == pytest.approx(0.0, abs=1e-12)
        assert report.credit_positive is False
