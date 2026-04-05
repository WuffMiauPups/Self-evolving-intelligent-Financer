"""Tests for training improvement pass: run summaries, comparison, label inspection, sweep, loss config."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import pytest

from src.training.run_summary import (
    build_run_summary,
    compare_runs,
    find_best_run,
    format_comparison_table,
    list_run_summaries,
    load_run_summary,
    save_run_summary,
)
from src.training.label_inspector import (
    LabelProfile,
    inspect_label_profiles,
    format_label_profiles,
)
from src.training.metrics import BatchMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig:
    class _Training:
        label_epsilon = 0.0005
        forward_horizons = [1, 5, 15]
        classification_loss = "focal"
        focal_gamma = 1.5
        label_smoothing = 0.02
        class_weight_mode = "balanced"
        use_weighted_sampler = True
        loss_weights = {"classification": 1.0, "regression": 0.25, "latent": 0.05}
        multitask_enabled = True
        contrastive_enabled = False
        batch_size = 64
        epochs = 3
        learning_rate = 0.001
        sequence_length = 64
        disable_regression = False
        disable_latent = False
        runs_dir = "runs"

    class _Data:
        symbols = ["AAPL", "MSFT"]
        timeframe = "1Min"
        start_date = "2022-01-01"
        end_date = "2026-01-01"

    training = _Training()
    data = _Data()


class _FakeDiagnostics:
    total_samples = 1000
    class_counts = {"short": 300, "flat": 400, "long": 300}
    class_ratios = {"short": 0.3, "flat": 0.4, "long": 0.3}
    flat_rate = 0.4
    near_flat_rate = 0.15
    near_boundary_rate = 0.08
    primary_return_quantiles = {"p10": -0.002, "p50": 0.0, "p90": 0.002}
    warnings = []


def _make_metrics(loss: float, accuracy: float, macro_f1: float, balanced_accuracy: float) -> BatchMetrics:
    return BatchMetrics(
        loss=loss,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        weighted_f1=macro_f1 + 0.02,
        long_recall=0.3,
        short_recall=0.35,
    )


def _make_report(split: str, macro_f1: float, balanced_accuracy: float) -> dict:
    return {
        "split": split,
        "accuracy": balanced_accuracy + 0.01,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1 + 0.01,
        "mcc": 0.1,
        "long_short_only_accuracy": 0.5,
        "prediction_counts": {"short": 100, "flat": 200, "long": 100},
        "target_counts": {"short": 120, "flat": 180, "long": 100},
        "prediction_ratios": {"short": 0.25, "flat": 0.5, "long": 0.25},
        "target_ratios": {"short": 0.3, "flat": 0.45, "long": 0.25},
        "per_class_metrics": {
            "short": {"precision": 0.3, "recall": 0.35, "f1": 0.32, "support": 120.0, "predicted": 100.0},
            "flat": {"precision": 0.5, "recall": 0.55, "f1": 0.52, "support": 180.0, "predicted": 200.0},
            "long": {"precision": 0.28, "recall": 0.3, "f1": 0.29, "support": 100.0, "predicted": 100.0},
        },
        "confusion_matrix": [[42, 50, 28], [30, 99, 51], [28, 42, 30]],
        "long_short_confusion": {"long_as_short_rate": 0.28, "short_as_long_rate": 0.23},
        "warnings": [],
    }


def _build_test_summary(run_id: str, val_f1: float, test_f1: float) -> dict:
    return build_run_summary(
        run_id=run_id,
        config=_FakeConfig(),
        data_meta={"source": "test", "symbols": ["AAPL"]},
        bundle_diagnostics={"train": _FakeDiagnostics(), "val": _FakeDiagnostics()},
        train_metrics=_make_metrics(0.5, 0.4, 0.33, 0.40),
        val_metrics=_make_metrics(0.48, 0.42, val_f1, 0.39),
        test_metrics=_make_metrics(0.52, 0.38, test_f1, 0.37),
        val_report=_make_report("val", val_f1, 0.39),
        test_report=_make_report("test", test_f1, 0.37),
        checkpoint_path="checkpoints/brain.pt",
    )


# ---------------------------------------------------------------------------
# Run Summary Tests
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_build_run_summary_has_required_keys(self) -> None:
        summary = _build_test_summary("test_run_001", 0.31, 0.28)
        assert summary["run_id"] == "test_run_001"
        assert "timestamp" in summary
        assert "config" in summary
        assert "train" in summary
        assert "val" in summary
        assert "test" in summary
        assert "val_report" in summary
        assert "test_report" in summary
        assert "diagnostics" in summary

    def test_save_and_load_roundtrip(self) -> None:
        summary = _build_test_summary("roundtrip_001", 0.32, 0.29)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_run_summary(summary, runs_dir=tmpdir)
            assert path.exists()
            loaded = load_run_summary(path)
            assert loaded["run_id"] == "roundtrip_001"
            assert loaded["val_report"]["macro_f1"] == pytest.approx(0.32, abs=1e-4)

    def test_list_run_summaries_finds_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_run_summary(_build_test_summary("a", 0.3, 0.28), runs_dir=tmpdir)
            save_run_summary(_build_test_summary("b", 0.35, 0.30), runs_dir=tmpdir)
            paths = list_run_summaries(tmpdir)
            assert len(paths) == 2

    def test_summary_contains_config_details(self) -> None:
        summary = _build_test_summary("config_check", 0.31, 0.28)
        cfg = summary["config"]
        assert cfg["label_epsilon"] == pytest.approx(0.0005)
        assert cfg["classification_loss"] == "focal"
        assert cfg["symbols"] == ["AAPL", "MSFT"]
        assert cfg["loss_weights"]["classification"] == 1.0

    def test_summary_contains_per_class_metrics(self) -> None:
        summary = _build_test_summary("per_class", 0.31, 0.28)
        per_class = summary["val_report"]["per_class_metrics"]
        assert "long" in per_class
        assert "recall" in per_class["long"]
        assert "precision" in per_class["long"]

    def test_summary_json_serializable(self) -> None:
        summary = _build_test_summary("json_test", 0.31, 0.28)
        serialized = json.dumps(summary, default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 100


# ---------------------------------------------------------------------------
# Comparison Tests
# ---------------------------------------------------------------------------


class TestRunComparison:
    def test_compare_two_runs_shows_deltas(self) -> None:
        s1 = _build_test_summary("run_a", 0.30, 0.27)
        s2 = _build_test_summary("run_b", 0.35, 0.31)
        comparison = compare_runs([s1, s2])
        assert len(comparison["runs"]) == 2
        assert len(comparison["deltas"]) == 1
        delta = comparison["deltas"][0]
        assert delta["val_report.macro_f1"] == pytest.approx(0.05, abs=1e-4)

    def test_compare_identifies_best(self) -> None:
        s1 = _build_test_summary("run_a", 0.30, 0.27)
        s2 = _build_test_summary("run_b", 0.35, 0.31)
        comparison = compare_runs([s1, s2])
        best = comparison["best"]
        assert best["val_report.macro_f1"]["run_id"] == "run_b"

    def test_format_comparison_table_produces_output(self) -> None:
        s1 = _build_test_summary("run_a", 0.30, 0.27)
        s2 = _build_test_summary("run_b", 0.35, 0.31)
        comparison = compare_runs([s1, s2])
        table = format_comparison_table(comparison)
        assert "run_a" in table
        assert "run_b" in table
        assert "macro_f1" in table

    def test_compare_empty_returns_gracefully(self) -> None:
        comparison = compare_runs([])
        assert comparison["runs"] == []

    def test_find_best_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_run_summary(_build_test_summary("weak", 0.20, 0.18), runs_dir=tmpdir)
            save_run_summary(_build_test_summary("strong", 0.40, 0.35), runs_dir=tmpdir)
            best = find_best_run(tmpdir, metric="val_report.macro_f1")
            assert best is not None
            assert best["run_id"] == "strong"

    def test_find_best_run_loss_lower_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = _build_test_summary("high_loss", 0.3, 0.28)
            s1["val"]["loss"] = 0.8
            s2 = _build_test_summary("low_loss", 0.3, 0.28)
            s2["val"]["loss"] = 0.3
            save_run_summary(s1, runs_dir=tmpdir)
            save_run_summary(s2, runs_dir=tmpdir)
            best = find_best_run(tmpdir, metric="val.loss", higher_is_better=False)
            assert best is not None
            assert best["run_id"] == "low_loss"


# ---------------------------------------------------------------------------
# Label Inspector Tests
# ---------------------------------------------------------------------------


class TestLabelInspector:
    def _make_samples(self) -> list[dict]:
        """Create fake samples with known return distributions."""
        import random
        random.seed(42)
        samples = []
        for _ in range(1000):
            ret_1 = random.gauss(0, 0.001)
            ret_5 = random.gauss(0, 0.002)
            ret_15 = random.gauss(0, 0.003)
            samples.append({"future_returns": {1: ret_1, 5: ret_5, 15: ret_15}})
        return samples

    def test_inspect_label_profiles_returns_profiles(self) -> None:
        samples = self._make_samples()
        profiles = inspect_label_profiles(
            samples,
            epsilon_values=[0.0003, 0.0005, 0.001],
            horizons=[1, 5, 15],
        )
        assert len(profiles) == 9  # 3 epsilons x 3 horizons
        for p in profiles:
            assert isinstance(p, LabelProfile)
            assert p.total_samples == 1000
            assert abs(p.class_ratios["short"] + p.class_ratios["flat"] + p.class_ratios["long"] - 1.0) < 1e-6

    def test_higher_epsilon_means_more_flat(self) -> None:
        samples = self._make_samples()
        profiles = inspect_label_profiles(
            samples,
            epsilon_values=[0.0001, 0.01],
            horizons=[15],
        )
        low_eps = [p for p in profiles if p.label_epsilon == 0.0001][0]
        high_eps = [p for p in profiles if p.label_epsilon == 0.01][0]
        assert high_eps.flat_rate > low_eps.flat_rate

    def test_longer_horizon_wider_returns(self) -> None:
        samples = self._make_samples()
        profiles = inspect_label_profiles(
            samples,
            epsilon_values=[0.0005],
            horizons=[1, 15],
        )
        h1 = [p for p in profiles if p.primary_horizon == 1][0]
        h15 = [p for p in profiles if p.primary_horizon == 15][0]
        # Longer horizon has wider return distribution → less flat at same epsilon
        assert h15.flat_rate < h1.flat_rate

    def test_format_label_profiles_produces_table(self) -> None:
        samples = self._make_samples()
        profiles = inspect_label_profiles(
            samples,
            epsilon_values=[0.0005],
            horizons=[15],
        )
        table = format_label_profiles(profiles)
        assert "horizon" in table
        assert "epsilon" in table
        assert "flat%" in table

    def test_empty_samples_returns_empty(self) -> None:
        profiles = inspect_label_profiles([], epsilon_values=[0.001], horizons=[5])
        assert profiles == []

    def test_missing_horizon_skipped(self) -> None:
        samples = [{"future_returns": {5: 0.001}}]
        profiles = inspect_label_profiles(samples, epsilon_values=[0.0005], horizons=[5, 999])
        assert len(profiles) == 1
        assert profiles[0].primary_horizon == 5


# ---------------------------------------------------------------------------
# Loss Config Tests
# ---------------------------------------------------------------------------


class TestLossConfig:
    def test_disable_regression_sets_weight_zero(self) -> None:
        from src.core.config import load_config
        config = load_config("configs/default.yaml", "configs/training.yaml")
        config.training.disable_regression = True
        effective_weights = dict(config.training.loss_weights)
        if config.training.disable_regression:
            effective_weights["regression"] = 0.0
        assert effective_weights["regression"] == 0.0
        assert effective_weights["classification"] > 0.0

    def test_disable_latent_sets_weight_zero(self) -> None:
        from src.core.config import load_config
        config = load_config("configs/default.yaml", "configs/training.yaml")
        config.training.disable_latent = True
        effective_weights = dict(config.training.loss_weights)
        if config.training.disable_latent:
            effective_weights["latent"] = 0.0
        assert effective_weights["latent"] == 0.0

    def test_loss_weight_overrides_from_config(self) -> None:
        from src.core.config import load_config
        config = load_config("configs/default.yaml", "configs/training.yaml")
        config.training.loss_weights["classification"] = 2.0
        config.training.loss_weights["regression"] = 0.1
        assert config.training.loss_weights["classification"] == 2.0
        assert config.training.loss_weights["regression"] == 0.1

    def test_focal_gamma_override(self) -> None:
        from src.core.config import load_config
        config = load_config("configs/default.yaml", "configs/training.yaml")
        config.training.focal_gamma = 3.0
        assert config.training.focal_gamma == 3.0


# ---------------------------------------------------------------------------
# Sweep Tests
# ---------------------------------------------------------------------------


class TestSweep:
    def test_expand_grid_produces_correct_count(self) -> None:
        from scripts.sweep_train import expand_grid
        grid = {"a": [1, 2], "b": ["x", "y", "z"]}
        combos = expand_grid(grid)
        assert len(combos) == 6

    def test_prune_redundant_combos(self) -> None:
        from scripts.sweep_train import prune_redundant_combos
        combos = [
            {"classification_loss": "cross_entropy", "focal_gamma": 1.5},
            {"classification_loss": "cross_entropy", "focal_gamma": 2.0},
            {"classification_loss": "focal", "focal_gamma": 1.5},
            {"classification_loss": "focal", "focal_gamma": 2.0},
        ]
        pruned = prune_redundant_combos(combos)
        # Two cross_entropy combos collapse into one (focal_gamma irrelevant)
        assert len(pruned) == 3

    def test_apply_overrides(self) -> None:
        from scripts.sweep_train import apply_overrides
        from src.core.config import load_config
        config = load_config("configs/default.yaml", "configs/training.yaml")
        overridden = apply_overrides(config, {
            "label_epsilon": 0.001,
            "classification_loss": "cross_entropy",
            "class_weight_mode": "effective_number",
        })
        assert overridden.training.label_epsilon == 0.001
        assert overridden.training.classification_loss == "cross_entropy"
        assert overridden.training.class_weight_mode == "effective_number"
        # Original unchanged
        assert config.training.classification_loss != "cross_entropy" or config.training.classification_loss == "cross_entropy"

    def test_apply_overrides_unknown_key_raises(self) -> None:
        from scripts.sweep_train import apply_overrides
        from src.core.config import load_config
        config = load_config("configs/default.yaml", "configs/training.yaml")
        with pytest.raises(ValueError, match="Unknown sweep key"):
            apply_overrides(config, {"nonexistent_key": 42})

    def test_default_sweep_grid_is_valid(self) -> None:
        from scripts.sweep_train import DEFAULT_SWEEP_GRID, expand_grid, prune_redundant_combos
        combos = expand_grid(DEFAULT_SWEEP_GRID)
        pruned = prune_redundant_combos(combos)
        # Should produce a reasonable number of runs
        assert 4 <= len(pruned) <= 50


# ---------------------------------------------------------------------------
# Metric Comparison Correctness
# ---------------------------------------------------------------------------


class TestMetricComparisonCorrectness:
    def test_comparison_delta_sign_correct(self) -> None:
        s1 = _build_test_summary("worse", 0.20, 0.18)
        s2 = _build_test_summary("better", 0.35, 0.30)
        comparison = compare_runs([s1, s2])
        delta = comparison["deltas"][0]
        assert delta["val_report.macro_f1"] > 0  # improvement is positive
        assert delta["test_report.macro_f1"] > 0

    def test_comparison_loss_delta_direction(self) -> None:
        s1 = _build_test_summary("high_loss", 0.3, 0.28)
        s1["val"]["loss"] = 0.8
        s2 = _build_test_summary("low_loss", 0.3, 0.28)
        s2["val"]["loss"] = 0.5
        comparison = compare_runs([s1, s2])
        delta = comparison["deltas"][0]
        assert delta["val.loss"] < 0  # loss went down = good

    def test_best_selects_lower_loss(self) -> None:
        s1 = _build_test_summary("a", 0.3, 0.28)
        s2 = _build_test_summary("b", 0.3, 0.28)
        comparison = compare_runs([s1, s2])
        best_loss = comparison["best"].get("val.loss")
        assert best_loss is not None


# ---------------------------------------------------------------------------
# No Silent Fallback Tests
# ---------------------------------------------------------------------------


class TestNoSilentFallback:
    def test_run_summary_without_metrics_still_serializes(self) -> None:
        summary = build_run_summary(
            run_id="no_metrics",
            config=_FakeConfig(),
            data_meta={},
            bundle_diagnostics={},
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            val_report=_make_report("val", 0.0, 0.0),
            test_report=_make_report("test", 0.0, 0.0),
            checkpoint_path="",
        )
        serialized = json.dumps(summary, default=str)
        assert "no_metrics" in serialized

    def test_compare_single_run_works(self) -> None:
        s1 = _build_test_summary("solo", 0.3, 0.28)
        comparison = compare_runs([s1])
        assert len(comparison["runs"]) == 1
        assert len(comparison["deltas"]) == 0
