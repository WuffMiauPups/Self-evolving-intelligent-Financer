"""Tests for state + future-path training design.

Tests cover:
- Primary + auxiliary horizon target generation
- Path label generation correctness
- Expanded regression targets (per-horizon returns, MFE/MAE/volatility/drawdown)
- Loss configuration behavior with new weights
- Path label classification loss
- Future return and risk path regression losses
- Diagnostics reporting path label distributions
- Run summary / checkpoint metadata completeness
- Config backwards compatibility
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import torch
import pytest

pytestmark = pytest.mark.slow

from src.core.config import load_config
from src.data.dataset import TrainingSample, collate_training_samples
from src.data.schemas import MarketBar
from src.memory.outcomes import compute_outcome_summary
from src.training.class_balance import compute_target_diagnostics
from src.training.losses import (
    direction_classification_loss,
    future_return_regression_loss,
    risk_path_regression_loss,
)
from src.training.metrics import BatchMetrics
from src.training.targets import PATH_LABEL_VOCABULARY, build_training_targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(count: int = 100, base_price: float = 100.0) -> list[MarketBar]:
    """Generate simple bars with a known pattern: slight uptrend with noise."""
    import random
    random.seed(42)
    bars = []
    price = base_price
    for i in range(count):
        noise = random.gauss(0, 0.3)
        price = price + 0.01 + noise  # slight uptrend
        price = max(price, 1.0)
        ts = datetime(2026, 1, 1, 10, 0, tzinfo=UTC) + timedelta(minutes=i)
        bars.append(MarketBar("TEST", ts, price + 0.5, price + 1.0, price - 0.5, price, 1000.0))
    return bars


def _make_sample(
    label: int = 1,
    direction_name: str = "flat",
    path_name: str = "chop",
    path_label: int = 2,
    primary_return: float = 0.0,
    future_returns: dict[int, float] | None = None,
    regression_targets: dict[str, float] | None = None,
    symbol: str = "TEST",
) -> TrainingSample:
    return TrainingSample(
        features=torch.zeros(4),
        branch_inputs={"trend": torch.zeros(2)},
        regime_input=torch.zeros(2),
        temporal_tensor=torch.zeros(2, 4),
        label=torch.tensor(label, dtype=torch.long),
        symbol=symbol,
        direction_name=direction_name,
        path_label=torch.tensor(path_label, dtype=torch.long),
        path_name=path_name,
        regression_targets=regression_targets or {"primary_return": primary_return},
        future_returns=future_returns or {15: primary_return},
    )


# ---------------------------------------------------------------------------
# Primary + Auxiliary Horizon Target Generation
# ---------------------------------------------------------------------------


class TestHorizonTargets:
    def test_primary_horizon_respected(self) -> None:
        bars = _make_bars(100)
        outcome = compute_outcome_summary(
            bars, 10, horizons=[5, 15, 30], path_label_rules={
                "breakout_return_threshold": 0.003,
                "mean_reversion_return_threshold": 0.0015,
                "flat_return_threshold": 0.00075,
                "time_to_move_threshold": 0.002,
            }
        )
        targets_h15 = build_training_targets(
            outcome, label_epsilon=0.001, regression_targets=["primary_return"],
            primary_horizon=15,
        )
        targets_h30 = build_training_targets(
            outcome, label_epsilon=0.001, regression_targets=["primary_return"],
            primary_horizon=30,
        )
        # Primary return should differ when using different horizons
        assert targets_h15.primary_return == pytest.approx(outcome.future_returns[15], abs=1e-8)
        assert targets_h30.primary_return == pytest.approx(outcome.future_returns[30], abs=1e-8)

    def test_auxiliary_horizons_add_return_targets(self) -> None:
        bars = _make_bars(100)
        outcome = compute_outcome_summary(
            bars, 10, horizons=[5, 15, 30], path_label_rules={
                "breakout_return_threshold": 0.003,
                "mean_reversion_return_threshold": 0.0015,
                "flat_return_threshold": 0.00075,
                "time_to_move_threshold": 0.002,
            }
        )
        targets = build_training_targets(
            outcome, label_epsilon=0.001, regression_targets=["primary_return"],
            primary_horizon=15,
            enable_future_return_targets=True,
        )
        assert "return_h5" in targets.regression_targets
        assert "return_h15" in targets.regression_targets
        assert "return_h30" in targets.regression_targets

    def test_risk_path_targets_added(self) -> None:
        bars = _make_bars(100)
        outcome = compute_outcome_summary(
            bars, 10, horizons=[5, 15], path_label_rules={
                "breakout_return_threshold": 0.003,
                "mean_reversion_return_threshold": 0.0015,
                "flat_return_threshold": 0.00075,
                "time_to_move_threshold": 0.002,
            }
        )
        targets = build_training_targets(
            outcome, label_epsilon=0.001, regression_targets=["primary_return"],
            enable_risk_path_targets=True,
        )
        assert "mfe" in targets.regression_targets
        assert "mae" in targets.regression_targets
        assert "volatility" in targets.regression_targets
        assert "path_drawdown" in targets.regression_targets

    def test_all_horizons_property(self) -> None:
        config = load_config("configs/default.yaml", "configs/training.yaml")
        config.training.forward_horizons = [1, 5, 15]
        config.training.auxiliary_horizons = [30, 60]
        assert config.training.all_horizons == [1, 5, 15, 30, 60]

    def test_effective_primary_horizon_default(self) -> None:
        config = load_config("configs/default.yaml")
        config.training.forward_horizons = [1, 5, 15]
        config.training.primary_direction_horizon = None
        assert config.training.effective_primary_horizon == 15

    def test_effective_primary_horizon_explicit(self) -> None:
        config = load_config("configs/default.yaml")
        config.training.primary_direction_horizon = 30
        assert config.training.effective_primary_horizon == 30


# ---------------------------------------------------------------------------
# Path Label Generation
# ---------------------------------------------------------------------------


class TestPathLabels:
    def test_path_label_vocabulary(self) -> None:
        assert len(PATH_LABEL_VOCABULARY) == 5
        assert "breakout_up" in PATH_LABEL_VOCABULARY
        assert "breakout_down" in PATH_LABEL_VOCABULARY
        assert "chop" in PATH_LABEL_VOCABULARY
        assert "mean_reversion_up" in PATH_LABEL_VOCABULARY
        assert "mean_reversion_down" in PATH_LABEL_VOCABULARY

    def test_path_label_always_populated(self) -> None:
        bars = _make_bars(50)
        outcome = compute_outcome_summary(
            bars, 5, horizons=[5, 15], path_label_rules={
                "breakout_return_threshold": 0.003,
                "mean_reversion_return_threshold": 0.0015,
                "flat_return_threshold": 0.00075,
                "time_to_move_threshold": 0.002,
            }
        )
        targets = build_training_targets(
            outcome, label_epsilon=0.001, regression_targets=["primary_return"],
        )
        assert targets.path_name in PATH_LABEL_VOCABULARY
        assert 0 <= targets.path_label < len(PATH_LABEL_VOCABULARY)

    def test_path_label_deterministic(self) -> None:
        bars = _make_bars(50)
        outcome = compute_outcome_summary(
            bars, 5, horizons=[5, 15], path_label_rules={
                "breakout_return_threshold": 0.003,
                "mean_reversion_return_threshold": 0.0015,
                "flat_return_threshold": 0.00075,
                "time_to_move_threshold": 0.002,
            }
        )
        t1 = build_training_targets(outcome, label_epsilon=0.001, regression_targets=[])
        t2 = build_training_targets(outcome, label_epsilon=0.001, regression_targets=[])
        assert t1.path_label == t2.path_label
        assert t1.path_name == t2.path_name

    def test_path_label_config_driven(self) -> None:
        bars = _make_bars(50)
        # Tight breakout threshold
        outcome_tight = compute_outcome_summary(
            bars, 5, horizons=[15], path_label_rules={
                "breakout_return_threshold": 0.0001,
                "mean_reversion_return_threshold": 0.0001,
                "flat_return_threshold": 0.00001,
                "time_to_move_threshold": 0.00001,
            }
        )
        # Loose breakout threshold
        outcome_loose = compute_outcome_summary(
            bars, 5, horizons=[15], path_label_rules={
                "breakout_return_threshold": 10.0,
                "mean_reversion_return_threshold": 10.0,
                "flat_return_threshold": 10.0,
                "time_to_move_threshold": 10.0,
            }
        )
        t_tight = build_training_targets(outcome_tight, label_epsilon=0.001, regression_targets=[])
        t_loose = build_training_targets(outcome_loose, label_epsilon=0.001, regression_targets=[])
        # With very loose thresholds, everything should be chop
        assert t_loose.path_name == "chop"


# ---------------------------------------------------------------------------
# Loss Configuration
# ---------------------------------------------------------------------------


class TestLossConfig:
    def test_new_loss_weights_in_config(self) -> None:
        config = load_config("configs/default.yaml", "configs/training.yaml")
        lw = config.training.loss_weights
        assert "classification" in lw
        assert "future_return" in lw
        assert "risk_path" in lw
        assert "regression" in lw
        assert "latent" in lw
        # path_label was killed when the head was removed (val acc < majority,
        # 7.7pt train→val overfit). Validator pops it silently if present.
        assert "path_label" not in lw

    def test_old_3key_config_migrated(self) -> None:
        """Old configs with only {classification, regression, latent} should still work."""
        config = load_config("configs/default.yaml")
        # Force old-style
        config.training.loss_weights = {
            "classification": 1.0, "regression": 0.35, "latent": 0.10
        }
        # Trigger validation
        config.training.validate_training_config()
        assert "future_return" in config.training.loss_weights
        assert "risk_path" in config.training.loss_weights
        assert "path_label" not in config.training.loss_weights

    def test_legacy_config_with_path_label_pops_silently(self) -> None:
        """Old configs that still set path_label should not raise."""
        config = load_config("configs/default.yaml")
        config.training.loss_weights = {
            "classification": 1.0, "regression": 0.35, "latent": 0.10,
            "path_label": 0.30, "future_return": 0.20, "risk_path": 0.15,
        }
        config.training.validate_training_config()
        assert "path_label" not in config.training.loss_weights


# ---------------------------------------------------------------------------
# Future Return & Risk Path Losses
# ---------------------------------------------------------------------------


class TestFuturePathLosses:
    def _make_output(self) -> object:
        class FakeOutput:
            action_logits = torch.randn(4, 3)
            target_fraction = torch.randn(4, 1)
            value = torch.randn(4, 1)
            confidence = torch.randn(4, 1)
            fused_representation = torch.randn(4, 16)
        return FakeOutput()

    def test_future_return_regression_loss(self) -> None:
        output = self._make_output()
        targets = {
            "return_h5": torch.randn(4),
            "return_h15": torch.randn(4),
            "primary_return": torch.randn(4),  # should be ignored
        }
        loss, components = future_return_regression_loss(output, targets)
        assert float(loss.item()) > 0.0
        assert "return_h5" in components
        assert "return_h15" in components
        assert "primary_return" not in components

    def test_risk_path_regression_loss(self) -> None:
        output = self._make_output()
        targets = {
            "mfe": torch.randn(4),
            "mae": torch.randn(4),
            "volatility": torch.randn(4),
            "path_drawdown": torch.randn(4),
        }
        loss, components = risk_path_regression_loss(output, targets)
        assert float(loss.item()) > 0.0
        assert len(components) >= 3  # at least mfe/mae/vol + total

    def test_empty_targets_return_zero(self) -> None:
        output = self._make_output()
        loss, components = future_return_regression_loss(output, {})
        assert float(loss.item()) == 0.0
        loss2, components2 = risk_path_regression_loss(output, {})
        assert float(loss2.item()) == 0.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestPathDiagnostics:
    def test_diagnostics_include_path_label_counts(self) -> None:
        samples = [
            _make_sample(path_name="breakout_up", path_label=4),
            _make_sample(path_name="breakout_up", path_label=4),
            _make_sample(path_name="chop", path_label=2),
            _make_sample(path_name="mean_reversion_down", path_label=1),
        ]
        diagnostics = compute_target_diagnostics(
            samples, split="train", minority_class_floor_warning=0.05,
        )
        assert diagnostics.path_label_counts["breakout_up"] == 2
        assert diagnostics.path_label_counts["chop"] == 1
        assert diagnostics.path_label_ratios["breakout_up"] == pytest.approx(0.5)

    def test_diagnostics_include_per_horizon_returns(self) -> None:
        samples = [
            _make_sample(future_returns={5: 0.001, 15: 0.003}),
            _make_sample(future_returns={5: -0.002, 15: -0.001}),
        ]
        diagnostics = compute_target_diagnostics(
            samples, split="train", minority_class_floor_warning=0.05,
        )
        assert "h5" in diagnostics.per_horizon_return_quantiles
        assert "h15" in diagnostics.per_horizon_return_quantiles
        assert "p50" in diagnostics.per_horizon_return_quantiles["h5"]

    def test_diagnostics_include_risk_path_summaries(self) -> None:
        samples = [
            _make_sample(regression_targets={
                "primary_return": 0.002, "mfe": 0.005, "mae": -0.003,
                "volatility": 0.01, "path_drawdown": -0.004,
            }),
            _make_sample(regression_targets={
                "primary_return": -0.001, "mfe": 0.003, "mae": -0.005,
                "volatility": 0.015, "path_drawdown": -0.006,
            }),
        ]
        diagnostics = compute_target_diagnostics(
            samples, split="train", minority_class_floor_warning=0.05,
        )
        assert "mfe" in diagnostics.risk_path_summaries
        assert "mae" in diagnostics.risk_path_summaries


# ---------------------------------------------------------------------------
# Collate Compatibility
# ---------------------------------------------------------------------------


class TestCollateCompatibility:
    def test_collate_handles_path_label(self) -> None:
        samples = [
            _make_sample(path_label=0, path_name="breakout_down"),
            _make_sample(path_label=4, path_name="breakout_up"),
        ]
        batch = collate_training_samples(samples)
        assert "path_label" in batch
        assert batch["path_label"].shape == (2,)
        assert int(batch["path_label"][0].item()) == 0
        assert int(batch["path_label"][1].item()) == 4

    def test_collate_handles_expanded_regression(self) -> None:
        samples = [
            _make_sample(regression_targets={
                "primary_return": 0.001, "return_h5": 0.0005,
                "mfe": 0.003, "mae": -0.002,
            }),
            _make_sample(regression_targets={
                "primary_return": -0.001, "return_h5": -0.0003,
                "mfe": 0.002, "mae": -0.004,
            }),
        ]
        batch = collate_training_samples(samples)
        reg = batch["regression_targets"]
        assert "return_h5" in reg
        assert "mfe" in reg
        assert "mae" in reg
        assert reg["return_h5"].shape == (2,)


# ---------------------------------------------------------------------------
# Run Summary Metadata
# ---------------------------------------------------------------------------


class TestRunSummaryMetadata:
    def test_summary_includes_state_path_config(self) -> None:
        from src.training.run_summary import build_run_summary

        class FakeConfig:
            class training:
                label_epsilon = 0.001
                forward_horizons = [1, 5, 15]
                classification_loss = "focal"
                focal_gamma = 1.5
                label_smoothing = 0.02
                class_weight_mode = "balanced"
                use_weighted_sampler = True
                loss_weights = {"classification": 1.0, "regression": 0.25, "latent": 0.05,
                                "path_label": 0.3, "future_return": 0.2, "risk_path": 0.15}
                multitask_enabled = True
                contrastive_enabled = False
                batch_size = 64
                epochs = 3
                learning_rate = 0.001
                sequence_length = 64
                disable_regression = False
                disable_latent = False
                runs_dir = "runs"
                primary_direction_horizon = 15
                auxiliary_horizons = [30, 60]
                all_horizons = [1, 5, 15, 30, 60]
                effective_primary_horizon = 15
                enable_future_return_targets = True
                enable_risk_path_targets = True
                path_label_rules = {"breakout_return_threshold": 0.003}
            class data:
                symbols = ["AAPL"]
                timeframe = "1Min"
                start_date = "2022-01-01"
                end_date = None

        class FakeDiag:
            total_samples = 100
            class_counts = {"short": 30, "flat": 40, "long": 30}
            class_ratios = {"short": 0.3, "flat": 0.4, "long": 0.3}
            flat_rate = 0.4
            near_flat_rate = 0.1
            near_boundary_rate = 0.05
            primary_return_quantiles = {"p50": 0.0}
            path_label_counts = {"chop": 40, "breakout_up": 20}
            path_label_ratios = {"chop": 0.4, "breakout_up": 0.2}
            per_horizon_return_quantiles = {"h15": {"p50": 0.001}}
            warnings = []

        report = {"accuracy": 0.4, "balanced_accuracy": 0.39, "macro_f1": 0.31,
                  "weighted_f1": 0.32, "mcc": 0.1, "long_short_only_accuracy": 0.5,
                  "prediction_counts": {}, "target_counts": {},
                  "prediction_ratios": {}, "target_ratios": {},
                  "per_class_metrics": {}, "confusion_matrix": [],
                  "long_short_confusion": {}, "warnings": []}

        summary = build_run_summary(
            run_id="test",
            config=FakeConfig(),
            data_meta={},
            bundle_diagnostics={"train": FakeDiag()},
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            val_report=report,
            test_report=report,
            checkpoint_path="test.pt",
        )
        cfg = summary["config"]
        assert cfg["primary_direction_horizon"] == 15
        assert cfg["auxiliary_horizons"] == [30, 60]
        assert cfg["enable_future_return_targets"] is True
        assert cfg["enable_risk_path_targets"] is True

        diag = summary["diagnostics"]["train"]
        assert diag["path_label_counts"]["chop"] == 40
        assert "h15" in diag["per_horizon_return_quantiles"]


# ---------------------------------------------------------------------------
# BatchMetrics new fields
# ---------------------------------------------------------------------------


class TestBatchMetricsNewFields:
    def test_batch_metrics_has_path_fields(self) -> None:
        m = BatchMetrics(
            loss=0.5, accuracy=0.4,
            future_return_loss=0.1, risk_path_loss=0.05,
        )
        assert m.future_return_loss == 0.1
        assert m.risk_path_loss == 0.05

    def test_batch_metrics_defaults_zero(self) -> None:
        m = BatchMetrics(loss=0.5, accuracy=0.4)
        assert m.future_return_loss == 0.0
        assert m.risk_path_loss == 0.0


# ---------------------------------------------------------------------------
# End-to-end: config + targets + collate round-trip
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_target_generation_with_state_path_config(self) -> None:
        config = load_config("configs/default.yaml", "configs/training.yaml")
        bars = _make_bars(150)
        all_horizons = config.training.all_horizons
        max_horizon = max(all_horizons)
        outcome = compute_outcome_summary(
            bars, 20,
            horizons=all_horizons,
            path_label_rules=config.training.path_label_rules,
        )
        targets = build_training_targets(
            outcome,
            label_epsilon=config.training.label_epsilon,
            regression_targets=config.training.regression_targets,
            primary_horizon=config.training.effective_primary_horizon,
            enable_future_return_targets=config.training.enable_future_return_targets,
            enable_risk_path_targets=config.training.enable_risk_path_targets,
        )
        # Direction label
        assert targets.direction_name in ("short", "flat", "long")
        # Path label
        assert targets.path_name in PATH_LABEL_VOCABULARY
        # Future returns at all horizons
        for h in all_horizons:
            if h <= 130:  # within range
                assert h in targets.future_returns
        # Regression targets include expanded ones
        if config.training.enable_future_return_targets:
            return_keys = [k for k in targets.regression_targets if k.startswith("return_h")]
            assert len(return_keys) > 0
        if config.training.enable_risk_path_targets:
            assert "mfe" in targets.regression_targets
            assert "mae" in targets.regression_targets
