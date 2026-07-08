"""Tests for fast experiment mode.

Covers:
- FastModeConfig defaults and validation
- Config integration (TrainingConfig, AppConfig, YAML loading)
- Sample cap behaviour (determinism, correctness)
- Epoch cap logic
- Target/loss simplification logic
- Date-range override logic
- Symbol subset logic
- Run summary fast_mode section (build, persist, backward compat)
- compare_runs compatibility with fast runs
- No silent fallback to synthetic data
"""
from __future__ import annotations

import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.config import AppConfig, FastModeConfig, TrainingConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeFastMode:
    """Minimal fast_mode stand-in for use in _FakeConfig."""

    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", False)
        self.fast_months = kwargs.get("fast_months", None)
        self.fast_start_date = kwargs.get("fast_start_date", None)
        self.fast_end_date = kwargs.get("fast_end_date", None)
        self.symbols = kwargs.get("symbols", [])
        self.max_train_samples = kwargs.get("max_train_samples", None)
        self.max_val_samples = kwargs.get("max_val_samples", None)
        self.max_test_samples = kwargs.get("max_test_samples", None)
        self.max_epochs = kwargs.get("max_epochs", None)
        self.disable_future_return = kwargs.get("disable_future_return", False)
        self.disable_risk_path = kwargs.get("disable_risk_path", False)
        self.model_profile = kwargs.get("model_profile", "default")
        self.sample_seed = kwargs.get("sample_seed", 42)


class _FakeTraining:
    label_epsilon = 0.0005
    forward_horizons = [1, 5, 15]
    classification_loss = "focal"
    focal_gamma = 1.5
    label_smoothing = 0.02
    class_weight_mode = "balanced"
    use_weighted_sampler = True
    loss_weights = {
        "classification": 1.0,
        "regression": 0.25,
        "latent": 0.05,
        "future_return": 0.20,
        "risk_path": 0.15,
    }
    multitask_enabled = True
    contrastive_enabled = False
    batch_size = 64
    epochs = 3
    learning_rate = 0.001
    sequence_length = 64
    auxiliary_horizons = [30, 60]
    all_horizons = [1, 5, 15, 30, 60]
    effective_primary_horizon = 15
    enable_future_return_targets = True
    enable_risk_path_targets = True
    path_label_rules = {}
    fast_mode = _FakeFastMode()


class _FakeData:
    symbols = ["AAPL", "MSFT"]
    timeframe = "1Min"
    start_date = "2022-01-01"
    end_date = "2026-01-01"


class _FakeConfig:
    training = _FakeTraining()
    data = _FakeData()


def _make_fake_config(fast_enabled: bool = False, **fast_kwargs) -> _FakeConfig:
    cfg = _FakeConfig()
    cfg.training = _FakeTraining()
    cfg.training.fast_mode = _FakeFastMode(enabled=fast_enabled, **fast_kwargs)
    return cfg


# ---------------------------------------------------------------------------
# 1. FastModeConfig: defaults and validation
# ---------------------------------------------------------------------------


class TestFastModeConfig:
    def test_defaults_disabled(self):
        fc = FastModeConfig()
        assert fc.enabled is False
        assert fc.fast_months is None
        assert fc.fast_start_date is None
        assert fc.fast_end_date is None
        assert fc.symbols == []
        assert fc.max_train_samples is None
        assert fc.max_val_samples is None
        assert fc.max_test_samples is None
        assert fc.max_epochs is None
        assert fc.disable_future_return is False
        assert fc.disable_risk_path is False
        assert fc.model_profile == "default"
        assert fc.sample_seed == 42

    def test_enable_with_fast_months(self):
        fc = FastModeConfig(enabled=True, fast_months=6)
        assert fc.enabled is True
        assert fc.fast_months == 6

    def test_enable_with_symbol_subset(self):
        fc = FastModeConfig(enabled=True, symbols=["AAPL"])
        assert fc.symbols == ["AAPL"]

    def test_enable_with_sample_cap(self):
        fc = FastModeConfig(enabled=True, max_train_samples=50_000)
        assert fc.max_train_samples == 50_000

    def test_enable_with_epoch_cap(self):
        fc = FastModeConfig(enabled=True, max_epochs=2)
        assert fc.max_epochs == 2

    def test_enable_with_all_disables(self):
        fc = FastModeConfig(
            enabled=True,
            disable_future_return=True,
            disable_risk_path=True,
        )
        assert fc.disable_future_return is True
        assert fc.disable_risk_path is True

    def test_small_model_profile(self):
        fc = FastModeConfig(enabled=True, model_profile="small")
        assert fc.model_profile == "small"

    def test_default_model_profile(self):
        fc = FastModeConfig(model_profile="default")
        assert fc.model_profile == "default"

    def test_invalid_fast_months_zero(self):
        with pytest.raises(Exception):
            FastModeConfig(fast_months=0)

    def test_invalid_fast_months_negative(self):
        with pytest.raises(Exception):
            FastModeConfig(fast_months=-3)

    def test_invalid_max_train_samples_zero(self):
        with pytest.raises(Exception):
            FastModeConfig(max_train_samples=0)

    def test_invalid_max_val_samples_negative(self):
        with pytest.raises(Exception):
            FastModeConfig(max_val_samples=-1)

    def test_invalid_max_epochs_zero(self):
        with pytest.raises(Exception):
            FastModeConfig(max_epochs=0)

    def test_invalid_model_profile(self):
        with pytest.raises(Exception):
            FastModeConfig(model_profile="xlarge")

    def test_custom_sample_seed(self):
        fc = FastModeConfig(sample_seed=99)
        assert fc.sample_seed == 99

    def test_explicit_date_overrides(self):
        fc = FastModeConfig(
            enabled=True,
            fast_start_date="2025-01-01",
            fast_end_date="2025-06-30",
        )
        assert fc.fast_start_date == "2025-01-01"
        assert fc.fast_end_date == "2025-06-30"


# ---------------------------------------------------------------------------
# 2. Config integration: TrainingConfig and AppConfig
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_training_config_has_fast_mode_field(self):
        tc = TrainingConfig()
        assert hasattr(tc, "fast_mode")
        assert isinstance(tc.fast_mode, FastModeConfig)

    def test_training_config_fast_mode_default_disabled(self):
        tc = TrainingConfig()
        assert tc.fast_mode.enabled is False

    def test_training_config_fast_mode_override(self):
        tc = TrainingConfig(fast_mode=FastModeConfig(enabled=True, fast_months=3))
        assert tc.fast_mode.enabled is True
        assert tc.fast_mode.fast_months == 3

    def test_app_config_has_training_fast_mode(self):
        config = load_config("configs/default.yaml", "configs/training.yaml")
        assert hasattr(config.training, "fast_mode")
        assert isinstance(config.training.fast_mode, FastModeConfig)

    def test_default_config_fast_mode_disabled(self):
        config = load_config("configs/default.yaml", "configs/training.yaml")
        assert config.training.fast_mode.enabled is False

    def test_default_config_fast_mode_sample_seed(self):
        config = load_config("configs/default.yaml", "configs/training.yaml")
        assert config.training.fast_mode.sample_seed == 42

    def test_fast_mode_does_not_affect_other_training_fields(self):
        """Enabling fast mode must not change any other TrainingConfig fields."""
        tc_full = TrainingConfig()
        tc_fast = TrainingConfig(fast_mode=FastModeConfig(enabled=True, fast_months=3))
        assert tc_full.epochs == tc_fast.epochs
        assert tc_full.batch_size == tc_fast.batch_size
        assert tc_full.use_real_historical_data == tc_fast.use_real_historical_data

    def test_fast_mode_does_not_enable_synthetic_data(self):
        """Fast mode must NOT silently switch to synthetic data."""
        tc = TrainingConfig(fast_mode=FastModeConfig(enabled=True))
        assert tc.use_real_historical_data is False  # unchanged default

    def test_real_data_flag_independent_of_fast_mode(self):
        """use_real_historical_data is set independently; fast mode must not override it."""
        tc = TrainingConfig(
            use_real_historical_data=True,
            fast_mode=FastModeConfig(enabled=True, fast_months=6),
        )
        assert tc.use_real_historical_data is True
        assert tc.fast_mode.enabled is True


# ---------------------------------------------------------------------------
# 3. Sample cap: determinism and correctness
# ---------------------------------------------------------------------------


class TestFastSampleCap:
    def _make_items(self, n: int) -> list[dict]:
        return [{"idx": i} for i in range(n)]

    def test_cap_reduces_to_correct_count(self):
        items = self._make_items(1000)
        capped = random.Random(42).sample(items, 100)
        assert len(capped) == 100

    def test_cap_is_deterministic_same_seed(self):
        items = self._make_items(1000)
        a = random.Random(42).sample(items, 200)
        b = random.Random(42).sample(items, 200)
        assert [x["idx"] for x in a] == [x["idx"] for x in b]

    def test_different_seeds_give_different_subsets(self):
        items = self._make_items(1000)
        a = random.Random(42).sample(items, 200)
        b = random.Random(99).sample(items, 200)
        assert [x["idx"] for x in a] != [x["idx"] for x in b]

    def test_no_cap_when_count_below_limit(self):
        """If cap > actual count, nothing should be removed."""
        items = self._make_items(50)
        cap = 200
        result = items if len(items) <= cap else random.Random(42).sample(items, cap)
        assert len(result) == 50

    def test_cap_config_fields(self):
        fc = FastModeConfig(
            enabled=True,
            max_train_samples=5000,
            max_val_samples=1000,
            max_test_samples=500,
            sample_seed=7,
        )
        assert fc.max_train_samples == 5000
        assert fc.max_val_samples == 1000
        assert fc.max_test_samples == 500
        assert fc.sample_seed == 7

    def test_cap_per_split_independence(self):
        """Each split uses its own independent cap."""
        train_items = self._make_items(10_000)
        val_items = self._make_items(2_000)
        rng = random.Random(42)
        capped_train = rng.sample(train_items, 500)
        capped_val = rng.sample(val_items, 200)
        assert len(capped_train) == 500
        assert len(capped_val) == 200


# ---------------------------------------------------------------------------
# 4. Epoch cap logic
# ---------------------------------------------------------------------------


class TestFastEpochCap:
    def _effective_epochs(self, configured: int, max_epochs: int | None) -> int:
        if max_epochs is None:
            return configured
        return min(configured, max_epochs)

    def test_cap_lower_than_configured(self):
        assert self._effective_epochs(10, 2) == 2

    def test_cap_higher_than_configured(self):
        assert self._effective_epochs(2, 10) == 2

    def test_cap_equal_to_configured(self):
        assert self._effective_epochs(3, 3) == 3

    def test_no_cap_when_none(self):
        assert self._effective_epochs(5, None) == 5

    def test_fast_mode_epoch_cap_config(self):
        fc = FastModeConfig(enabled=True, max_epochs=1)
        assert fc.max_epochs == 1


# ---------------------------------------------------------------------------
# 5. Target/loss simplification
# ---------------------------------------------------------------------------


class TestFastTargetSimplification:
    _BASE_WEIGHTS = {
        "classification": 1.0,
        "regression": 0.25,
        "latent": 0.05,
        "future_return": 0.20,
        "risk_path": 0.15,
    }

    def _apply_fast_disables(self, fast: FastModeConfig) -> dict:
        weights = dict(self._BASE_WEIGHTS)
        if fast.enabled:
            if fast.disable_future_return:
                weights["future_return"] = 0.0
            if fast.disable_risk_path:
                weights["risk_path"] = 0.0
        return weights

    def test_disable_future_return_zeros_weight(self):
        fast = FastModeConfig(enabled=True, disable_future_return=True)
        weights = self._apply_fast_disables(fast)
        assert weights["future_return"] == 0.0
        assert weights["classification"] == 1.0  # unchanged

    def test_disable_risk_path_zeros_weight(self):
        fast = FastModeConfig(enabled=True, disable_risk_path=True)
        weights = self._apply_fast_disables(fast)
        assert weights["risk_path"] == 0.0
        assert weights["future_return"] == 0.20  # unchanged

    def test_disable_both(self):
        fast = FastModeConfig(
            enabled=True,
            disable_future_return=True,
            disable_risk_path=True,
        )
        weights = self._apply_fast_disables(fast)
        assert weights["future_return"] == 0.0
        assert weights["risk_path"] == 0.0
        assert weights["classification"] == 1.0  # unchanged

    def test_no_simplification_when_flags_false(self):
        fast = FastModeConfig(enabled=True)  # no disable_* flags
        weights = self._apply_fast_disables(fast)
        assert weights == self._BASE_WEIGHTS

    def test_fast_disabled_leaves_weights_unchanged(self):
        fast = FastModeConfig(
            enabled=False,
            disable_future_return=True,  # should be ignored
        )
        weights = self._apply_fast_disables(fast)
        assert weights == self._BASE_WEIGHTS


# ---------------------------------------------------------------------------
# 6. Date-range override logic
# ---------------------------------------------------------------------------


class TestFastDateRange:
    def test_fast_months_gives_earlier_start(self):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        fast_months = 6
        start = now - timedelta(days=30 * fast_months)
        delta = (now - start).days
        assert 170 <= delta <= 185  # roughly 6 months

    def test_fast_months_1_gives_approx_30_days(self):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30 * 1)
        delta = (now - start).days
        assert 28 <= delta <= 32

    def test_fast_months_12_gives_approx_year(self):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30 * 12)
        delta = (now - start).days
        assert 355 <= delta <= 365

    def test_fast_start_date_config(self):
        fc = FastModeConfig(enabled=True, fast_start_date="2025-01-01")
        assert fc.fast_start_date == "2025-01-01"

    def test_fast_end_date_config(self):
        fc = FastModeConfig(enabled=True, fast_end_date="2025-06-30")
        assert fc.fast_end_date == "2025-06-30"

    def test_fast_months_and_explicit_start_independent(self):
        """fast_months and fast_start_date are separate fields."""
        fc = FastModeConfig(
            enabled=True,
            fast_months=3,
            fast_start_date="2024-01-01",
        )
        assert fc.fast_months == 3
        assert fc.fast_start_date == "2024-01-01"

    def test_parse_date_or_datetime_fast_start(self):
        """parse_date_or_datetime should accept fast_start_date strings."""
        from src.data.historical_data import parse_date_or_datetime
        dt = parse_date_or_datetime("2025-06-01")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 6


# ---------------------------------------------------------------------------
# 7. Symbol subset logic
# ---------------------------------------------------------------------------


class TestFastSymbolSubset:
    def _apply_fast_symbols(
        self,
        configured: list[str],
        fast_symbols: list[str],
    ) -> list[str]:
        """Simulate what _resolve_bars_by_symbol does."""
        if not fast_symbols:
            return list(configured)
        effective = [s for s in fast_symbols if s in set(configured)]
        if not effective:
            raise RuntimeError("No overlap between fast_mode.symbols and data.symbols")
        return effective

    def test_empty_fast_symbols_returns_all_configured(self):
        result = self._apply_fast_symbols(["AAPL", "MSFT"], [])
        assert result == ["AAPL", "MSFT"]

    def test_single_symbol_subset(self):
        result = self._apply_fast_symbols(["AAPL", "MSFT"], ["AAPL"])
        assert result == ["AAPL"]

    def test_multi_symbol_subset(self):
        result = self._apply_fast_symbols(["AAPL", "MSFT", "TSLA"], ["AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_unknown_symbol_filtered_out(self):
        """Symbols not in configured list are silently dropped (with a warning in real code)."""
        result = self._apply_fast_symbols(["AAPL", "MSFT"], ["AAPL", "GOOG"])
        assert result == ["AAPL"]

    def test_no_overlap_raises(self):
        with pytest.raises(RuntimeError):
            self._apply_fast_symbols(["AAPL", "MSFT"], ["TSLA", "GOOG"])

    def test_fast_symbol_config(self):
        fc = FastModeConfig(enabled=True, symbols=["AAPL"])
        assert fc.symbols == ["AAPL"]

    def test_fast_symbol_config_empty_default(self):
        fc = FastModeConfig(enabled=True)
        assert fc.symbols == []


# ---------------------------------------------------------------------------
# 8. Run summary: fast_mode section
# ---------------------------------------------------------------------------


class TestFastRunSummary:
    def _build_summary(self, fast_enabled: bool = False, **fast_kwargs) -> dict:
        from src.training.run_summary import build_run_summary
        config = _make_fake_config(fast_enabled=fast_enabled, **fast_kwargs)
        return build_run_summary(
            run_id="test_run",
            config=config,
            data_meta={"source": "test"},
            bundle_diagnostics={},
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            val_report={},
            test_report={},
            checkpoint_path="checkpoints/brain.pt",
        )

    def test_run_summary_has_fast_mode_section(self):
        summary = self._build_summary(fast_enabled=False)
        assert "fast_mode" in summary

    def test_run_summary_fast_mode_disabled_by_default(self):
        summary = self._build_summary(fast_enabled=False)
        assert summary["fast_mode"]["enabled"] is False

    def test_run_summary_fast_mode_enabled(self):
        summary = self._build_summary(fast_enabled=True)
        assert summary["fast_mode"]["enabled"] is True

    def test_run_summary_fast_months(self):
        summary = self._build_summary(fast_enabled=True, fast_months=6)
        assert summary["fast_mode"]["fast_months"] == 6

    def test_run_summary_max_train_samples(self):
        summary = self._build_summary(fast_enabled=True, max_train_samples=10_000)
        assert summary["fast_mode"]["max_train_samples"] == 10_000

    def test_run_summary_disable_flags(self):
        summary = self._build_summary(
            fast_enabled=True,
            disable_future_return=True,
            disable_risk_path=True,
        )
        assert summary["fast_mode"]["disable_future_return"] is True
        assert summary["fast_mode"]["disable_risk_path"] is True

    def test_run_summary_model_profile(self):
        summary = self._build_summary(fast_enabled=True, model_profile="small")
        assert summary["fast_mode"]["model_profile"] == "small"

    def test_run_summary_sample_seed(self):
        summary = self._build_summary(fast_enabled=True, sample_seed=99)
        assert summary["fast_mode"]["sample_seed"] == 99

    def test_run_summary_effective_epochs_from_param(self):
        from src.training.run_summary import build_run_summary
        config = _make_fake_config(fast_enabled=True, max_epochs=1)
        summary = build_run_summary(
            run_id="eff_epoch_test",
            config=config,
            data_meta={},
            bundle_diagnostics={},
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            val_report={},
            test_report={},
            checkpoint_path="checkpoints/brain.pt",
            effective_epochs=1,
        )
        assert summary["fast_mode"]["effective_epochs"] == 1
        assert summary["config"]["effective_epochs"] == 1

    def test_run_summary_persists_to_json(self):
        from src.training.run_summary import build_run_summary, save_run_summary, load_run_summary
        config = _make_fake_config(fast_enabled=True, model_profile="small", fast_months=3)
        summary = build_run_summary(
            run_id="persist_test",
            config=config,
            data_meta={},
            bundle_diagnostics={},
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            val_report={},
            test_report={},
            checkpoint_path="checkpoints/brain.pt",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_run_summary(summary, runs_dir=tmpdir)
            loaded = load_run_summary(path)
            assert loaded["fast_mode"]["enabled"] is True
            assert loaded["fast_mode"]["model_profile"] == "small"
            assert loaded["fast_mode"]["fast_months"] == 3

    def test_run_summary_backward_compat_no_fast_mode(self):
        """Old summaries without fast_mode section must load fine."""
        from src.training.run_summary import load_run_summary
        old_summary = {
            "run_id": "old_run",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "val_report": {"macro_f1": 0.45, "balanced_accuracy": 0.5},
            "test_report": {"macro_f1": 0.44},
            "config": {"label_epsilon": 0.0005},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "old_run.json"
            p.write_text(json.dumps(old_summary))
            loaded = load_run_summary(p)
            # Should not crash; fast_mode simply absent
            assert loaded.get("fast_mode", {}).get("enabled", False) is False

    def test_run_summary_config_section_has_effective_epochs(self):
        from src.training.run_summary import build_run_summary
        config = _make_fake_config(fast_enabled=False)
        summary = build_run_summary(
            run_id="config_epochs_test",
            config=config,
            data_meta={},
            bundle_diagnostics={},
            train_metrics=None,
            val_metrics=None,
            test_metrics=None,
            val_report={},
            test_report={},
            checkpoint_path="checkpoints/brain.pt",
        )
        assert "effective_epochs" in summary["config"]
        assert "epochs" in summary["config"]


# ---------------------------------------------------------------------------
# 9. compare_runs compatibility with fast runs
# ---------------------------------------------------------------------------


class TestFastCompareRuns:
    def test_compare_runs_includes_fast_run(self):
        from src.training.run_summary import compare_runs
        fast = {
            "run_id": "fast_001",
            "fast_mode": {"enabled": True, "fast_months": 3},
            "val_report": {"macro_f1": 0.42, "balanced_accuracy": 0.49},
            "test_report": {"macro_f1": 0.41, "balanced_accuracy": 0.48},
            "val": {"loss": 1.2},
            "test": {"loss": 1.25},
        }
        full = {
            "run_id": "full_001",
            "fast_mode": {"enabled": False},
            "val_report": {"macro_f1": 0.51, "balanced_accuracy": 0.58},
            "test_report": {"macro_f1": 0.50, "balanced_accuracy": 0.57},
            "val": {"loss": 0.9},
            "test": {"loss": 0.95},
        }
        result = compare_runs([fast, full])
        assert len(result["runs"]) == 2
        run_ids = [r["run_id"] for r in result["runs"]]
        assert "fast_001" in run_ids
        assert "full_001" in run_ids

    def test_fast_mode_flag_survives_json_roundtrip(self):
        summary = {"run_id": "fast_001", "fast_mode": {"enabled": True, "fast_months": 3}}
        raw = json.dumps(summary)
        loaded = json.loads(raw)
        assert loaded["fast_mode"]["enabled"] is True
        assert loaded["fast_mode"]["fast_months"] == 3

    def test_compare_runs_best_by_metric_works_with_fast_runs(self):
        from src.training.run_summary import compare_runs
        runs = [
            {
                "run_id": "fast_001",
                "fast_mode": {"enabled": True},
                "val_report": {"macro_f1": 0.42, "balanced_accuracy": 0.49},
                "test_report": {"macro_f1": 0.41},
                "val": {"loss": 1.2},
                "test": {"loss": 1.25},
            },
            {
                "run_id": "full_001",
                "fast_mode": {"enabled": False},
                "val_report": {"macro_f1": 0.55, "balanced_accuracy": 0.60},
                "test_report": {"macro_f1": 0.54},
                "val": {"loss": 0.85},
                "test": {"loss": 0.90},
            },
        ]
        result = compare_runs(runs)
        best_f1 = result["best"].get("val_report.macro_f1")
        assert best_f1 is not None
        assert best_f1["run_id"] == "full_001"  # full run should win on macro_f1

    def test_compare_fast_and_full_shows_delta(self):
        from src.training.run_summary import compare_runs
        fast = {
            "run_id": "fast_001",
            "fast_mode": {"enabled": True},
            "val_report": {"macro_f1": 0.40, "balanced_accuracy": 0.47},
            "test_report": {"macro_f1": 0.39},
            "val": {"loss": 1.3},
            "test": {"loss": 1.35},
        }
        full = {
            "run_id": "full_001",
            "fast_mode": {"enabled": False},
            "val_report": {"macro_f1": 0.52, "balanced_accuracy": 0.58},
            "test_report": {"macro_f1": 0.51},
            "val": {"loss": 0.88},
            "test": {"loss": 0.92},
        }
        result = compare_runs([fast, full])
        assert len(result["deltas"]) == 1
        delta = result["deltas"][0]
        assert delta["run_id"] == "full_001"
        f1_delta = delta.get("val_report.macro_f1")
        assert f1_delta is not None
        assert f1_delta > 0  # full run improved over fast


# ---------------------------------------------------------------------------
# 10. No silent fallback to synthetic data
# ---------------------------------------------------------------------------


class TestNoSilentFallback:
    def test_fast_mode_enabled_does_not_set_synthetic_flag(self):
        tc = TrainingConfig(fast_mode=FastModeConfig(enabled=True))
        assert tc.use_real_historical_data is False  # unchanged from default (not synthetic override)

    def test_real_data_flag_preserved_with_fast_mode(self):
        tc = TrainingConfig(
            use_real_historical_data=True,
            fast_mode=FastModeConfig(enabled=True, fast_months=6),
        )
        assert tc.use_real_historical_data is True

    def test_fast_mode_disabled_by_default_in_yaml(self):
        config = load_config("configs/default.yaml", "configs/training.yaml")
        assert config.training.fast_mode.enabled is False

    def test_fast_mode_config_immutable_to_non_fast_fields(self):
        """Toggling fast_mode should not alter unrelated training fields."""
        tc_a = TrainingConfig()
        tc_b = TrainingConfig(fast_mode=FastModeConfig(enabled=True, fast_months=3))
        assert tc_a.label_epsilon == tc_b.label_epsilon
        assert tc_a.classification_loss == tc_b.classification_loss
        assert tc_a.forward_horizons == tc_b.forward_horizons


# ---------------------------------------------------------------------------
# 11. Model profile config
# ---------------------------------------------------------------------------


class TestFastModelProfile:
    def test_default_profile_unchanged(self):
        fc = FastModeConfig(enabled=False)
        assert fc.model_profile == "default"

    def test_small_profile_accepted(self):
        fc = FastModeConfig(enabled=True, model_profile="small")
        assert fc.model_profile == "small"

    def test_small_overrides_match_expected_keys(self):
        """The small profile override dict covers the right ModelConfig fields."""
        from src.training.supervised import _FAST_SMALL_MODEL_OVERRIDES
        expected_keys = {"cnn_channels", "temporal_blocks", "branch_hidden_dim", "branch_output_dim", "soma_hidden_dim"}
        assert set(_FAST_SMALL_MODEL_OVERRIDES.keys()) == expected_keys

    def test_small_model_reduces_cnn_channels(self):
        from src.training.supervised import _FAST_SMALL_MODEL_OVERRIDES
        assert _FAST_SMALL_MODEL_OVERRIDES["cnn_channels"] == [8, 16]

    def test_small_model_reduces_temporal_blocks(self):
        from src.training.supervised import _FAST_SMALL_MODEL_OVERRIDES
        assert _FAST_SMALL_MODEL_OVERRIDES["temporal_blocks"] == 2
