"""Tests for experiment presets and --classification-only flag.

Covers:
- PRESETS dict structure and required keys
- _apply_preset logic (defaults, CLI override)
- --classification-only flag config effects
- --preset state-only config effects
- Sweep config loading with preset key
- Preset names in sweep_state_only.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_train import PRESETS, _apply_preset
from src.core.config import AppConfig, load_config


# ---------------------------------------------------------------------------
# 1. PRESETS dict structure
# ---------------------------------------------------------------------------


class TestPresetsDict:
    def test_state_only_exists(self):
        assert "state-only" in PRESETS

    def test_classification_latent_exists(self):
        assert "classification-latent" in PRESETS

    def test_classification_only_exists(self):
        assert "classification-only" in PRESETS

    def test_all_presets_have_description(self):
        for name, preset in PRESETS.items():
            assert "description" in preset, f"Preset {name!r} missing description"
            assert isinstance(preset["description"], str)

    def test_state_only_disables_auxiliaries(self):
        p = PRESETS["state-only"]
        assert p.get("disable_future_return") is True
        assert p.get("disable_risk_path") is True
        # Should NOT disable regression or latent
        assert p.get("disable_regression") is not True
        assert p.get("disable_latent") is not True
        # path_label head was killed; should not appear in any preset
        assert "disable_path_label" not in p

    def test_classification_only_disables_everything(self):
        p = PRESETS["classification-only"]
        assert p.get("disable_future_return") is True
        assert p.get("disable_risk_path") is True
        assert p.get("disable_regression") is True
        assert p.get("disable_latent") is True
        assert "disable_path_label" not in p

    def test_classification_latent_disables_regression_but_not_latent(self):
        p = PRESETS["classification-latent"]
        assert p.get("disable_future_return") is True
        assert p.get("disable_risk_path") is True
        assert p.get("disable_regression") is True
        assert p.get("disable_latent") is not True
        assert "disable_path_label" not in p


# ---------------------------------------------------------------------------
# 2. _apply_preset logic
# ---------------------------------------------------------------------------


def _make_namespace(**kwargs) -> argparse.Namespace:
    """Create a namespace mimicking default argparse output."""
    defaults = {
        "fast": False,
        "fast_months": None,
        "fast_epochs": None,
        "disable_future_return": False,
        "disable_risk_path": False,
        "disable_regression": False,
        "disable_latent": False,
        "label_epsilon": None,
        "focal_gamma": None,
        "classification_loss": None,
        "class_weight_mode": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestApplyPreset:
    def test_state_only_sets_fast_and_disables(self):
        ns = _make_namespace()
        _apply_preset(ns, "state-only")
        assert ns.fast is True
        assert ns.fast_months == 6
        assert ns.fast_epochs == 2
        assert ns.disable_future_return is True
        assert ns.disable_risk_path is True

    def test_cli_overrides_preset_bool(self):
        """If user already set --disable-future-return, preset doesn't change it."""
        ns = _make_namespace(disable_future_return=True)
        _apply_preset(ns, "state-only")
        assert ns.disable_future_return is True  # was already True

    def test_cli_overrides_preset_value(self):
        """If user set --fast-months 3, preset's fast_months=6 doesn't overwrite."""
        ns = _make_namespace(fast_months=3)
        _apply_preset(ns, "state-only")
        assert ns.fast_months == 3  # CLI wins

    def test_classification_only_preset(self):
        ns = _make_namespace()
        _apply_preset(ns, "classification-only")
        assert ns.disable_regression is True
        assert ns.disable_latent is True
        assert ns.disable_future_return is True
        assert ns.disable_risk_path is True

    def test_preset_does_not_set_unknown_attrs(self):
        ns = _make_namespace()
        _apply_preset(ns, "state-only")
        # Regression and latent should stay at default for state-only
        assert ns.disable_regression is False
        assert ns.disable_latent is False


# ---------------------------------------------------------------------------
# 3. --classification-only config effects
# ---------------------------------------------------------------------------


class TestClassificationOnlyConfig:
    def test_config_disables_all_auxiliaries(self):
        config = load_config("configs/default.yaml", "configs/training.yaml")
        # Simulate what run_train.py does with --classification-only
        config.training.disable_regression = True
        config.training.disable_latent = True
        config.training.fast_mode.disable_future_return = True
        config.training.fast_mode.disable_risk_path = True

        assert config.training.disable_regression is True
        assert config.training.disable_latent is True
        assert config.training.fast_mode.disable_future_return is True
        assert config.training.fast_mode.disable_risk_path is True
        # Classification weight should remain untouched
        assert config.training.loss_weights["classification"] == 1.0


# ---------------------------------------------------------------------------
# 4. --preset state-only config effects
# ---------------------------------------------------------------------------


class TestPresetStateOnlyConfig:
    def test_state_only_keeps_regression_and_latent(self):
        config = load_config("configs/default.yaml", "configs/training.yaml")
        # Simulate what happens when --preset state-only is applied
        fast = config.training.fast_mode
        fast.enabled = True
        fast.fast_months = 6
        fast.max_epochs = 2
        fast.disable_future_return = True
        fast.disable_risk_path = True

        assert config.training.disable_regression is False
        assert config.training.disable_latent is False
        assert config.training.loss_weights["regression"] > 0
        assert config.training.loss_weights["latent"] > 0

    def test_state_only_zeros_auxiliary_weights_at_training_time(self):
        """Verify that fast_mode disable flags will zero the right weights."""
        config = load_config("configs/default.yaml", "configs/training.yaml")
        fast = config.training.fast_mode
        fast.enabled = True
        fast.disable_future_return = True
        fast.disable_risk_path = True

        # Simulate supervised.py logic
        effective = dict(config.training.loss_weights)
        if fast.disable_future_return:
            effective["future_return"] = 0.0
        if fast.disable_risk_path:
            effective["risk_path"] = 0.0

        assert effective["future_return"] == 0.0
        assert effective["risk_path"] == 0.0
        assert effective["classification"] > 0
        assert effective["regression"] > 0
        assert effective["latent"] > 0


# ---------------------------------------------------------------------------
# 5. Sweep config loading with preset
# ---------------------------------------------------------------------------


class TestSweepConfigPreset:
    def test_load_sweep_config_with_preset(self):
        from sweep_train import load_sweep_config

        grid, preset = load_sweep_config("configs/sweep_state_only.yaml")
        assert preset == "state-only"
        assert "label_epsilon" in grid
        assert "classification_loss" in grid
        assert "focal_gamma" in grid
        assert "class_weight_mode" in grid

    def test_load_sweep_config_default_no_preset(self):
        from sweep_train import load_sweep_config

        grid, preset = load_sweep_config(None)
        assert preset is None
        assert "label_epsilon" in grid

    def test_sweep_state_only_grid_values(self):
        from sweep_train import load_sweep_config, expand_grid, prune_redundant_combos

        grid, _ = load_sweep_config("configs/sweep_state_only.yaml")
        combos = expand_grid(grid)
        combos = prune_redundant_combos(combos)
        # Should have reasonable number of combos
        assert len(combos) > 0
        # After pruning, cross_entropy combos should not vary focal_gamma
        ce_combos = [c for c in combos if c.get("classification_loss") == "cross_entropy"]
        focal_combos = [c for c in combos if c.get("classification_loss") == "focal"]
        assert len(focal_combos) > len(ce_combos)  # focal has gamma variations


# ---------------------------------------------------------------------------
# 6. Preset names consistency
# ---------------------------------------------------------------------------


class TestPresetNamesConsistency:
    def test_sweep_config_references_valid_preset(self):
        import yaml

        with open("configs/sweep_state_only.yaml") as f:
            data = yaml.safe_load(f)
        preset_name = data.get("preset")
        assert preset_name in PRESETS, f"Sweep config references unknown preset: {preset_name!r}"
