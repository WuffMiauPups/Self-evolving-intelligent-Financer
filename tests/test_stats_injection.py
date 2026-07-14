"""Normalization-stats injection for checkpoint evaluation (MASTER_PLAN §4.6.4).

``evaluate_checkpoint`` used to re-derive train-split z-score stats from
whatever data range was configured, so evaluating one checkpoint against
two ranges normalized inputs differently — metrics were not comparable.
The fix threads a ``normalization_override`` through the bundle build:
the checkpoint's persisted stats are applied to every split instead of
deriving new ones.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.features.normalization import normalization_stats_max_rel_diff
from src.training.data_module import build_real_data_training_bundle
from src.training.sample_columns import apply_normalization_stats, normalize_columns

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_sample_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SETAI_DISABLE_SAMPLE_CACHE", "1")


def _config():
    config = load_config("configs/default.yaml")
    config.runtime.warmup_bars = 32
    config.data.symbols = ["AAPL"]
    config.training.fast_mode.enabled = False
    return config


def _bundle(config, bars, override=None):
    return build_real_data_training_bundle(
        config=config,
        bars_by_symbol={"AAPL": bars},
        pipeline=FeaturePipeline(config.features, device=torch.device("cpu")),
        normalization_override=override,
    )


def test_apply_stats_reproduces_derivation_transform() -> None:
    """Applying derived stats to a raw copy == the in-place derivation."""
    from tests.test_normalization_parity import _columns

    derived, raw_copy = _columns(11), _columns(11)
    stats = normalize_columns(derived)
    apply_normalization_stats(stats, raw_copy)
    torch.testing.assert_close(raw_copy.features, derived.features)
    torch.testing.assert_close(raw_copy.regime_input, derived.regime_input)
    torch.testing.assert_close(raw_copy.temporal_tensor, derived.temporal_tensor)
    for name in derived.branch_inputs:
        torch.testing.assert_close(raw_copy.branch_inputs[name], derived.branch_inputs[name])


def test_two_data_ranges_apply_identical_injected_stats() -> None:
    """The §4.6.4 verify criterion: same checkpoint stats, two ranges."""
    config = _config()
    bars = generate_synthetic_bars("AAPL", 400)

    checkpoint_stats = _bundle(config, bars[:300]).normalization_stats
    assert checkpoint_stats is not None

    # A DIFFERENT data range would derive materially different stats...
    rederived = _bundle(config, bars[100:]).normalization_stats
    diffs = normalization_stats_max_rel_diff(checkpoint_stats, rederived)
    assert any(value > 0.0 for value in diffs.values()), (
        "sanity: two ranges should not derive identical stats"
    )

    # ...but with the override, the bundle carries the injected stats
    # unchanged: evaluation normalizes with the checkpoint's transform.
    injected = _bundle(config, bars[100:], override=checkpoint_stats)
    assert injected.normalization_stats is checkpoint_stats


def test_evaluate_checkpoint_injects_artifact_stats() -> None:
    """Source contract: evaluate_checkpoint passes the checkpoint's stats
    into the bundle build (the full path needs a real checkpoint + data;
    the mechanism is proven by the bundle-level test above)."""
    source = (REPO_ROOT / "src" / "training" / "supervised.py").read_text(encoding="utf-8")
    assert "normalization_override=artifacts.normalization" in source
