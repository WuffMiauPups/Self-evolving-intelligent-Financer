"""Tests for the normalization-stats comparison seam (MASTER_PLAN 4.6.4).

``evaluate_checkpoint`` rebuilds the training bundle and re-derives
normalization stats from the CURRENT data range; the checkpoint carries the
stats it was trained with.  ``normalization_stats_max_rel_diff`` measures
per-tensor divergence and ``_warn_on_normalization_stats_mismatch`` (in
src/training/supervised.py) turns material divergence into a loud warning
about metric comparability.
"""
from __future__ import annotations

import math

import torch

from src.features.normalization import (
    NormalizationStats,
    normalization_stats_max_rel_diff,
)


def _make_stats(**overrides) -> NormalizationStats:
    base = dict(
        features_mean=torch.tensor([1.0, -2.0, 0.5]),
        features_std=torch.tensor([0.5, 1.5, 2.0]),
        regime_mean=torch.tensor([0.1, 0.2]),
        regime_std=torch.tensor([1.0, 1.0]),
        temporal_mean=torch.tensor([0.0, 1.0]),
        temporal_std=torch.tensor([1.0, 2.0]),
        branch_means={"trend": torch.tensor([0.3, 0.4])},
        branch_stds={"trend": torch.tensor([1.1, 0.9])},
    )
    base.update(overrides)
    return NormalizationStats(**base)


def test_identical_stats_have_zero_diff() -> None:
    a = _make_stats()
    b = _make_stats()
    diffs = normalization_stats_max_rel_diff(a, b)
    assert diffs, "expected per-tensor entries"
    assert all(value == 0.0 for value in diffs.values()), diffs


def test_material_difference_is_flagged_per_tensor() -> None:
    a = _make_stats()
    # 5% relative perturbation on one tensor only.
    b = _make_stats(features_mean=torch.tensor([1.05, -2.0, 0.5]))
    diffs = normalization_stats_max_rel_diff(a, b)
    assert diffs["features_mean"] > 0.01
    assert diffs["features_std"] == 0.0
    assert diffs["regime_mean"] == 0.0


def test_sub_threshold_noise_not_material() -> None:
    a = _make_stats()
    b = _make_stats(features_std=torch.tensor([0.5001, 1.5, 2.0]))  # ~0.02% rel
    diffs = normalization_stats_max_rel_diff(a, b)
    assert diffs["features_std"] < 0.01


def test_denominator_floor_suppresses_fp_noise_near_zero() -> None:
    a = _make_stats(regime_mean=torch.tensor([1.0e-9, 0.0]))
    b = _make_stats(regime_mean=torch.tensor([-1.0e-9, 0.0]))
    diffs = normalization_stats_max_rel_diff(a, b)
    assert diffs["regime_mean"] < 0.01


def test_shape_mismatch_scores_inf() -> None:
    a = _make_stats()
    b = _make_stats(features_mean=torch.tensor([1.0, -2.0]), features_std=torch.tensor([0.5, 1.5]))
    diffs = normalization_stats_max_rel_diff(a, b)
    assert math.isinf(diffs["features_mean"])
    assert math.isinf(diffs["features_std"])


def test_branch_key_missing_on_one_side_scores_inf() -> None:
    a = _make_stats()
    b = _make_stats(branch_means={}, branch_stds={})
    diffs = normalization_stats_max_rel_diff(a, b)
    assert math.isinf(diffs["branch_mean[trend]"])
    assert math.isinf(diffs["branch_std[trend]"])


# ---------------------------------------------------------------------------
# The evaluate_checkpoint warning seam
# ---------------------------------------------------------------------------


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []
        self.infos: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs) -> None:
        self.warnings.append((event, kwargs))

    def info(self, event: str, **kwargs) -> None:
        self.infos.append((event, kwargs))


def test_warn_helper_fires_loud_warning_on_material_mismatch() -> None:
    from src.training.supervised import _warn_on_normalization_stats_mismatch

    logger = _RecordingLogger()
    material = _warn_on_normalization_stats_mismatch(
        logger,
        checkpoint_path="checkpoints/brain.pt",
        checkpoint_stats=_make_stats(),
        bundle_stats=_make_stats(features_mean=torch.tensor([1.2, -2.0, 0.5])),
    )
    assert "features_mean" in material
    assert len(logger.warnings) == 1
    event, payload = logger.warnings[0]
    assert event == "checkpoint_normalization_stats_mismatch"
    assert "features_mean" in payload["max_rel_diff_by_tensor"]
    assert "NOT COMPARABLE" in payload["consequence"]


def test_warn_helper_silent_when_stats_match() -> None:
    from src.training.supervised import _warn_on_normalization_stats_mismatch

    logger = _RecordingLogger()
    material = _warn_on_normalization_stats_mismatch(
        logger,
        checkpoint_path="checkpoints/brain.pt",
        checkpoint_stats=_make_stats(),
        bundle_stats=_make_stats(),
    )
    assert material == {}
    assert logger.warnings == []


def test_warn_helper_skips_when_stats_missing() -> None:
    from src.training.supervised import _warn_on_normalization_stats_mismatch

    logger = _RecordingLogger()
    material = _warn_on_normalization_stats_mismatch(
        logger,
        checkpoint_path="checkpoints/brain.pt",
        checkpoint_stats=None,
        bundle_stats=_make_stats(),
    )
    assert material == {}
    assert logger.warnings == []
    assert logger.infos and logger.infos[0][0] == "checkpoint_normalization_stats_comparison_skipped"
