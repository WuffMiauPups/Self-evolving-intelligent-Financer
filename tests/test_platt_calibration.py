"""Platt calibration correctness (MASTER_PLAN §4.2).

Covers:
  1. Direction-aware fit — calibrated confidence approximates P(chosen
     direction wins), not P(price up); FLAT predictions are excluded.
  2. Artifact isolation — the calibrator is written next to the checkpoint
     it calibrates, so isolated experiment runs never clobber the canonical
     artifact the runtime loads.
  3. Fingerprint binding — loaders resolve the artifact relative to the
     checkpoint and refuse a fingerprint mismatch (passthrough fallback);
     legacy artifacts without a fingerprint load with a warning.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.training.platt_calibration import (
    PLATT_CALIBRATION_FILENAME,
    PlattCalibrator,
    direction_aware_outcomes,
    load_calibrator_for_checkpoint,
    platt_calibration_path_for_checkpoint,
)
from src.training.supervised import resolve_checkpoint_save_dir

# Direction classes (src/training/targets.py)
SHORT, FLAT, LONG = 0, 1, 2


# ---------------------------------------------------------------------------
# 1. Direction-aware fit
# ---------------------------------------------------------------------------


def test_direction_aware_outcomes_signs_costs_and_flat_exclusion() -> None:
    actions = np.array([SHORT, FLAT, LONG])
    returns = np.array([-0.01, 0.005, 0.02])
    traded, outcomes = direction_aware_outcomes(actions, returns, per_trade_cost=0.001)
    assert traded.tolist() == [True, False, True]
    # Correct short: predicted_sign(-1) x (-0.01) - 0.001 = +0.009 (a win)
    assert outcomes[0] == pytest.approx(0.009)
    # Long: +0.02 - 0.001 = +0.019
    assert outcomes[2] == pytest.approx(0.019)


def test_direction_aware_outcomes_cost_can_flip_marginal_win_to_loss() -> None:
    actions = np.array([LONG] * 25)
    returns = np.full(25, 0.0005)  # +5bps move
    _, outcomes = direction_aware_outcomes(actions, returns, per_trade_cost=0.001)
    assert (outcomes < 0.0).all()  # 10bps cost eats the 5bps win


def test_shorts_80pct_correct_calibrate_near_08_not_02() -> None:
    """§4.2.1 verify: shorts that are 80% right calibrate to ~0.8.

    Under the old direction-agnostic labels (`primary_return > 0`) the same
    set calibrated to ~0.2 — the price went DOWN on the winning shorts, so
    correct short calls were counted as losses.
    """
    rng = np.random.default_rng(7)
    n = 500
    actions = np.full(n, SHORT)
    returns = np.where(np.arange(n) < int(n * 0.8), -0.01, 0.01).astype(float)
    rng.shuffle(returns)
    confidences = rng.uniform(0.55, 0.95, size=n)

    traded, outcomes = direction_aware_outcomes(actions, returns, per_trade_cost=0.001)
    calibrator = PlattCalibrator()
    calibrator.fit(confidences[traded], outcomes[traded])
    assert calibrator.is_fitted
    calibrated = np.array([calibrator.calibrate(c) for c in confidences])
    assert abs(float(calibrated.mean()) - 0.8) < 0.05

    # The old labels are anti-calibrated for the same shorts.
    old_calibrator = PlattCalibrator()
    old_calibrator.fit(confidences, returns)
    old_calibrated = np.array([old_calibrator.calibrate(c) for c in confidences])
    assert float(old_calibrated.mean()) < 0.35


# ---------------------------------------------------------------------------
# 2. Artifact isolation
# ---------------------------------------------------------------------------


def _fit_dummy_calibrator(fingerprint: str | None = None) -> PlattCalibrator:
    rng = np.random.default_rng(3)
    calibrator = PlattCalibrator()
    calibrator.fit(rng.uniform(0.4, 0.9, 64), rng.normal(0.0, 0.01, 64))
    assert calibrator.is_fitted
    calibrator.encoder_fingerprint = fingerprint
    calibrator.checkpoint_filename = "brain.pt"
    return calibrator


def test_isolated_run_calibrator_lands_in_run_dir_canonical_untouched(tmp_path: Path) -> None:
    """§4.2.2 verify: isolated artifact in isolated dir; canonical bytes intact."""
    base_dir = tmp_path / "checkpoints"
    base_dir.mkdir()
    canonical_platt = platt_calibration_path_for_checkpoint(base_dir / "brain.pt")
    _fit_dummy_calibrator(fingerprint="canonical-fp").save(canonical_platt)
    canonical_bytes = canonical_platt.read_bytes()

    # Isolated run (explicit run_id → auto-isolation, same as supervised.py)
    isolated_dir, isolated = resolve_checkpoint_save_dir(
        base_dir=base_dir,
        run_id="exp_direction_labels",
        explicit_run_id=True,
        fast_mode_enabled=False,
        isolate_checkpoint=None,
    )
    assert isolated
    isolated_checkpoint = isolated_dir / "brain.pt"
    _fit_dummy_calibrator(fingerprint="experiment-fp").save(
        platt_calibration_path_for_checkpoint(isolated_checkpoint)
    )

    assert (base_dir / "exp_direction_labels" / PLATT_CALIBRATION_FILENAME).exists()
    assert canonical_platt.read_bytes() == canonical_bytes


def test_canonical_run_calibrator_path_is_next_to_canonical_checkpoint(tmp_path: Path) -> None:
    base_dir = tmp_path / "checkpoints"
    target_dir, isolated = resolve_checkpoint_save_dir(
        base_dir=base_dir,
        run_id="20260710_120000",
        explicit_run_id=False,
        fast_mode_enabled=False,
        isolate_checkpoint=None,
    )
    assert not isolated
    assert (
        platt_calibration_path_for_checkpoint(target_dir / "brain.pt")
        == base_dir / PLATT_CALIBRATION_FILENAME
    )


# ---------------------------------------------------------------------------
# 3. Config-driven load + fingerprint binding
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_provenance(tmp_path: Path) -> None:
    calibrator = _fit_dummy_calibrator(fingerprint="fp-1234")
    path = tmp_path / PLATT_CALIBRATION_FILENAME
    calibrator.save(path)
    loaded = PlattCalibrator()
    loaded.load(path)
    assert loaded.is_fitted
    assert loaded.a == calibrator.a
    assert loaded.b == calibrator.b
    assert loaded.encoder_fingerprint == "fp-1234"
    assert loaded.checkpoint_filename == "brain.pt"


def test_load_for_checkpoint_matching_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "brain.pt"
    calibrator = _fit_dummy_calibrator(fingerprint="fp-match")
    calibrator.save(platt_calibration_path_for_checkpoint(checkpoint))
    loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp-match")
    assert loaded.is_fitted
    assert loaded.a == calibrator.a
    assert loaded.encoder_fingerprint == "fp-match"


def test_load_for_checkpoint_mismatch_falls_back_to_passthrough(
    tmp_path: Path, caplog
) -> None:
    checkpoint = tmp_path / "brain.pt"
    _fit_dummy_calibrator(fingerprint="fp-stale").save(
        platt_calibration_path_for_checkpoint(checkpoint)
    )
    with caplog.at_level(logging.WARNING, logger="src.training.platt_calibration"):
        loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp-new")
    assert not loaded.is_fitted
    assert loaded.calibrate(0.7) == 0.7  # passthrough, not stale calibration
    assert any("fingerprint_mismatch" in record.getMessage() for record in caplog.records)


def test_load_for_checkpoint_legacy_file_loads_with_warning(tmp_path: Path, caplog) -> None:
    checkpoint = tmp_path / "brain.pt"
    # Legacy pre-binding artifact: no fingerprint keys at all.
    legacy_path = platt_calibration_path_for_checkpoint(checkpoint)
    legacy_path.write_text(
        json.dumps({"a": 2.0, "b": -0.5, "is_fitted": True, "n_samples": 128})
    )
    with caplog.at_level(logging.WARNING, logger="src.training.platt_calibration"):
        loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp-new")
    assert loaded.is_fitted
    assert loaded.a == 2.0
    assert loaded.b == -0.5
    assert loaded.encoder_fingerprint is None
    assert any("legacy" in record.getMessage() for record in caplog.records)


def test_load_for_checkpoint_missing_artifact_is_passthrough(tmp_path: Path) -> None:
    loaded = load_calibrator_for_checkpoint(
        tmp_path / "brain.pt", expected_fingerprint="fp-any"
    )
    assert not loaded.is_fitted
    assert loaded.calibrate(0.42) == 0.42


def test_load_for_checkpoint_no_checkpoint_path_is_passthrough() -> None:
    loaded = load_calibrator_for_checkpoint(None, expected_fingerprint=None)
    assert not loaded.is_fitted
