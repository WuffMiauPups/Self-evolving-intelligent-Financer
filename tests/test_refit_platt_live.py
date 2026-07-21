"""Live Platt refit tooling (§7.0.3).

Joins decisions (raw confidence, client_order_id) with the outcome stream
(decision_id, realized net PnL) and fits on RESOLVED trades only: opening
fills (realized_pnl == 0.0) mean "unresolved", never "loss". The refit is
refused below the minimum sample count, and the artifact is isolated —
the canonical sidecar moves only through the promotion gate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "refit_platt_live", _ROOT / "scripts" / "refit_platt_live.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(order_id: str, confidence: float, *, semantics: str | None = "current") -> dict:
    """Post-fix ledger row by default; pass ``semantics=None`` for a pre-fix
    row (they lack the key entirely)."""
    from src.core.constants import CONFIDENCE_SEMANTICS

    row = {"client_order_id": order_id, "confidence": confidence, "approved": True}
    if semantics == "current":
        row["confidence_semantics"] = CONFIDENCE_SEMANTICS
    elif semantics is not None:
        row["confidence_semantics"] = semantics
    return row


def _observation(order_id: str, realized: float) -> dict:
    return {"decision_id": order_id, "realized_pnl": realized}


def test_join_excludes_opening_fills_and_unmatched() -> None:
    module = _load_module()
    decisions = [_decision("o1", 0.51), _decision("o2", 0.49), _decision("o3", 0.50)]
    observations = [
        _observation("o1", 0.0),     # opening fill: unresolved, NOT a loss
        _observation("o2", +1.25),   # resolved win
        _observation("o3", -0.80),   # resolved loss
        _observation("ghost", 2.0),  # no matching decision
    ]
    confidences, outcomes, stats = module.join_resolved_trades(decisions, observations)
    assert stats["resolved_trades"] == 2
    assert stats["unresolved_opening_fills"] == 1
    assert stats["unmatched_observations"] == 1
    assert confidences.tolist() == [0.49, 0.50]
    assert outcomes.tolist() == [1.25, -0.80]


def test_join_excludes_pre_fix_semantics_rows() -> None:
    """Semantics boundary (2026-07-21): rows written before the confidence
    fix recorded the legacy sigmoid-head quantity and MUST NOT enter a fit
    on P(resolved class) rows — mixing the two random variables would mint
    a falsely-stamped calibrator that the loader gate then accepts."""
    module = _load_module()
    decisions = [
        _decision("pre1", 0.51, semantics=None),        # pre-fix: no marker
        _decision("pre2", 0.49, semantics="something"),  # foreign marker
        _decision("post1", 0.62),                        # post-fix
    ]
    observations = [
        _observation("pre1", +1.0),
        _observation("pre2", -1.0),
        _observation("post1", +2.0),
    ]
    confidences, outcomes, stats = module.join_resolved_trades(decisions, observations)
    assert stats["pre_fix_semantics_excluded"] == 2
    assert stats["resolved_trades"] == 1
    assert confidences.tolist() == [0.62]
    assert outcomes.tolist() == [2.0]


def test_fit_on_synthetic_outcomes_improves_calibration_and_widens_range() -> None:
    from src.training.platt_calibration import PlattCalibrator

    rng = np.random.default_rng(7)
    n = 400
    # The live pathology: confidence pinned near 0.5 while the realized
    # win rate is NOT 0.5 — the passthrough is miscalibrated by a level
    # shift, exactly what Platt's b parameter corrects.
    confidences = rng.uniform(0.485, 0.515, n)
    outcomes = np.where(rng.uniform(size=n) < 0.75, 1.0, -1.0)

    passthrough = PlattCalibrator()
    fitted = PlattCalibrator()
    fitted.fit(confidences, outcomes)
    assert fitted.is_fitted
    assert fitted.brier_score(confidences, outcomes) < passthrough.brier_score(
        confidences, outcomes
    )
    # The level shift is learned: calibrated output moves toward the true
    # win rate instead of sitting at the pinned raw value.
    assert fitted.calibrate(0.5) > 0.6


def test_reliability_table_bins_are_consistent() -> None:
    module = _load_module()
    from src.training.platt_calibration import PlattCalibrator

    confidences = np.array([0.2, 0.4, 0.6, 0.8] * 10)
    outcomes = np.array([-1.0, -1.0, 1.0, 1.0] * 10)
    table = module.reliability_table(PlattCalibrator(), confidences, outcomes)
    assert sum(row["n"] for row in table) == 40
    for row in table:
        assert 0.0 <= row["win_rate"] <= 1.0


def test_script_refuses_below_min_samples_end_to_end(tmp_path: Path) -> None:
    """The refusal path — exactly what running it today should do."""
    import json
    import subprocess
    import sys

    decisions = tmp_path / "decisions.jsonl"
    observations = tmp_path / "observations.jsonl"
    decisions.write_text(
        json.dumps({"client_order_id": "o1", "confidence": 0.5}) + "\n", encoding="utf-8"
    )
    observations.write_text(
        json.dumps({"decision_id": "o1", "realized_pnl": 1.0}) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "refit_platt_live.py"),
            "--decisions",
            str(decisions),
            "--observations",
            str(observations),
            "--report",
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        timeout=120,
    )
    assert result.returncode == 2
    assert "REFUSED" in result.stdout
    assert not (tmp_path / "report.json").exists()


def test_artifact_path_is_isolated_not_canonical_source_contract() -> None:
    source = (_ROOT / "scripts" / "refit_platt_live.py").read_text(encoding="utf-8")
    assert 'DEFAULT_REPORT_PATH = Path("runs/platt_live_refit.json")' in source
    # It must never write the canonical checkpoint-adjacent sidecar.
    assert "platt_calibration_path_for_checkpoint" not in source
    assert "checkpoints/platt_calibration.json" not in source
