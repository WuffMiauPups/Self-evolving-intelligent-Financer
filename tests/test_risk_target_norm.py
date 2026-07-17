"""Per-symbol risk-target normalization (§7.0.4 fix).

The EV-input audit measured the risk head as a cross-sectional vol prior
(per-symbol rank correlation ≈ 0). With training.risk_target_per_symbol_norm
on, mfe/mae targets are divided by the symbol's train-split median
|excursion| (post-cache trainer transform), the scales persist in
checkpoint meta, and BrainPolicy multiplies the head's outputs back per
symbol. Old checkpoints (no scales in meta) are untouched end to end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.core.config import load_config
from src.models.factory import LoadedModelArtifacts
from src.strategies.brain_policy import BrainPolicy
from src.training.risk_target_norm import (
    compute_risk_target_scales,
    normalize_risk_targets,
)


def _columns(symbols: list[str], mfe: list[float], mae: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        symbols=symbols,
        regression_targets={
            "mfe": torch.tensor(mfe, dtype=torch.float32),
            "mae": torch.tensor(mae, dtype=torch.float32),
        },
    )


def test_scales_are_per_symbol_train_medians() -> None:
    # MSTR excursions ~10x SPY's — the exact vol-level asymmetry the head
    # was exploiting.
    train = _columns(
        ["SPY", "SPY", "SPY", "MSTR", "MSTR", "MSTR"],
        mfe=[0.001, 0.002, 0.003, 0.010, 0.020, 0.030],
        mae=[-0.001, -0.002, -0.003, -0.010, -0.020, -0.030],
    )
    scales = compute_risk_target_scales(train)
    assert scales["SPY"]["mfe"] == pytest.approx(0.002)
    assert scales["MSTR"]["mfe"] == pytest.approx(0.020)
    assert scales["SPY"]["mae"] == pytest.approx(0.002)  # median of |values|
    assert scales["MSTR"]["mae"] == pytest.approx(0.020)


def test_normalization_removes_the_level_answer_across_splits() -> None:
    train = _columns(
        ["SPY", "SPY", "SPY", "MSTR", "MSTR", "MSTR"],
        mfe=[0.001, 0.002, 0.003, 0.010, 0.020, 0.030],
        mae=[-0.001, -0.002, -0.003, -0.010, -0.020, -0.030],
    )
    val = _columns(["SPY", "MSTR"], mfe=[0.002, 0.020], mae=[-0.002, -0.020])
    scales = normalize_risk_targets(train, val)
    # After normalization both symbols' typical |mfe| is ~1.0 — a constant
    # per-symbol prediction can no longer minimize the loss.
    spy = train.regression_targets["mfe"][:3]
    mstr = train.regression_targets["mfe"][3:]
    assert float(spy.abs().median()) == pytest.approx(1.0)
    assert float(mstr.abs().median()) == pytest.approx(1.0)
    # Val used the TRAIN-derived scale (no leakage), signs preserved.
    assert val.regression_targets["mfe"].tolist() == pytest.approx([1.0, 1.0])
    assert val.regression_targets["mae"].tolist() == pytest.approx([-1.0, -1.0])
    assert set(scales) == {"SPY", "MSTR"}


def test_degenerate_scale_floors_instead_of_exploding() -> None:
    train = _columns(["DEAD", "DEAD"], mfe=[0.0, 0.0], mae=[0.0, 0.0])
    scales = compute_risk_target_scales(train)
    assert scales["DEAD"]["mfe"] == pytest.approx(1.0e-5)
    normalize_risk_targets(train)
    assert torch.isfinite(train.regression_targets["mfe"]).all()


def _policy(scales: dict | None) -> BrainPolicy:
    return BrainPolicy(
        model=SimpleNamespace(risk_head=object()),
        device=torch.device("cpu"),
        risk_target_scales=scales,
    )


def test_policy_denormalizes_per_symbol_and_passes_through_otherwise() -> None:
    policy = _policy({"MSTR": {"mfe": 0.020, "mae": 0.030}})
    # Model outputs ~1.2 "typical units" -> extractor gives 12_000 bps; the
    # de-norm restores raw fractions: 12_000 * 0.020 = 240 bps.
    gain, loss = policy._denormalize_ev("MSTR", 12_000.0, 10_000.0)
    assert gain == pytest.approx(240.0)
    assert loss == pytest.approx(300.0)
    # Unknown symbol: pass through.
    assert policy._denormalize_ev("SPY", 100.0, 50.0) == (100.0, 50.0)
    # Pre-normalization checkpoint (scales None): pass through.
    assert _policy(None)._denormalize_ev("MSTR", 100.0, 50.0) == (100.0, 50.0)


def test_artifacts_parse_scales_from_meta_and_default_none() -> None:
    with_scales = LoadedModelArtifacts(
        model=None,
        feature_contract=None,
        meta={"risk_target_scales": {"SPY": {"mfe": 0.002, "mae": 0.0025}}},
        model_kind="brain",
    )
    assert with_scales.risk_target_scales == {"SPY": {"mfe": 0.002, "mae": 0.0025}}
    without = LoadedModelArtifacts(model=None, feature_contract=None, meta={}, model_kind="brain")
    assert without.risk_target_scales is None
    legacy_none = LoadedModelArtifacts(
        model=None, feature_contract=None, meta={"risk_target_scales": None}, model_kind="brain"
    )
    assert legacy_none.risk_target_scales is None


def test_both_inference_paths_denormalize_source_contract() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "strategies" / "brain_policy.py"
    ).read_text(encoding="utf-8")
    single_path = source.split("def propose(", 1)[1].split("def propose_batch(", 1)[0]
    batch_path = source.split("def propose_batch(", 1)[1]
    assert "_denormalize_ev(" in single_path
    assert "_denormalize_ev(" in batch_path


def test_config_pair_defaults_off() -> None:
    config = load_config("configs/default.yaml")
    assert config.training.risk_target_per_symbol_norm is False
