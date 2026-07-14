"""Encoder-fingerprint identity (2026-07-14 collision fix).

Observed in the ⟲3 run: the ⟲2 (44-dim contract) and ⟲3 (52-dim contract)
candidates shared encoder fingerprint 4575582c9cd97795 because the hash
reduced to (version, kind, model_config) — the "epoch"/"training_loss"
keys it referenced never existed in meta (the save path writes
"train_loss"), and neither the feature contract nor the weights entered
the hash. A colliding fingerprint silently defeats the Platt calibrator
binding and memory.require_encoder_match.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.memory.memory_bank import compute_encoder_fingerprint
from src.models.factory import state_dict_digest

REPO_ROOT = Path(__file__).resolve().parents[1]

_BASE_META = {
    "checkpoint_version": "2.0",
    "model_kind": "brain",
    "model_config": {"decoupled_heads": True, "soma_hidden_dim": 192},
    "feature_contract": {"feature_dim": 44, "regime_dim": 10},
    "weights_digest": "abc123",
}


def test_fingerprint_is_deterministic() -> None:
    assert compute_encoder_fingerprint(dict(_BASE_META)) == compute_encoder_fingerprint(
        dict(_BASE_META)
    )


def test_different_feature_contract_changes_fingerprint() -> None:
    """The exact ⟲2-vs-⟲3 collision: same model config, 44 vs 52 dims."""
    widened = dict(_BASE_META)
    widened["feature_contract"] = {"feature_dim": 52, "regime_dim": 18}
    assert compute_encoder_fingerprint(dict(_BASE_META)) != compute_encoder_fingerprint(widened)


def test_different_weights_change_fingerprint() -> None:
    retrained = dict(_BASE_META)
    retrained["weights_digest"] = "def456"
    assert compute_encoder_fingerprint(dict(_BASE_META)) != compute_encoder_fingerprint(retrained)


def test_state_dict_digest_tracks_weights_and_names() -> None:
    g = torch.Generator().manual_seed(11)
    state = {"linear.weight": torch.randn(4, 3, generator=g), "linear.bias": torch.zeros(4)}
    same = {name: tensor.clone() for name, tensor in state.items()}
    assert state_dict_digest(state) == state_dict_digest(same)

    perturbed = {name: tensor.clone() for name, tensor in state.items()}
    perturbed["linear.weight"][0, 0] += 1.0e-3
    assert state_dict_digest(state) != state_dict_digest(perturbed)

    renamed = {"other.weight": state["linear.weight"], "linear.bias": state["linear.bias"]}
    assert state_dict_digest(state) != state_dict_digest(renamed)


def test_save_path_stamps_weights_digest_source_contract() -> None:
    source = (REPO_ROOT / "src" / "training" / "supervised.py").read_text(encoding="utf-8")
    assert "weights_digest=state_dict_digest(model.state_dict())" in source
