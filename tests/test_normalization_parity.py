"""Train/serve normalization parity tests (C2 fix).

Training z-scores every model input with train-split statistics.  These
tests pin the two halves of the fix: (1) the stats returned by
``normalize_columns`` reproduce the in-place training transform exactly,
and (2) the inference paths (BrainPolicy, ModelStateEncoder) apply those
stats to raw inputs before the model forward.  Together they guarantee
the deployed model sees the same input distribution it was trained on.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import torch

from src.features.normalization import NormalizationStats
from src.features.feature_pipeline import FeatureSnapshot
from src.models.base import ModelOutput
from src.models.state_encoder import ModelStateEncoder
from src.strategies.brain_policy import BrainPolicy
from src.training.sample_columns import ColumnarSamples, normalize_columns


N, F, R, C, T, D = 64, 7, 4, 3, 12, 5
BRANCHES = ("trend", "volatility")


def _columns(seed: int, n: int = N) -> ColumnarSamples:
    g = torch.Generator().manual_seed(seed)

    def r(*shape: int, scale: float = 1.0, offset: float = 0.0) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * scale + offset

    return ColumnarSamples(
        features=r(n, F, scale=3.0, offset=5.0),
        label=torch.randint(0, 3, (n,), generator=g),
        regime_input=r(n, R, scale=0.5, offset=-2.0),
        temporal_tensor=r(n, C, T, scale=1e-3),  # tiny scale like return channels
        branch_inputs={name: r(n, D, scale=2.0, offset=1.0) for name in BRANCHES},
    )


def _clone_inputs(cols: ColumnarSamples) -> dict[str, object]:
    return {
        "features": cols.features.clone(),
        "regime_input": cols.regime_input.clone(),
        "temporal_tensor": cols.temporal_tensor.clone(),
        "branch_inputs": {k: v.clone() for k, v in cols.branch_inputs.items()},
    }


def test_stats_reproduce_training_transform_exactly() -> None:
    train, val = _columns(1), _columns(2, n=32)
    raw_train, raw_val = _clone_inputs(train), _clone_inputs(val)
    stats = normalize_columns(train, val)

    for raw, cols in ((raw_train, train), (raw_val, val)):
        torch.testing.assert_close(stats.apply_features(raw["features"]), cols.features)
        torch.testing.assert_close(stats.apply_regime(raw["regime_input"]), cols.regime_input)
        torch.testing.assert_close(stats.apply_temporal(raw["temporal_tensor"]), cols.temporal_tensor)
        for name in BRANCHES:
            torch.testing.assert_close(
                stats.apply_branch(name, raw["branch_inputs"][name]),
                cols.branch_inputs[name],
            )


def test_stats_survive_json_round_trip() -> None:
    train = _columns(3)
    raw = _clone_inputs(train)
    stats = normalize_columns(train)
    revived = NormalizationStats.from_dict(json.loads(json.dumps(stats.to_dict())))
    torch.testing.assert_close(revived.apply_features(raw["features"]), train.features)
    torch.testing.assert_close(revived.apply_temporal(raw["temporal_tensor"]), train.temporal_tensor)
    for name in BRANCHES:
        torch.testing.assert_close(
            revived.apply_branch(name, raw["branch_inputs"][name]),
            train.branch_inputs[name],
        )


def test_from_meta_none_for_pre_fix_checkpoints() -> None:
    assert NormalizationStats.from_meta(None) is None
    assert NormalizationStats.from_meta({}) is None
    assert NormalizationStats.from_meta({"normalization": None}) is None


class _CapturingModel:
    """Records the tensors the forward actually received."""

    def __init__(self) -> None:
        self.captured: dict[str, object] = {}

    def __call__(
        self,
        *,
        branch_inputs: dict[str, torch.Tensor],
        regime_input: torch.Tensor,
        memory_state: torch.Tensor | None = None,
        full_features: torch.Tensor | None = None,
        temporal_tensor: torch.Tensor | None = None,
    ) -> ModelOutput:
        self.captured = {
            "branch_inputs": {k: v.clone() for k, v in branch_inputs.items()},
            "regime_input": regime_input.clone(),
            "full_features": full_features.clone(),
            "temporal_tensor": temporal_tensor.clone(),
        }
        batch = regime_input.shape[0]
        latent = torch.zeros(batch, 8)
        return ModelOutput(
            action_logits=torch.tensor([[0.1, 0.9, 0.2]]).repeat(batch, 1),
            confidence=torch.full((batch, 1), 0.7),
            target_fraction=torch.full((batch, 1), 0.05),
            fused_representation=latent,
            branch_embeddings={},
            branch_confidences={},
            soma_weights=torch.zeros(batch, 1),
            soma_activation=torch.full((batch, 1), 1.0),
        )


def _stats() -> NormalizationStats:
    return NormalizationStats(
        features_mean=torch.full((F,), 5.0),
        features_std=torch.full((F,), 2.0),
        regime_mean=torch.full((R,), -2.0),
        regime_std=torch.full((R,), 0.5),
        temporal_mean=torch.full((C,), 1e-3),
        temporal_std=torch.full((C,), 1e-3),
        branch_means={name: torch.full((D,), 1.0) for name in BRANCHES},
        branch_stds={name: torch.full((D,), 2.0) for name in BRANCHES},
        clamp=10.0,
    )


def _snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 7, 6, 14, 30, tzinfo=UTC),
        last_price=100.0,
        spread_bps=1.0,
        features=torch.full((F,), 7.0),
        branch_inputs={name: torch.full((D,), 3.0) for name in BRANCHES},
        regime_input=torch.full((R,), -1.0),
        temporal_tensor=torch.full((C, T), 2e-3),
        feature_names=[f"f{i}" for i in range(F)],
        branch_feature_names={name: [f"{name}{i}" for i in range(D)] for name in BRANCHES},
        regime_feature_names=[f"r{i}" for i in range(R)],
        temporal_feature_names=[f"t{i}" for i in range(C)],
    )


def test_brain_policy_normalizes_model_inputs() -> None:
    model = _CapturingModel()
    policy = BrainPolicy(
        model=model,
        device=torch.device("cpu"),
        normalization=_stats(),
    )
    policy.propose(_snapshot())
    torch.testing.assert_close(
        model.captured["full_features"],
        torch.full((1, F), (7.0 - 5.0) / 2.0),
    )
    torch.testing.assert_close(
        model.captured["regime_input"],
        torch.full((1, R), (-1.0 - -2.0) / 0.5),
    )
    # ensure_batch_dimension leaves an already-2D (C, T) tensor unbatched
    # on the single-sample path; only the values matter here.
    torch.testing.assert_close(
        model.captured["temporal_tensor"].reshape(C, T),
        torch.full((C, T), (2e-3 - 1e-3) / 1e-3),
    )
    for name in BRANCHES:
        torch.testing.assert_close(
            model.captured["branch_inputs"][name],
            torch.full((1, D), (3.0 - 1.0) / 2.0),
        )


def test_brain_policy_batch_matches_single_under_normalization() -> None:
    stats = _stats()
    single_model = _CapturingModel()
    batch_model = _CapturingModel()
    single = BrainPolicy(model=single_model, device=torch.device("cpu"), normalization=stats)
    batched = BrainPolicy(model=batch_model, device=torch.device("cpu"), normalization=stats)
    snap = _snapshot()
    single.propose(snap)
    batched.propose_batch([snap])
    torch.testing.assert_close(
        batch_model.captured["full_features"], single_model.captured["full_features"]
    )
    torch.testing.assert_close(
        batch_model.captured["temporal_tensor"][0],
        single_model.captured["temporal_tensor"].reshape(C, T),
    )
    torch.testing.assert_close(
        batch_model.captured["regime_input"], single_model.captured["regime_input"]
    )


def test_state_encoder_normalizes_model_inputs() -> None:
    model = _CapturingModel()
    encoder = ModelStateEncoder(
        model=model,
        device=torch.device("cpu"),
        normalization=_stats(),
    )
    encoder.encode(_snapshot())
    torch.testing.assert_close(
        model.captured["full_features"],
        torch.full((1, F), (7.0 - 5.0) / 2.0),
    )
    torch.testing.assert_close(
        model.captured["temporal_tensor"],
        torch.full((1, C, T), (2e-3 - 1e-3) / 1e-3),
    )


def test_without_stats_inputs_pass_through_raw() -> None:
    model = _CapturingModel()
    policy = BrainPolicy(model=model, device=torch.device("cpu"), normalization=None)
    policy.propose(_snapshot())
    torch.testing.assert_close(model.captured["full_features"], torch.full((1, F), 7.0))


def test_clamp_matches_training_bounds() -> None:
    stats = _stats()
    wild = torch.full((1, F), 1e6)
    assert stats.apply_features(wild).max().item() == pytest.approx(10.0)
