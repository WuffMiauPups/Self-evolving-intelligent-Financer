"""Phase 2 §6.1 — model-head disentanglement.

Before this change three conflicting jobs shared two scalars:
- every per-horizon return target regressed onto the single
  ``target_fraction`` output (which is also the position-sizing signal);
- the ``stability`` statistic regressed onto ``confidence`` — the same
  scalar the Platt calibrator and runtime gating consume as
  "P(this trade wins)".

These tests pin the disentangled architecture AND that legacy checkpoints
still reconstruct their exact old architecture from their saved config.
"""
from __future__ import annotations

import torch

from src.core.config import FeatureConfig, ModelConfig
from src.models.brain_model import BrainTradingModel
from src.models.value_head import ReturnHeads
from src.training.losses import (
    future_return_regression_loss,
    multitask_regression_loss,
    risk_path_regression_loss,
)

_BRANCH_DIMS = {"trend": 4, "volume": 3}
_FLAT_DIM = sum(_BRANCH_DIMS.values())


def _synthetic_snapshot():
    """Build a real FeatureSnapshot (and thus a real FeatureContract)."""
    from datetime import UTC, datetime, timedelta

    from src.data.schemas import MarketBar
    from src.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(FeatureConfig(), device=torch.device("cpu"))
    bars = [
        MarketBar(
            symbol="AAPL",
            timestamp=datetime(2026, 7, 1, 14, 30, tzinfo=UTC) + timedelta(minutes=5 * i),
            open=100.0 + 0.3 * i,
            high=100.5 + 0.3 * i,
            low=99.5 + 0.3 * i,
            close=100.2 + 0.3 * i,
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(80)
    ]
    snapshot = pipeline.build(bars, symbol="AAPL")
    return pipeline, snapshot


def _build_model(**overrides) -> BrainTradingModel:
    config = ModelConfig(
        soma_hidden_dim=32,
        head_hidden_dim=16,
        use_cnn_encoder=False,
        **overrides,
    )
    return BrainTradingModel(branch_input_dims=_BRANCH_DIMS, config=config)


def _forward(model: BrainTradingModel, batch_size: int = 4):
    features = torch.randn(batch_size, _FLAT_DIM)
    return model(
        branch_inputs={},
        regime_input=torch.zeros(batch_size, 4),
        full_features=features,
    )


# ---------------------------------------------------------------------------
# Architecture: legacy config → legacy params; decoupled config → new params
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_legacy_defaults_reconstruct_old_architecture(self) -> None:
        """A checkpoint saved before this change carries no new config keys;
        the defaults must rebuild the exact old parameter set or every old
        artifact fails strict load_state_dict."""
        model = _build_model()
        keys = set(model.state_dict().keys())
        assert "policy_head.confidence.weight" in keys
        assert "policy_head.target_fraction.weight" in keys
        assert not any(k.startswith("return_heads.") for k in keys)

    def test_decoupled_config_builds_new_architecture(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[5, 15, 30])
        keys = set(model.state_dict().keys())
        assert "policy_head.confidence.weight" not in keys
        assert "policy_head.target_fraction.weight" not in keys
        assert "return_heads.heads.return_h5.0.weight" in keys
        assert "return_heads.heads.return_h15.0.weight" in keys
        assert "return_heads.heads.return_h30.0.weight" in keys

    def test_legacy_state_dict_round_trips(self) -> None:
        saved = _build_model().state_dict()
        _build_model().load_state_dict(saved)  # strict — raises on mismatch

    def test_decoupled_state_dict_round_trips(self) -> None:
        kwargs = {"decoupled_heads": True, "return_head_horizons": [5, 15]}
        saved = _build_model(**kwargs).state_dict()
        _build_model(**kwargs).load_state_dict(saved)


# ---------------------------------------------------------------------------
# Forward semantics of the decoupled heads
# ---------------------------------------------------------------------------


class TestDecoupledForward:
    def test_confidence_is_probability_of_chosen_class(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[15])
        model.eval()
        output = _forward(model)
        expected = torch.softmax(output.action_logits, dim=-1).amax(dim=-1, keepdim=True)
        assert torch.allclose(output.confidence, expected)
        # A softmax max over 3 classes lives in [1/3, 1].
        assert output.confidence.min().item() >= 1.0 / 3.0 - 1e-6
        assert output.confidence.max().item() <= 1.0 + 1e-6

    def test_target_fraction_is_not_learned(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[15])
        output = _forward(model)
        assert torch.count_nonzero(output.target_fraction) == 0

    def test_horizon_returns_keys_match_config(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[30, 5, 15])
        output = _forward(model, batch_size=2)
        assert set(output.horizon_returns.keys()) == {"return_h5", "return_h15", "return_h30"}
        for tensor in output.horizon_returns.values():
            assert tensor.shape == (2, 1)

    def test_legacy_forward_unchanged(self) -> None:
        model = _build_model()
        output = _forward(model)
        assert output.horizon_returns is None
        # Legacy confidence is a sigmoid of a learned head, not the softmax max.
        softmax_max = torch.softmax(output.action_logits, dim=-1).amax(dim=-1, keepdim=True)
        assert not torch.allclose(output.confidence, softmax_max)


# ---------------------------------------------------------------------------
# Loss routing: the entanglements are actually dead
# ---------------------------------------------------------------------------


class TestLossRouting:
    def test_per_horizon_targets_train_their_own_heads(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[5, 15])
        output = _forward(model, batch_size=3)
        targets = {
            "return_h5": torch.zeros(3),
            "return_h15": torch.zeros(3),
        }
        loss, components = future_return_regression_loss(output, targets)
        loss.backward()
        # Gradients land in the per-horizon heads, not in the policy head.
        h5_grad = model.return_heads.heads["return_h5"][0].weight.grad
        assert h5_grad is not None and torch.count_nonzero(h5_grad) > 0
        assert model.policy_head.action.weight.grad is None
        assert "return_h5" in components and "return_h15" in components

    def test_target_without_head_is_surfaced_not_trained(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[5])
        output = _forward(model)
        loss, components = future_return_regression_loss(
            output, {"return_h99": torch.zeros(4)}
        )
        assert float(loss) == 0.0
        assert "return_h99_no_head" in components

    def test_legacy_models_keep_target_fraction_path(self) -> None:
        model = _build_model()
        output = _forward(model)
        loss, components = future_return_regression_loss(
            output, {"return_h15": torch.zeros(4)}
        )
        loss.backward()
        assert model.policy_head.target_fraction.weight.grad is not None
        assert "return_h15" in components

    def test_stability_no_longer_trains_confidence(self) -> None:
        """The defining entanglement: stability regressed onto the same
        scalar the calibrator treats as win probability."""
        model = _build_model(decoupled_heads=True, return_head_horizons=[15])
        output = _forward(model)
        loss, components = multitask_regression_loss(
            output,
            {"stability": torch.zeros(4), "primary_return": torch.zeros(4)},
            decoupled=True,
        )
        assert float(loss) == 0.0
        assert "stability" not in components
        assert "primary_return" not in components
        # In the decoupled model, confidence is derived from the logits — if
        # stability WERE regressed onto it, gradients would corrupt the
        # classifier itself.
        assert model.policy_head.action.weight.grad is None

    def test_drawdown_keeps_its_dedicated_value_head(self) -> None:
        model = _build_model(decoupled_heads=True, return_head_horizons=[15])
        output = _forward(model)
        loss, components = multitask_regression_loss(
            output, {"drawdown": torch.zeros(4)}, decoupled=True
        )
        assert "drawdown" in components
        loss.backward()
        assert model.value_head.network[0].weight.grad is not None

    def test_risk_path_fallbacks_disabled_when_decoupled(self) -> None:
        model = _build_model(
            decoupled_heads=True, return_head_horizons=[15], separate_risk_heads=False
        )
        output = _forward(model)
        assert output.risk_predictions is None
        loss, components = risk_path_regression_loss(
            output,
            {"mfe": torch.zeros(4), "mae": torch.zeros(4)},
            decoupled=True,
        )
        # No dedicated risk head → nothing trains; the old fallback would
        # have regressed mfe onto confidence and mae onto value.
        assert float(loss) == 0.0
        assert not components


# ---------------------------------------------------------------------------
# Checkpoint round trip through the factory (save → reconstruct → load)
# ---------------------------------------------------------------------------


class TestFactoryRoundTrip:
    def test_decoupled_checkpoint_reconstructs_and_matches(self, tmp_path) -> None:
        from src.models.factory import build_checkpoint_meta, load_model_artifacts

        _, snapshot = _synthetic_snapshot()
        contract = snapshot.contract
        config = ModelConfig(
            decoupled_heads=True,
            return_head_horizons=[5, 15],
            use_cnn_encoder=False,
        )
        from src.models.factory import build_model_from_contract

        model = build_model_from_contract(
            model_kind="brain", model_config=config, feature_contract=contract
        )
        model.eval()
        meta = build_checkpoint_meta(
            model_kind="brain",
            model_config=config,
            feature_config=FeatureConfig(),
            feature_contract=contract,
        )
        path = tmp_path / "brain.pt"
        torch.save({"model_state": model.state_dict(), "meta": meta}, path)

        artifacts = load_model_artifacts(path, device=torch.device("cpu"))
        assert getattr(artifacts.model, "decoupled_heads") is True
        assert artifacts.model.return_heads is not None

        features = torch.randn(2, contract.feature_dim)
        original = model(branch_inputs={}, regime_input=torch.zeros(2, contract.regime_dim), full_features=features)
        reloaded = artifacts.model(
            branch_inputs={}, regime_input=torch.zeros(2, contract.regime_dim), full_features=features
        )
        assert torch.allclose(original.action_logits, reloaded.action_logits)
        assert torch.allclose(original.confidence, reloaded.confidence)
        for name in original.horizon_returns:
            assert torch.allclose(original.horizon_returns[name], reloaded.horizon_returns[name])

    def test_pre_phase2_checkpoint_meta_still_loads(self, tmp_path) -> None:
        """Simulate a checkpoint saved BEFORE this change: its stored
        model_config has no decoupled_heads / return_head_horizons keys."""
        from src.models.factory import (
            build_checkpoint_meta,
            build_model_from_contract,
            load_model_artifacts,
        )

        _, snapshot = _synthetic_snapshot()
        contract = snapshot.contract
        config = ModelConfig(use_cnn_encoder=False)
        model = build_model_from_contract(
            model_kind="brain", model_config=config, feature_contract=contract
        )
        meta = build_checkpoint_meta(
            model_kind="brain",
            model_config=config,
            feature_config=FeatureConfig(),
            feature_contract=contract,
        )
        # Strip the new keys the way an old artifact would lack them.
        for key in ("decoupled_heads", "return_head_horizons"):
            meta["model_config"].pop(key, None)
        path = tmp_path / "old_brain.pt"
        torch.save({"model_state": model.state_dict(), "meta": meta}, path)

        artifacts = load_model_artifacts(path, device=torch.device("cpu"))
        assert getattr(artifacts.model, "decoupled_heads") is False
        assert artifacts.model.return_heads is None
        keys = set(artifacts.model.state_dict().keys())
        assert "policy_head.confidence.weight" in keys


# ---------------------------------------------------------------------------
# Policy integration: sizing separation at the proposal level
# ---------------------------------------------------------------------------


def test_policy_uses_default_fraction_and_honest_confidence() -> None:
    from src.strategies.brain_policy import BrainPolicy

    torch.manual_seed(7)
    _, snapshot = _synthetic_snapshot()
    contract = snapshot.contract
    config = ModelConfig(
        decoupled_heads=True,
        return_head_horizons=[15],
        use_cnn_encoder=False,
    )
    from src.models.factory import build_model_from_contract

    model = build_model_from_contract(
        model_kind="brain", model_config=config, feature_contract=contract
    )
    model.eval()
    policy = BrainPolicy(model=model, device=torch.device("cpu"), feature_contract=contract)
    proposal, _diagnostics = policy.propose(snapshot)
    # Honest confidence: a 3-class softmax max.
    assert 1.0 / 3.0 - 1e-6 <= proposal.confidence <= 1.0
    # Sizing separation: the model emits zero target_fraction, so a
    # directional proposal falls back to the policy default; flat is zero.
    if proposal.action == 0:
        assert proposal.target_fraction == 0.0
    else:
        assert abs(proposal.target_fraction) == policy.default_target_fraction
