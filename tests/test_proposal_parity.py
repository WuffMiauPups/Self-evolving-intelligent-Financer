"""Backtest-vs-runtime proposal parity + EV-gate fail-fast (§4.1.4, §4.4.4).

Parity: the same checkpoint + the same snapshot must yield an IDENTICAL
proposal whether the policy was constructed through the backtest factory
(``policy_factory.build_brain_policy``) or the runtime bootstrap
(``runtime_bootstrap.build_runtime_brain_policy``). The flip-flag
retirement (§4.1.4) promised this test; the audit found it missing.

Fail-fast: a model without a risk head feeds (0, 0) into the EV gate,
which then rejects every trade — a silent zero-trade session. Production
construction paths must refuse such a model unless
``risk.ev_gate_requires_signals`` is explicitly disabled.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.backtest.policy_factory as policy_factory
from src.core.config import load_config
from src.core.exceptions import ConfigurationError
from src.execution.runtime_bootstrap import build_runtime_brain_policy
from src.features.feature_pipeline import FeatureSnapshot
from src.models.base import ModelOutput
from src.strategies.brain_policy import BrainPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]

F_DIM, R_DIM, C_DIM, T_DIM, D_DIM = 7, 4, 3, 12, 5
BRANCHES = ("trend", "volatility")


class _StubBrainModel:
    """Deterministic model with a risk head; emits target_fraction=0 so the
    policy's ``default_target_fraction`` path is exercised (a parity-relevant
    config field the factory used to drop)."""

    risk_head = object()

    def __call__(
        self,
        *,
        branch_inputs,
        regime_input,
        memory_state=None,
        full_features=None,
        temporal_tensor=None,
    ) -> ModelOutput:
        batch = regime_input.shape[0]
        return ModelOutput(
            action_logits=torch.tensor([[0.1, 0.2, 1.5]]).repeat(batch, 1),
            confidence=torch.full((batch, 1), 0.7),
            target_fraction=torch.zeros(batch, 1),
            fused_representation=torch.zeros(batch, 8),
            branch_embeddings={},
            branch_confidences={},
            soma_weights=torch.zeros(batch, 1),
            soma_activation=torch.full((batch, 1), 1.0),
            risk_predictions={
                "mfe": torch.full((batch, 1), 0.0020),
                "mae": torch.full((batch, 1), -0.0010),
            },
        )


def _snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 7, 14, 14, 30, tzinfo=UTC),
        last_price=100.0,
        spread_bps=1.0,
        features=torch.full((F_DIM,), 0.5),
        branch_inputs={name: torch.full((D_DIM,), 0.25) for name in BRANCHES},
        regime_input=torch.full((R_DIM,), -0.5),
        temporal_tensor=torch.full((C_DIM, T_DIM), 1e-3),
        feature_names=[f"f{i}" for i in range(F_DIM)],
        branch_feature_names={name: [f"{name}{i}" for i in range(D_DIM)] for name in BRANCHES},
        regime_feature_names=[f"r{i}" for i in range(R_DIM)],
        temporal_feature_names=[f"t{i}" for i in range(C_DIM)],
    )


def _fake_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        model=_StubBrainModel(),
        feature_contract=None,
        normalization=None,
        model_kind="brain",
        encoder_fingerprint="stub",
    )


def test_same_checkpoint_same_snapshot_identical_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/default.yaml")
    config.device.preferred = "cpu"
    # Non-default values prove the fields are actually threaded on BOTH paths.
    config.runtime.default_target_fraction = 0.07
    config.runtime.state_reset_gap_sec = 123

    artifacts = _fake_artifacts()
    monkeypatch.setattr(policy_factory, "_resolve_artifacts", lambda *a, **k: (artifacts, None))
    backtest_policy = policy_factory.build_brain_policy(
        config, bars_by_symbol={"AAPL": []}, pipeline=None
    )
    runtime_policy = build_runtime_brain_policy(config, artifacts, torch.device("cpu"))

    # Construction parity: every config-driven field identical.
    for field_name in (
        "force_directional_flip",
        "default_target_fraction",
        "max_state_gap_sec",
        "ev_gate_requires_signals",
        "normalization",
    ):
        assert getattr(backtest_policy, field_name) == getattr(runtime_policy, field_name), field_name

    # Behavioral parity: identical proposal for the identical snapshot.
    proposal_a, _ = backtest_policy.propose(_snapshot())
    proposal_b, _ = runtime_policy.propose(_snapshot())
    assert proposal_a.action == proposal_b.action
    assert proposal_a.confidence == pytest.approx(proposal_b.confidence)
    assert proposal_a.target_fraction == pytest.approx(proposal_b.target_fraction)
    assert proposal_a.expected_gain_bps == pytest.approx(proposal_b.expected_gain_bps)
    assert proposal_a.expected_loss_bps == pytest.approx(proposal_b.expected_loss_bps)
    assert proposal_a.strategy_id == proposal_b.strategy_id
    # The zero target_fraction from the model must resolve to the CONFIG
    # default fraction (long => positive) on both paths.
    assert proposal_a.target_fraction == pytest.approx(0.07)


# ── §4.4.4: EV-gate fail-fast at construction ───────────────────────────


class _NoRiskHeadModel:
    def __call__(self, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("construction must fail before any forward")


def test_risk_head_less_model_raises_at_construction() -> None:
    with pytest.raises(ConfigurationError, match="risk head"):
        BrainPolicy(
            model=_NoRiskHeadModel(),
            device=torch.device("cpu"),
            ev_gate_requires_signals=True,
        )


def test_bypass_flag_accepts_risk_head_less_model() -> None:
    policy = BrainPolicy(
        model=_NoRiskHeadModel(),
        device=torch.device("cpu"),
        ev_gate_requires_signals=False,
    )
    assert policy.ev_gate_requires_signals is False


def test_model_with_risk_head_constructs_under_fail_fast() -> None:
    policy = BrainPolicy(
        model=_StubBrainModel(),
        device=torch.device("cpu"),
        ev_gate_requires_signals=True,
    )
    assert policy.ev_gate_requires_signals is True


def test_production_paths_thread_the_fail_fast_flag() -> None:
    expected = "ev_gate_requires_signals=config.risk.ev_gate_requires_signals"
    for consumer in (
        "src/backtest/policy_factory.py",
        "src/execution/runtime_bootstrap.py",
        "scripts/run_backtest.py",
    ):
        source = (REPO_ROOT / consumer).read_text(encoding="utf-8")
        assert expected in source or expected.replace("config.", "cfg.") in source, (
            f"{consumer} does not thread risk.ev_gate_requires_signals"
        )
    config = load_config("configs/default.yaml")
    assert config.risk.ev_gate_requires_signals is True
