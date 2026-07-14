"""Trainer hygiene regression tests (MASTER_PLAN §4.7).

Pins the five trainer-hygiene items the 2026-07-14 plan audit found
missing despite the "Phase 1 code-complete" banner:

1. validation/test epochs run under ``torch.no_grad`` (no autograd graph);
2. ``ModelStateEncoder.encode_batch`` forces eval-mode normalization and
   restores the caller's mode;
3. the training DataLoader drops a would-be size-1 remainder batch
   (BatchNorm guard) without starving single-batch smoke datasets;
4. the training seed comes from config, not a hardcoded literal;
5. collate-time ``nan_to_num`` sanitation is surfaced with counts, not silent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.config import load_config
from src.core.config.training import TrainingConfig
from src.data.dataset import count_nonfinite_columns, make_columnar_collate
from src.features.feature_pipeline import FeatureSnapshot
from src.models.base import ModelOutput
from src.models.state_encoder import ModelStateEncoder
from src.training.data_module import train_drop_last
from src.training.sample_columns import ColumnarSamples, IndexDataset
from src.training.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Validation epochs must not build autograd graphs
# ---------------------------------------------------------------------------


@dataclass
class _StubOutput:
    action_logits: torch.Tensor


class _GradRecordingModel(nn.Module):
    """Records grad-enabled state and train/eval mode at forward time."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 3)
        self.grad_enabled_seen: list[bool] = []
        self.training_mode_seen: list[bool] = []

    def forward(self, *, branch_inputs, regime_input, full_features, temporal_tensor=None):
        self.grad_enabled_seen.append(torch.is_grad_enabled())
        self.training_mode_seen.append(self.training)
        return _StubOutput(action_logits=self.linear(full_features))


def _batch(feature_dim: int, batch_size: int = 8) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "features": torch.randn(batch_size, feature_dim, generator=g),
        "branch_inputs": {},
        "regime_input": torch.zeros(batch_size, 2),
        "label": torch.randint(0, 3, (batch_size,), generator=g),
    }


def _trainer(model: nn.Module) -> Trainer:
    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
        use_amp=False,
        multitask_enabled=False,
        augmentation_enabled=False,
    )


def test_val_epoch_runs_without_grad_and_in_eval_mode() -> None:
    feature_dim = 5
    model = _GradRecordingModel(feature_dim)
    trainer = _trainer(model)
    trainer._run_epoch([_batch(feature_dim)], training=False, stage="val")
    assert model.grad_enabled_seen == [False]
    assert model.training_mode_seen == [False]


def test_train_epoch_keeps_grad_enabled() -> None:
    feature_dim = 5
    model = _GradRecordingModel(feature_dim)
    trainer = _trainer(model)
    metrics = trainer._run_epoch([_batch(feature_dim)], training=True, stage="train")
    assert model.grad_enabled_seen == [True]
    assert model.training_mode_seen == [True]
    assert metrics.loss > 0.0  # backward path actually ran


# ---------------------------------------------------------------------------
# 2. encode_batch: eval-mode forward, caller's mode restored
# ---------------------------------------------------------------------------

F_DIM, R_DIM, C_DIM, T_DIM, D_DIM = 7, 4, 3, 12, 5
BRANCHES = ("trend", "volatility")


class _ModeRecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # A parameter so .train()/.eval() are meaningful on a real module.
        self.dummy = nn.Linear(1, 1)
        self.training_mode_seen: list[bool] = []

    def forward(
        self,
        *,
        branch_inputs,
        regime_input,
        memory_state=None,
        full_features=None,
        temporal_tensor=None,
    ) -> ModelOutput:
        self.training_mode_seen.append(self.training)
        batch = regime_input.shape[0]
        return ModelOutput(
            action_logits=torch.tensor([[0.1, 0.9, 0.2]]).repeat(batch, 1),
            confidence=torch.full((batch, 1), 0.7),
            target_fraction=torch.full((batch, 1), 0.05),
            fused_representation=torch.zeros(batch, 8),
            branch_embeddings={},
            branch_confidences={},
            soma_weights=torch.zeros(batch, 1),
            soma_activation=torch.full((batch, 1), 1.0),
        )


def _snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 7, 14, 14, 30, tzinfo=UTC),
        last_price=100.0,
        spread_bps=1.0,
        features=torch.full((F_DIM,), 7.0),
        branch_inputs={name: torch.full((D_DIM,), 3.0) for name in BRANCHES},
        regime_input=torch.full((R_DIM,), -1.0),
        temporal_tensor=torch.full((C_DIM, T_DIM), 2e-3),
        feature_names=[f"f{i}" for i in range(F_DIM)],
        branch_feature_names={name: [f"{name}{i}" for i in range(D_DIM)] for name in BRANCHES},
        regime_feature_names=[f"r{i}" for i in range(R_DIM)],
        temporal_feature_names=[f"t{i}" for i in range(C_DIM)],
    )


def test_encode_batch_forces_eval_and_restores_training_mode() -> None:
    model = _ModeRecordingModel()
    model.train(True)
    encoder = ModelStateEncoder(model=model, device=torch.device("cpu"))
    encoder.encode_batch([_snapshot()])
    assert model.training_mode_seen == [False], "forward must run in eval mode"
    assert model.training is True, "caller's train mode must be restored"


def test_encode_batch_leaves_eval_model_in_eval() -> None:
    model = _ModeRecordingModel()
    model.eval()
    encoder = ModelStateEncoder(model=model, device=torch.device("cpu"))
    encoder.encode_batch([_snapshot()])
    assert model.training_mode_seen == [False]
    assert model.training is False


# ---------------------------------------------------------------------------
# 3. BatchNorm guard: drop_last on the training loader
# ---------------------------------------------------------------------------


def test_train_drop_last_decision() -> None:
    # Multi-batch datasets drop the remainder (guards size-1 batches).
    assert train_drop_last(sample_count=385, batch_size=384) is True
    assert train_drop_last(sample_count=769, batch_size=384) is True
    # Single-batch (smoke/fast-mode) datasets keep everything: drop_last
    # would leave zero batches and silently zero the epoch metrics.
    assert train_drop_last(sample_count=384, batch_size=384) is False
    assert train_drop_last(sample_count=10, batch_size=384) is False


def test_train_loader_never_yields_size_one_batch() -> None:
    # 9 samples, batch 4: without drop_last the final batch has size 1 —
    # exactly the BatchNorm failure mode. Mirror the builder's wiring.
    sample_count, batch_size = 9, 4
    loader = DataLoader(
        IndexDataset(sample_count),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda indices: list(indices),
        drop_last=train_drop_last(sample_count, batch_size),
    )
    batch_sizes = [len(batch) for batch in loader]
    assert batch_sizes == [4, 4]
    # Validation loaders keep every sample (eval-mode BatchNorm is safe).
    val_loader = DataLoader(
        IndexDataset(sample_count),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda indices: list(indices),
    )
    assert sum(len(batch) for batch in val_loader) == sample_count


# ---------------------------------------------------------------------------
# 4. Config-driven training seed
# ---------------------------------------------------------------------------


def test_training_seed_is_config_driven() -> None:
    assert TrainingConfig().seed == 7
    config = load_config("configs/default.yaml")
    assert config.training.seed == 7
    # Source contract: build_training_components must read the config
    # seed; the historical hardcoded literal must be gone (it zeroed
    # seed variance for every sweep routed through it).
    source = (REPO_ROOT / "src" / "training" / "supervised.py").read_text(encoding="utf-8")
    assert "seed_everything(config.training.seed)" in source
    assert "seed_everything(7)" not in source


# ---------------------------------------------------------------------------
# 5. Collate nan_to_num sanitation is surfaced
# ---------------------------------------------------------------------------


def _columns_with(n: int = 4, *, poison: bool) -> ColumnarSamples:
    features = torch.ones(n, F_DIM)
    regression = {"primary_return": torch.zeros(n)}
    if poison:
        features[0, 0] = float("nan")
        features[1, 2] = float("inf")
        regression["primary_return"][3] = float("-inf")
    return ColumnarSamples(
        features=features,
        label=torch.zeros(n, dtype=torch.long),
        regime_input=torch.zeros(n, R_DIM),
        temporal_tensor=torch.zeros(n, C_DIM, T_DIM),
        branch_inputs={name: torch.zeros(n, D_DIM) for name in BRANCHES},
        regression_targets=regression,
        symbols=["AAPL"] * n,
        timestamps=[datetime(2026, 7, 14, tzinfo=UTC)] * n,
        direction_names=["flat"] * n,
        path_names=[""] * n,
    )


def test_count_nonfinite_columns_reports_exact_counts() -> None:
    counts = count_nonfinite_columns(_columns_with(poison=True))
    assert counts == {"features": 2, "regression_targets.primary_return": 1}
    assert count_nonfinite_columns(_columns_with(poison=False)) == {}


def test_collate_surfaces_nonfinite_counts_and_sanitizes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.data.dataset"):
        collate = make_columnar_collate(_columns_with(poison=True))
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("collate_nonfinite_inputs" in message for message in warnings)
    batch = collate([0, 1, 2, 3])
    assert torch.isfinite(batch["features"]).all()
    assert torch.isfinite(batch["regression_targets"]["primary_return"]).all()


def test_collate_stays_quiet_on_clean_columns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.data.dataset"):
        make_columnar_collate(_columns_with(poison=False))
    assert not [r for r in caplog.records if "collate_nonfinite_inputs" in r.getMessage()]
