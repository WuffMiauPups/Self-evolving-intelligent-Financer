from __future__ import annotations

import torch

from src.training.class_balance import build_class_weights, build_weighted_sampler, compute_target_diagnostics
from src.training.losses import direction_classification_loss
from src.training.metrics import build_collapse_warnings, compute_classification_report, compute_logit_diagnostics
from src.data.dataset import TrainingSample


def _sample(label: int, direction_name: str, primary_return: float, *, symbol: str = "AAPL") -> TrainingSample:
    return TrainingSample(
        features=torch.zeros(2),
        branch_inputs={"trend": torch.zeros(1)},
        regime_input=torch.zeros(1),
        temporal_tensor=torch.zeros(2, 4),
        label=torch.tensor(label, dtype=torch.long),
        symbol=symbol,
        direction_name=direction_name,
        regression_targets={"primary_return": primary_return},
        future_returns={5: primary_return},
    )


def test_classification_report_exposes_per_class_behavior() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.5, -1.0],
            [0.2, 1.5, 0.1],
            [-1.0, 0.2, 2.0],
            [1.2, 0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    report = compute_classification_report(logits=logits, labels=labels)
    assert report["per_class"]["long"]["recall"] < 1.0
    assert report["balanced_accuracy"] > 0.0
    assert report["macro_f1"] > 0.0
    assert report["confusion_matrix"][2][0] == 1


def test_logit_diagnostics_surface_long_noncompetitive_behavior() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],
            [2.5, 0.8, -1.5],
            [2.8, 0.9, -1.8],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([2, 2, 2], dtype=torch.long)
    diagnostics = compute_logit_diagnostics(logits=logits, labels=labels)
    assert diagnostics["long_noncompetitive_rate"] > 0.9
    assert "long" in diagnostics["logit_summary"]


def test_collapse_warnings_trigger_on_long_prediction_floor() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [2.1, 0.3, -0.8],
            [1.9, 0.4, -0.7],
            [2.2, 0.1, -1.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([2, 2, 0, 1], dtype=torch.long)
    report = compute_classification_report(logits=logits, labels=labels)
    warnings = build_collapse_warnings(
        classification_report=report,
        thresholds={"long_prediction_floor": 0.10, "minority_recall_floor": 0.20},
    )
    assert any("long_prediction_collapse_warning" in warning for warning in warnings)


def test_effective_number_class_weights_and_weighted_sampler_work() -> None:
    samples = [_sample(0, "short", -0.01) for _ in range(8)] + [_sample(2, "long", 0.02) for _ in range(2)]
    weights = build_class_weights(samples, mode="effective_number", cap=2.5)
    sampler = build_weighted_sampler(samples, mode="effective_number")
    assert weights is not None
    assert float(weights[2].item()) > float(weights[0].item())
    assert sampler is not None


def test_inverse_frequency_sampler_mode_is_supported() -> None:
    samples = [_sample(0, "short", -0.01) for _ in range(6)] + [_sample(2, "long", 0.02) for _ in range(2)]
    sampler = build_weighted_sampler(samples, mode="inverse_frequency")
    assert sampler is not None


def test_focal_loss_path_runs() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 1.0, -0.1]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = direction_classification_loss(
        logits,
        labels,
        loss_mode="focal",
        focal_gamma=1.5,
        label_smoothing=0.01,
    )
    assert float(loss.item()) > 0.0


def test_target_diagnostics_report_boundary_heaviness() -> None:
    samples = [
        _sample(1, "flat", 0.0004),
        _sample(1, "flat", -0.0003),
        _sample(2, "long", 0.0006),
        _sample(0, "short", -0.0006),
    ]
    diagnostics = compute_target_diagnostics(
        samples,
        split="train",
        minority_class_floor_warning=0.05,
        label_epsilon=0.0005,
        inspect_target_boundaries=True,
    )
    assert diagnostics.near_boundary_rate > 0.0
    assert diagnostics.primary_return_quantiles
