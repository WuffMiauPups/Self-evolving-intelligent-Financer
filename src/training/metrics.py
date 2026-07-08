from __future__ import annotations

from dataclasses import dataclass, field

import torch


CLASS_NAMES: tuple[str, ...] = ("short", "flat", "long")


@dataclass
class BatchMetrics:
    loss: float
    accuracy: float
    classification_loss: float = 0.0
    regression_loss: float = 0.0
    latent_loss: float = 0.0
    future_return_loss: float = 0.0
    risk_path_loss: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    long_recall: float = 0.0
    short_recall: float = 0.0
    prediction_distribution: dict[str, float] = field(default_factory=dict)
    target_distribution: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0


def compute_classification_report(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, object]:
    if logits.numel() == 0 or labels.numel() == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "mcc": 0.0,
            "prediction_counts": {name: 0 for name in class_names},
            "target_counts": {name: 0 for name in class_names},
            "prediction_ratios": {name: 0.0 for name in class_names},
            "target_ratios": {name: 0.0 for name in class_names},
            "per_class": {name: _empty_class_metrics() for name in class_names},
            "confusion_matrix": [[0 for _ in class_names] for _ in class_names],
        }

    predictions = logits.argmax(dim=-1)
    class_count = len(class_names)
    confusion = torch.zeros((class_count, class_count), dtype=torch.float32)
    for label, prediction in zip(labels.detach().cpu(), predictions.detach().cpu()):
        confusion[int(label.item()), int(prediction.item())] += 1.0

    total = float(confusion.sum().item())
    per_class: dict[str, dict[str, float]] = {}
    weighted_f1_sum = 0.0
    recalls: list[float] = []
    f1_values: list[float] = []
    prediction_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for class_index, class_name in enumerate(class_names):
        tp = float(confusion[class_index, class_index].item())
        fp = float(confusion[:, class_index].sum().item() - tp)
        fn = float(confusion[class_index, :].sum().item() - tp)
        tn = float(total - tp - fp - fn)
        support = int(confusion[class_index, :].sum().item())
        predicted = int(confusion[:, class_index].sum().item())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        one_vs_rest_accuracy = (tp + tn) / max(total, 1.0)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "predicted": float(predicted),
            "one_vs_rest_accuracy": one_vs_rest_accuracy,
        }
        prediction_counts[class_name] = predicted
        target_counts[class_name] = support
        recalls.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support

    accuracy = float(confusion.diag().sum().item() / max(total, 1.0))
    balanced_accuracy = sum(recalls) / max(len(recalls), 1)
    macro_f1 = sum(f1_values) / max(len(f1_values), 1)
    weighted_f1 = weighted_f1_sum / max(total, 1.0)
    prediction_ratios = {
        name: prediction_counts[name] / max(total, 1.0)
        for name in class_names
    }
    target_ratios = {
        name: target_counts[name] / max(total, 1.0)
        for name in class_names
    }
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": _multiclass_mcc(confusion),
        "prediction_counts": prediction_counts,
        "target_counts": target_counts,
        "prediction_ratios": prediction_ratios,
        "target_ratios": target_ratios,
        "per_class": per_class,
        "confusion_matrix": confusion.to(dtype=torch.int64).tolist(),
    }


def compute_logit_diagnostics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, object]:
    if logits.numel() == 0 or labels.numel() == 0:
        return {
            "logit_summary": {},
            "probability_summary": {},
            "top_class_margin": {},
            "long_near_miss_rate": 0.0,
            "long_noncompetitive_rate": 0.0,
        }
    probabilities = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probabilities, k=min(2, probabilities.shape[-1]), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else torch.zeros_like(top2[:, 0]))
    predictions = probabilities.argmax(dim=-1)
    logit_summary = {
        class_name: _tensor_summary(logits[:, class_index])
        for class_index, class_name in enumerate(class_names)
    }
    probability_summary = {
        class_name: _tensor_summary(probabilities[:, class_index])
        for class_index, class_name in enumerate(class_names)
    }
    long_index = class_names.index("long") if "long" in class_names else len(class_names) - 1
    long_probabilities = probabilities[:, long_index]
    long_true_mask = labels == long_index
    long_lost_mask = long_true_mask & (predictions != long_index)
    long_near_miss_rate = 0.0
    long_noncompetitive_rate = 0.0
    if long_lost_mask.any():
        top_probs = probabilities.max(dim=-1).values
        long_near_miss_rate = float(
            ((top_probs[long_lost_mask] - long_probabilities[long_lost_mask]) <= 0.05).float().mean().item()
        )
        long_noncompetitive_rate = float(
            (long_probabilities[long_lost_mask] <= 0.10).float().mean().item()
        )
    return {
        "logit_summary": logit_summary,
        "probability_summary": probability_summary,
        "top_class_margin": _tensor_summary(margin),
        "long_near_miss_rate": long_near_miss_rate,
        "long_noncompetitive_rate": long_noncompetitive_rate,
    }


def build_collapse_warnings(
    *,
    classification_report: dict[str, object],
    thresholds: dict[str, float],
    flat_dominance_threshold: float = 0.75,
) -> list[str]:
    warnings: list[str] = []
    prediction_ratios = classification_report["prediction_ratios"]
    target_ratios = classification_report["target_ratios"]
    per_class = classification_report["per_class"]
    long_floor = float(thresholds["long_prediction_floor"])
    minority_recall_floor = float(thresholds["minority_recall_floor"])
    if prediction_ratios.get("long", 0.0) < long_floor:
        warnings.append(f"long_prediction_collapse_warning:{prediction_ratios.get('long', 0.0):.4f}")
    for class_name in ("short", "flat", "long"):
        recall = float(per_class.get(class_name, {}).get("recall", 0.0))
        if recall < minority_recall_floor:
            warnings.append(f"minority_class_recall_warning:{class_name}={recall:.4f}")
    for class_name in CLASS_NAMES:
        gap = abs(prediction_ratios.get(class_name, 0.0) - target_ratios.get(class_name, 0.0))
        if gap > 0.20:
            warnings.append(f"class_distribution_mismatch_warning:{class_name}={gap:.4f}")
    if prediction_ratios.get("flat", 0.0) > flat_dominance_threshold:
        warnings.append(f"flat_dominance_warning:{prediction_ratios.get('flat', 0.0):.4f}")
    return warnings


def _tensor_summary(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().cpu().to(dtype=torch.float32)
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def _multiclass_mcc(confusion: torch.Tensor) -> float:
    total = float(confusion.sum().item())
    if total <= 0.0:
        return 0.0
    trace = float(torch.trace(confusion).item())
    row_sums = confusion.sum(dim=1)
    col_sums = confusion.sum(dim=0)
    numerator = (trace * total) - float((row_sums * col_sums).sum().item())
    denominator_left = (total**2) - float((col_sums**2).sum().item())
    denominator_right = (total**2) - float((row_sums**2).sum().item())
    denominator = max(denominator_left * denominator_right, 0.0) ** 0.5
    if denominator <= 1.0e-12:
        return 0.0
    return numerator / denominator


def _empty_class_metrics() -> dict[str, float]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0.0,
        "predicted": 0.0,
        "one_vs_rest_accuracy": 0.0,
    }
