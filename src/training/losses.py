from __future__ import annotations

import torch
import torch.nn.functional as F


def direction_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    loss_mode: str = "cross_entropy",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    weights = None if class_weights is None else class_weights.to(device=logits.device, dtype=logits.dtype)
    if loss_mode == "cross_entropy":
        return F.cross_entropy(logits, labels, weight=weights, label_smoothing=label_smoothing)
    if loss_mode == "focal":
        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        probabilities = torch.softmax(logits, dim=-1)
        pt = probabilities.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).clamp_min(1.0e-8)
        focal_factor = (1.0 - pt) ** focal_gamma
        return (focal_factor * ce_loss).mean()
    raise ValueError(f"Unsupported classification loss mode: {loss_mode}")


def path_label_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int = 5,
    loss_mode: str = "cross_entropy",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Classification loss for path labels (breakout_up/down, mean_reversion_up/down, chop).

    Uses the same loss modes as direction classification. No class weights
    by default since path labels have a natural distribution that should be
    learned as-is.
    """
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.new_zeros(())
    if loss_mode == "cross_entropy":
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    if loss_mode == "focal":
        ce_loss = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
        probabilities = torch.softmax(logits, dim=-1)
        pt = probabilities.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).clamp_min(1.0e-8)
        focal_factor = (1.0 - pt) ** focal_gamma
        return (focal_factor * ce_loss).mean()
    raise ValueError(f"Unsupported path label loss mode: {loss_mode}")


def future_return_regression_loss(
    output: object,
    regression_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression loss specifically for per-horizon future return targets.

    Targets have names like 'return_h5', 'return_h15', 'return_h30', etc.
    Uses the model's target_fraction output as the prediction head.
    """
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    for name, target in regression_targets.items():
        if not name.startswith("return_h"):
            continue
        # Use target_fraction as the primary return prediction
        prediction = output.target_fraction.reshape(-1)
        target_tensor = target.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
        loss = F.smooth_l1_loss(prediction, target_tensor)
        losses.append(loss)
        components[name] = float(loss.item())
    if not losses:
        zero = output.action_logits.new_zeros(())
        return zero, components
    total = torch.stack(losses).mean()
    components["future_return_total"] = float(total.item())
    return total, components


def risk_path_regression_loss(
    output: object,
    regression_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression loss for risk/path statistics: MFE, MAE, volatility, drawdown.

    Uses the model's value and confidence heads as prediction outputs.
    """
    risk_target_names = {"mfe", "mae", "volatility", "path_drawdown"}
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    for name in risk_target_names:
        if name not in regression_targets:
            continue
        target = regression_targets[name]
        # Map risk targets to model output heads
        if name in ("mae", "path_drawdown") and output.value is not None:
            prediction = output.value.reshape(-1)
        elif name in ("mfe", "volatility"):
            prediction = output.confidence.reshape(-1)
        else:
            continue
        target_tensor = target.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
        loss = F.smooth_l1_loss(prediction, target_tensor)
        losses.append(loss)
        components[name] = float(loss.item())
    if not losses:
        zero = output.action_logits.new_zeros(())
        return zero, components
    total = torch.stack(losses).mean()
    components["risk_path_total"] = float(total.item())
    return total, components


def feature_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def multitask_regression_loss(
    output: object,
    regression_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    if "primary_return" in regression_targets:
        target = regression_targets["primary_return"].to(
            device=output.target_fraction.device,
            dtype=output.target_fraction.dtype,
        )
        loss = F.smooth_l1_loss(output.target_fraction.reshape(-1), target.reshape(-1))
        losses.append(loss)
        components["primary_return"] = float(loss.item())
    if "drawdown" in regression_targets and output.value is not None:
        target = regression_targets["drawdown"].to(device=output.value.device, dtype=output.value.dtype)
        loss = F.smooth_l1_loss(output.value.reshape(-1), target.reshape(-1))
        losses.append(loss)
        components["drawdown"] = float(loss.item())
    if "stability" in regression_targets:
        target = regression_targets["stability"].to(device=output.confidence.device, dtype=output.confidence.dtype)
        loss = F.smooth_l1_loss(output.confidence.reshape(-1), target.reshape(-1))
        losses.append(loss)
        components["stability"] = float(loss.item())
    if not losses:
        zero = output.action_logits.new_zeros(())
        return zero, components
    total = torch.stack(losses).mean()
    components["regression_total"] = float(total.item())
    return total, components


def latent_consistency_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    normalized = F.normalize(embeddings, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive_mask = label_matrix & ~eye
    negative_mask = ~label_matrix
    positive_loss = (
        (1.0 - similarity[positive_mask]).mean()
        if positive_mask.any()
        else similarity.new_zeros(())
    )
    negative_loss = (
        F.relu(similarity[negative_mask] - margin).mean()
        if negative_mask.any()
        else similarity.new_zeros(())
    )
    return positive_loss + negative_loss
