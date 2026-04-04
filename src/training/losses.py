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
