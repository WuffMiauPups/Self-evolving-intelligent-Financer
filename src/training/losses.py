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


def future_return_regression_loss(
    output: object,
    regression_targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression loss specifically for per-horizon future return targets.

    Targets have names like 'return_h5', 'return_h15', 'return_h30', etc.
    With per-horizon return heads (Phase 2 §6.1) each target trains its own
    head; a single scalar cannot fit several horizons at once. Legacy models
    without the heads fall back to the old (entangled) target_fraction path.
    """
    horizon_returns = getattr(output, "horizon_returns", None)
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    for name, target in regression_targets.items():
        if not name.startswith("return_h"):
            continue
        if horizon_returns is not None:
            head_prediction = horizon_returns.get(name)
            if head_prediction is None:
                # A target without a matching head trains nothing; surface
                # it in components so a horizon mismatch is visible.
                components[f"{name}_no_head"] = 0.0
                continue
            prediction = head_prediction.reshape(-1)
        else:
            # Legacy: use target_fraction as the (shared) return prediction
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
    *,
    decoupled: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression loss for risk/path statistics: MFE, MAE, volatility, drawdown.

    Uses dedicated risk heads when available, otherwise falls back to the
    model's value and confidence heads as prediction outputs. With
    ``decoupled=True`` (Phase 2 §6.1) the fallbacks are disabled: the
    derived confidence is a softmax probability, so regressing a risk
    statistic onto it would push gradients into the action logits and
    re-entangle the classifier with risk prediction.
    """
    risk_target_names = {"mfe", "mae", "volatility", "path_drawdown"}
    risk_preds = getattr(output, "risk_predictions", None)
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    for name in risk_target_names:
        if name not in regression_targets:
            continue
        target = regression_targets[name]
        # Use dedicated risk heads when available
        if risk_preds is not None:
            prediction = risk_preds[name].reshape(-1)
        elif decoupled:
            continue
        elif name in ("mae", "path_drawdown") and output.value is not None:
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
    *,
    decoupled: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Legacy multitask regression terms.

    With ``decoupled=True`` (Phase 2 §6.1) the two entangling terms are
    disabled: ``primary_return`` no longer trains ``target_fraction`` (the
    per-horizon return heads cover returns and sizing is not learned), and
    ``stability`` no longer trains ``confidence`` (the derived confidence
    is P(chosen class); regressing a stability statistic onto it was the
    reason calibrated confidence never approximated win probability).
    ``drawdown`` keeps its own dedicated value head in both modes.
    """
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    if "primary_return" in regression_targets and not decoupled:
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
    if "stability" in regression_targets and not decoupled:
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


# latent_consistency_loss and soma_gating_confidence_loss were removed in
# the Great Simplification (Phase 8). The former was a metric-learning loss
# with no downstream consumer; the latter was a regularizer for the soma
# aggregator which no longer exists.
