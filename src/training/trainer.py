from __future__ import annotations

import copy
from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.core.logging import get_logger
from src.core.device import maybe_autocast
from src.models.base import TradingModelProtocol
from src.training.augmentation import TrainingAugmenter
from src.training.losses import (
    direction_classification_loss,
    future_return_regression_loss,
    multitask_regression_loss,
    risk_path_regression_loss,
)
from src.training.metrics import BatchMetrics, build_collapse_warnings, compute_classification_report
from src.utils.torch_utils import move_batch_to_device


def _move_batch_non_blocking(batch, device: torch.device):
    """Like move_batch_to_device but uses non_blocking=True on CUDA so that
    pinned-memory host->device copies overlap with the previous step's kernel.
    Falls back to synchronous copies on CPU-only runs (where non_blocking is a
    no-op anyway)."""
    from collections.abc import Mapping
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {k: _move_batch_non_blocking(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_batch_non_blocking(v, device) for v in batch)
    if isinstance(batch, list):
        return [_move_batch_non_blocking(v, device) for v in batch]
    return batch


@dataclass
class TrainingHistory:
    train: list[BatchMetrics] = field(default_factory=list)
    val: list[BatchMetrics] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: TradingModelProtocol | torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        use_amp: bool,
        class_weights: torch.Tensor | None = None,
        multitask_enabled: bool = True,
        contrastive_enabled: bool = False,
        contrastive_margin: float = 0.1,
        classification_loss_mode: str = "cross_entropy",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        loss_weights: dict[str, float] | None = None,
        collapse_warning_thresholds: dict[str, float] | None = None,
        report_prediction_distribution: bool = True,
        report_per_class_metrics: bool = True,
        # State + future-path config
        enable_future_return_targets: bool = False,
        enable_risk_path_targets: bool = False,
        adaptive_focal_gamma: bool = False,
        focal_gamma_base: float = 1.5,
        focal_gamma_boost: float = 2.0,
        scheduler: object | None = None,
        early_stopping_patience: int = 0,
        restore_best_on_finish: bool = True,
        model_selection_metric: str = "val_loss",
        metrics_flat_dominance_threshold: float = 0.75,
        augmentation_enabled: bool = True,
        noise_std: float = 0.01,
        temporal_jitter_prob: float = 0.0,
        # Forgetting-safe retraining (§4.5.6): anchor to the incumbent.
        anchor_state_dict: dict[str, torch.Tensor] | None = None,
        distill_lambda_kd: float = 0.0,
        proximal_lambda_l2: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.class_weights = class_weights
        self.multitask_enabled = multitask_enabled
        self.contrastive_enabled = contrastive_enabled
        self.contrastive_margin = contrastive_margin
        self.classification_loss_mode = classification_loss_mode
        self.focal_gamma = focal_gamma
        self.adaptive_focal_gamma = adaptive_focal_gamma
        self.focal_gamma_base = focal_gamma_base
        self.focal_gamma_boost = focal_gamma_boost
        self.label_smoothing = label_smoothing
        # Defaults are merged with the caller-supplied dict so new loss terms
        # (like gating_confidence, added 2026-04-07) automatically apply to
        # configs that predate them. Caller overrides win where keys overlap.
        default_loss_weights = {
            "classification": 1.0,
            "regression": 0.35,
            "future_return": 0.20,
            "risk_path": 0.15,
        }
        self.loss_weights = {**default_loss_weights, **dict(loss_weights or {})}
        self.collapse_warning_thresholds = dict(
            collapse_warning_thresholds or {"long_prediction_floor": 0.08, "minority_recall_floor": 0.15}
        )
        self.report_prediction_distribution = report_prediction_distribution
        self.report_per_class_metrics = report_per_class_metrics
        self.enable_future_return_targets = enable_future_return_targets
        self.enable_risk_path_targets = enable_risk_path_targets
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_on_finish = restore_best_on_finish
        self.model_selection_metric = model_selection_metric
        self.metrics_flat_dominance_threshold = metrics_flat_dominance_threshold
        self.augmentation_enabled = augmentation_enabled
        self.augmenter = TrainingAugmenter(
            noise_std=noise_std,
            temporal_jitter_prob=temporal_jitter_prob,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
        # --- Incumbent anchor (forgetting-safe retraining, §4.5.6) ---
        # KD: KL(incumbent || student) on each training batch, incumbent
        # frozen/no-grad.  L2: proximal penalty pulling student params
        # toward the anchor.  Both off (0.0) unless configured.
        self.distill_lambda_kd = float(distill_lambda_kd)
        self.proximal_lambda_l2 = float(proximal_lambda_l2)
        self._anchor_model: torch.nn.Module | None = None
        self._anchor_params: dict[str, torch.Tensor] = {}
        if anchor_state_dict is not None and (self.distill_lambda_kd > 0.0 or self.proximal_lambda_l2 > 0.0):
            if self.proximal_lambda_l2 > 0.0:
                self._anchor_params = {
                    name: tensor.detach().clone().to(device)
                    for name, tensor in anchor_state_dict.items()
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                }
            if self.distill_lambda_kd > 0.0:
                anchor_model = copy.deepcopy(model)
                anchor_model.load_state_dict(anchor_state_dict)
                anchor_model.to(device)
                anchor_model.eval()
                for parameter in anchor_model.parameters():
                    parameter.requires_grad_(False)
                self._anchor_model = anchor_model
        self.logger = get_logger("trainer")

    def _update_focal_gamma(self, val_metrics) -> None:
        """Adapt focal gamma based on minority class recall.

        Formula: gamma = base + boost * (1 - min_class_recall)
        When minority recall is 0% → gamma = base + boost (most aggressive).
        When minority recall is 100% → gamma = base (standard).
        """
        if not self.adaptive_focal_gamma:
            return
        long_recall = getattr(val_metrics, 'long_recall', 0.0) or 0.0
        short_recall = getattr(val_metrics, 'short_recall', 0.0) or 0.0
        min_recall = min(long_recall, short_recall)
        new_gamma = self.focal_gamma_base + self.focal_gamma_boost * (1.0 - min_recall)
        self.logger.info(
            "adaptive_focal_gamma_update",
            old_gamma=self.focal_gamma,
            new_gamma=round(new_gamma, 4),
            min_class_recall=round(min_recall, 4),
            long_recall=round(long_recall, 4),
            short_recall=round(short_recall, 4),
        )
        self.focal_gamma = new_gamma

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
    ) -> TrainingHistory:
        history = TrainingHistory()
        best_score = float("-inf")
        patience_counter = 0
        best_model_state: dict | None = None
        # Epoch-aware samplers (CurriculumSampler, DistributedSampler, ...)
        # need the current epoch to pick their subset/permutation.
        train_sampler = getattr(train_loader, "sampler", None)
        for epoch_index in range(epochs):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch_index)
            self.logger.info("epoch_start", epoch=epoch_index + 1, epochs=epochs)
            train_metrics = self._run_epoch(
                train_loader,
                training=True,
                stage="train",
                epoch_index=epoch_index + 1,
                epoch_count=epochs,
            )
            val_metrics = self._run_epoch(
                val_loader,
                training=False,
                stage="val",
                epoch_index=epoch_index + 1,
                epoch_count=epochs,
            )
            history.train.append(train_metrics)
            history.val.append(val_metrics)
            # --- Best-epoch selection + early stopping ---
            # Selection uses the configured metric (default: balanced
            # accuracy) instead of raw multi-task loss — the loss rewards
            # flat-hedging, which is exactly the degenerate optimum the
            # class-imbalance machinery exists to avoid.
            score = self._selection_score(val_metrics)
            if score > best_score:
                best_score = score
                patience_counter = 0
                if self.restore_best_on_finish:
                    best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    self.logger.info(
                        "early_stopping",
                        epoch=epoch_index + 1,
                        metric=self.model_selection_metric,
                        best_score=best_score,
                    )
                    break
            self._update_focal_gamma(val_metrics)
            # Step the learning rate scheduler (if configured)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.loss)
                else:
                    self.scheduler.step()
            self.logger.info(
                "epoch_complete",
                epoch=epoch_index + 1,
                epochs=epochs,
                current_lr=self.optimizer.param_groups[0]["lr"],
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                train_macro_f1=train_metrics.macro_f1,
                train_balanced_accuracy=train_metrics.balanced_accuracy,
                train_long_recall=train_metrics.long_recall,
                train_short_recall=train_metrics.short_recall,
                train_prediction_distribution=(
                    train_metrics.prediction_distribution if self.report_prediction_distribution else {}
                ),
                train_per_class_metrics=(train_metrics.per_class_metrics if self.report_per_class_metrics else {}),
                train_warnings=train_metrics.warnings,
                val_loss=val_metrics.loss,
                val_accuracy=val_metrics.accuracy,
                val_macro_f1=val_metrics.macro_f1,
                val_balanced_accuracy=val_metrics.balanced_accuracy,
                val_long_recall=val_metrics.long_recall,
                val_short_recall=val_metrics.short_recall,
                val_prediction_distribution=(
                    val_metrics.prediction_distribution if self.report_prediction_distribution else {}
                ),
                val_per_class_metrics=(val_metrics.per_class_metrics if self.report_per_class_metrics else {}),
                val_warnings=val_metrics.warnings,
            )
        if self.restore_best_on_finish and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return history

    def _selection_score(self, metrics: object) -> float:
        """Higher-is-better score for best-epoch selection / early stopping."""
        if self.model_selection_metric == "val_balanced_accuracy":
            return float(metrics.balanced_accuracy)
        if self.model_selection_metric == "val_macro_f1":
            return float(metrics.macro_f1)
        # "val_loss" (legacy): lower loss is better.
        return -float(metrics.loss)

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        stage: str = "train",
        epoch_index: int | None = None,
        epoch_count: int | None = None,
    ) -> BatchMetrics:
        self.model.train(training)
        # --- Accumulate loss components on-device ---------------------------
        # Previously each batch called ``.item()`` 8 times to extract scalar
        # losses for logging. Each ``.item()`` is a GPU->CPU sync that drains
        # the CUDA stream and blocks the next batch. We now accumulate every
        # scalar loss on-device in a single tensor and sync once at the end
        # of the epoch, eliminating 8*N_batches sync points.
        device = self.device
        loss_accum = torch.zeros(8, device=device)  # idx 0..7 map to the counters below
        LOSS_IDX_TOTAL = 0
        LOSS_IDX_ACC = 1
        LOSS_IDX_CLS = 2
        LOSS_IDX_REG = 3
        # slot 4 intentionally unused (was LOSS_IDX_LAT, removed in Phase 8)
        # slot 5 intentionally unused (was LOSS_IDX_PL, removed when path_label head was killed)
        LOSS_IDX_FR = 6
        LOSS_IDX_RP = 7
        component_totals: dict[str, torch.Tensor] = {}
        logits_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        grad_norms: list[float] = []
        batches = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        progress_interval = 200 if training else 100
        started_at = perf_counter()
        _t_iter = perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            _t_data = perf_counter() - _t_iter  # time spent in DataLoader iterator
            batch = _move_batch_non_blocking(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            # --- Training-time augmentation: add Gaussian noise to features.
            # Only fires during training, never during validation.
            if training and self.augmentation_enabled:
                batch["features"] = self.augmenter.augment(batch["features"])
            # Validation/test epochs must not build autograd graphs (§4.7):
            # the backward pass only runs when training, so grad tracking
            # outside training is pure memory/compute waste per batch.
            with torch.set_grad_enabled(training), maybe_autocast(self.device, self.use_amp and training):
                output = self.model(
                    branch_inputs=batch["branch_inputs"],
                    regime_input=batch["regime_input"],
                    full_features=batch["features"],
                    temporal_tensor=batch.get("temporal_tensor"),
                )

                # --- A. Direction classification loss ---
                classification_loss = direction_classification_loss(
                    output.action_logits,
                    batch["label"],
                    class_weights=self.class_weights,
                    loss_mode=self.classification_loss_mode,
                    focal_gamma=self.focal_gamma,
                    label_smoothing=self.label_smoothing,
                )

                # --- C. Legacy multitask regression loss ---
                decoupled = bool(getattr(self.model, "decoupled_heads", False))
                regression_loss = output.action_logits.new_zeros(())
                regression_components: dict[str, float] = {}
                if self.multitask_enabled:
                    regression_loss, regression_components = multitask_regression_loss(
                        output,
                        batch.get("regression_targets", {}),
                        decoupled=decoupled,
                    )

                # --- D. Future return regression loss ---
                future_return_loss = output.action_logits.new_zeros(())
                if self.enable_future_return_targets:
                    fr_loss, fr_components = future_return_regression_loss(
                        output, batch.get("regression_targets", {}),
                    )
                    future_return_loss = fr_loss
                    regression_components.update(fr_components)

                # --- E. Risk/path regression loss ---
                risk_path_loss = output.action_logits.new_zeros(())
                if self.enable_risk_path_targets:
                    rp_loss, rp_components = risk_path_regression_loss(
                        output, batch.get("regression_targets", {}),
                        decoupled=decoupled,
                    )
                    risk_path_loss = rp_loss
                    regression_components.update(rp_components)

                # --- Weighted total loss ---
                loss = (
                    self.loss_weights.get("classification", 1.0) * classification_loss
                    + self.loss_weights.get("regression", 0.0) * regression_loss
                    + self.loss_weights.get("future_return", 0.0) * future_return_loss
                    + self.loss_weights.get("risk_path", 0.0) * risk_path_loss
                )

                # --- Incumbent anchor terms (training only, §4.5.6) ---
                if training and self._anchor_model is not None:
                    with torch.no_grad():
                        anchor_output = self._anchor_model(
                            branch_inputs=batch["branch_inputs"],
                            regime_input=batch["regime_input"],
                            full_features=batch["features"],
                            temporal_tensor=batch.get("temporal_tensor"),
                        )
                    # KL(incumbent || student): standard distillation direction.
                    kd_loss = torch.nn.functional.kl_div(
                        torch.log_softmax(output.action_logits, dim=-1),
                        torch.softmax(anchor_output.action_logits.detach(), dim=-1),
                        reduction="batchmean",
                    )
                    loss = loss + self.distill_lambda_kd * kd_loss
                    regression_components["anchor_kd"] = kd_loss.detach()
                if training and self._anchor_params:
                    l2_sum = output.action_logits.new_zeros(())
                    for name, parameter in self.model.named_parameters():
                        anchor_parameter = self._anchor_params.get(name)
                        if anchor_parameter is not None and anchor_parameter.shape == parameter.shape:
                            l2_sum = l2_sum + (parameter - anchor_parameter).pow(2).sum()
                    loss = loss + self.proximal_lambda_l2 * l2_sum
                    regression_components["anchor_l2"] = l2_sum.detach()

            if training:
                if torch.isnan(loss):
                    # Diagnose which component produced NaN
                    _nan_parts = []
                    for _name, _val in [
                        ("classification", classification_loss),
                        ("regression", regression_loss),
                        ("future_return", future_return_loss),
                        ("risk_path", risk_path_loss),
                    ]:
                        if torch.isnan(_val):
                            _nan_parts.append(_name)
                    raise RuntimeError(
                        f"NaN loss at batch {batch_index}: "
                        f"nan_components={_nan_parts}, "
                        f"total={loss.item()}"
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    continue
                grad_norms.append(float(total_norm.item()))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            # Accumulate on-device; no sync here.
            with torch.no_grad():
                predictions = output.action_logits.detach().argmax(dim=-1)
                accuracy = (predictions == batch["label"]).float().mean()
                loss_accum[LOSS_IDX_TOTAL] += loss.detach()
                loss_accum[LOSS_IDX_ACC] += accuracy
                loss_accum[LOSS_IDX_CLS] += classification_loss.detach()
                loss_accum[LOSS_IDX_REG] += regression_loss.detach()
                loss_accum[LOSS_IDX_FR] += future_return_loss.detach()
                loss_accum[LOSS_IDX_RP] += risk_path_loss.detach()
                # Keep full logits/labels for end-of-epoch metrics, but stay
                # on-device — the .cpu() call is deferred to after the loop.
                logits_batches.append(output.action_logits.detach())
                label_batches.append(batch["label"].detach())
                for name, value in regression_components.items():
                    v = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=device)
                    if name not in component_totals:
                        component_totals[name] = v.detach().clone()
                    else:
                        component_totals[name] += v.detach()
            batches += 1
            _t_iter = perf_counter()  # reset for next iteration's data timing
            if batch_index == 1 or batch_index % progress_interval == 0 or (total_batches is not None and batch_index == total_batches):
                elapsed = perf_counter() - started_at
                # Progress logging is the *only* place we sync mid-epoch.
                # Only fire at sparse intervals (every 200 train batches).
                avg = (loss_accum[[LOSS_IDX_TOTAL, LOSS_IDX_ACC]] / batches).tolist()
                self.logger.info(
                    "epoch_progress",
                    stage=stage,
                    epoch=epoch_index,
                    epochs=epoch_count,
                    batch=batch_index,
                    total_batches=total_batches,
                    avg_loss=avg[0],
                    avg_accuracy=avg[1],
                    elapsed_sec=elapsed,
                    ms_per_batch=round(elapsed / batches * 1000, 1),
                    last_data_ms=round(_t_data * 1000, 1),
                )
        if batches == 0:
            return BatchMetrics(loss=0.0, accuracy=0.0)
        # Single host-sync of the whole accumulator at the end of the epoch.
        loss_totals = loss_accum.detach().cpu().tolist()
        total_loss = loss_totals[LOSS_IDX_TOTAL]
        total_accuracy = loss_totals[LOSS_IDX_ACC]
        total_classification_loss = loss_totals[LOSS_IDX_CLS]
        total_regression_loss = loss_totals[LOSS_IDX_REG]
        total_future_return_loss = loss_totals[LOSS_IDX_FR]
        total_risk_path_loss = loss_totals[LOSS_IDX_RP]
        epoch_logits = torch.cat(logits_batches, dim=0).cpu()
        epoch_labels = torch.cat(label_batches, dim=0).cpu()
        classification_report = compute_classification_report(logits=epoch_logits, labels=epoch_labels)
        warnings = build_collapse_warnings(
            classification_report=classification_report,
            thresholds=self.collapse_warning_thresholds,
            flat_dominance_threshold=self.metrics_flat_dominance_threshold,
        )

        return BatchMetrics(
            loss=total_loss / batches,
            accuracy=total_accuracy / batches,
            classification_loss=total_classification_loss / batches,
            regression_loss=total_regression_loss / batches,
            future_return_loss=total_future_return_loss / batches,
            risk_path_loss=total_risk_path_loss / batches,
            balanced_accuracy=float(classification_report["balanced_accuracy"]),
            macro_f1=float(classification_report["macro_f1"]),
            weighted_f1=float(classification_report["weighted_f1"]),
            long_recall=float(classification_report["per_class"]["long"]["recall"]),
            short_recall=float(classification_report["per_class"]["short"]["recall"]),
            prediction_distribution=dict(classification_report["prediction_ratios"]),
            target_distribution=dict(classification_report["target_ratios"]),
            per_class_metrics=dict(classification_report["per_class"]),
            warnings=warnings,
            components={name: float(value.detach().cpu()) / batches for name, value in component_totals.items()},
            grad_norm_mean=sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            grad_norm_max=max(grad_norms) if grad_norms else 0.0,
        )
