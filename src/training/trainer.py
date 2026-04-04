from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.core.logging import get_logger
from src.core.device import maybe_autocast
from src.models.base import TradingModelProtocol
from src.training.losses import direction_classification_loss, latent_consistency_loss, multitask_regression_loss
from src.training.metrics import BatchMetrics, build_collapse_warnings, compute_classification_report
from src.utils.torch_utils import move_batch_to_device


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
        classification_loss_mode: str = "cross_entropy",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        loss_weights: dict[str, float] | None = None,
        collapse_warning_thresholds: dict[str, float] | None = None,
        report_prediction_distribution: bool = True,
        report_per_class_metrics: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.class_weights = class_weights
        self.multitask_enabled = multitask_enabled
        self.contrastive_enabled = contrastive_enabled
        self.classification_loss_mode = classification_loss_mode
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.loss_weights = dict(loss_weights or {"classification": 1.0, "regression": 0.35, "latent": 0.10})
        self.collapse_warning_thresholds = dict(
            collapse_warning_thresholds or {"long_prediction_floor": 0.08, "minority_recall_floor": 0.15}
        )
        self.report_prediction_distribution = report_prediction_distribution
        self.report_per_class_metrics = report_per_class_metrics
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
        self.logger = get_logger("trainer")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
    ) -> TrainingHistory:
        history = TrainingHistory()
        for epoch_index in range(epochs):
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
            self.logger.info(
                "epoch_complete",
                epoch=epoch_index + 1,
                epochs=epochs,
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
        return history

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
        total_loss = 0.0
        total_accuracy = 0.0
        total_classification_loss = 0.0
        total_regression_loss = 0.0
        total_latent_loss = 0.0
        component_totals: dict[str, float] = {}
        logits_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        batches = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        progress_interval = 200 if training else 100
        started_at = perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(self.device, self.use_amp and training):
                output = self.model(
                    branch_inputs=batch["branch_inputs"],
                    regime_input=batch["regime_input"],
                    full_features=batch["features"],
                    temporal_tensor=batch.get("temporal_tensor"),
                )
                classification_loss = direction_classification_loss(
                    output.action_logits,
                    batch["label"],
                    class_weights=self.class_weights,
                    loss_mode=self.classification_loss_mode,
                    focal_gamma=self.focal_gamma,
                    label_smoothing=self.label_smoothing,
                )
                regression_loss = output.action_logits.new_zeros(())
                latent_loss = output.action_logits.new_zeros(())
                regression_components: dict[str, float] = {}
                if self.multitask_enabled:
                    regression_loss, regression_components = multitask_regression_loss(
                        output,
                        batch.get("regression_targets", {}),
                    )
                    if self.contrastive_enabled:
                        latent_loss = latent_consistency_loss(output.fused_representation, batch["label"])
                loss = (
                    self.loss_weights["classification"] * classification_loss
                    + self.loss_weights["regression"] * regression_loss
                    + self.loss_weights["latent"] * latent_loss
                )
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            predictions = output.action_logits.detach().argmax(dim=-1)
            accuracy = (predictions == batch["label"]).float().mean().item()
            logits_batches.append(output.action_logits.detach().cpu())
            label_batches.append(batch["label"].detach().cpu())
            total_loss += float(loss.detach().item())
            total_accuracy += float(accuracy)
            total_classification_loss += float(classification_loss.detach().item())
            total_regression_loss += float(regression_loss.detach().item())
            total_latent_loss += float(latent_loss.detach().item())
            for name, value in regression_components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value)
            batches += 1
            if batch_index == 1 or batch_index % progress_interval == 0 or (total_batches is not None and batch_index == total_batches):
                elapsed = perf_counter() - started_at
                self.logger.info(
                    "epoch_progress",
                    stage=stage,
                    epoch=epoch_index,
                    epochs=epoch_count,
                    batch=batch_index,
                    total_batches=total_batches,
                    avg_loss=total_loss / batches,
                    avg_accuracy=total_accuracy / batches,
                    elapsed_sec=elapsed,
                )
        if batches == 0:
            return BatchMetrics(loss=0.0, accuracy=0.0)
        epoch_logits = torch.cat(logits_batches, dim=0)
        epoch_labels = torch.cat(label_batches, dim=0)
        classification_report = compute_classification_report(logits=epoch_logits, labels=epoch_labels)
        warnings = build_collapse_warnings(
            classification_report=classification_report,
            thresholds=self.collapse_warning_thresholds,
        )
        return BatchMetrics(
            loss=total_loss / batches,
            accuracy=total_accuracy / batches,
            classification_loss=total_classification_loss / batches,
            regression_loss=total_regression_loss / batches,
            latent_loss=total_latent_loss / batches,
            balanced_accuracy=float(classification_report["balanced_accuracy"]),
            macro_f1=float(classification_report["macro_f1"]),
            weighted_f1=float(classification_report["weighted_f1"]),
            long_recall=float(classification_report["per_class"]["long"]["recall"]),
            short_recall=float(classification_report["per_class"]["short"]["recall"]),
            prediction_distribution=dict(classification_report["prediction_ratios"]),
            target_distribution=dict(classification_report["target_ratios"]),
            per_class_metrics=dict(classification_report["per_class"]),
            warnings=warnings,
            components={name: value / batches for name, value in component_totals.items()},
        )
