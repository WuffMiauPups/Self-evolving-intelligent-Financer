from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.core.logging import get_logger
from src.core.device import maybe_autocast
from src.models.base import TradingModelProtocol
from src.training.losses import (
    direction_classification_loss,
    future_return_regression_loss,
    latent_consistency_loss,
    multitask_regression_loss,
    path_label_classification_loss,
    risk_path_regression_loss,
    soma_gating_confidence_loss,
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
        classification_loss_mode: str = "cross_entropy",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        loss_weights: dict[str, float] | None = None,
        collapse_warning_thresholds: dict[str, float] | None = None,
        report_prediction_distribution: bool = True,
        report_per_class_metrics: bool = True,
        # State + future-path config
        enable_path_label_head: bool = False,
        enable_future_return_targets: bool = False,
        enable_risk_path_targets: bool = False,
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
        # Defaults are merged with the caller-supplied dict so new loss terms
        # (like gating_confidence, added 2026-04-07) automatically apply to
        # configs that predate them. Caller overrides win where keys overlap.
        default_loss_weights = {
            "classification": 1.0,
            "regression": 0.35,
            "latent": 0.10,
            "path_label": 0.30,
            "future_return": 0.20,
            "risk_path": 0.15,
            # Penalises soma gating that puts weight on low-confidence branches.
            # Small weight: it's a regularizer, not the primary objective.
            "gating_confidence": 0.05,
        }
        self.loss_weights = {**default_loss_weights, **dict(loss_weights or {})}
        self.collapse_warning_thresholds = dict(
            collapse_warning_thresholds or {"long_prediction_floor": 0.08, "minority_recall_floor": 0.15}
        )
        self.report_prediction_distribution = report_prediction_distribution
        self.report_per_class_metrics = report_per_class_metrics
        self.enable_path_label_head = enable_path_label_head
        self.enable_future_return_targets = enable_future_return_targets
        self.enable_risk_path_targets = enable_risk_path_targets
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
        LOSS_IDX_LAT = 4
        LOSS_IDX_PL = 5
        LOSS_IDX_FR = 6
        LOSS_IDX_RP = 7
        component_totals: dict[str, torch.Tensor] = {}
        logits_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        path_label_logits_batches: list[torch.Tensor] = []
        path_label_batches: list[torch.Tensor] = []
        batches = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        progress_interval = 200 if training else 100
        started_at = perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            batch = _move_batch_non_blocking(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(self.device, self.use_amp and training):
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

                # --- B. Path label classification loss ---
                path_label_loss = output.action_logits.new_zeros(())
                if self.enable_path_label_head and "path_label" in batch:
                    # Use regime_weights as path label logits if available and right shape,
                    # otherwise reuse action_logits projected to 5 classes
                    path_logits = self._get_path_logits(output, batch)
                    if path_logits is not None:
                        path_label_loss = path_label_classification_loss(
                            path_logits,
                            batch["path_label"],
                            loss_mode=self.classification_loss_mode,
                            focal_gamma=self.focal_gamma,
                            label_smoothing=self.label_smoothing,
                        )
                        path_label_logits_batches.append(path_logits.detach())
                        path_label_batches.append(batch["path_label"].detach())

                # --- C. Legacy multitask regression loss ---
                regression_loss = output.action_logits.new_zeros(())
                regression_components: dict[str, float] = {}
                if self.multitask_enabled:
                    regression_loss, regression_components = multitask_regression_loss(
                        output,
                        batch.get("regression_targets", {}),
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
                    )
                    risk_path_loss = rp_loss
                    regression_components.update(rp_components)

                # --- F. Latent consistency loss ---
                latent_loss = output.action_logits.new_zeros(())
                if self.multitask_enabled and self.contrastive_enabled:
                    latent_loss = latent_consistency_loss(output.fused_representation, batch["label"])

                # --- G. Soma gating-confidence regularization ---
                # Pulls the soma weights toward high-confidence branches so the
                # gate can no longer learn to over-weight the regime branch
                # while ignoring its own confidence (the observed runtime bug).
                # Skipped for the baseline model which has no real branch
                # confidences (it ships an empty dict).
                gating_loss = output.action_logits.new_zeros(())
                if output.branch_confidences:
                    bc_stack = torch.stack(
                        [output.branch_confidences[name].squeeze(-1) for name in output.branch_confidences],
                        dim=-1,
                    )
                    if output.soma_weights.shape == bc_stack.shape:
                        gating_loss = soma_gating_confidence_loss(output.soma_weights, bc_stack)

                # --- Weighted total loss ---
                loss = (
                    self.loss_weights.get("classification", 1.0) * classification_loss
                    + self.loss_weights.get("regression", 0.0) * regression_loss
                    + self.loss_weights.get("latent", 0.0) * latent_loss
                    + self.loss_weights.get("path_label", 0.0) * path_label_loss
                    + self.loss_weights.get("future_return", 0.0) * future_return_loss
                    + self.loss_weights.get("risk_path", 0.0) * risk_path_loss
                    + self.loss_weights.get("gating_confidence", 0.0) * gating_loss
                )

            if training:
                self.scaler.scale(loss).backward()
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
                loss_accum[LOSS_IDX_LAT] += latent_loss.detach()
                loss_accum[LOSS_IDX_PL] += path_label_loss.detach()
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
                )
        if batches == 0:
            return BatchMetrics(loss=0.0, accuracy=0.0)
        # Single host-sync of the whole accumulator at the end of the epoch.
        loss_totals = loss_accum.detach().cpu().tolist()
        total_loss = loss_totals[LOSS_IDX_TOTAL]
        total_accuracy = loss_totals[LOSS_IDX_ACC]
        total_classification_loss = loss_totals[LOSS_IDX_CLS]
        total_regression_loss = loss_totals[LOSS_IDX_REG]
        total_latent_loss = loss_totals[LOSS_IDX_LAT]
        total_path_label_loss = loss_totals[LOSS_IDX_PL]
        total_future_return_loss = loss_totals[LOSS_IDX_FR]
        total_risk_path_loss = loss_totals[LOSS_IDX_RP]
        epoch_logits = torch.cat(logits_batches, dim=0).cpu()
        epoch_labels = torch.cat(label_batches, dim=0).cpu()
        classification_report = compute_classification_report(logits=epoch_logits, labels=epoch_labels)
        warnings = build_collapse_warnings(
            classification_report=classification_report,
            thresholds=self.collapse_warning_thresholds,
        )

        # Path label accuracy (if tracked)
        path_label_accuracy = 0.0
        if path_label_logits_batches:
            all_pl_logits = torch.cat(path_label_logits_batches, dim=0).cpu()
            all_pl_labels = torch.cat(path_label_batches, dim=0).cpu()
            pl_preds = all_pl_logits.argmax(dim=-1)
            path_label_accuracy = float((pl_preds == all_pl_labels).float().mean())

        return BatchMetrics(
            loss=total_loss / batches,
            accuracy=total_accuracy / batches,
            classification_loss=total_classification_loss / batches,
            regression_loss=total_regression_loss / batches,
            latent_loss=total_latent_loss / batches,
            path_label_loss=total_path_label_loss / batches,
            future_return_loss=total_future_return_loss / batches,
            risk_path_loss=total_risk_path_loss / batches,
            path_label_accuracy=path_label_accuracy,
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
        )

    def _get_path_logits(self, output: object, batch: dict) -> torch.Tensor | None:
        """Extract or construct path-label logits from model output.

        The model has regime_weights (shape [batch, regime_count]) which we can
        repurpose if regime_count == 5 (matching PATH_LABEL_VOCABULARY size).
        Otherwise, we reuse action_logits expanded to 5 classes via a simple
        linear projection on the fused representation.
        """
        # Try regime_weights first (naturally 4-class but close to 5)
        if hasattr(output, "regime_weights") and output.regime_weights is not None:
            rw = output.regime_weights
            if rw.shape[-1] == 5:
                return rw
        # Fall back: use fused_representation to produce 5-class logits
        # We use a simple linear combination from the 3-class action logits + bias
        # This gives the optimizer a gradient path through the existing representation
        logits_3 = output.action_logits  # [batch, 3]
        batch_size = logits_3.shape[0]
        # Map 3 direction logits → 5 path logits via a fixed deterministic expansion:
        # breakout_down ← short logit, mean_rev_down ← (short + flat)/2,
        # chop ← flat logit, mean_rev_up ← (flat + long)/2, breakout_up ← long logit
        short_l = logits_3[:, 0:1]
        flat_l = logits_3[:, 1:2]
        long_l = logits_3[:, 2:3]
        path_logits = torch.cat([
            short_l,                        # breakout_down
            (short_l + flat_l) * 0.5,       # mean_reversion_down
            flat_l,                         # chop
            (flat_l + long_l) * 0.5,        # mean_reversion_up
            long_l,                         # breakout_up
        ], dim=-1)
        return path_logits
