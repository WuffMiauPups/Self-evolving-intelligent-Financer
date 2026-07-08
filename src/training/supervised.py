from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.config import AppConfig
from src.core.device import build_device_context
from src.core.device import maybe_autocast
from src.core.logging import configure_logging, get_logger
from src.core.seeds import seed_everything
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import build_checkpoint_meta, build_model_from_contract, load_model_artifacts
from src.training.checkpoints import save_checkpoint
from src.training.data_loading import _build_training_client, _resolve_bars_by_symbol, _FAST_SMALL_MODEL_OVERRIDES
from src.training.data_module import TrainingDataBundle, build_real_data_training_bundle
from src.training.memory_builder import build_post_training_memory
from src.training.metrics import build_collapse_warnings, compute_classification_report, compute_logit_diagnostics
from src.training.run_summary import build_run_summary, save_run_summary
from src.training.trainer import Trainer
from src.utils.torch_utils import move_batch_to_device


def build_training_components(
    config: AppConfig,
    *,
    use_baseline: bool = False,
) -> tuple[torch.nn.Module, TrainingDataBundle, dict[str, list[MarketBar]], dict[str, object]]:
    logger = get_logger("training")
    seed_everything(7)
    device_ctx = build_device_context(config.device.preferred, config.training.use_amp or config.device.use_amp)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bars_by_symbol, data_meta = _resolve_bars_by_symbol(config)
    bundle = build_real_data_training_bundle(
        config=config,
        bars_by_symbol=bars_by_symbol,
        pipeline=pipeline,
    )
    # Fast-mode: small model profile reduces capacity for rapid experiments
    effective_model_config = config.model
    fast = config.training.fast_mode
    if fast.enabled and fast.model_profile == "small" and not use_baseline:
        effective_model_config = config.model.model_copy(update=_FAST_SMALL_MODEL_OVERRIDES)
        logger.info(
            "fast_mode_small_model_profile",
            overrides=_FAST_SMALL_MODEL_OVERRIDES,
            note="Reduced model capacity for fast experiment. NOT for production.",
        )
    model = build_model_from_contract(
        model_kind="baseline" if use_baseline else "brain",
        model_config=effective_model_config,
        feature_contract=bundle.feature_contract,
    )
    model = model.to(device_ctx.device)
    return model, bundle, bars_by_symbol, data_meta


def inspect_training_data(config: AppConfig) -> dict[str, object]:
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("training")
    _, bundle, _, data_meta = build_training_components(config, use_baseline=False)
    for split_name, diagnostics in bundle.diagnostics.items():
        logger.info(
            "target_diagnostics",
            split=split_name,
            total_samples=diagnostics.total_samples,
            class_counts=diagnostics.class_counts,
            class_ratios=diagnostics.class_ratios,
            flat_rate=diagnostics.flat_rate,
            symbol_counts=diagnostics.symbol_counts,
            flat_rate_by_symbol=diagnostics.flat_rate_by_symbol,
            primary_return_quantiles=diagnostics.primary_return_quantiles,
            near_flat_rate=diagnostics.near_flat_rate,
            near_boundary_rate=diagnostics.near_boundary_rate,
            path_label_counts=diagnostics.path_label_counts,
            path_label_ratios=diagnostics.path_label_ratios,
            per_horizon_return_quantiles=diagnostics.per_horizon_return_quantiles,
            risk_path_summaries={
                name: summary.__dict__
                for name, summary in diagnostics.risk_path_summaries.items()
            },
            regression_summaries={
                name: summary.__dict__
                for name, summary in diagnostics.regression_summaries.items()
            },
            warnings=diagnostics.warnings,
        )
    return {
        "data_meta": data_meta,
        "diagnostics": bundle.diagnostics,
        "feature_contract": bundle.feature_contract.to_dict(),
    }


def resolve_checkpoint_save_dir(
    *,
    base_dir: Path,
    run_id: str,
    explicit_run_id: bool,
    fast_mode_enabled: bool,
    isolate_checkpoint: bool | None,
) -> tuple[Path, bool]:
    """Resolve where this training run should save its checkpoint.

    Returns (target_dir, isolated) where ``isolated`` is True iff the
    target_dir was nested under ``base_dir / run_id`` to protect the
    canonical checkpoint at ``base_dir / brain.pt`` from being overwritten.

    Auto-isolation kicks in when the caller passed an explicit run_id
    (signalling "this is a labeled experiment, do not touch the canonical
    checkpoint") or when fast mode is enabled (fast-mode checkpoints must
    never be promoted to production per --fast help text). Callers that
    manage their own per-run paths (e.g. sweep_train.py) can pass
    ``isolate_checkpoint=False`` to opt out.
    """
    if isolate_checkpoint is None:
        isolate_checkpoint = explicit_run_id or fast_mode_enabled
    if isolate_checkpoint:
        return base_dir / run_id, True
    return base_dir, False


def run_supervised_training(
    config: AppConfig,
    *,
    use_baseline: bool = False,
    run_id: str | None = None,
    isolate_checkpoint: bool | None = None,
) -> dict[str, object]:
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("training")

    # Auto-cleanup training cache if it exceeds the size limit (default 10 GB).
    from src.training.sample_cache import enforce_cache_limit
    freed = enforce_cache_limit()
    if freed > 0:
        logger.info("training_cache_freed_on_startup", freed_gb=round(freed / 1_073_741_824, 2))

    explicit_run_id = run_id is not None
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_start = datetime.now(UTC)
    device_ctx = build_device_context(config.device.preferred, config.training.use_amp or config.device.use_amp)
    model, bundle, bars_by_symbol, data_meta = build_training_components(config, use_baseline=use_baseline)
    # Fast-mode: log clear banner and compute effective settings
    fast = config.training.fast_mode
    if fast.enabled:
        logger.warning(
            "FAST_EXPERIMENT_MODE_ACTIVE",
            note=(
                "This is a FAST EXPERIMENT RUN. Results are approximate and intended for "
                "research/comparison only. Do NOT promote this checkpoint to production."
            ),
            fast_months=fast.fast_months,
            fast_start_date=fast.fast_start_date,
            fast_end_date=fast.fast_end_date,
            fast_symbols=list(fast.symbols) if fast.symbols else "(all configured symbols)",
            max_train_samples=fast.max_train_samples,
            max_val_samples=fast.max_val_samples,
            max_test_samples=fast.max_test_samples,
            max_epochs=fast.max_epochs,
            disable_future_return=fast.disable_future_return,
            disable_risk_path=fast.disable_risk_path,
            model_profile=fast.model_profile,
            sample_seed=fast.sample_seed,
        )

    # Effective epochs: respect fast_mode.max_epochs cap
    effective_epochs = config.training.epochs
    if fast.enabled and fast.max_epochs is not None:
        effective_epochs = min(effective_epochs, fast.max_epochs)
        if effective_epochs < config.training.epochs:
            logger.info(
                "fast_mode_epoch_cap_applied",
                configured_epochs=config.training.epochs,
                effective_epochs=effective_epochs,
            )

    logger.info(
        "training_strategy",
        classification_loss=config.training.classification_loss,
        class_weight_mode=config.training.class_weight_mode,
        class_weight_cap=config.training.class_weight_cap,
        use_weighted_sampler=config.training.use_weighted_sampler,
        weighted_sampler_mode=config.training.weighted_sampler_mode,
        label_smoothing=config.training.label_smoothing,
        focal_gamma=config.training.focal_gamma,
        loss_weights=config.training.loss_weights,
        fast_mode=fast.enabled,
        effective_epochs=effective_epochs,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    # --- Learning rate scheduler ---
    scheduler = None
    sched_type = config.training.scheduler_type
    if sched_type == "cosine":
        t_max = config.training.scheduler_t_max or effective_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(t_max, 1), eta_min=config.training.scheduler_min_lr,
        )
    elif sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.scheduler_min_lr,
        )
    effective_multitask = config.training.multitask_enabled and not config.training.disable_regression
    effective_contrastive = config.training.contrastive_enabled and not config.training.disable_latent
    effective_loss_weights = dict(config.training.loss_weights)
    if config.training.disable_regression:
        effective_loss_weights["regression"] = 0.0
    if config.training.disable_latent:
        effective_loss_weights["latent"] = 0.0
    # Fast-mode: optional target/loss simplification
    effective_enable_future_return = config.training.enable_future_return_targets
    effective_enable_risk_path = config.training.enable_risk_path_targets
    if fast.enabled:
        if fast.disable_future_return:
            effective_enable_future_return = False
            effective_loss_weights["future_return"] = 0.0
        if fast.disable_risk_path:
            effective_enable_risk_path = False
            effective_loss_weights["risk_path"] = 0.0
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device_ctx.device,
        use_amp=device_ctx.use_amp,
        class_weights=bundle.class_weights if isinstance(bundle.class_weights, torch.Tensor) else None,
        multitask_enabled=effective_multitask,
        contrastive_enabled=effective_contrastive,
        contrastive_margin=config.training.contrastive_margin,
        classification_loss_mode=config.training.classification_loss,
        focal_gamma=config.training.focal_gamma,
        label_smoothing=config.training.label_smoothing,
        loss_weights=effective_loss_weights,
        collapse_warning_thresholds=config.training.collapse_warning_thresholds,
        report_prediction_distribution=config.training.report_prediction_distribution,
        report_per_class_metrics=config.training.report_per_class_metrics,
        enable_future_return_targets=effective_enable_future_return,
        enable_risk_path_targets=effective_enable_risk_path,
        adaptive_focal_gamma=config.training.adaptive_focal_gamma,
        focal_gamma_base=config.training.focal_gamma_base,
        focal_gamma_boost=config.training.focal_gamma_boost,
        scheduler=scheduler,
        early_stopping_patience=config.training.early_stopping_patience,
        restore_best_on_finish=config.training.restore_best_model,
        metrics_flat_dominance_threshold=config.training.metrics_flat_dominance_threshold,
        augmentation_enabled=config.training.augmentation_enabled,
        noise_std=config.training.noise_std,
        temporal_jitter_prob=config.training.temporal_jitter_prob,
    )
    for split_name, symbols in bundle.split_ranges.items():
        for symbol, split in symbols.items():
            logger.info(
                "training_split",
                split=split_name,
                symbol=symbol,
                sample_count=split.sample_count,
                start_timestamp=None if split.start_timestamp is None else split.start_timestamp.isoformat(),
                end_timestamp=None if split.end_timestamp is None else split.end_timestamp.isoformat(),
            )
    if config.training.report_target_stats:
        for split_name, diagnostics in bundle.diagnostics.items():
            logger.info(
                "target_diagnostics",
                split=split_name,
                total_samples=diagnostics.total_samples,
                class_counts=diagnostics.class_counts,
                class_ratios=diagnostics.class_ratios,
                flat_rate=diagnostics.flat_rate,
                symbol_counts=diagnostics.symbol_counts,
                flat_rate_by_symbol=diagnostics.flat_rate_by_symbol,
                primary_return_quantiles=diagnostics.primary_return_quantiles,
                near_flat_rate=diagnostics.near_flat_rate,
                near_boundary_rate=diagnostics.near_boundary_rate,
                path_label_counts=diagnostics.path_label_counts,
                path_label_ratios=diagnostics.path_label_ratios,
                per_horizon_return_quantiles=diagnostics.per_horizon_return_quantiles,
                risk_path_summaries={
                    name: summary.__dict__
                    for name, summary in diagnostics.risk_path_summaries.items()
                },
                regression_summaries={
                    name: summary.__dict__
                    for name, summary in diagnostics.regression_summaries.items()
                },
                warnings=diagnostics.warnings,
            )
    history = trainer.fit(bundle.train_loader, bundle.val_loader, epochs=effective_epochs)
    test_metrics = trainer._run_epoch(bundle.test_loader, training=False, stage="test")
    val_report = _evaluate_detailed_split(
        loader=bundle.val_loader,
        model=model,
        device=device_ctx.device,
        use_amp=device_ctx.use_amp,
        split="val",
        collapse_warning_thresholds=config.training.collapse_warning_thresholds,
        metrics_flat_dominance_threshold=config.training.metrics_flat_dominance_threshold,
    )
    test_report = _evaluate_detailed_split(
        loader=bundle.test_loader,
        model=model,
        device=device_ctx.device,
        use_amp=device_ctx.use_amp,
        split="test",
        collapse_warning_thresholds=config.training.collapse_warning_thresholds,
        metrics_flat_dominance_threshold=config.training.metrics_flat_dominance_threshold,
    )
    _log_evaluation_report(
        logger,
        val_report,
        report_logit_stats=config.training.report_logit_stats,
        report_prediction_distribution=config.training.report_prediction_distribution,
        report_per_class_metrics=config.training.report_per_class_metrics,
    )
    _log_evaluation_report(
        logger,
        test_report,
        report_logit_stats=config.training.report_logit_stats,
        report_prediction_distribution=config.training.report_prediction_distribution,
        report_per_class_metrics=config.training.report_per_class_metrics,
    )
    base_checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir, checkpoint_isolated = resolve_checkpoint_save_dir(
        base_dir=base_checkpoint_dir,
        run_id=run_id,
        explicit_run_id=explicit_run_id,
        fast_mode_enabled=fast.enabled,
        isolate_checkpoint=isolate_checkpoint,
    )
    checkpoint_filename = "baseline.pt" if use_baseline else "brain.pt"
    checkpoint_path = checkpoint_dir / checkpoint_filename
    if checkpoint_isolated:
        canonical_path = base_checkpoint_dir / checkpoint_filename
        logger.warning(
            "checkpoint_path_isolated",
            run_id=run_id,
            reason="explicit_run_id" if explicit_run_id else "fast_mode",
            isolated_path=str(checkpoint_path),
            canonical_path_protected=str(canonical_path),
            note="Set isolate_checkpoint=False to write to canonical_path instead.",
        )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        meta=build_checkpoint_meta(
            model_kind="baseline" if use_baseline else "brain",
            model_config=config.model,
            feature_config=config.features,
            feature_contract=bundle.feature_contract,
            extra={
                "train_loss": history.train[-1].loss if history.train else 0.0,
                "val_loss": history.val[-1].loss if history.val else 0.0,
                "test_loss": test_metrics.loss,
                "train_accuracy": history.train[-1].accuracy if history.train else 0.0,
                "val_accuracy": history.val[-1].accuracy if history.val else 0.0,
                "test_accuracy": test_metrics.accuracy,
                "train_macro_f1": history.train[-1].macro_f1 if history.train else 0.0,
                "val_macro_f1": history.val[-1].macro_f1 if history.val else 0.0,
                "test_macro_f1": float(test_report["macro_f1"]),
                "train_balanced_accuracy": history.train[-1].balanced_accuracy if history.train else 0.0,
                "val_balanced_accuracy": history.val[-1].balanced_accuracy if history.val else 0.0,
                "test_balanced_accuracy": float(test_report["balanced_accuracy"]),
                "evaluation_reports": {
                    "val": val_report,
                    "test": test_report,
                },
                "collapse_warnings": {
                    "val": list(val_report["warnings"]),
                    "test": list(test_report["warnings"]),
                },
                "data_config": config.data.model_dump(mode="json"),
                "training_config": config.training.model_dump(mode="json"),
                "symbol_universe": list(config.data.symbols),
                "training_split_ranges": {
                    split_name: {
                        symbol: {
                            "sample_count": split.sample_count,
                            "start_timestamp": (
                                None if split.start_timestamp is None else split.start_timestamp.isoformat()
                            ),
                            "end_timestamp": (
                                None if split.end_timestamp is None else split.end_timestamp.isoformat()
                            ),
                        }
                        for symbol, split in symbol_splits.items()
                    }
                    for split_name, symbol_splits in bundle.split_ranges.items()
                },
                "target_diagnostics": {
                    split: {
                        "total_samples": diagnostics.total_samples,
                        "class_counts": diagnostics.class_counts,
                        "class_ratios": diagnostics.class_ratios,
                        "flat_rate": diagnostics.flat_rate,
                        "symbol_counts": diagnostics.symbol_counts,
                        "flat_rate_by_symbol": diagnostics.flat_rate_by_symbol,
                        "primary_return_quantiles": diagnostics.primary_return_quantiles,
                        "near_flat_count": diagnostics.near_flat_count,
                        "near_flat_rate": diagnostics.near_flat_rate,
                        "near_boundary_count": diagnostics.near_boundary_count,
                        "near_boundary_rate": diagnostics.near_boundary_rate,
                        "path_label_counts": diagnostics.path_label_counts,
                        "path_label_ratios": diagnostics.path_label_ratios,
                        "per_horizon_return_quantiles": diagnostics.per_horizon_return_quantiles,
                        "risk_path_summaries": {
                            name: summary.__dict__
                            for name, summary in diagnostics.risk_path_summaries.items()
                        },
                        "regression_summaries": {
                            name: summary.__dict__
                            for name, summary in diagnostics.regression_summaries.items()
                        },
                        "warnings": list(diagnostics.warnings),
                    }
                    for split, diagnostics in bundle.diagnostics.items()
                },
                "state_path_design": {
                    "primary_direction_horizon": config.training.effective_primary_horizon,
                    "auxiliary_horizons": list(config.training.auxiliary_horizons),
                    "all_horizons": config.training.all_horizons,
                    "enable_future_return_targets": config.training.enable_future_return_targets,
                    "enable_risk_path_targets": config.training.enable_risk_path_targets,
                    "path_label_rules": dict(config.training.path_label_rules),
                    "loss_weights": dict(config.training.loss_weights),
                },
                "data_meta": data_meta,
            },
        ),
    )

    # Save structured run summary
    elapsed = (datetime.now(UTC) - run_start).total_seconds()
    run_summary = build_run_summary(
        run_id=run_id,
        config=config,
        data_meta=data_meta,
        bundle_diagnostics=bundle.diagnostics,
        train_metrics=history.train[-1] if history.train else None,
        val_metrics=history.val[-1] if history.val else None,
        test_metrics=test_metrics,
        val_report=val_report,
        test_report=test_report,
        checkpoint_path=str(checkpoint_path),
        elapsed_seconds=elapsed,
        effective_epochs=effective_epochs,
    )
    summary_path = save_run_summary(run_summary, runs_dir=config.training.runs_dir)
    logger.info("run_summary_saved", path=str(summary_path), run_id=run_id)

    # --- Platt calibration fitting ---
    if config.training.platt_calibration_enabled:
        from src.training.platt_calibration import PlattCalibrator
        import numpy as np

        model.eval()
        all_confidences: list[float] = []
        all_primary_returns: list[float] = []
        with torch.no_grad():
            for batch in bundle.val_loader:
                batch_on_device = move_batch_to_device(batch, device_ctx.device)
                with maybe_autocast(device_ctx.device, device_ctx.use_amp):
                    output = model(
                        branch_inputs=batch_on_device["branch_inputs"],
                        regime_input=batch_on_device["regime_input"],
                        full_features=batch_on_device["features"],
                        temporal_tensor=batch_on_device.get("temporal_tensor"),
                    )
                all_confidences.extend(output.confidence.cpu().numpy().flatten().tolist())
                reg_targets = batch.get("regression_targets", {})
                if "primary_return" in reg_targets:
                    all_primary_returns.extend(
                        reg_targets["primary_return"].cpu().numpy().flatten().tolist()
                    )
        if len(all_confidences) >= 20 and len(all_primary_returns) == len(all_confidences):
            calibrator = PlattCalibrator()
            calibrator.fit(np.array(all_confidences), np.array(all_primary_returns))
            platt_path = Path(config.training.runs_dir) / "platt_calibration.json"
            calibrator.save(platt_path)
            logger.info(
                "platt_calibration_fitted",
                a=calibrator.a,
                b=calibrator.b,
                n=calibrator.n_samples,
                path=str(platt_path),
            )
        else:
            logger.warning(
                "platt_calibration_skipped",
                n_confidences=len(all_confidences),
                n_returns=len(all_primary_returns),
                reason="insufficient_samples" if len(all_confidences) < 20 else "length_mismatch",
            )

    memory_path = build_post_training_memory(
        config=config,
        model=model,
        bundle=bundle,
        bars_by_symbol=bars_by_symbol,
        device=device_ctx.device,
    )

    logger.info(
        "training_complete",
        device=str(device_ctx.device),
        train_loss=history.train[-1].loss if history.train else 0.0,
        val_loss=history.val[-1].loss if history.val else 0.0,
        test_loss=test_metrics.loss,
        train_accuracy=history.train[-1].accuracy if history.train else 0.0,
        train_macro_f1=history.train[-1].macro_f1 if history.train else 0.0,
        train_balanced_accuracy=history.train[-1].balanced_accuracy if history.train else 0.0,
        val_accuracy=history.val[-1].accuracy if history.val else 0.0,
        val_macro_f1=history.val[-1].macro_f1 if history.val else 0.0,
        val_balanced_accuracy=history.val[-1].balanced_accuracy if history.val else 0.0,
        test_accuracy=test_metrics.accuracy,
        test_macro_f1=test_report["macro_f1"],
        test_balanced_accuracy=test_report["balanced_accuracy"],
        primary_direction_horizon=config.training.effective_primary_horizon,
        enable_future_return_targets=config.training.enable_future_return_targets,
        enable_risk_path_targets=config.training.enable_risk_path_targets,
        class_weight_mode=config.training.class_weight_mode,
        weighted_sampler=config.training.use_weighted_sampler,
        classification_loss=config.training.classification_loss,
        checkpoint=str(checkpoint_path),
        memory_path=memory_path,
    )
    return {
        "history": history,
        "checkpoint": str(checkpoint_path),
        "device": str(device_ctx.device),
        "test_metrics": test_metrics,
        "memory_path": memory_path,
        "diagnostics": bundle.diagnostics,
        "evaluation_reports": {"val": val_report, "test": test_report},
        "run_id": run_id,
        "run_summary_path": str(summary_path),
    }


def inspect_label_sensitivity(
    config: AppConfig,
    *,
    epsilon_values: list[float] | None = None,
    horizon_values: list[int] | None = None,
) -> list[object]:
    """Sweep epsilon/horizon values and report class balance profiles."""
    from src.training.label_inspector import inspect_label_profiles, format_label_profiles

    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("training")
    _, bundle, _, data_meta = build_training_components(config, use_baseline=False)

    # Collect future returns from all training samples
    samples_with_returns: list[dict[str, object]] = []
    for sample in bundle.samples_by_split.get("train", []):
        fr = dict(getattr(sample, "future_returns", {}))
        if fr:
            samples_with_returns.append({"future_returns": {int(k): float(v) for k, v in fr.items()}})

    if epsilon_values is None:
        base = config.training.label_epsilon
        epsilon_values = [base * 0.25, base * 0.5, base, base * 2.0, base * 4.0]
    if horizon_values is None:
        horizon_values = list(config.training.forward_horizons)

    profiles = inspect_label_profiles(
        samples_with_returns,
        epsilon_values=epsilon_values,
        horizons=horizon_values,
    )

    logger.info("label_sensitivity_report", sample_count=len(samples_with_returns))
    print(format_label_profiles(profiles))
    return profiles


def evaluate_checkpoint(config: AppConfig, checkpoint_path: str) -> dict[str, object]:
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("training")
    device_ctx = build_device_context(config.device.preferred, config.training.use_amp or config.device.use_amp)
    _, bundle, _, data_meta = build_training_components(config, use_baseline=False)
    artifacts = load_model_artifacts(checkpoint_path, device=device_ctx.device)
    artifacts.feature_contract.assert_matches(
        bundle.feature_contract,
        context=f"checkpoint_eval[{Path(checkpoint_path).name}]",
    )
    logger.info(
        "checkpoint_evaluation_start",
        checkpoint=str(checkpoint_path),
        model_kind=artifacts.model_kind,
        data_source=data_meta.get("source"),
        symbols=list(config.data.symbols),
    )
    val_report = _evaluate_detailed_split(
        loader=bundle.val_loader,
        model=artifacts.model,
        device=device_ctx.device,
        use_amp=device_ctx.use_amp,
        split="val",
        collapse_warning_thresholds=config.training.collapse_warning_thresholds,
        metrics_flat_dominance_threshold=config.training.metrics_flat_dominance_threshold,
    )
    test_report = _evaluate_detailed_split(
        loader=bundle.test_loader,
        model=artifacts.model,
        device=device_ctx.device,
        use_amp=device_ctx.use_amp,
        split="test",
        collapse_warning_thresholds=config.training.collapse_warning_thresholds,
        metrics_flat_dominance_threshold=config.training.metrics_flat_dominance_threshold,
    )
    _log_evaluation_report(
        logger,
        val_report,
        report_logit_stats=config.training.report_logit_stats,
        report_prediction_distribution=config.training.report_prediction_distribution,
        report_per_class_metrics=config.training.report_per_class_metrics,
    )
    _log_evaluation_report(
        logger,
        test_report,
        report_logit_stats=config.training.report_logit_stats,
        report_prediction_distribution=config.training.report_prediction_distribution,
        report_per_class_metrics=config.training.report_per_class_metrics,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "device": str(device_ctx.device),
        "data_meta": data_meta,
        "evaluation_reports": {"val": val_report, "test": test_report},
    }


def _evaluate_detailed_split(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    split: str,
    collapse_warning_thresholds: dict[str, float],
    metrics_flat_dominance_threshold: float = 0.75,
) -> dict[str, object]:
    class_names = ("short", "flat", "long")
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_symbols: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            symbols = list(batch["symbol"])
            labels_cpu = batch["label"].clone().cpu()
            batch = move_batch_to_device(batch, device)
            with maybe_autocast(device, use_amp):
                output = model(
                    branch_inputs=batch["branch_inputs"],
                    regime_input=batch["regime_input"],
                    full_features=batch["features"],
                    temporal_tensor=batch.get("temporal_tensor"),
                )
            all_logits.append(output.action_logits.detach().cpu())
            all_labels.append(labels_cpu)
            all_symbols.extend(symbols)

    epoch_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, 3), dtype=torch.float32)
    epoch_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
    classification_report = compute_classification_report(logits=epoch_logits, labels=epoch_labels, class_names=class_names)
    logit_diagnostics = compute_logit_diagnostics(logits=epoch_logits, labels=epoch_labels, class_names=class_names)
    warnings = build_collapse_warnings(
        classification_report=classification_report,
        thresholds=collapse_warning_thresholds,
        flat_dominance_threshold=metrics_flat_dominance_threshold,
    )
    predictions = epoch_logits.argmax(dim=-1) if epoch_logits.numel() > 0 else torch.empty((0,), dtype=torch.long)
    symbol_correct: dict[str, int] = {}
    symbol_total: dict[str, int] = {}
    for idx, symbol in enumerate(all_symbols):
        symbol_total[symbol] = symbol_total.get(symbol, 0) + 1
        if int(predictions[idx].item()) == int(epoch_labels[idx].item()):
            symbol_correct[symbol] = symbol_correct.get(symbol, 0) + 1
        else:
            symbol_correct.setdefault(symbol, 0)
    per_symbol_accuracy = {
        symbol: symbol_correct.get(symbol, 0) / max(total, 1)
        for symbol, total in sorted(symbol_total.items())
    }
    short_index = class_names.index("short")
    long_index = class_names.index("long")
    long_short_mask = (epoch_labels == short_index) | (epoch_labels == long_index)
    long_short_accuracy = 0.0
    if bool(long_short_mask.any()):
        long_short_accuracy = float((predictions[long_short_mask] == epoch_labels[long_short_mask]).float().mean().item())
    confusion_matrix = classification_report["confusion_matrix"]
    long_short_confusion = {
        "long_as_short_rate": (
            confusion_matrix[long_index][short_index] / max(classification_report["target_counts"]["long"], 1)
        ),
        "short_as_long_rate": (
            confusion_matrix[short_index][long_index] / max(classification_report["target_counts"]["short"], 1)
        ),
    }
    return {
        "split": split,
        "total_samples": int(epoch_labels.numel()),
        "accuracy": float(classification_report["accuracy"]),
        "balanced_accuracy": float(classification_report["balanced_accuracy"]),
        "macro_f1": float(classification_report["macro_f1"]),
        "weighted_f1": float(classification_report["weighted_f1"]),
        "mcc": float(classification_report["mcc"]),
        "prediction_counts": classification_report["prediction_counts"],
        "target_counts": classification_report["target_counts"],
        "prediction_ratios": classification_report["prediction_ratios"],
        "target_ratios": classification_report["target_ratios"],
        "per_class_metrics": classification_report["per_class"],
        "confusion_matrix": classification_report["confusion_matrix"],
        "per_symbol_accuracy": per_symbol_accuracy,
        "long_short_only_accuracy": long_short_accuracy,
        "long_short_confusion": long_short_confusion,
        "logit_diagnostics": logit_diagnostics,
        "warnings": warnings,
    }


def _log_evaluation_report(
    logger: object,
    report: dict[str, object],
    *,
    report_logit_stats: bool,
    report_prediction_distribution: bool,
    report_per_class_metrics: bool,
) -> None:
    payload = {
        "split": report["split"],
        "total_samples": report["total_samples"],
        "accuracy": report["accuracy"],
        "balanced_accuracy": report["balanced_accuracy"],
        "macro_f1": report["macro_f1"],
        "weighted_f1": report["weighted_f1"],
        "mcc": report["mcc"],
        "long_short_only_accuracy": report["long_short_only_accuracy"],
        "long_short_confusion": report["long_short_confusion"],
        "prediction_counts": report["prediction_counts"],
        "target_counts": report["target_counts"],
        "per_symbol_accuracy": report["per_symbol_accuracy"],
        "confusion_matrix": report["confusion_matrix"],
        "warnings": report["warnings"],
    }
    if report_prediction_distribution:
        payload["prediction_ratios"] = report["prediction_ratios"]
        payload["target_ratios"] = report["target_ratios"]
    if report_per_class_metrics:
        payload["per_class_metrics"] = report["per_class_metrics"]
    if report_logit_stats:
        payload["logit_diagnostics"] = report["logit_diagnostics"]
    logger.info("evaluation_report", **payload)
