from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import AppConfig, load_config
from src.training.supervised import (
    evaluate_checkpoint,
    inspect_label_sensitivity,
    inspect_training_data,
    run_supervised_training,
)

CANONICAL_CHECKPOINT_FILENAME = "brain.pt"

# ---------------------------------------------------------------------------
# Experiment presets: named configurations for reproducible baselines
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, object]] = {
    "state-only": {
        "description": (
            "State-only baseline: classification + regression + latent only. "
            "Disables future-return and risk-path auxiliary losses."
        ),
        "fast": True,
        "fast_months": 6,
        "fast_epochs": 2,
        "disable_future_return": True,
        "disable_risk_path": True,
    },
    "classification-latent": {
        "description": (
            "Classification + latent only: disables regression and all auxiliary losses."
        ),
        "fast": True,
        "fast_months": 6,
        "fast_epochs": 2,
        "disable_future_return": True,
        "disable_risk_path": True,
        "disable_regression": True,
    },
    "classification-only": {
        "description": (
            "Pure classification: only direction loss, all auxiliaries disabled."
        ),
        "fast": True,
        "fast_months": 6,
        "fast_epochs": 2,
        "disable_future_return": True,
        "disable_risk_path": True,
        "disable_regression": True,
        "disable_latent": True,
    },
}


def _apply_preset(args: argparse.Namespace, preset_name: str) -> None:
    """Apply a named preset to the parsed args (CLI flags override preset values)."""
    preset = PRESETS[preset_name]
    # Map preset keys to argparse attribute names (hyphens → underscores)
    for key, value in preset.items():
        if key == "description":
            continue
        attr = key.replace("-", "_")
        # Only apply if the CLI did not explicitly set the value
        current = getattr(args, attr, None)
        if isinstance(value, bool):
            # For bool flags: only apply if the user didn't set the flag
            if not current:
                setattr(args, attr, value)
        else:
            # For optional values: only apply if user left it at default (None)
            if current is None:
                setattr(args, attr, value)


def _check_retrain_trigger_cli(config: AppConfig, observations_path: str) -> int:
    """Evaluate the retrain trigger from the persisted outcome stream.

    Exit code 0 = trigger fired (retrain recommended), 1 = no trigger.
    """
    from src.core.logging import configure_logging, get_logger
    from src.training.replay_buffer import load_observations
    from src.training.retrain import check_retrain_trigger_from_observations

    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("run_train")
    observations = load_observations(observations_path)
    should_retrain, reason = check_retrain_trigger_from_observations(
        observations,
        thresholds=config.training.retrain_triggers,
    )
    logger.info(
        "retrain_trigger_check",
        observations_path=str(observations_path),
        observation_count=len(observations),
        thresholds=dict(config.training.retrain_triggers),
        should_retrain=should_retrain,
        reason=reason,
    )
    return 0 if should_retrain else 1


def _checkpoint_meta_extra(checkpoint_path: Path) -> dict[str, object]:
    """Best-effort read of a checkpoint's meta (fingerprint + flat metrics)."""
    from src.training.checkpoints import load_checkpoint_payload

    try:
        meta = load_checkpoint_payload(checkpoint_path).get("meta", {})
    except Exception:  # noqa: BLE001
        return {}
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    return {
        "encoder_fingerprint": meta.get("encoder_fingerprint") if isinstance(meta, dict) else None,
        "extra": extra if isinstance(extra, dict) else {},
    }


def _candidate_metrics_from_result(result: dict[str, object]) -> dict[str, float]:
    """Dotted gate metrics from a run_supervised_training result."""
    val_report = result.get("evaluation_reports", {}).get("val", {})  # type: ignore[union-attr]
    per_class = val_report.get("per_class_metrics", {})
    history = result.get("history")
    val_history = getattr(history, "val", None) or []
    return {
        "val_report.balanced_accuracy": float(val_report.get("balanced_accuracy", 0.0)),
        "val_report.macro_f1": float(val_report.get("macro_f1", 0.0)),
        "val_report.per_class_metrics.long.recall": float(
            per_class.get("long", {}).get("recall", 0.0)
        ),
        "val_report.per_class_metrics.short.recall": float(
            per_class.get("short", {}).get("recall", 0.0)
        ),
        "val.loss": float(val_history[-1].loss) if val_history else 0.0,
    }


def _benchmark_candidate_vs_incumbent(
    config: AppConfig,
    candidate_path: Path,
    incumbent_path: Path,
    logger,
):
    """Scorecard rows for candidate (and incumbent, if any) on real OOS bars.

    Returns (candidate_row, incumbent_row); (None, None) with a loud
    warning when no real OOS data exists (synthetic/test runs) or the
    benchmark fails — the classification gates still apply then.
    """
    if not config.training.use_real_historical_data:
        logger.warning(
            "scorecard_gate_skipped",
            reason="no real OOS data (synthetic run) — classification gates still apply",
        )
        return None, None
    try:
        from src.evaluation.benchmark import run_benchmark
        from src.training.data_loading import _resolve_bars_by_symbol

        bars_by_symbol, _ = _resolve_bars_by_symbol(config)
        candidate_config = config.model_copy(deep=True)
        candidate_config.model.checkpoint_path = str(candidate_path)
        candidate_row = run_benchmark(candidate_config, bars_by_symbol)
        incumbent_row = None
        if incumbent_path.exists():
            incumbent_config = config.model_copy(deep=True)
            incumbent_config.model.checkpoint_path = str(incumbent_path)
            incumbent_row = run_benchmark(incumbent_config, bars_by_symbol)
        return candidate_row, incumbent_row
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scorecard_gate_skipped",
            reason=f"benchmark unavailable: {exc}",
            note="classification gates still apply",
        )
        return None, None


def _promote_platt_sidecar(candidate_path: Path, target_path: Path, logger) -> None:
    """Carry the candidate's Platt calibrator to the canonical location."""
    import os
    import shutil

    from src.training.platt_calibration import platt_calibration_path_for_checkpoint

    source = platt_calibration_path_for_checkpoint(candidate_path)
    if not source.exists():
        return
    target = platt_calibration_path_for_checkpoint(target_path)
    tmp = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)
    logger.info("platt_calibrator_promoted", source=str(source), target=str(target))


def run_canonical_training_with_gate(config: AppConfig, *, force_promote: bool = False) -> int:
    """Canonical (non-isolated) brain training: stage a candidate, gate, promote.

    The freshly trained checkpoint is staged under a per-run directory (a
    CANDIDATE); the canonical ``brain.pt`` is only touched by
    ``promote_checkpoint`` after the gate passes (or ``--force-promote``
    bypasses it with a loud log).  An ungated/refused run leaves the
    canonical checkpoint byte-identical.
    """
    from datetime import UTC, datetime

    from src.core.logging import get_logger
    from src.training.checkpoint_promotion import (
        CheckpointVersion,
        PromotionGate,
        evaluate_promotion,
        promote_checkpoint,
    )

    result = run_supervised_training(config, use_baseline=False, run_id=None, isolate_checkpoint=True)
    logger = get_logger("run_train")
    candidate_path = Path(str(result["checkpoint"]))
    base_dir = Path(config.training.checkpoint_dir)
    canonical_path = base_dir / CANONICAL_CHECKPOINT_FILENAME
    registry_path = base_dir / "registry.json"

    candidate_meta = _checkpoint_meta_extra(candidate_path)
    incumbent_meta = _checkpoint_meta_extra(canonical_path) if canonical_path.exists() else {}
    candidate_metrics = _candidate_metrics_from_result(result)
    # Incumbent metrics come flat from the checkpoint meta; the gate's
    # key adapter maps them onto the dotted schema.
    parent_metrics = None
    incumbent_extra = incumbent_meta.get("extra") or {}
    if incumbent_extra:
        parent_metrics = {
            key: float(value)
            for key, value in incumbent_extra.items()
            if isinstance(value, (int, float))
        }

    candidate_row, incumbent_row = _benchmark_candidate_vs_incumbent(
        config, candidate_path, canonical_path, logger
    )

    promotion_config = config.evaluation.promotion
    gate = PromotionGate(
        min_benchmark_balanced_accuracy_delta=promotion_config.min_balanced_accuracy_delta,
        min_benchmark_expectancy_delta=promotion_config.min_expectancy_delta,
        min_excess_vs_buy_hold=promotion_config.min_excess_vs_buy_hold,
        backward_transfer_floor=promotion_config.backward_transfer_floor,
    )
    decision = evaluate_promotion(
        candidate_metrics=candidate_metrics,
        parent_metrics=parent_metrics,
        gate=gate,
        candidate_benchmark=candidate_row,
        incumbent_benchmark=incumbent_row,
    )

    if not decision.eligible and not force_promote:
        logger.warning(
            "promotion_refused",
            candidate=str(candidate_path),
            canonical_untouched=str(canonical_path),
            reasons=decision.reasons,
            note=(
                "The candidate remains staged at its run directory. Re-run with "
                "--force-promote to bypass the gate (not recommended)."
            ),
        )
        return 0
    if not decision.eligible and force_promote:
        logger.warning(
            "FORCE_PROMOTE_OVERRIDE",
            note=(
                "PROMOTION GATE BYPASSED with --force-promote. The candidate FAILED "
                "the gate and is being promoted anyway."
            ),
            reasons=decision.reasons,
            candidate=str(candidate_path),
        )

    metrics_for_registry = dict(candidate_metrics)
    if candidate_row is not None:
        metrics_for_registry.update(
            {
                "benchmark.balanced_accuracy": float(candidate_row.balanced_accuracy),
                "benchmark.expectancy": float(candidate_row.expectancy),
                "benchmark.excess_vs_buy_hold": float(candidate_row.excess_vs_buy_hold),
            }
        )
    version = CheckpointVersion(
        version_id=str(result.get("run_id", "unknown")),
        checkpoint_path=str(candidate_path),
        fingerprint=str(candidate_meta.get("encoder_fingerprint") or "unknown"),
        parent_fingerprint=(
            str(incumbent_meta["encoder_fingerprint"])
            if incumbent_meta.get("encoder_fingerprint")
            else None
        ),
        created_at=datetime.now(UTC).isoformat(),
        retrain_id=None,
        metrics=metrics_for_registry,
        status="candidate",
        notes="run_train canonical promotion" + (" (FORCED past gate)" if not decision.eligible else ""),
    )
    promote_checkpoint(
        source_path=candidate_path,
        target_path=canonical_path,
        version=version,
        registry_path=registry_path,
    )
    _promote_platt_sidecar(candidate_path, canonical_path, logger)
    logger.info(
        "canonical_checkpoint_promoted",
        candidate=str(candidate_path),
        canonical=str(canonical_path),
        forced=not decision.eligible,
        reasons=decision.reasons,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run supervised training")
    parser.add_argument("--baseline", action="store_true", help="Train the baseline MLP instead of the brain model")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help=(
            "Apply a named experiment preset. Presets set sensible defaults; "
            "explicit CLI flags still override preset values. "
            "Available: " + ", ".join(sorted(PRESETS.keys()))
        ),
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available experiment presets and exit.",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Train from cached/fetched Alpaca historical bars instead of synthetic bars.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Build the dataset and emit target/class diagnostics without training.",
    )
    parser.add_argument(
        "--inspect-labels",
        action="store_true",
        help="Sweep epsilon/horizon values and report class balance profiles.",
    )
    parser.add_argument(
        "--save-memory",
        action="store_true",
        help="Build and persist an associative memory bank from the trained checkpoint immediately after training.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced", "effective_number"],
        default=None,
        help="Override training.class_weight_mode for quick comparison runs.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Enable the weighted sampler for the current run.",
    )
    parser.add_argument(
        "--classification-loss",
        choices=["cross_entropy", "focal"],
        default=None,
        help="Override training.classification_loss for the current run.",
    )
    parser.add_argument(
        "--label-epsilon",
        type=float,
        default=None,
        help="Override training.label_epsilon for the current run.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Override training.focal_gamma for the current run.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Override training.label_smoothing for the current run.",
    )
    parser.add_argument(
        "--disable-regression",
        action="store_true",
        help="Disable regression auxiliary loss for the current run.",
    )
    parser.add_argument(
        "--disable-latent",
        action="store_true",
        help="Disable latent consistency loss for the current run.",
    )
    parser.add_argument(
        "--loss-weight-classification",
        type=float,
        default=None,
        help="Override loss_weights.classification.",
    )
    parser.add_argument(
        "--loss-weight-regression",
        type=float,
        default=None,
        help="Override loss_weights.regression.",
    )
    parser.add_argument(
        "--loss-weight-latent",
        type=float,
        default=None,
        help="Override loss_weights.latent.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs.",
    )
    # --- Fast experiment mode ---
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Enable fast experiment mode. Results are approximate and clearly marked. "
            "Do NOT promote a fast-mode checkpoint to production."
        ),
    )
    parser.add_argument(
        "--fast-months",
        type=int,
        default=None,
        metavar="N",
        help="Restrict training data to last N months (fast mode, overrides data.start_date).",
    )
    parser.add_argument(
        "--fast-symbols",
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="Train on this symbol subset only (fast mode), e.g. --fast-symbols AAPL MSFT.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap training samples at N (fast mode, deterministic with fixed seed).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap validation samples at N (fast mode).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap test samples at N (fast mode).",
    )
    parser.add_argument(
        "--fast-epochs",
        type=int,
        default=None,
        metavar="N",
        help="Cap epochs at N (fast mode, takes min of this and training.epochs).",
    )
    parser.add_argument(
        "--disable-future-return",
        action="store_true",
        help="Disable future-return auxiliary loss for this run (fast mode).",
    )
    parser.add_argument(
        "--disable-risk-path",
        action="store_true",
        help="Disable risk-path auxiliary loss for this run (fast mode).",
    )
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help="Convenience: disable regression, latent, path-label, future-return, and risk-path (pure direction classification).",
    )
    parser.add_argument(
        "--fast-model",
        action="store_true",
        help="Use smaller experimental model profile (fast mode): reduced CNN, temporal, and branch dims.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Custom run ID for this training run.",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        default=None,
        help="Load an existing checkpoint and run the richer val/test evaluation without retraining.",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="Compare run summaries (paths or 'all' for all runs in runs/).",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all saved run summaries.",
    )
    parser.add_argument(
        "--best-run",
        default=None,
        help="Find best run by metric (e.g. val_report.macro_f1).",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Config file(s), later files override earlier files. "
            "Default: configs/default.yaml + configs/training.yaml. "
            "Passing --config REPLACES the default stack (it does not append to it)."
        ),
    )
    parser.add_argument(
        "--check-retrain-trigger",
        action="store_true",
        help=(
            "Evaluate the retrain drift trigger against the persisted per-decision "
            "outcome stream and exit: 0 = trigger fired (retrain recommended), "
            "1 = no trigger. Thresholds come from training.retrain_triggers."
        ),
    )
    parser.add_argument(
        "--observations-path",
        default="runs/observations.jsonl",
        help="JSONL outcome stream for --check-retrain-trigger (DecisionObservation records).",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help=(
            "Bypass the promotion gate after a canonical training run and promote "
            "the candidate to the canonical checkpoint regardless of the decision. "
            "Logged loudly; use only when you understand why the gate refused."
        ),
    )
    parser.add_argument(
        "--anchor-checkpoint",
        default=None,
        help=(
            "Incumbent checkpoint to anchor retraining against (forgetting-safe "
            "retraining). Sets training.anchor_checkpoint_path; pair with "
            "training.distill_lambda_kd / training.proximal_lambda_l2."
        ),
    )
    args = parser.parse_args()
    # argparse action='append' appends to a non-None default instead of
    # replacing it, so the default stack is applied post-parse.
    if args.config is None:
        args.config = ["configs/default.yaml", "configs/training.yaml"]

    # Handle --list-presets
    if args.list_presets:
        print("\nAvailable experiment presets:\n")
        for name in sorted(PRESETS):
            preset = PRESETS[name]
            desc = preset.get("description", "")
            flags = {k: v for k, v in preset.items() if k != "description"}
            print(f"  {name}")
            print(f"    {desc}")
            print(f"    Flags: {flags}\n")
        return 0

    # Apply preset defaults (CLI flags still override)
    if args.preset is not None:
        _apply_preset(args, args.preset)

    # Handle comparison/listing modes that don't need full config
    if args.list_runs:
        from src.training.run_summary import list_run_summaries, load_run_summary
        paths = list_run_summaries()
        if not paths:
            print("No runs found.")
            return 0
        for p in paths:
            s = load_run_summary(p)
            val_f1 = s.get("val_report", {}).get("macro_f1", "n/a")
            val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
            print(f"  {s.get('run_id', p.stem):30s}  val_macro_f1={val_f1_str}  {s.get('timestamp', '')[:19]}")
        return 0

    if args.best_run:
        from src.training.run_summary import find_best_run, _extract_nested
        higher = "loss" not in args.best_run
        best = find_best_run(metric=args.best_run, higher_is_better=higher)
        if best is None:
            print(f"No valid runs for metric: {args.best_run}")
            return 1
        value = _extract_nested(best, args.best_run)
        print(f"Best: {best['run_id']}  {args.best_run}={value}  checkpoint={best.get('checkpoint_path', 'n/a')}")
        return 0

    if args.compare is not None:
        from src.training.run_summary import (
            compare_runs,
            format_comparison_table,
            list_run_summaries,
            load_run_summary,
        )
        if args.compare:
            summaries = [load_run_summary(p) for p in args.compare]
        else:
            summaries = [load_run_summary(p) for p in list_run_summaries()]
        if not summaries:
            print("No runs to compare.")
            return 1
        print(format_comparison_table(compare_runs(summaries)))
        return 0

    config = load_config(*args.config)
    if args.anchor_checkpoint is not None:
        config.training.anchor_checkpoint_path = args.anchor_checkpoint
    if args.check_retrain_trigger:
        return _check_retrain_trigger_cli(config, args.observations_path)
    if args.real_data:
        config.training.use_real_historical_data = True
    if args.save_memory:
        config.training.save_memory_after_training = True
    if args.class_weight_mode is not None:
        config.training.class_weight_mode = args.class_weight_mode
    if args.use_weighted_sampler:
        config.training.use_weighted_sampler = True
    if args.classification_loss is not None:
        config.training.classification_loss = args.classification_loss
    if args.label_epsilon is not None:
        config.training.label_epsilon = args.label_epsilon
    if args.focal_gamma is not None:
        config.training.focal_gamma = args.focal_gamma
    if args.label_smoothing is not None:
        config.training.label_smoothing = args.label_smoothing
    if args.disable_regression:
        config.training.disable_regression = True
    if args.disable_latent:
        config.training.disable_latent = True
    if args.classification_only:
        config.training.disable_regression = True
        config.training.disable_latent = True
        config.training.fast_mode.disable_future_return = True
        config.training.fast_mode.disable_risk_path = True
    if args.loss_weight_classification is not None:
        config.training.loss_weights["classification"] = args.loss_weight_classification
    if args.loss_weight_regression is not None:
        config.training.loss_weights["regression"] = args.loss_weight_regression
    if args.loss_weight_latent is not None:
        config.training.loss_weights["latent"] = args.loss_weight_latent
    if args.epochs is not None:
        config.training.epochs = args.epochs
    # --- Fast experiment mode ---
    if args.fast:
        config.training.fast_mode.enabled = True
    if args.fast_months is not None:
        config.training.fast_mode.fast_months = args.fast_months
        config.training.fast_mode.enabled = True
    if args.fast_symbols is not None:
        config.training.fast_mode.symbols = list(args.fast_symbols)
        config.training.fast_mode.enabled = True
    if args.max_train_samples is not None:
        config.training.fast_mode.max_train_samples = args.max_train_samples
        config.training.fast_mode.enabled = True
    if args.max_val_samples is not None:
        config.training.fast_mode.max_val_samples = args.max_val_samples
        config.training.fast_mode.enabled = True
    if args.max_test_samples is not None:
        config.training.fast_mode.max_test_samples = args.max_test_samples
        config.training.fast_mode.enabled = True
    if args.fast_epochs is not None:
        config.training.fast_mode.max_epochs = args.fast_epochs
        config.training.fast_mode.enabled = True
    if args.disable_future_return:
        config.training.fast_mode.disable_future_return = True
    if args.disable_risk_path:
        config.training.fast_mode.disable_risk_path = True
    if args.fast_model:
        config.training.fast_mode.model_profile = "small"
        config.training.fast_mode.enabled = True

    if args.evaluate_checkpoint:
        evaluate_checkpoint(config, args.evaluate_checkpoint)
    elif args.inspect_labels:
        inspect_label_sensitivity(config)
    elif args.inspect_only:
        inspect_training_data(config)
    else:
        # Canonical brain runs (no explicit run_id, no fast mode) go through
        # the promotion gate: the checkpoint is staged as a candidate and the
        # canonical brain.pt is only replaced by promote_checkpoint (§4.5.4).
        # Isolated/fast/baseline runs keep the previous direct behavior — they
        # never touch the canonical checkpoint.
        canonical = (
            not args.baseline
            and args.run_id is None
            and not config.training.fast_mode.enabled
        )
        if canonical:
            return run_canonical_training_with_gate(config, force_promote=args.force_promote)
        run_supervised_training(config, use_baseline=args.baseline, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
