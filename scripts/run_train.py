from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.training.supervised import (
    evaluate_checkpoint,
    inspect_label_sensitivity,
    inspect_training_data,
    run_supervised_training,
)

# ---------------------------------------------------------------------------
# Experiment presets: named configurations for reproducible baselines
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, object]] = {
    "state-only": {
        "description": (
            "State-only baseline: classification + regression + latent only. "
            "Disables path-label, future-return, and risk-path auxiliary losses."
        ),
        "fast": True,
        "fast_months": 6,
        "fast_epochs": 2,
        "disable_path_label": True,
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
        "disable_path_label": True,
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
        "disable_path_label": True,
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
        "--disable-path-label",
        action="store_true",
        help="Disable path-label auxiliary loss for this run (fast mode).",
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
        default=["configs/default.yaml", "configs/training.yaml"],
        help="Config file(s), later files override earlier files.",
    )
    args = parser.parse_args()

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
        config.training.fast_mode.disable_path_label = True
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
    if args.disable_path_label:
        config.training.fast_mode.disable_path_label = True
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
        run_supervised_training(config, use_baseline=args.baseline, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
