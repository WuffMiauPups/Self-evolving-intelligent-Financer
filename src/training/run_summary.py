"""Lightweight structured run summary persistence and comparison."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_run_summary(
    *,
    run_id: str,
    config: object,
    data_meta: dict[str, Any],
    bundle_diagnostics: dict[str, object],
    train_metrics: object | None,
    val_metrics: object | None,
    test_metrics: object | None,
    val_report: dict[str, Any],
    test_report: dict[str, Any],
    checkpoint_path: str,
    elapsed_seconds: float = 0.0,
    effective_epochs: int | None = None,
) -> dict[str, Any]:
    """Build a machine-readable run summary dict."""

    def _metrics_dict(m: object | None) -> dict[str, float]:
        if m is None:
            return {}
        return {
            "loss": float(getattr(m, "loss", 0.0)),
            "accuracy": float(getattr(m, "accuracy", 0.0)),
            "balanced_accuracy": float(getattr(m, "balanced_accuracy", 0.0)),
            "macro_f1": float(getattr(m, "macro_f1", 0.0)),
            "weighted_f1": float(getattr(m, "weighted_f1", 0.0)),
            "long_recall": float(getattr(m, "long_recall", 0.0)),
            "short_recall": float(getattr(m, "short_recall", 0.0)),
            "classification_loss": float(getattr(m, "classification_loss", 0.0)),
            "regression_loss": float(getattr(m, "regression_loss", 0.0)),
            "latent_loss": float(getattr(m, "latent_loss", 0.0)),
            "future_return_loss": float(getattr(m, "future_return_loss", 0.0)),
            "risk_path_loss": float(getattr(m, "risk_path_loss", 0.0)),
        }

    def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
        return {
            "accuracy": float(report.get("accuracy", 0.0)),
            "balanced_accuracy": float(report.get("balanced_accuracy", 0.0)),
            "macro_f1": float(report.get("macro_f1", 0.0)),
            "weighted_f1": float(report.get("weighted_f1", 0.0)),
            "mcc": float(report.get("mcc", 0.0)),
            "long_short_only_accuracy": float(report.get("long_short_only_accuracy", 0.0)),
            "prediction_counts": report.get("prediction_counts", {}),
            "target_counts": report.get("target_counts", {}),
            "prediction_ratios": _floatify(report.get("prediction_ratios", {})),
            "target_ratios": _floatify(report.get("target_ratios", {})),
            "per_class_metrics": _floatify_nested(report.get("per_class_metrics", {})),
            "confusion_matrix": report.get("confusion_matrix", []),
            "long_short_confusion": _floatify(report.get("long_short_confusion", {})),
            "warnings": list(report.get("warnings", [])),
        }

    def _diagnostics_summary(diagnostics: dict[str, object]) -> dict[str, Any]:
        result = {}
        for split_name, diag in diagnostics.items():
            result[split_name] = {
                "total_samples": int(getattr(diag, "total_samples", 0)),
                "class_counts": dict(getattr(diag, "class_counts", {})),
                "class_ratios": _floatify(getattr(diag, "class_ratios", {})),
                "flat_rate": float(getattr(diag, "flat_rate", 0.0)),
                "near_flat_rate": float(getattr(diag, "near_flat_rate", 0.0)),
                "near_boundary_rate": float(getattr(diag, "near_boundary_rate", 0.0)),
                "primary_return_quantiles": _floatify(getattr(diag, "primary_return_quantiles", {})),
                "path_label_counts": dict(getattr(diag, "path_label_counts", {})),
                "path_label_ratios": _floatify(getattr(diag, "path_label_ratios", {})),
                "per_horizon_return_quantiles": {
                    k: _floatify(v) for k, v in dict(getattr(diag, "per_horizon_return_quantiles", {})).items()
                },
                "warnings": list(getattr(diag, "warnings", [])),
            }
        return result

    training_config = config.training if hasattr(config, "training") else None
    data_config = config.data if hasattr(config, "data") else None
    fast_mode_config = getattr(training_config, "fast_mode", None)
    _configured_epochs = int(getattr(training_config, "epochs", 0)) if training_config else 0
    _effective_epochs = effective_epochs if effective_epochs is not None else _configured_epochs

    return {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "checkpoint_path": checkpoint_path,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "config": {
            "symbols": list(getattr(data_config, "symbols", [])) if data_config else [],
            "timeframe": str(getattr(data_config, "timeframe", "")) if data_config else "",
            "start_date": str(getattr(data_config, "start_date", "")) if data_config else "",
            "end_date": str(getattr(data_config, "end_date", "")) if data_config else "",
            "label_epsilon": float(getattr(training_config, "label_epsilon", 0.0)) if training_config else 0.0,
            "forward_horizons": list(getattr(training_config, "forward_horizons", [])) if training_config else [],
            "classification_loss": str(getattr(training_config, "classification_loss", "")) if training_config else "",
            "focal_gamma": float(getattr(training_config, "focal_gamma", 0.0)) if training_config else 0.0,
            "label_smoothing": float(getattr(training_config, "label_smoothing", 0.0)) if training_config else 0.0,
            "class_weight_mode": str(getattr(training_config, "class_weight_mode", "")) if training_config else "",
            "use_weighted_sampler": bool(getattr(training_config, "use_weighted_sampler", False)) if training_config else False,
            "loss_weights": dict(getattr(training_config, "loss_weights", {})) if training_config else {},
            "multitask_enabled": bool(getattr(training_config, "multitask_enabled", True)) if training_config else True,
            "contrastive_enabled": bool(getattr(training_config, "contrastive_enabled", False)) if training_config else False,
            "batch_size": int(getattr(training_config, "batch_size", 0)) if training_config else 0,
            "epochs": _configured_epochs,
            "effective_epochs": _effective_epochs,
            "learning_rate": float(getattr(training_config, "learning_rate", 0.0)) if training_config else 0.0,
            "sequence_length": int(getattr(training_config, "sequence_length", 0)) if training_config else 0,
            # State + future-path design
            "primary_direction_horizon": int(getattr(training_config, "effective_primary_horizon", 0)) if training_config else 0,
            "auxiliary_horizons": list(getattr(training_config, "auxiliary_horizons", [])) if training_config else [],
            "all_horizons": list(getattr(training_config, "all_horizons", [])) if training_config else [],
            "enable_future_return_targets": bool(getattr(training_config, "enable_future_return_targets", False)) if training_config else False,
            "enable_risk_path_targets": bool(getattr(training_config, "enable_risk_path_targets", False)) if training_config else False,
            "path_label_rules": dict(getattr(training_config, "path_label_rules", {})) if training_config else {},
        },
        "data_meta": data_meta,
        "diagnostics": _diagnostics_summary(bundle_diagnostics),
        "train": _metrics_dict(train_metrics),
        "val": _metrics_dict(val_metrics),
        "test": _metrics_dict(test_metrics),
        "val_report": _report_summary(val_report),
        "test_report": _report_summary(test_report),
        "fast_mode": {
            "enabled": bool(getattr(fast_mode_config, "enabled", False)) if fast_mode_config else False,
            "fast_months": getattr(fast_mode_config, "fast_months", None) if fast_mode_config else None,
            "fast_start_date": getattr(fast_mode_config, "fast_start_date", None) if fast_mode_config else None,
            "fast_end_date": getattr(fast_mode_config, "fast_end_date", None) if fast_mode_config else None,
            "symbols": list(getattr(fast_mode_config, "symbols", [])) if fast_mode_config else [],
            "max_train_samples": getattr(fast_mode_config, "max_train_samples", None) if fast_mode_config else None,
            "max_val_samples": getattr(fast_mode_config, "max_val_samples", None) if fast_mode_config else None,
            "max_test_samples": getattr(fast_mode_config, "max_test_samples", None) if fast_mode_config else None,
            "max_epochs": getattr(fast_mode_config, "max_epochs", None) if fast_mode_config else None,
            "effective_epochs": _effective_epochs,
            "disable_future_return": bool(getattr(fast_mode_config, "disable_future_return", False)) if fast_mode_config else False,
            "disable_risk_path": bool(getattr(fast_mode_config, "disable_risk_path", False)) if fast_mode_config else False,
            "model_profile": str(getattr(fast_mode_config, "model_profile", "default")) if fast_mode_config else "default",
            "sample_seed": int(getattr(fast_mode_config, "sample_seed", 42)) if fast_mode_config else 42,
        },
    }


def save_run_summary(summary: dict[str, Any], *, runs_dir: str = "runs") -> Path:
    """Save a run summary as JSON. Returns the path written."""
    path = Path(runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{summary['run_id']}.json"
    filepath = path / filename
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return filepath


def load_run_summary(path: str | Path) -> dict[str, Any]:
    """Load a run summary from JSON."""
    with open(path) as f:
        return json.load(f)


def list_run_summaries(runs_dir: str = "runs") -> list[Path]:
    """List all run summary files sorted by modification time (newest first)."""
    path = Path(runs_dir)
    if not path.exists():
        return []
    files = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def compare_runs(
    summaries: list[dict[str, Any]],
    *,
    focus_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compare multiple run summaries and compute deltas.

    Returns a dict with per-run metrics and delta analysis.
    """
    if not summaries:
        return {"runs": [], "comparison": {}}

    default_metrics = [
        "val_report.balanced_accuracy",
        "val_report.macro_f1",
        "val_report.per_class_metrics.long.recall",
        "val_report.per_class_metrics.long.precision",
        "val_report.per_class_metrics.short.recall",
        "val_report.per_class_metrics.short.precision",
        "val_report.long_short_only_accuracy",
        "test_report.balanced_accuracy",
        "test_report.macro_f1",
        "test_report.per_class_metrics.long.recall",
        "test_report.per_class_metrics.long.precision",
        "test_report.per_class_metrics.short.recall",
        "test_report.per_class_metrics.short.precision",
        "test_report.long_short_only_accuracy",
        "val.loss",
        "test.loss",
    ]
    metrics = focus_metrics or default_metrics

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {
            "run_id": summary.get("run_id", "unknown"),
            "timestamp": summary.get("timestamp", ""),
        }
        for metric_path in metrics:
            row[metric_path] = _extract_nested(summary, metric_path)
        rows.append(row)

    # Compute deltas between consecutive runs
    deltas: list[dict[str, float | None]] = []
    for i in range(1, len(rows)):
        delta: dict[str, float | None] = {"run_id": rows[i]["run_id"]}
        for metric_path in metrics:
            prev = rows[i - 1].get(metric_path)
            curr = rows[i].get(metric_path)
            if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
                delta[metric_path] = round(curr - prev, 6)
            else:
                delta[metric_path] = None
        deltas.append(delta)

    # Find best run per metric
    best: dict[str, dict[str, Any]] = {}
    for metric_path in metrics:
        values = [(row["run_id"], row.get(metric_path)) for row in rows]
        valid = [(rid, v) for rid, v in values if isinstance(v, (int, float))]
        if valid:
            # Lower is better for loss, higher for everything else
            is_loss = "loss" in metric_path
            best_run = min(valid, key=lambda x: x[1]) if is_loss else max(valid, key=lambda x: x[1])
            best[metric_path] = {"run_id": best_run[0], "value": best_run[1]}

    return {
        "runs": rows,
        "deltas": deltas,
        "best": best,
        "metrics": metrics,
    }


def format_comparison_table(comparison: dict[str, Any]) -> str:
    """Format a comparison result as a human-readable table."""
    runs = comparison.get("runs", [])
    metrics = comparison.get("metrics", [])
    best = comparison.get("best", {})
    if not runs:
        return "No runs to compare."

    # Build header
    run_ids = [r["run_id"] for r in runs]
    col_width = max(16, max((len(rid) for rid in run_ids), default=16) + 2)
    metric_width = max(len(m) for m in metrics) + 2 if metrics else 40

    lines: list[str] = []
    header = f"{'metric':<{metric_width}}"
    for rid in run_ids:
        header += f"{rid:>{col_width}}"
    header += f"{'best':>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    for metric_path in metrics:
        row = f"{metric_path:<{metric_width}}"
        for run in runs:
            value = run.get(metric_path)
            if isinstance(value, float):
                row += f"{value:>{col_width}.4f}"
            elif value is not None:
                row += f"{str(value):>{col_width}}"
            else:
                row += f"{'n/a':>{col_width}}"
        best_info = best.get(metric_path)
        if best_info:
            row += f"{best_info['run_id']:>{col_width}}"
        else:
            row += f"{'':>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def find_best_run(
    runs_dir: str = "runs",
    *,
    metric: str = "val_report.macro_f1",
    higher_is_better: bool = True,
) -> dict[str, Any] | None:
    """Find the best run by a given metric."""
    paths = list_run_summaries(runs_dir)
    if not paths:
        return None
    best_summary: dict[str, Any] | None = None
    best_value: float | None = None
    for path in paths:
        summary = load_run_summary(path)
        value = _extract_nested(summary, metric)
        if not isinstance(value, (int, float)):
            continue
        if best_value is None or (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
            best_value = value
            best_summary = summary
    return best_summary


def _extract_nested(data: dict[str, Any], path: str) -> Any:
    """Extract a value from a nested dict using dot-separated path."""
    keys = path.split(".")
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _floatify(d: Any) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    return {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in d.items()}


def _floatify_nested(d: Any) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[str(k)] = _floatify(v)
        else:
            result[str(k)] = v
    return result
