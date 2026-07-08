"""Experiment reporting: aggregation, comparison tables, and file output."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.walk_forward import FoldResult, WalkForwardResult

logger = logging.getLogger(__name__)


def aggregate_fold_metrics(
    fold_results: list[FoldResult],
) -> dict[str, dict[str, float]]:
    """Compute per-metric aggregate statistics across folds.

    Returns {metric: {mean, std, median, min, max, count, profitable_folds}}.
    """
    import math
    import statistics

    if not fold_results:
        return {}

    all_keys: set[str] = set()
    for fr in fold_results:
        all_keys.update(fr.test_metrics.keys())

    aggregate: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [
            fr.test_metrics[key]
            for fr in fold_results
            if key in fr.test_metrics and math.isfinite(fr.test_metrics[key])
        ]
        if not values:
            continue
        mean_val = sum(values) / len(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        median_val = statistics.median(values)
        aggregate[key] = {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "min": min(values),
            "max": max(values),
            "count": float(len(values)),
        }
        if key == "total_return":
            aggregate[key]["profitable_folds"] = float(sum(1 for v in values if v > 0))
            aggregate[key]["profitable_pct"] = aggregate[key]["profitable_folds"] / len(values)

    return aggregate


def build_comparison_table(
    experiment_results: dict[str, WalkForwardResult | dict[str, float]],
) -> dict[str, Any]:
    """Build a structured comparison across named experiments.

    Accepts either WalkForwardResult (uses aggregate mean) or a raw
    metrics dict (single-backtest mode).
    """
    # Key metrics to compare (ordered by importance)
    comparison_keys = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "win_rate", "profit_factor", "expectancy",
        "fill_count", "round_trip_count", "avg_exposure", "flat_fraction",
        "long_count", "short_count", "avg_holding_bars", "total_fees",
    ]

    rows: dict[str, dict[str, float]] = {}
    for name, result in experiment_results.items():
        if isinstance(result, WalkForwardResult):
            metrics = {
                k: v.get("mean", 0.0) for k, v in result.aggregate_metrics.items()
            }
        else:
            metrics = result
        rows[name] = {k: metrics.get(k, 0.0) for k in comparison_keys}

    return {
        "experiments": list(rows.keys()),
        "metrics": comparison_keys,
        "values": rows,
    }


def format_ranking_table(
    experiment_results: dict[str, WalkForwardResult | dict[str, float]],
) -> str:
    """Format a human-readable ranking table of experiments.

    Ranks by Sharpe ratio by default. Shows key metrics side-by-side.
    """
    comparison = build_comparison_table(experiment_results)
    rows = comparison["values"]
    if not rows:
        return "No experiments to compare."

    # Sort by sharpe_ratio descending
    sorted_names = sorted(rows.keys(), key=lambda n: rows[n].get("sharpe_ratio", 0.0), reverse=True)

    display_metrics = [
        ("return", "total_return", "+.4f"),
        ("sharpe", "sharpe_ratio", "+.3f"),
        ("sortino", "sortino_ratio", "+.3f"),
        ("maxDD", "max_drawdown", ".4f"),
        ("winrate", "win_rate", ".3f"),
        ("PF", "profit_factor", ".3f"),
        ("fills", "fill_count", ".0f"),
        ("flat%", "flat_fraction", ".3f"),
    ]

    # Header
    col_w = 10
    name_w = 25
    header = f"{'Experiment':<{name_w}}" + "".join(f"{label:>{col_w}}" for label, _, _ in display_metrics)
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for rank, name in enumerate(sorted_names, 1):
        vals = rows[name]
        cells = []
        for _, key, fmt in display_metrics:
            v = vals.get(key, 0.0)
            cells.append(f"{v:{fmt}}")
        line = f"{rank}. {name:<{name_w - 3}}" + "".join(f"{c:>{col_w}}" for c in cells)
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)


def save_experiment_results(
    result: WalkForwardResult | dict[str, float],
    output_dir: Path,
    experiment_name: str,
    *,
    git_hash: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Save experiment artifacts to a timestamped directory.

    Structure::

        {output_dir}/{experiment_name}/{timestamp}/
            config.json
            fold_definitions.json
            metrics.json
            summary.txt
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(result, WalkForwardResult):
        # Config snapshot
        (run_dir / "config.json").write_text(
            json.dumps(result.config_snapshot, indent=2, default=str),
            encoding="utf-8",
        )

        # Fold definitions
        fold_data = [
            {
                "fold_index": f.fold_index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "val_start": f.val_start,
                "val_end": f.val_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
            }
            for f in result.fold_definitions
        ]
        (run_dir / "fold_definitions.json").write_text(
            json.dumps(fold_data, indent=2), encoding="utf-8",
        )

        # Per-fold metrics
        fold_metrics = [
            {
                "fold_index": fr.fold_definition.fold_index,
                "elapsed_seconds": fr.elapsed_seconds,
                **fr.test_metrics,
            }
            for fr in result.folds
        ]
        metrics_data = {
            "experiment_name": experiment_name,
            "git_hash": git_hash,
            "timestamp": timestamp,
            "fold_count": len(result.folds),
            "total_elapsed_seconds": result.total_elapsed_seconds,
            "aggregate": result.aggregate_metrics,
            "per_fold": fold_metrics,
        }
    else:
        # Single-backtest mode
        metrics_data = {
            "experiment_name": experiment_name,
            "git_hash": git_hash,
            "timestamp": timestamp,
            "fold_count": 0,
            "metrics": result,
        }

    if extra_meta:
        metrics_data["extra"] = extra_meta

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_data, indent=2, default=str), encoding="utf-8",
    )

    # Human-readable summary
    summary_lines = [
        f"Experiment: {experiment_name}",
        f"Timestamp: {timestamp}",
        f"Git hash: {git_hash or 'unknown'}",
        "",
    ]
    if isinstance(result, WalkForwardResult):
        summary_lines.append(f"Folds: {len(result.folds)}")
        summary_lines.append(f"Total time: {result.total_elapsed_seconds:.1f}s")
        summary_lines.append("")
        for key in ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
                     "win_rate", "profit_factor", "fill_count"]:
            agg = result.aggregate_metrics.get(key, {})
            if agg:
                summary_lines.append(
                    f"  {key}: mean={agg.get('mean', 0):.4f}  "
                    f"std={agg.get('std', 0):.4f}  "
                    f"min={agg.get('min', 0):.4f}  "
                    f"max={agg.get('max', 0):.4f}"
                )
    else:
        summary_lines.append("Mode: single backtest")
        for key in sorted(result.keys()):
            summary_lines.append(f"  {key}: {result[key]:.6f}")

    (run_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("experiment_saved path=%s", run_dir)
    return run_dir


def save_comparison_csv(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a comparison table to CSV for spreadsheet analysis."""
    experiments = comparison["experiments"]
    metrics = comparison["metrics"]
    values = comparison["values"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + metrics)
        for name in experiments:
            row = [name] + [values[name].get(m, 0.0) for m in metrics]
            writer.writerow(row)
