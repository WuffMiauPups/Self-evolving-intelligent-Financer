"""Controlled training sweep runner.

Runs a small grid of training experiments over configurable axes,
produces per-run JSON summaries, and identifies the best checkpoint.

Example usage:
    python scripts/sweep_train.py --real-data
    python scripts/sweep_train.py --real-data --sweep-config configs/sweep.yaml
    python scripts/sweep_train.py --real-data --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import AppConfig, load_config
from src.core.logging import configure_logging, get_logger
from src.training.run_summary import (
    compare_runs,
    find_best_run,
    format_comparison_table,
    load_run_summary,
    list_run_summaries,
)


DEFAULT_SWEEP_GRID: dict[str, list[object]] = {
    "label_epsilon": [0.0003, 0.0005, 0.001],
    "classification_loss": ["focal", "cross_entropy"],
    "focal_gamma": [1.5, 2.0],
    "class_weight_mode": ["balanced", "effective_number"],
}


def load_sweep_grid(path: str | None) -> dict[str, list[object]]:
    """Load sweep grid from YAML or return default."""
    if path is None:
        return dict(DEFAULT_SWEEP_GRID)
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    grid = data.get("sweep", data.get("grid", data))
    if not isinstance(grid, dict):
        raise ValueError(f"Sweep config must be a dict, got {type(grid)}")
    return grid


def expand_grid(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    """Expand a grid dict into a list of config override dicts."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def apply_overrides(config: AppConfig, overrides: dict[str, object]) -> AppConfig:
    """Apply sweep overrides to a config copy."""
    config = deepcopy(config)
    training = config.training
    for key, value in overrides.items():
        if key == "label_epsilon":
            training.label_epsilon = float(value)
        elif key == "classification_loss":
            training.classification_loss = str(value)
        elif key == "focal_gamma":
            training.focal_gamma = float(value)
        elif key == "label_smoothing":
            training.label_smoothing = float(value)
        elif key == "class_weight_mode":
            training.class_weight_mode = str(value)
        elif key == "use_weighted_sampler":
            training.use_weighted_sampler = bool(value)
        elif key == "loss_weight_classification":
            training.loss_weights["classification"] = float(value)
        elif key == "loss_weight_regression":
            training.loss_weights["regression"] = float(value)
        elif key == "loss_weight_latent":
            training.loss_weights["latent"] = float(value)
        elif key == "disable_regression":
            training.disable_regression = bool(value)
        elif key == "disable_latent":
            training.disable_latent = bool(value)
        elif key == "forward_horizons":
            training.forward_horizons = list(value)
        elif key == "epochs":
            training.epochs = int(value)
        elif key == "batch_size":
            training.batch_size = int(value)
        elif key == "learning_rate":
            training.learning_rate = float(value)
        else:
            raise ValueError(f"Unknown sweep key: {key}")
    return config


def prune_redundant_combos(combos: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove combos that are redundant (e.g. focal_gamma irrelevant when loss=cross_entropy)."""
    seen: set[str] = set()
    pruned: list[dict[str, object]] = []
    for combo in combos:
        effective = dict(combo)
        if effective.get("classification_loss") == "cross_entropy":
            effective.pop("focal_gamma", None)
        key = json.dumps(effective, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            pruned.append(combo)
    return pruned


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a controlled training sweep")
    parser.add_argument("--real-data", action="store_true", help="Use cached real data")
    parser.add_argument("--sweep-config", default=None, help="Path to sweep grid YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print the sweep plan without running")
    parser.add_argument(
        "--rank-by",
        default="val_report.macro_f1",
        help="Metric to rank runs by (default: val_report.macro_f1)",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/default.yaml", "configs/training.yaml"],
        help="Base config file(s)",
    )
    args = parser.parse_args()

    base_config = load_config(*args.config)
    if args.real_data:
        base_config.training.use_real_historical_data = True

    grid = load_sweep_grid(args.sweep_config)
    combos = expand_grid(grid)
    combos = prune_redundant_combos(combos)

    configure_logging(
        base_config.logging.level,
        base_config.logging.json_logs,
        file_enabled=base_config.logging.file_enabled,
        file_path=base_config.logging.file_path,
    )
    logger = get_logger("sweep")
    logger.info("sweep_plan", total_runs=len(combos), grid_keys=sorted(grid.keys()))

    if args.dry_run:
        print(f"\nSweep plan: {len(combos)} runs")
        print(f"Grid keys: {sorted(grid.keys())}")
        for i, combo in enumerate(combos):
            print(f"  Run {i + 1}: {combo}")
        return 0

    # Import training here to avoid slow imports during --dry-run
    from src.training.supervised import run_supervised_training

    sweep_id = datetime.now(UTC).strftime("sweep_%Y%m%d_%H%M%S")
    summary_paths: list[str] = []

    for i, combo in enumerate(combos):
        run_id = f"{sweep_id}_run{i + 1:03d}"
        logger.info("sweep_run_start", run=i + 1, total=len(combos), run_id=run_id, overrides=combo)

        try:
            run_config = apply_overrides(base_config, combo)
            # Use a unique checkpoint path per run
            run_config.training.checkpoint_dir = f"checkpoints/{sweep_id}"
            checkpoint_name = f"brain_run{i + 1:03d}.pt"
            result = run_supervised_training(run_config, run_id=run_id)
            summary_paths.append(result.get("run_summary_path", ""))

            # Rename checkpoint to unique name
            original_ckpt = Path(run_config.training.checkpoint_dir) / "brain.pt"
            target_ckpt = Path(run_config.training.checkpoint_dir) / checkpoint_name
            if original_ckpt.exists():
                original_ckpt.rename(target_ckpt)

            logger.info(
                "sweep_run_complete",
                run=i + 1,
                run_id=run_id,
                checkpoint=str(target_ckpt),
            )
        except Exception as e:
            logger.error("sweep_run_failed", run=i + 1, run_id=run_id, error=str(e))
            continue

    # Compare all runs
    if summary_paths:
        summaries = []
        for path in summary_paths:
            if path and Path(path).exists():
                summaries.append(load_run_summary(path))
        if summaries:
            comparison = compare_runs(summaries)
            print("\n" + "=" * 80)
            print("SWEEP COMPARISON")
            print("=" * 80)
            print(format_comparison_table(comparison))

            best = find_best_run(
                base_config.training.runs_dir,
                metric=args.rank_by,
            )
            if best:
                print(f"\nBest run by {args.rank_by}: {best['run_id']}")
                print(f"  Checkpoint: {best.get('checkpoint_path', 'n/a')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
