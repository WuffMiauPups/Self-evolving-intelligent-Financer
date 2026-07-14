"""Perturbed-R² signal diagnostic (MASTER_PLAN §6.4).

Backtests the checkpoint on OOS bars with its direction logits mixed
toward calibrated noise (R² = 1.0 → 0.0) and reports the performance
curve. Healthy models degrade to zero edge; profits that survive pure
noise indicate leakage; a flat curve means structure, not signal.

Usage:
    python scripts/run_perturbation.py --real-data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger

DEFAULT_REPORT_PATH = Path("runs/perturbation_curve.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Perturbed-R² signal diagnostic")
    parser.add_argument(
        "--config",
        nargs="*",
        default=None,
        help=(
            "Config file(s), later files override earlier files. Default: "
            "configs/default.yaml. Passing --config REPLACES the default."
        ),
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override")
    parser.add_argument("--real-data", action="store_true", help="Use cached real historical bars")
    parser.add_argument(
        "--r2-levels",
        nargs="*",
        type=float,
        default=None,
        help="Signal R² levels (default: 1.0 0.5 0.1 0.01 0.0)",
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT_PATH), help="Where to write the JSON report"
    )
    args = parser.parse_args()

    config_paths = args.config if args.config else ["configs/default.yaml"]
    config = load_config(*config_paths)
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("run_perturbation")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path:
        logger.warning("perturbation_requires_checkpoint")
        return 2
    if not args.real_data:
        logger.warning(
            "perturbation_synthetic_refused",
            note="the diagnostic is only meaningful on real OOS data; pass --real-data",
        )
        return 2

    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.oos import enforce_out_of_sample
    from src.evaluation.perturbation import DEFAULT_R2_LEVELS, run_perturbation_curve
    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    bars_by_symbol, _ = enforce_out_of_sample(
        bars_by_symbol, checkpoint_path=config.model.checkpoint_path
    )

    levels = tuple(args.r2_levels) if args.r2_levels else DEFAULT_R2_LEVELS
    report = run_perturbation_curve(
        config,
        bars_by_symbol,
        lambda arm_config, bars, pipeline: build_brain_policy(
            arm_config, bars_by_symbol=bars, pipeline=pipeline
        ),
        r2_levels=levels,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("\n  PERTURBED-R2 CURVE (signal share of variance -> performance)")
    print("  | R2    | sharpe   | total_ret | expectancy | fills |")
    print("  |-------|----------|-----------|------------|-------|")
    for point in report.points:
        m = point.metrics
        print(
            f"  | {point.signal_r2:<5.2f} | {m['sharpe_ratio']:+8.3f} | {m['total_return']:+9.4f} "
            f"| {m['expectancy']:+10.3f} | {int(m['fill_count']):5d} |"
        )
    for note in report.observations:
        print(f"  -> {note}")
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
