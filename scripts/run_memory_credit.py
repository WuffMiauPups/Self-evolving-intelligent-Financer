"""Counterfactual memory-credit evaluation (MASTER_PLAN §6.2).

Runs paired brain-only vs memory-damped backtests over out-of-sample
folds and reports whether memory influence EARNS a positive credit.
This is the gate for flipping ``memory.influence_mode`` to "damping":
exit code 0 = credit positive, 1 = keep memory influence off.

Usage:
    python scripts/run_memory_credit.py --real-data
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

DEFAULT_REPORT_PATH = Path("runs/memory_credit.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Counterfactual memory-credit evaluation")
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
    parser.add_argument("--folds", type=int, default=4, help="Number of contiguous OOS folds")
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
    logger = get_logger("run_memory_credit")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path:
        logger.warning("memory_credit_requires_checkpoint")
        return 2
    if not args.real_data:
        logger.warning(
            "memory_credit_synthetic_refused",
            note="the credit gate is only meaningful on real OOS data; pass --real-data",
        )
        return 2
    config.memory.enabled = True

    from src.backtest.policy_factory import build_brain_memory_policy
    from src.evaluation.memory_credit import evaluate_memory_credit
    from src.evaluation.oos import enforce_out_of_sample
    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    bars_by_symbol, _ = enforce_out_of_sample(
        bars_by_symbol, checkpoint_path=config.model.checkpoint_path
    )

    def policy_builder(arm_config, fold_bars, pipeline):
        return build_brain_memory_policy(
            arm_config, bars_by_symbol=fold_bars, pipeline=pipeline
        )

    report = evaluate_memory_credit(
        config, bars_by_symbol, policy_builder, n_folds=args.folds
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(
        f"\n  MEMORY CREDIT: {'POSITIVE' if report.credit_positive else 'NOT EARNED'}"
        f" | mean dTotalReturn={report.mean_delta_total_return:+.5f}"
        f" | positive folds {report.positive_folds}/{report.active_folds}"
        f" (degenerate: {report.degenerate_folds})"
        f" | sign-test p={report.sign_test_p_value:.3f}"
        f"\n  Report -> {report_path}"
    )
    if report.credit_positive:
        print("  -> memory.influence_mode=damping is justified by this data.")
        return 0
    print("  -> keep memory.influence_mode=off (§6.2 gate).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
