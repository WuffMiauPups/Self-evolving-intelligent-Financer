"""Compare training runs side by side.

Usage:
    python scripts/compare_runs.py                          # compare all runs
    python scripts/compare_runs.py runs/run1.json runs/run2.json  # specific runs
    python scripts/compare_runs.py --best val_report.macro_f1     # find best run
    python scripts/compare_runs.py --list                         # list all runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.run_summary import (
    compare_runs,
    find_best_run,
    format_comparison_table,
    list_run_summaries,
    load_run_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare training runs")
    parser.add_argument("runs", nargs="*", help="Run summary JSON files to compare")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run summaries")
    parser.add_argument("--list", action="store_true", help="List all available runs")
    parser.add_argument("--best", default=None, help="Find best run by metric (e.g. val_report.macro_f1)")
    parser.add_argument("--last", type=int, default=None, help="Compare the last N runs")
    args = parser.parse_args()

    if args.list:
        paths = list_run_summaries(args.runs_dir)
        if not paths:
            print("No runs found.")
            return 0
        print(f"Found {len(paths)} run(s) in {args.runs_dir}/:")
        for p in paths:
            summary = load_run_summary(p)
            rid = summary.get("run_id", p.stem)
            ts = summary.get("timestamp", "")
            val_f1 = summary.get("val_report", {}).get("macro_f1", "n/a")
            test_f1 = summary.get("test_report", {}).get("macro_f1", "n/a")
            val_ba = summary.get("val_report", {}).get("balanced_accuracy", "n/a")
            eps = summary.get("config", {}).get("label_epsilon", "n/a")
            loss_type = summary.get("config", {}).get("classification_loss", "n/a")
            val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
            test_f1_str = f"{test_f1:.4f}" if isinstance(test_f1, float) else str(test_f1)
            val_ba_str = f"{val_ba:.4f}" if isinstance(val_ba, float) else str(val_ba)
            print(
                f"  {rid:30s} {ts[:19]:>20s}  "
                f"val_f1={val_f1_str}  test_f1={test_f1_str}  val_ba={val_ba_str}  "
                f"eps={eps}  loss={loss_type}"
            )
        return 0

    if args.best:
        higher = "loss" not in args.best
        best = find_best_run(args.runs_dir, metric=args.best, higher_is_better=higher)
        if best is None:
            print(f"No valid runs found for metric: {args.best}")
            return 1
        print(f"Best run by {args.best}: {best['run_id']}")
        print(f"  Checkpoint: {best.get('checkpoint_path', 'n/a')}")
        from src.training.run_summary import _extract_nested
        value = _extract_nested(best, args.best)
        print(f"  {args.best}: {value}")
        return 0

    # Load summaries
    if args.runs:
        summaries = [load_run_summary(p) for p in args.runs]
    else:
        paths = list_run_summaries(args.runs_dir)
        if args.last:
            paths = paths[:args.last]
        summaries = [load_run_summary(p) for p in paths]

    if not summaries:
        print("No runs to compare. Run training first or specify run files.")
        return 1

    comparison = compare_runs(summaries)
    print(format_comparison_table(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
