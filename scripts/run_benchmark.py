"""Run the accuracy scorecard benchmark (MASTER_PLAN §2.1).

Grades the configured checkpoint strictly out of sample with next-bar
fills and real costs, compares against tradeable baselines, and appends
one deterministic row to docs/analysis/SCORECARD.md.

Usage:
    python scripts/run_benchmark.py --real-data
    python scripts/run_benchmark.py --checkpoint checkpoints/brain.pt --real-data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.core.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Accuracy scorecard benchmark")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Config file(s), later files override earlier files. "
            "Default: configs/default.yaml. "
            "Passing --config REPLACES the default (it does not append to it)."
        ),
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to grade (default: config.model.checkpoint_path)")
    parser.add_argument("--real-data", action="store_true", help="Use cached/fetched real historical bars")
    parser.add_argument("--allow-mock", action="store_true", help="Allow mock bars without credentials (testing only)")
    parser.add_argument(
        "--allow-in-sample",
        action="store_true",
        help="Permit in-sample evaluation; the scorecard row is tagged IN-SAMPLE.",
    )
    parser.add_argument("--scorecard", default=None, help="Override the scorecard path (default docs/analysis/SCORECARD.md)")
    args = parser.parse_args()
    if args.config is None:
        args.config = ["configs/default.yaml"]
    config = load_config(*args.config)
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint

    # Reuse the backtest CLI's bar loader so both entry points resolve
    # data identically.
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_backtest import _load_bars_by_symbol  # noqa: E402

    from src.evaluation.benchmark import DEFAULT_SCORECARD_PATH, append_scorecard_row, run_benchmark

    use_real_data = args.real_data or config.training.use_real_historical_data
    bars_by_symbol = _load_bars_by_symbol(config, use_real_data=use_real_data, allow_mock=args.allow_mock)
    row = run_benchmark(config, bars_by_symbol, allow_in_sample=args.allow_in_sample)
    path = append_scorecard_row(row, path=args.scorecard or DEFAULT_SCORECARD_PATH)
    print(f"\n  SCORECARD ROW APPENDED -> {path}")
    print(f"  {row.to_markdown()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
