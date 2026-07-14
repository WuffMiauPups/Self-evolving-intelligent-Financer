"""Cap-parameter sweep (MASTER_PLAN §6.4).

Backtests identical OOS bars across a range of values for one risk cap
and reports the knee — the measured operating point that replaces the
hand-picked YAML value. Never mutates config; apply the knee manually.

Usage:
    python scripts/run_cap_sweep.py --real-data --parameter max_position_notional \
        --values 3000 6000 12000 24000 48000
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk-cap sweep with knee selection")
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
        "--parameter",
        default="max_position_notional",
        help="RiskConfig field to sweep (default: max_position_notional)",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        required=True,
        help="Values to sweep, e.g. --values 3000 6000 12000 24000 48000",
    )
    parser.add_argument("--report", default=None, help="JSON report path override")
    args = parser.parse_args()

    config_paths = args.config if args.config else ["configs/default.yaml"]
    config = load_config(*config_paths)
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("run_cap_sweep")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path:
        logger.warning("cap_sweep_requires_checkpoint")
        return 2
    if not args.real_data:
        logger.warning(
            "cap_sweep_synthetic_refused",
            note="knees picked on synthetic data are fiction; pass --real-data",
        )
        return 2

    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.cap_sweep import run_cap_sweep
    from src.evaluation.oos import enforce_out_of_sample
    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    bars_by_symbol, _ = enforce_out_of_sample(
        bars_by_symbol, checkpoint_path=config.model.checkpoint_path
    )

    report = run_cap_sweep(
        config,
        bars_by_symbol,
        lambda arm_config, bars, pipeline: build_brain_policy(
            arm_config, bars_by_symbol=bars, pipeline=pipeline
        ),
        parameter=args.parameter,
        values=list(args.values),
    )

    report_path = Path(args.report or f"runs/cap_sweep_{args.parameter}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print(f"\n  CAP SWEEP: risk.{report.parameter} (current: {report.current_value:g})")
    print("  | value      | total_ret | sharpe   | maxDD    | CVaR95    | ret/|CVaR| | fills |")
    print("  |------------|-----------|----------|----------|-----------|------------|-------|")
    for p in report.points:
        m = p.metrics
        marker = " <-- knee" if p.value == report.knee_value else ""
        print(
            f"  | {p.value:<10g} | {m['total_return']:+9.4f} | {m['sharpe_ratio']:+8.3f} "
            f"| {m['max_drawdown']:+8.4f} | {m['cvar_95']:+9.5f} | {p.risk_adjusted_return:+10.3f} "
            f"| {int(m['fill_count']):5d} |{marker}"
        )
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
