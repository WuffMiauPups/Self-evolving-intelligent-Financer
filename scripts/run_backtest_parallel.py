"""Parallel backtest runner - runs N complete backtests across processes.

SAFE PATTERN: each worker runs a FULL independent backtest (different
configs). Risk constraints stay intact within each worker because the
whole portfolio is in one process.

DOES NOT split a single backtest across symbols - that would violate
global risk constraints (max_concurrent_positions, max_aggregate_notional,
max_daily_loss). If you need symbol-level parallelism, you need a
coordinator process. Not supported here.

Usage:
    python scripts/run_backtest_parallel.py --configs \\
        configs/backtest_quick.yaml \\
        configs/backtest_memory_off.yaml \\
        configs/backtest_baseline.yaml \\
        --policy brain+memory --real-data --workers 3

Each config produces a separate Latest Result + JSON row under runs/backtests/.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_one_backtest(
    config_paths: list[str],
    policy_kind: str,
    use_real_data: bool,
    walk_forward: bool,
    allow_mock: bool,
    label: str,
) -> dict:
    """Run a single backtest in the worker process.

    Imports are done here, NOT at module top, so that spawn-based
    multiprocessing (Windows) only pays the import cost once per worker.
    """
    from src.core.logging import configure_logging
    from src.gui_services.backtest_service import run_backtest

    configure_logging(level="INFO", json_logs=False, file_enabled=False)

    try:
        row = run_backtest(
            config_paths=config_paths,
            policy_kind=policy_kind,
            use_real_data=use_real_data,
            walk_forward=walk_forward,
            allow_mock=allow_mock,
            label=label,
        )
        return {
            "ok": True,
            "label": label,
            "total_return": row.total_return,
            "max_drawdown": row.max_drawdown,
            "win_rate": row.win_rate,
            "fill_count": row.fill_count,
            "profit_factor": row.profit_factor,
            "sharpe_ratio": getattr(row, "sharpe_ratio", 0.0),
            "sortino_ratio": getattr(row, "sortino_ratio", 0.0),
            "config_paths": config_paths,
        }
    except Exception as exc:
        return {
            "ok": False,
            "label": label,
            "error": str(exc),
            "config_paths": config_paths,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run N backtests in parallel (one config per worker).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of config files or full config-stacks (use +-separated list for stacks).",
    )
    parser.add_argument(
        "--policy",
        default="brain+memory",
        choices=["baseline", "brain", "brain+memory"],
    )
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: min(len(configs), cpu_count()/2)).",
    )
    args = parser.parse_args()

    # Each `--configs` entry can be a single path or a +-separated stack
    config_stacks: list[list[str]] = []
    for entry in args.configs:
        paths = entry.split("+")
        # Ensure configs/default.yaml is always the base
        if paths[0] != "configs/default.yaml":
            paths = ["configs/default.yaml"] + paths
        config_stacks.append(paths)

    workers = args.workers or min(len(config_stacks), max(1, (mp.cpu_count() // 2)))
    workers = min(workers, len(config_stacks))

    print(f"Running {len(config_stacks)} backtests across {workers} workers", flush=True)
    for i, stack in enumerate(config_stacks):
        print(f"  [{i}] {stack}", flush=True)

    t0 = time.perf_counter()

    ctx = mp.get_context("spawn")  # Windows-safe
    with ctx.Pool(workers) as pool:
        results = []
        for i, stack in enumerate(config_stacks):
            label = f"parallel_{i:02d}_{Path(stack[-1]).stem}"
            async_res = pool.apply_async(
                _run_one_backtest,
                kwds=dict(
                    config_paths=stack,
                    policy_kind=args.policy,
                    use_real_data=args.real_data,
                    walk_forward=args.walk_forward,
                    allow_mock=args.allow_mock,
                    label=label,
                ),
            )
            results.append((i, label, async_res))

        final = []
        for i, label, async_res in results:
            try:
                res = async_res.get(timeout=3600 * 6)  # 6 hour ceiling
                final.append(res)
            except Exception as exc:
                final.append({"ok": False, "label": label, "error": str(exc)})

    elapsed = time.perf_counter() - t0
    print(f"\nAll backtests complete in {elapsed:.1f}s\n", flush=True)
    print(f"{'Label':<40} {'Status':<6} {'Return':>10} {'Sharpe':>8} {'Fills':>8}")
    print("-" * 80)
    for res in final:
        if res.get("ok"):
            print(
                f"{res['label']:<40} OK     "
                f"{res['total_return']:>+10.4f} "
                f"{res['sharpe_ratio']:>8.2f} "
                f"{res['fill_count']:>8}"
            )
        else:
            print(f"{res['label']:<40} FAIL   {res.get('error', 'unknown')}")

    return 0 if all(r.get("ok") for r in final) else 1


if __name__ == "__main__":
    mp.freeze_support()  # Windows-required for spawn
    raise SystemExit(main())
