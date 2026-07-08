"""Profile a minimal backtest. Real numbers, no guessing.

Uses synthetic bars (FAST — no network). Uses the same policy path as
a real backtest. Runs for ~200 bars × 5 symbols = 1000 bar-evals.

At the end prints:
- Total wall-clock
- Per-bar cost (µs)
- Top 30 functions by cumulative time
- Top 30 functions by own time
- Top 30 functions by call count
- Device used
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.core.logging import configure_logging
from src.gui_services.backtest_service import run_backtest


def main() -> int:
    configure_logging(level="INFO", json_logs=False, file_enabled=False)

    print(f"=== DEVICE INFO ===")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device count={torch.cuda.device_count()}")
        print(f"cuda device={torch.cuda.get_device_name(0)}")
    print()

    # Use synthetic bars (no network, reproducible). Reuse backtest_quick preset
    # for tight date range, but synthetic overrides the data source.
    config_paths = ["configs/default.yaml", "configs/backtest_quick.yaml"]

    print(f"=== STARTING PROFILED BACKTEST ===")
    print(f"policy=brain+memory, real_data=False (synthetic), walk_forward=False")
    print()

    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()

    try:
        row = run_backtest(
            config_paths=config_paths,
            policy_kind="brain+memory",
            use_real_data=False,   # synthetic — FAST
            walk_forward=False,
            allow_mock=True,
            label="profile_small",
        )
    except Exception as exc:
        profiler.disable()
        wallclock = time.perf_counter() - t_start
        print(f"\n=== BACKTEST FAILED after {wallclock:.1f}s ===")
        print(f"Error: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    profiler.disable()
    wallclock = time.perf_counter() - t_start

    # Report summary
    print(f"\n=== BACKTEST COMPLETE ===")
    print(f"wallclock={wallclock:.2f}s")
    print(f"label={row.label}")
    print(f"total_return={row.total_return:.4f}")
    print(f"max_drawdown={row.max_drawdown:.4f}")
    print(f"win_rate={row.win_rate:.4f}")
    print(f"fill_count={row.fill_count}")
    print(f"profit_factor={row.profit_factor:.2f}")
    print(f"sharpe={getattr(row, 'sharpe_ratio', 0):.2f}")

    def _print_top(label: str, sort_by: str, n: int = 30):
        print(f"\n=== TOP {n} — sorted by {sort_by} ===")
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats(sort_by)
        stats.print_stats(n)
        print(buf.getvalue())

    _print_top("cumulative time (tottime + callees)", "cumulative", 30)
    _print_top("own time (tottime only)", "tottime", 30)
    _print_top("call count", "calls", 30)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
