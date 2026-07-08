"""Profile the backtest hot path. Run:
    python scripts/profile_backtest.py

Outputs:
- Top 20 functions by cumulative time
- Top 20 functions by own time (excluding callees)
- Estimated per-bar cost for each hot function
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# torch.compile with the inductor backend requires a C++ compiler (MSVC `cl`
# on Windows).  The profile script is meant to be runnable on any dev box, so
# neutralize torch.compile up-front — we are interested in the PYTHON-level
# hot path, not the compiled-kernel path anyway.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch._dynamo  # noqa: E402

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Also neutralize `torch.compile` directly so the call inside the model
# factory becomes a no-op rather than a dynamo-wrapped forward.  This keeps
# the timings in the cProfile report attributable to the real Python
# functions rather than to dynamo/inductor internals.
import torch  # noqa: E402

torch.compile = lambda model, **_kwargs: model  # type: ignore[assignment]

from src.core.logging import configure_logging  # noqa: E402
from src.core.config import load_config  # noqa: E402,F401  (kept per spec)
from src.gui_services.backtest_service import run_backtest  # noqa: E402


def main():
    # Keep logging at INFO so we can see phase timings
    configure_logging(level="INFO", json_logs=False, file_enabled=False)

    print("=== Starting backtest profile ===", flush=True)

    profiler = cProfile.Profile()
    profiler.enable()

    # Run a small real-data backtest (quick preset for speed)
    row = run_backtest(
        config_paths=["configs/default.yaml", "configs/backtest_quick.yaml"],
        policy_kind="brain+memory",
        use_real_data=True,
        walk_forward=False,
        allow_mock=True,
        label="profile_run",
    )

    profiler.disable()

    # ---- Report ----
    print(f"\n=== Backtest complete ===", flush=True)
    print(f"return={row.total_return:.4f} fills={row.fill_count} winrate={row.win_rate:.2%}")

    print("\n=== TOP 20 by cumulative time (tottime + callees) ===")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)

    print("\n=== TOP 20 by own time (tottime only) ===")
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(20)

    print("\n=== TOP 20 by call count ===")
    stats = pstats.Stats(profiler).sort_stats("calls")
    stats.print_stats(20)


if __name__ == "__main__":
    raise SystemExit(main())
