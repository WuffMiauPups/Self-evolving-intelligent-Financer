from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.execution.trade_engine import run_trade_loop


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paper trading loop")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load for the neural policy")
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Enable associative market memory decision support during the runtime loop.",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Second explicit confirmation required before live mode can start.",
    )
    parser.add_argument(
        "--memory-update-mode",
        choices=["disabled", "pending_only", "append_when_ready"],
        default=None,
        help="Override memory runtime_update_mode (default: from config).",
    )
    parser.add_argument(
        "--freeze-memory",
        action="store_true",
        help="Disable all online memory updates (read-only memory).",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Config file(s), later files override earlier files. "
            "Default: configs/default.yaml + configs/paper_trade.yaml. "
            "Passing --config REPLACES the default stack (it does not append to it)."
        ),
    )
    args = parser.parse_args()
    # argparse action='append' appends to a non-None default instead of
    # replacing it, so the default stack is applied post-parse.
    if args.config is None:
        args.config = ["configs/default.yaml", "configs/paper_trade.yaml"]
    config = load_config(*args.config)
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if args.use_memory:
        config.memory.enabled = True
    if args.freeze_memory:
        config.runtime.allow_online_memory_updates = False
    if args.memory_update_mode is not None:
        config.memory.runtime_update_mode = args.memory_update_mode
    run_trade_loop(config, confirm_live=args.confirm_live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
