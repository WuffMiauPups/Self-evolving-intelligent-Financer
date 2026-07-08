from __future__ import annotations

import argparse

from src.core.config import AppConfig, load_config
from src.execution.trade_engine import run_trade_loop
from src.training.supervised import run_supervised_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SEIF trading brain entrypoint")
    parser.add_argument(
        "command",
        choices=["paper", "train"],
        help="Run paper trading or training.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Config path(s), later files override earlier files. "
            "REPLACES the default stack (configs/default.yaml) when given."
        ),
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Second explicit confirmation required before live mode can start.",
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Enable associative market memory during supported runtime commands.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_paths = args.config if args.config else ["configs/default.yaml"]
    config: AppConfig = load_config(*config_paths)
    if args.use_memory:
        config.memory.enabled = True
    if args.command == "paper":
        run_trade_loop(config, confirm_live=args.confirm_live)
        return 0
    if args.command == "train":
        run_supervised_training(config)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
