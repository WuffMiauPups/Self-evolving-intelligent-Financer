from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.training.supervised import evaluate_checkpoint, inspect_training_data, run_supervised_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Run supervised training")
    parser.add_argument("--baseline", action="store_true", help="Train the baseline MLP instead of the brain model")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Train from cached/fetched Alpaca historical bars instead of synthetic bars.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Build the dataset and emit target/class diagnostics without training.",
    )
    parser.add_argument(
        "--save-memory",
        action="store_true",
        help="Build and persist an associative memory bank from the trained checkpoint immediately after training.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced", "effective_number"],
        default=None,
        help="Override training.class_weight_mode for quick comparison runs.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Enable the weighted sampler for the current run.",
    )
    parser.add_argument(
        "--classification-loss",
        choices=["cross_entropy", "focal"],
        default=None,
        help="Override training.classification_loss for the current run.",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        default=None,
        help="Load an existing checkpoint and run the richer val/test evaluation without retraining.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/default.yaml", "configs/training.yaml"],
        help="Config file(s), later files override earlier files.",
    )
    args = parser.parse_args()
    config = load_config(*args.config)
    if args.real_data:
        config.training.use_real_historical_data = True
    if args.save_memory:
        config.training.save_memory_after_training = True
    if args.class_weight_mode is not None:
        config.training.class_weight_mode = args.class_weight_mode
    if args.use_weighted_sampler:
        config.training.use_weighted_sampler = True
    if args.classification_loss is not None:
        config.training.classification_loss = args.classification_loss
    if args.evaluate_checkpoint:
        evaluate_checkpoint(config, args.evaluate_checkpoint)
    elif args.inspect_only:
        inspect_training_data(config)
    else:
        run_supervised_training(config, use_baseline=args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
