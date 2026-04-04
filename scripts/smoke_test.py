from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.execution.trade_engine import run_trade_loop
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager
from src.strategies.baseline_policy import BaselineMomentumPolicy
from src.training.supervised import run_supervised_training


def main() -> int:
    config = load_config("configs/default.yaml", "configs/paper_trade.yaml", "configs/training.yaml")
    config.runtime.symbols = ["AAPL"]
    config.runtime.warmup_bars = 32
    config.runtime.max_runtime_steps = 3
    config.runtime.require_market_open = False
    config.runtime.allowed_sessions = []
    config.broker.polling_interval_sec = 0.01
    config.training.epochs = 1
    config.training.batch_size = 8

    training_result = run_supervised_training(config)
    config.model.checkpoint_path = str(training_result["checkpoint"])

    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bars_by_symbol = {"AAPL": generate_synthetic_bars("AAPL", 160)}
    report = BacktestEngine(config, pipeline, RiskManager(config.risk)).run(
        bars_by_symbol,
        BaselineMomentumPolicy(),
    )
    print(f"backtest_total_return={report.metrics['total_return']:.4f}")

    original_key = os.environ.pop("ALPACA_API_KEY", None)
    original_secret = os.environ.pop("ALPACA_API_SECRET", None)
    try:
        run_trade_loop(config)
    finally:
        if original_key is not None:
            os.environ["ALPACA_API_KEY"] = original_key
        if original_secret is not None:
            os.environ["ALPACA_API_SECRET"] = original_secret
    print("smoke_test=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
