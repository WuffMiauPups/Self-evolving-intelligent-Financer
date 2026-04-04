import torch

from src.backtest.engine import BacktestEngine
from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager
from src.strategies.baseline_policy import BaselineMomentumPolicy


def test_backtest_basic_flow() -> None:
    config = load_config("configs/default.yaml")
    config.runtime.symbols = ["AAPL"]
    config.runtime.warmup_bars = 32
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    engine = BacktestEngine(config, pipeline, RiskManager(config.risk))
    report = engine.run({"AAPL": generate_synthetic_bars("AAPL", 180)}, BaselineMomentumPolicy())
    assert report.equity_curve
    assert "total_return" in report.metrics
    assert report.metrics["fill_count"] >= 0.0

