import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline


def test_feature_pipeline_builds_expected_groups() -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 120)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    assert snapshot.features.ndim == 1
    assert snapshot.regime_input.numel() >= 4
    assert "momentum_5" in snapshot.feature_names
    assert snapshot.branch_inputs["trend"].numel() > 0
    assert snapshot.branch_inputs["volatility"].numel() > 0
    assert snapshot.contract.feature_dim == snapshot.features.numel()


def test_feature_contract_is_consistent_across_training_backtest_and_runtime() -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 160)

    training_pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    # The training feature contract is a property of the pipeline+config, so
    # source it straight from the pipeline (the retired BarWindowDataset only
    # ever surfaced this same contract).
    training_contract = training_pipeline.build(bars, symbol="AAPL").contract
    assert training_contract is not None

    runtime_pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    runtime_snapshot = runtime_pipeline.build(bars[:120], symbol="AAPL")
    backtest_pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    backtest_snapshot = backtest_pipeline.build(bars[:140], symbol="AAPL")

    training_contract.assert_matches(runtime_snapshot.contract, context="runtime")
    training_contract.assert_matches(backtest_snapshot.contract, context="backtest")
