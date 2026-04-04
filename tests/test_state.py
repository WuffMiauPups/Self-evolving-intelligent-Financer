from dataclasses import replace
from datetime import timedelta

import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import build_model_from_contract
from src.strategies.brain_policy import BrainPolicy


def test_brain_policy_resets_state_on_sequence_gap() -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 140)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    first_snapshot = pipeline.build(bars[:100], symbol="AAPL")
    second_snapshot = pipeline.build(bars[:101], symbol="AAPL")
    second_snapshot = replace(
        second_snapshot,
        timestamp=first_snapshot.timestamp + timedelta(seconds=(config.runtime.state_reset_gap_sec or 0) + 5),
    )
    model = build_model_from_contract(
        model_kind="brain",
        model_config=config.model,
        feature_contract=first_snapshot.contract,
    )
    model.eval()
    policy = BrainPolicy(
        model=model,
        device=torch.device("cpu"),
        feature_contract=first_snapshot.contract,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
    )
    policy.propose(first_snapshot)
    assert "AAPL" in policy.memory_state_by_symbol
    assert policy.request_seq_by_symbol["AAPL"] == 1

    _, diagnostics = policy.propose(second_snapshot)
    assert diagnostics["state_reset_reason"] == "state_gap_exceeded"
    assert policy.request_seq_by_symbol["AAPL"] == 1

    policy.reset_all_state(reason="test_end")
    assert not policy.memory_state_by_symbol
    assert not policy.last_timestamp_by_symbol


def test_brain_policy_does_not_reset_on_equal_timestamp_poll() -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 140)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    first_snapshot = pipeline.build(bars[:100], symbol="AAPL")
    second_snapshot = replace(
        pipeline.build(bars[:101], symbol="AAPL"),
        timestamp=first_snapshot.timestamp,
    )
    model = build_model_from_contract(
        model_kind="brain",
        model_config=config.model,
        feature_contract=first_snapshot.contract,
    )
    model.eval()
    policy = BrainPolicy(
        model=model,
        device=torch.device("cpu"),
        feature_contract=first_snapshot.contract,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
    )
    policy.propose(first_snapshot)
    _, diagnostics = policy.propose(second_snapshot)
    assert diagnostics["state_reset_reason"] is None
    assert policy.request_seq_by_symbol["AAPL"] == 2
