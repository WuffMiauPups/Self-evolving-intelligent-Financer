import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.models.baseline import BaselineMLP
from src.models.brain_model import BrainTradingModel


def test_brain_model_forward_shapes() -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 120)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    model_config = config.model.model_copy(update={"regime_input_dim": snapshot.regime_input.numel()})
    model = BrainTradingModel(
        branch_input_dims={name: tensor.numel() for name, tensor in snapshot.branch_inputs.items()},
        config=model_config,
        temporal_input_dim=snapshot.temporal_tensor.shape[0],
    )
    output = model(
        branch_inputs={name: value.unsqueeze(0) for name, value in snapshot.branch_inputs.items()},
        regime_input=snapshot.regime_input.unsqueeze(0),
        full_features=snapshot.features.unsqueeze(0),
        temporal_tensor=snapshot.temporal_tensor.unsqueeze(0),
    )
    assert output.action_logits.shape == (1, 3)
    assert output.confidence.shape == (1, 1)
    assert output.target_fraction.shape == (1, 1)
    # Post-Simplification: memory_state removed (GRU gone).
    assert output.memory_state is None
    assert output.temporal_embedding is not None


def test_baseline_model_forward_shapes() -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 120)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    model = BaselineMLP(
        input_dim=snapshot.features.numel(),
        hidden_dim=config.model.baseline_hidden_dim,
        dropout=config.model.dropout,
    )
    output = model(
        branch_inputs={},
        regime_input=snapshot.regime_input.unsqueeze(0),
        full_features=snapshot.features.unsqueeze(0),
    )
    assert output.action_logits.shape == (1, 3)
