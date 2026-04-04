from pathlib import Path

import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import build_checkpoint_meta, build_model_from_contract, load_model_artifacts
from src.training.checkpoints import save_checkpoint


def test_checkpoint_roundtrip_preserves_model_outputs(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 120)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    model = build_model_from_contract(
        model_kind="brain",
        model_config=config.model,
        feature_contract=snapshot.contract,
    )
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    checkpoint_path = tmp_path / "brain.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        meta=build_checkpoint_meta(
            model_kind="brain",
            model_config=config.model,
            feature_config=config.features,
            feature_contract=snapshot.contract,
        ),
    )
    loaded = load_model_artifacts(checkpoint_path, device=torch.device("cpu"))
    with torch.no_grad():
        original = model(
            branch_inputs={name: value.unsqueeze(0) for name, value in snapshot.branch_inputs.items()},
            regime_input=snapshot.regime_input.unsqueeze(0),
            full_features=snapshot.features.unsqueeze(0),
            temporal_tensor=snapshot.temporal_tensor.unsqueeze(0),
        )
        restored = loaded.model(
            branch_inputs={name: value.unsqueeze(0) for name, value in snapshot.branch_inputs.items()},
            regime_input=snapshot.regime_input.unsqueeze(0),
            full_features=snapshot.features.unsqueeze(0),
            temporal_tensor=snapshot.temporal_tensor.unsqueeze(0),
        )
    assert loaded.feature_contract == snapshot.contract
    assert torch.allclose(original.action_logits, restored.action_logits)
    assert torch.allclose(original.confidence, restored.confidence)
    assert torch.allclose(original.target_fraction, restored.target_fraction)
