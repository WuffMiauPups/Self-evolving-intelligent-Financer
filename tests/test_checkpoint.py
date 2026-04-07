from pathlib import Path

import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import build_checkpoint_meta, build_model_from_contract, load_model_artifacts
from src.training.checkpoints import save_checkpoint
from src.training.supervised import resolve_checkpoint_save_dir


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


# ---------------------------------------------------------------------------
# Checkpoint isolation: protect canonical brain.pt from non-canonical runs
# ---------------------------------------------------------------------------
# Background: an earlier fast retrain accidentally clobbered the production
# checkpoint because --run-id only renamed the run-summary JSON, not the
# checkpoint. resolve_checkpoint_save_dir is the path-resolution helper that
# now isolates the checkpoint into a per-run subdirectory whenever the call
# is "non-canonical" (explicit run_id or fast mode), so the canonical
# brain.pt can never be overwritten by an experiment.


def test_resolve_checkpoint_save_dir_canonical_run_writes_to_base() -> None:
    """A plain `python scripts/run_train.py` (auto-generated run_id, no fast)
    must keep writing to checkpoint_dir/brain.pt — the canonical path."""
    target, isolated = resolve_checkpoint_save_dir(
        base_dir=Path("checkpoints"),
        run_id="20260407_120000",
        explicit_run_id=False,
        fast_mode_enabled=False,
        isolate_checkpoint=None,
    )
    assert target == Path("checkpoints")
    assert isolated is False


def test_resolve_checkpoint_save_dir_explicit_run_id_isolates() -> None:
    """When the caller passes --run-id, the checkpoint must land in a
    subdirectory so the canonical brain.pt is never touched."""
    target, isolated = resolve_checkpoint_save_dir(
        base_dir=Path("checkpoints"),
        run_id="experiment_v2",
        explicit_run_id=True,
        fast_mode_enabled=False,
        isolate_checkpoint=None,
    )
    assert target == Path("checkpoints") / "experiment_v2"
    assert isolated is True


def test_resolve_checkpoint_save_dir_fast_mode_isolates() -> None:
    """Fast mode produces non-production checkpoints (--fast help text:
    'Do NOT promote a fast-mode checkpoint to production'). They must
    never overwrite the canonical brain.pt, even with no explicit run_id."""
    target, isolated = resolve_checkpoint_save_dir(
        base_dir=Path("checkpoints"),
        run_id="20260407_120000",
        explicit_run_id=False,
        fast_mode_enabled=True,
        isolate_checkpoint=None,
    )
    assert target == Path("checkpoints") / "20260407_120000"
    assert isolated is True


def test_resolve_checkpoint_save_dir_explicit_false_overrides_auto() -> None:
    """sweep_train.py manages its own per-run checkpoint paths via
    checkpoint_dir override + post-rename. It needs to opt out of
    auto-isolation, otherwise the rename target path would be wrong."""
    target, isolated = resolve_checkpoint_save_dir(
        base_dir=Path("checkpoints/sweep_20260407"),
        run_id="sweep_20260407_run001",
        explicit_run_id=True,
        fast_mode_enabled=False,
        isolate_checkpoint=False,
    )
    assert target == Path("checkpoints/sweep_20260407")
    assert isolated is False


def test_resolve_checkpoint_save_dir_explicit_true_forces_isolation() -> None:
    """Force-isolation: the caller asks for a per-run subdirectory even
    when neither auto-trigger fires."""
    target, isolated = resolve_checkpoint_save_dir(
        base_dir=Path("checkpoints"),
        run_id="20260407_120000",
        explicit_run_id=False,
        fast_mode_enabled=False,
        isolate_checkpoint=True,
    )
    assert target == Path("checkpoints") / "20260407_120000"
    assert isolated is True
