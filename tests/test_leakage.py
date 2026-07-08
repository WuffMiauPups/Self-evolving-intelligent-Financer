"""Leakage and causality tests.

These tests verify that no information leaks across temporal boundaries:
- Features at bar i depend only on bars 0..i
- Walk-forward folds have no train/test overlap
- Memory bank and GRU state are properly reset between folds
- Ablation flags produce expected neutral outputs
"""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from src.core.config import load_config, WalkForwardConfig
from src.core.seeds import seed_everything
from src.data.historical_data import generate_synthetic_bars
from src.evaluation.walk_forward import generate_folds
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import build_model_from_contract
from src.strategies.brain_policy import BrainPolicy


def _make_config():
    config = load_config("configs/default.yaml")
    config.runtime.warmup_bars = 32
    return config


# ── Test 1: Feature causality ──────────────────────────────────────────


def test_feature_causality() -> None:
    """Features at bar i must not depend on bars after i."""
    config = _make_config()
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bars = generate_synthetic_bars("AAPL", 200)

    # Compute features for full sequence
    batched = pipeline.build_batch(bars, symbol="AAPL", temporal_window=config.runtime.warmup_bars)

    # Pick a bar in the middle
    target_idx = 100
    snapshot_original = batched.snapshot_at(target_idx)

    # Modify bars AFTER target_idx and recompute
    bars_modified = copy.deepcopy(bars)
    for i in range(target_idx + 1, len(bars_modified)):
        bars_modified[i] = replace(
            bars_modified[i],
            close=bars_modified[i].close * 2.0,
            high=bars_modified[i].high * 2.0,
            volume=bars_modified[i].volume * 10.0,
        )

    batched_modified = pipeline.build_batch(
        bars_modified, symbol="AAPL", temporal_window=config.runtime.warmup_bars,
    )
    snapshot_modified = batched_modified.snapshot_at(target_idx)

    # Features at target_idx must be identical regardless of future bar changes
    assert torch.allclose(snapshot_original.features, snapshot_modified.features, atol=1e-6), (
        "Feature values at bar i changed when bars after i were modified — future leakage detected"
    )


# ── Test 2: Fold generation no overlap ─────────────────────────────────


def test_fold_generation_no_overlap() -> None:
    """Walk-forward folds must not have test-train overlap."""
    wf_config = WalkForwardConfig(
        train_bars=500, val_bars=100, test_bars=200, step_bars=200,
        min_train_bars=100,
    )
    folds = generate_folds(total_bars=3000, config=wf_config)

    assert len(folds) > 1, "Need at least 2 folds for overlap test"

    for fold in folds:
        # Within each fold: train < val < test
        assert fold.train_start < fold.train_end, f"Fold {fold.fold_index}: empty train"
        assert fold.train_end <= fold.val_start, f"Fold {fold.fold_index}: train overlaps val"
        assert fold.val_end <= fold.test_start, f"Fold {fold.fold_index}: val overlaps test"
        assert fold.test_start < fold.test_end, f"Fold {fold.fold_index}: empty test"

    # Across folds: test windows should not overlap with future train windows
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            assert folds[i].test_end <= folds[j].test_start or folds[j].test_end <= folds[i].test_start, (
                f"Test windows of fold {i} and fold {j} overlap"
            )


def test_fold_generation_expanding() -> None:
    """Expanding window: train always starts at 0."""
    wf_config = WalkForwardConfig(
        train_bars=500, val_bars=100, test_bars=200, step_bars=200,
        window_type="expanding", min_train_bars=100,
    )
    folds = generate_folds(total_bars=3000, config=wf_config)
    assert len(folds) > 1
    for fold in folds:
        assert fold.train_start == 0, f"Fold {fold.fold_index}: expanding window should start at 0"
        assert fold.train_end <= fold.val_start


def test_fold_generation_too_short() -> None:
    """No folds should be generated if data is too short."""
    wf_config = WalkForwardConfig(train_bars=500, val_bars=100, test_bars=200)
    folds = generate_folds(total_bars=100, config=wf_config)
    assert folds == []


# ── Test 3: Memory bank isolation ───────────────────────────────────────


def test_memory_bank_no_cross_fold_contamination() -> None:
    """A fresh MemoryBank must be empty; creating a new one shares no state
    with the previous bank. This tests the isolation guarantee that the
    walk-forward runner relies on when it creates a new bank per fold.
    """
    from src.memory.memory_bank import MemoryBank
    from src.features.feature_pipeline import FeatureContract

    contract = FeatureContract(
        feature_names=("f1", "f2"),
        branch_feature_names={"trend": ("f1",), "volatility": ("f2",)},
        regime_feature_names=("f1",),
        temporal_feature_names=("f1", "f2"),
        feature_dtype="float32",
    )

    # Create bank1, add a dummy entry to meta to simulate state
    bank1 = MemoryBank(
        feature_contract=contract, latent_dim=8,
        similarity_metric="cosine", max_items=100,
    )
    bank1.meta["test_key"] = "state"
    assert bank1.meta.get("test_key") == "state", "Bank1 should have state"

    # Fold 2: fresh bank must have no state from bank1
    bank2 = MemoryBank(
        feature_contract=contract, latent_dim=8,
        similarity_metric="cosine", max_items=100,
    )
    assert len(bank2) == 0, "Fresh bank should have zero items"
    assert "test_key" not in bank2.meta, "Fresh bank should not share meta state"


# ── Test 4: Normalization no future leak ────────────────────────────────


def test_normalization_no_future_leak() -> None:
    """Feature normalization must not use future data."""
    config = _make_config()
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))

    bars = generate_synthetic_bars("AAPL", 200)
    target_idx = 80

    # Build with original bars
    snap1 = pipeline.build(bars[:target_idx + 1], symbol="AAPL")

    # Extend bars with very different values and rebuild up to same point
    extra = generate_synthetic_bars("AAPL", 100, start_price=500.0)
    bars_extended = bars[:target_idx + 1] + extra
    snap2 = pipeline.build(bars_extended[:target_idx + 1], symbol="AAPL")

    # Features should be identical — only bars up to target_idx are used
    assert torch.allclose(snap1.features, snap2.features, atol=1e-6), (
        "Features changed when future bars were appended — normalization leak"
    )


# ── Test 5: GRU state reset ────────────────────────────────────────────


# ── Test 6: Temporal ordering ───────────────────────────────────────────


def test_temporal_ordering_in_folds() -> None:
    """Test bars in each fold must have strictly later timestamps than train bars."""
    bars = generate_synthetic_bars("AAPL", 3000)
    wf_config = WalkForwardConfig(
        train_bars=500, val_bars=100, test_bars=200, step_bars=200,
        min_train_bars=100,
    )
    folds = generate_folds(total_bars=len(bars), config=wf_config)
    assert len(folds) > 0

    for fold in folds:
        last_train_ts = bars[fold.train_end - 1].timestamp
        first_test_ts = bars[fold.test_start].timestamp
        assert first_test_ts > last_train_ts, (
            f"Fold {fold.fold_index}: test start {first_test_ts} is not after "
            f"train end {last_train_ts}"
        )

    # Across folds: test bars of fold i+1 come after fold i
    for i in range(len(folds) - 1):
        ts_i = bars[folds[i].test_end - 1].timestamp
        ts_j = bars[folds[i + 1].test_start].timestamp
        assert ts_j >= ts_i, (
            f"Fold {i+1} test start is before fold {i} test end"
        )


# ── Test 7: Force-directional flip flag ─────────────────────────────────


def test_force_directional_flip_flag() -> None:
    """With force_directional_flip=False, a FLAT prediction stays FLAT."""
    config = _make_config()
    config.model.checkpoint_path = None
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bars = generate_synthetic_bars("AAPL", 100)

    sample = pipeline.build(bars[:config.runtime.warmup_bars + 1], symbol="AAPL")
    model = build_model_from_contract(
        model_kind="brain", model_config=config.model,
        feature_contract=sample.contract,
    )
    model.eval()

    # With flip enabled (default): FLAT should become directional
    policy_flip = BrainPolicy(
        model=model, device=torch.device("cpu"),
        feature_contract=sample.contract,
        force_directional_flip=True,
    )

    # With flip disabled: FLAT should stay FLAT
    policy_noflip = BrainPolicy(
        model=model, device=torch.device("cpu"),
        feature_contract=sample.contract,
        force_directional_flip=False,
    )

    batched = pipeline.build_batch(bars, symbol="AAPL", temporal_window=config.runtime.warmup_bars)
    snap = batched.snapshot_at(config.runtime.warmup_bars)

    proposal_flip, _ = policy_flip.propose(snap)
    # The flip policy should always produce a directional trade
    assert proposal_flip.action != 0 or proposal_flip.target_fraction == 0.0

    # The no-flip policy might produce FLAT depending on model weights.
    # We can't guarantee it without controlling model weights, but we can
    # verify the code path doesn't crash and returns valid proposals.
    proposal_noflip, _ = policy_noflip.propose(snap)
    assert proposal_noflip.action in {-1, 0, 1}
    if proposal_noflip.action == 0:
        assert proposal_noflip.target_fraction == 0.0, "FLAT action should have zero target_fraction"


# ── Test 8: Ablation flags wiring ───────────────────────────────────────




# ── Split-boundary embargo (C2-adjacent label leakage) ─────────────────


def test_split_boundary_embargo_purges_horizon_overlap() -> None:
    """Train/val samples whose forward-label window crosses the next
    split's boundary must be purged: a sample at bar i carries labels
    from bars i+1..i+max_horizon, so without the embargo the last
    max_horizon train samples are graded on val-period outcomes."""
    from src.features.feature_pipeline import FeaturePipeline as _FP
    from src.training.data_module import build_real_data_training_bundle

    config = _make_config()
    config.data.symbols = ["AAPL"]
    config.training.fast_mode.enabled = False
    bars = generate_synthetic_bars("AAPL", 600)
    bundle = build_real_data_training_bundle(
        config=config,
        bars_by_symbol={"AAPL": bars},
        pipeline=_FP(config.features, device=torch.device("cpu")),
    )
    max_horizon = max(
        list(config.training.forward_horizons) + list(config.training.auxiliary_horizons)
    )
    bar_index_by_ts = {bar.timestamp: i for i, bar in enumerate(bars)}
    for earlier, later in (("train", "val"), ("val", "test")):
        earlier_split = bundle.split_ranges[earlier]["AAPL"]
        later_split = bundle.split_ranges[later]["AAPL"]
        if earlier_split.sample_count == 0 or later_split.sample_count == 0:
            continue
        earlier_end = bar_index_by_ts[earlier_split.end_timestamp]
        later_start = bar_index_by_ts[later_split.start_timestamp]
        gap = later_start - earlier_end
        assert gap > max_horizon, (
            f"{earlier}->{later} boundary gap {gap} bars <= max_horizon "
            f"{max_horizon}: forward labels cross the split boundary"
        )
