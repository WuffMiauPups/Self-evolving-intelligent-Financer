"""Model input hardening and batch-composition invariance (§4.3.4, §4.3.5).

* ``BrainTradingModel.forward`` without ``full_features`` used to fall
  back to concatenating ``branch_inputs`` in branch order — a silent
  permutation of the contract's global (sorted) feature ordering.  It
  now raises.
* ``_extract_ev_signals_batch`` used to zero BOTH EV legs when only one
  of mfe/mae was present; the single-sample path handles each leg
  independently.  The batch path must match it.
* ``propose_batch`` used to left-zero-pad shorter temporal tensors to
  the batch max and convolve the padding, so a symbol's proposal
  depended on which other symbols shared its batch.  Length-bucketed
  sub-batches make ``propose_batch([a, b]) == [propose(a), propose(b)]``
  for heterogeneous temporal lengths.
"""
from __future__ import annotations

import pytest
import torch

from src.core.config import load_config
from src.core.seeds import seed_everything
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import build_model_from_contract
from src.strategies.brain_policy import (
    BrainPolicy,
    _extract_ev_signals_batch,
    _extract_ev_signals_single,
)

_TINY_MODEL_UPDATE = {
    "soma_hidden_dim": 16,
    "head_hidden_dim": 8,
    "head_depth": 1,
    "cnn_channels": [4, 8],
    "temporal_blocks": 2,
    "dropout": 0.0,
}


def _tiny_brain_setup():
    config = load_config("configs/default.yaml")
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bars = generate_synthetic_bars("AAPL", 100)
    contract = pipeline.build(bars, symbol="AAPL").contract
    seed_everything(11)
    model = build_model_from_contract(
        model_kind="brain",
        model_config=config.model.model_copy(update=_TINY_MODEL_UPDATE),
        feature_contract=contract,
    )
    model.eval()
    return model, pipeline, bars


# ── §4.3.4a: full_features is required ──────────────────────────────────


def test_brain_model_forward_without_full_features_raises() -> None:
    model, pipeline, bars = _tiny_brain_setup()
    snapshot = pipeline.build(bars, symbol="AAPL")
    with pytest.raises(ValueError, match="full_features"):
        model(
            branch_inputs={
                name: value.unsqueeze(0) for name, value in snapshot.branch_inputs.items()
            },
            regime_input=snapshot.regime_input.unsqueeze(0),
            temporal_tensor=snapshot.temporal_tensor.unsqueeze(0),
        )


# ── §4.3.4b: batch EV extraction handles each leg independently ─────────


@pytest.mark.parametrize(
    "present_legs",
    [("mfe", "mae"), ("mfe",), ("mae",), ()],
    ids=["both", "mfe_only", "mae_only", "neither"],
)
def test_batch_ev_extraction_matches_single_path_per_leg(
    present_legs: tuple[str, ...],
) -> None:
    batch_size = 3
    full = {
        "mfe": torch.tensor([[0.010], [-0.002], [0.030]]),
        "mae": torch.tensor([[-0.005], [-0.001], [0.002]]),
    }
    risk_predictions = {leg: full[leg] for leg in present_legs} or None

    gain_bps, loss_bps = _extract_ev_signals_batch(risk_predictions, batch_size)

    for row in range(batch_size):
        row_predictions = (
            {leg: full[leg][row] for leg in present_legs} if present_legs else None
        )
        single_gain, single_loss = _extract_ev_signals_single(row_predictions)
        assert gain_bps[row] == pytest.approx(single_gain, rel=1e-6, abs=1e-9)
        assert loss_bps[row] == pytest.approx(single_loss, rel=1e-6, abs=1e-9)

    # Regression pin: an mfe-only head keeps its gain leg (the old batch
    # path zeroed both legs unless BOTH were present).
    if present_legs == ("mfe",):
        assert gain_bps[0] == pytest.approx(100.0, rel=1e-4)
        assert loss_bps == [0.0] * batch_size
    if present_legs == ("mae",):
        assert gain_bps == [0.0] * batch_size
        assert loss_bps[0] == pytest.approx(50.0, rel=1e-4)


# ── §4.3.5: batch-composition invariance ────────────────────────────────


def test_propose_batch_matches_single_proposals_for_heterogeneous_lengths() -> None:
    """``propose_batch([a, b])`` must equal ``[propose(a), propose(b)]``
    when the snapshots carry DIFFERENT temporal lengths.  Pre-fix, the
    shorter snapshot was left-zero-padded to the longer one's length and
    the CNN convolved the padding, so its action/confidence/EV depended
    on batch composition."""
    model, pipeline, bars = _tiny_brain_setup()
    snapshot_long = pipeline.build(bars[:100], symbol="AAA")  # temporal length 100
    snapshot_short = pipeline.build(bars[:70], symbol="BBB")  # temporal length 70
    assert (
        snapshot_long.temporal_tensor.shape[-1]
        != snapshot_short.temporal_tensor.shape[-1]
    )

    device = torch.device("cpu")
    batch_policy = BrainPolicy(model=model, device=device, use_amp=False)
    single_policy = BrainPolicy(model=model, device=device, use_amp=False)

    batched = batch_policy.propose_batch([snapshot_long, snapshot_short])
    singles = [
        single_policy.propose(snapshot_long),
        single_policy.propose(snapshot_short),
    ]

    for (batch_proposal, _), (single_proposal, _) in zip(batched, singles):
        assert batch_proposal.symbol == single_proposal.symbol
        assert batch_proposal.action == single_proposal.action
        assert batch_proposal.confidence == pytest.approx(
            single_proposal.confidence, rel=1e-5, abs=1e-6
        )
        assert batch_proposal.target_fraction == pytest.approx(
            single_proposal.target_fraction, rel=1e-5, abs=1e-6
        )
        # EV signals are in bps (fraction * 1e4), so absolute fp32 noise
        # scales up by 1e4 too.
        assert batch_proposal.expected_gain_bps == pytest.approx(
            single_proposal.expected_gain_bps, rel=1e-4, abs=1e-2
        )
        assert batch_proposal.expected_loss_bps == pytest.approx(
            single_proposal.expected_loss_bps, rel=1e-4, abs=1e-2
        )


def test_propose_batch_matches_single_proposals_for_uniform_lengths() -> None:
    """The single-forward fast path (all lengths equal) must produce the
    same proposals as per-snapshot ``propose`` calls too."""
    model, pipeline, bars = _tiny_brain_setup()
    snapshot_a = pipeline.build(bars[:80], symbol="AAA")
    snapshot_b = pipeline.build(bars[20:100], symbol="BBB")
    assert (
        snapshot_a.temporal_tensor.shape[-1] == snapshot_b.temporal_tensor.shape[-1]
    )

    device = torch.device("cpu")
    batch_policy = BrainPolicy(model=model, device=device, use_amp=False)
    single_policy = BrainPolicy(model=model, device=device, use_amp=False)

    batched = batch_policy.propose_batch([snapshot_a, snapshot_b])
    singles = [single_policy.propose(snapshot_a), single_policy.propose(snapshot_b)]

    for (batch_proposal, _), (single_proposal, _) in zip(batched, singles):
        assert batch_proposal.action == single_proposal.action
        assert batch_proposal.confidence == pytest.approx(
            single_proposal.confidence, rel=1e-5, abs=1e-6
        )
        assert batch_proposal.target_fraction == pytest.approx(
            single_proposal.target_fraction, rel=1e-5, abs=1e-6
        )
