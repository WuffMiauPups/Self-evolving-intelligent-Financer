"""EV-leg direction semantics (§7.0.4 addendum, fixed 2026-07-21).

MFE/MAE are PRICE-PATH up/down excursions, so the extracted legs mean
"gain/loss for a LONG". The gate consumes "gain/loss FOR THE PROPOSED
TRADE": a short's gain is the down-move. The swap happens exactly once, in
``_build_proposal`` — the shared funnel both ``propose`` and
``propose_batch`` route through — the moment the action is known.

Measured pre-fix (runs/ev_direction_audit.json): 6,425 OOS short proposals
passed the pre-cost EV inequality at 0.3% as consumed vs 48.4% corrected;
longs 76.9% either way.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.strategies.brain_policy import BrainPolicy

T = datetime(2026, 7, 20, 15, 0, tzinfo=UTC)

# Distinct leg values so a swap is unambiguous: predicted up-move (MFE)
# 100 bps, predicted down-move (|MAE|) 200 bps.
UP_MOVE_BPS = 100.0
DOWN_MOVE_BPS = 200.0

LONG_LOGITS = [0.1, 0.2, 3.0]   # short, flat, long
SHORT_LOGITS = [3.0, 0.2, 0.1]
FLAT_LOGITS = [0.1, 3.0, 0.2]


def _policy(**overrides) -> BrainPolicy:
    return BrainPolicy(
        model=SimpleNamespace(risk_head=None),
        device=torch.device("cpu"),
        **overrides,
    )


def _snapshot(symbol: str = "NVDA") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, timestamp=T, last_price=100.0, spread_bps=1.0)


def _build(policy: BrainPolicy, logits: list[float]):
    return policy._build_proposal(
        snapshot=_snapshot(),
        logit_values=logits,
        confidence=0.6,
        target_fraction=0.05,
        soma_activation=1.0,
        branch_confidence_dict={},
        soma_weights=[1.0],
        regime_weights=[],
        state_reset_reason=None,
        expected_gain_bps=UP_MOVE_BPS,
        expected_loss_bps=DOWN_MOVE_BPS,
    )


def test_long_proposal_keeps_price_relative_legs() -> None:
    proposal, diagnostics = _build(_policy(), LONG_LOGITS)
    assert proposal.action == 1
    assert proposal.expected_gain_bps == pytest.approx(UP_MOVE_BPS)
    assert proposal.expected_loss_bps == pytest.approx(DOWN_MOVE_BPS)
    assert diagnostics["expected_gain_bps"] == pytest.approx(UP_MOVE_BPS)


def test_short_proposal_swaps_legs_to_trade_relative() -> None:
    """THE fix: a short's gain is the predicted down-move."""
    proposal, diagnostics = _build(_policy(), SHORT_LOGITS)
    assert proposal.action == -1
    assert proposal.expected_gain_bps == pytest.approx(DOWN_MOVE_BPS)
    assert proposal.expected_loss_bps == pytest.approx(UP_MOVE_BPS)
    # Diagnostics record the AS-CONSUMED (post-swap) values — the decision
    # ledger must reflect what the gate actually saw.
    assert diagnostics["expected_gain_bps"] == pytest.approx(DOWN_MOVE_BPS)
    assert diagnostics["expected_loss_bps"] == pytest.approx(UP_MOVE_BPS)


def test_flat_proposal_legs_pass_through() -> None:
    proposal, _ = _build(_policy(), FLAT_LOGITS)
    assert proposal.action == 0
    assert proposal.expected_gain_bps == pytest.approx(UP_MOVE_BPS)
    assert proposal.expected_loss_bps == pytest.approx(DOWN_MOVE_BPS)


def test_forced_flip_short_also_swaps() -> None:
    """force_directional_flip resolves FLAT to a direction AFTER logits —
    the swap must key off the RESOLVED action, not the argmax."""
    policy = _policy(force_directional_flip=True)
    flat_but_short_leaning = [2.0, 3.0, 0.1]  # flat wins, short > long
    proposal, _ = _build(policy, flat_but_short_leaning)
    assert proposal.action == -1
    assert proposal.expected_gain_bps == pytest.approx(DOWN_MOVE_BPS)
    assert proposal.expected_loss_bps == pytest.approx(UP_MOVE_BPS)


def test_swap_lives_only_in_the_shared_funnel_source_contract() -> None:
    """Both propose() and propose_batch() must route legs through
    _build_proposal, and the swap must exist exactly once, there."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "strategies" / "brain_policy.py"
    ).read_text(encoding="utf-8")
    assert source.count("self._build_proposal(") == 2  # single + batch paths
    swap_line = "expected_gain_bps, expected_loss_bps = expected_loss_bps, expected_gain_bps"
    assert source.count(swap_line) == 1
    assert swap_line in source.split("def _build_proposal", 1)[1].split("def propose", 1)[0]
