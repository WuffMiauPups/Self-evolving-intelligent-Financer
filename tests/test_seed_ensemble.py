"""Seed-ensemble policy (§9.4 arm 2).

Majority vote over same-architecture seeds: ≥2/3 agreement wins, 3-way
disagreement stands down to FLAT (disagreement IS the seed noise the arm
suppresses), target from the voting members only, confidence/EV legs
averaged over all members.
"""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.evaluation.seed_ensemble import SeedEnsemblePolicy
from src.risk.constraints import TradeProposal

T = datetime(2026, 7, 17, 15, 0, tzinfo=UTC)


def _proposal(action: int, confidence: float, target: float, gain: float = 100.0, loss: float = 50.0) -> TradeProposal:
    return TradeProposal(
        symbol="NVDA",
        timestamp=T,
        action=action,
        confidence=confidence,
        target_fraction=target,
        price=100.0,
        spread_bps=1.0,
        request_id="e-1",
        expected_gain_bps=gain,
        expected_loss_bps=loss,
    )


def _member(proposal: TradeProposal) -> SimpleNamespace:
    return SimpleNamespace(propose_batch=lambda snapshots: [(proposal, {})] * len(snapshots))


def test_majority_wins_with_voter_mean_target_and_all_member_means() -> None:
    ensemble = SeedEnsemblePolicy(
        members=[
            _member(_proposal(1, 0.52, 0.010, gain=120.0)),
            _member(_proposal(1, 0.50, 0.020, gain=90.0)),
            _member(_proposal(-1, 0.48, -0.030, gain=60.0)),
        ]
    )
    merged, diagnostics = ensemble.propose(SimpleNamespace(symbol="NVDA"))
    assert merged.action == 1
    assert merged.target_fraction == pytest.approx(0.015)  # voters only
    assert merged.confidence == pytest.approx((0.52 + 0.50 + 0.48) / 3)  # all members
    assert merged.expected_gain_bps == pytest.approx((120 + 90 + 60) / 3)
    assert diagnostics["votes"] == {1: 2, -1: 1}
    assert diagnostics["unanimous"] is False


def test_three_way_disagreement_stands_down_flat() -> None:
    ensemble = SeedEnsemblePolicy(
        members=[
            _member(_proposal(1, 0.51, 0.010)),
            _member(_proposal(0, 0.50, 0.000)),
            _member(_proposal(-1, 0.49, -0.010)),
        ]
    )
    merged, _ = ensemble.propose(SimpleNamespace(symbol="NVDA"))
    assert merged.action == 0
    assert merged.target_fraction == 0.0


def test_batch_merges_index_wise() -> None:
    long_p, short_p = _proposal(1, 0.51, 0.01), _proposal(-1, 0.49, -0.01)

    def member(first: TradeProposal, second: TradeProposal) -> SimpleNamespace:
        return SimpleNamespace(propose_batch=lambda snapshots: [(first, {}), (second, {})])

    ensemble = SeedEnsemblePolicy(
        members=[member(long_p, short_p), member(long_p, short_p), member(short_p, short_p)]
    )
    results = ensemble.propose_batch([SimpleNamespace(symbol="NVDA")] * 2)
    assert results[0][0].action == 1   # 2/3 long at index 0
    assert results[1][0].action == -1  # 3/3 short at index 1
    assert results[1][1]["unanimous"] is True
