"""Phase 2 §6.2-B — bounded, damping-only memory influence.

The safety contract these tests pin: with ``influence_mode="damping"``,
memory can only SHRINK a directional trade's target fraction, within
[damping_min_multiplier, 1.0]. It can never grow a position, flip or
veto an action, touch confidence, act on thin evidence, or act at all
in "off" mode.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import torch

from src.core.config import FeatureConfig, MemoryConfig
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.outcomes import build_risk_summary
from src.memory.schemas import MemoryItem, OutcomeSummary
from src.models.state_encoder import EncodedState
from src.risk.constraints import TradeProposal
from src.strategies.memory_policy import MemoryAwarePolicy

_T0 = datetime(2026, 6, 1, 14, 30, tzinfo=UTC)
_NOW = _T0 + timedelta(days=30)


def _contract():
    from src.data.schemas import MarketBar

    bars = [
        MarketBar(
            symbol="AAPL",
            timestamp=_T0 + timedelta(minutes=5 * i),
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.2,
            volume=1000.0,
            spread=0.05,
        )
        for i in range(80)
    ]
    pipeline = FeaturePipeline(FeatureConfig(), device=torch.device("cpu"))
    return pipeline.build(bars, symbol="AAPL").contract


_CONTRACT = _contract()


def _outcome(primary: float) -> OutcomeSummary:
    return OutcomeSummary(
        future_returns={1: primary},
        max_favorable_excursion=abs(primary) * 2,
        max_adverse_excursion=-abs(primary),
        realized_volatility=0.01,
        drawdown=0.0,
        path_label="chop",
        direction_class="flat",
        time_to_move=1,
    )


def _bank_with_neighbor_returns(returns: list[float]) -> MemoryBank:
    """A bank whose every item is maximally similar to torch.ones(8)."""
    bank = MemoryBank(feature_contract=_CONTRACT, latent_dim=8, max_items=1000)
    for i, ret in enumerate(returns):
        bank.append(
            MemoryItem(
                item_id=f"N-{i}",
                symbol="MSFT",
                timestamp=_T0 + timedelta(minutes=5 * i),
                latent=torch.ones(8),
                feature_contract_id="cid",
                regime_weights=None,
                branch_confidences={},
                branch_norms={},
                outcome=_outcome(ret),
                risk=build_risk_summary(_outcome(ret)),
            )
        )
    return bank


class _StubBasePolicy:
    """Base policy returning a fixed proposal."""

    def __init__(self, action: int = 1, target_fraction: float = 0.10, confidence: float = 0.7):
        self._action = action
        self._fraction = target_fraction
        self._confidence = confidence

    def propose(self, snapshot):
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=self._action,
            confidence=self._confidence,
            target_fraction=self._fraction,
            price=100.0,
            spread_bps=2.0,
            request_id=f"{snapshot.symbol}-req",
        )
        return proposal, {}


class _StubEncoder:
    def encode(self, snapshot):
        return EncodedState(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            latent=torch.ones(8),
            feature_contract_id="cid",
        )

    def encode_batch(self, snapshots):
        return [self.encode(s) for s in snapshots]


def _policy(
    bank: MemoryBank,
    *,
    mode: str = "damping",
    action: int = 1,
    fraction: float = 0.10,
    min_neighbors: int = 5,
) -> MemoryAwarePolicy:
    return MemoryAwarePolicy(
        base_policy=_StubBasePolicy(action=action, target_fraction=fraction),
        encoder=_StubEncoder(),
        bank=bank,
        knn_top_k=10,
        influence_mode=mode,
        damping_min_multiplier=0.5,
        damping_min_neighbors=min_neighbors,
        damping_full_at_return=0.002,
    )


def _snapshot(symbol: str = "AAPL"):
    return SimpleNamespace(symbol=symbol, timestamp=_NOW)


class TestDampingMath:
    def test_adverse_neighbors_shrink_the_trade(self) -> None:
        # Long proposal; neighbors averaged -10bps → halfway to the floor.
        bank = _bank_with_neighbor_returns([-0.001] * 8)
        proposal, diag = _policy(bank).propose(_snapshot())
        assert diag["memory_damping"]["applied"] is True
        assert diag["memory_damping"]["multiplier"] == pytest.approx(0.75)
        assert proposal.target_fraction == pytest.approx(0.075)

    def test_multiplier_saturates_at_floor(self) -> None:
        # Neighbors averaged -50bps, far past full_at_return=-20bps → 0.5, never lower.
        bank = _bank_with_neighbor_returns([-0.005] * 8)
        proposal, diag = _policy(bank).propose(_snapshot())
        assert diag["memory_damping"]["multiplier"] == pytest.approx(0.5)
        assert proposal.target_fraction == pytest.approx(0.05)

    def test_favorable_neighbors_never_grow_the_trade(self) -> None:
        bank = _bank_with_neighbor_returns([+0.005] * 8)
        proposal, diag = _policy(bank).propose(_snapshot())
        assert diag["memory_damping"]["applied"] is False
        assert diag["memory_damping"]["multiplier"] == 1.0
        assert proposal.target_fraction == pytest.approx(0.10)

    def test_short_side_alignment(self) -> None:
        # Short proposal; neighbors' PRICES rose (+10bps) → adverse for a
        # short → damped. Falling prices would be favorable → untouched.
        bank_adverse = _bank_with_neighbor_returns([+0.001] * 8)
        proposal, diag = _policy(bank_adverse, action=-1).propose(_snapshot())
        assert diag["memory_damping"]["applied"] is True
        assert abs(proposal.target_fraction) == pytest.approx(abs(0.10) * 0.75)

        bank_favorable = _bank_with_neighbor_returns([-0.001] * 8)
        proposal, diag = _policy(bank_favorable, action=-1).propose(_snapshot())
        assert diag["memory_damping"]["applied"] is False


class TestDampingSafetyRails:
    def test_action_and_confidence_are_never_touched(self) -> None:
        bank = _bank_with_neighbor_returns([-0.005] * 8)
        proposal, _ = _policy(bank).propose(_snapshot())
        assert proposal.action == 1
        assert proposal.confidence == pytest.approx(0.7)

    def test_flat_proposals_pass_through(self) -> None:
        bank = _bank_with_neighbor_returns([-0.005] * 8)
        proposal, diag = _policy(bank, action=0, fraction=0.0).propose(_snapshot())
        assert diag["memory_damping"]["applied"] is False
        assert proposal.target_fraction == 0.0

    def test_thin_evidence_means_no_influence(self) -> None:
        bank = _bank_with_neighbor_returns([-0.005] * 3)  # below min_neighbors=5
        proposal, diag = _policy(bank).propose(_snapshot())
        assert diag["memory_damping"]["applied"] is False
        assert diag["memory_damping"]["reason"] == "insufficient_neighbors"
        assert proposal.target_fraction == pytest.approx(0.10)

    def test_mode_off_is_purely_diagnostic(self) -> None:
        bank = _bank_with_neighbor_returns([-0.005] * 8)
        proposal, diag = _policy(bank, mode="off").propose(_snapshot())
        assert diag["memory_damping"]["applied"] is False
        assert proposal.target_fraction == pytest.approx(0.10)
        # Diagnostics still present (kNN stays observable).
        assert diag["memory_knn"]["neighbor_count"] > 0

    def test_temporal_cutoff_shields_damping_from_future_outcomes(self) -> None:
        # All adverse neighbors are AFTER the cutoff → invisible → no damping.
        bank = MemoryBank(feature_contract=_CONTRACT, latent_dim=8, max_items=100)
        for i in range(8):
            bank.append(
                MemoryItem(
                    item_id=f"F-{i}",
                    symbol="MSFT",
                    timestamp=_NOW + timedelta(days=1 + i),
                    latent=torch.ones(8),
                    feature_contract_id="cid",
                    regime_weights=None,
                    branch_confidences={},
                    branch_norms={},
                    outcome=_outcome(-0.005),
                    risk=build_risk_summary(_outcome(-0.005)),
                )
            )
        policy = _policy(bank)
        policy.set_temporal_cutoff(_NOW)
        proposal, diag = policy.propose(_snapshot())
        assert diag["memory_knn"]["neighbor_count"] == 0
        assert diag["memory_damping"]["applied"] is False
        assert proposal.target_fraction == pytest.approx(0.10)

    def test_multiplier_bounds_hold_for_any_neighbor_mix(self) -> None:
        import random

        rng = random.Random(7)
        for _ in range(25):
            returns = [rng.uniform(-0.02, 0.02) for _ in range(rng.randint(5, 12))]
            bank = _bank_with_neighbor_returns(returns)
            proposal, diag = _policy(bank).propose(_snapshot())
            multiplier = diag["memory_damping"]["multiplier"]
            assert 0.5 <= multiplier <= 1.0
            assert abs(proposal.target_fraction) <= 0.10 + 1e-12


def test_config_defaults_keep_memory_influence_off() -> None:
    config = MemoryConfig()
    assert config.influence_mode == "off"
    with pytest.raises(Exception):
        MemoryConfig(influence_mode="veto")
    with pytest.raises(Exception):
        MemoryConfig(damping_min_multiplier=0.0)
    with pytest.raises(Exception):
        MemoryConfig(damping_min_multiplier=1.5)
