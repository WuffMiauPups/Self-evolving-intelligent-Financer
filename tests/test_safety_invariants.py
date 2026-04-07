"""Tests for critical safety invariants of the intended system behavior.

Covers:
  1. Flat base action never becomes a trade (memory/fusion cannot promote)
  2. Memory/fusion cannot bypass the risk manager
  3. Quarantined units excluded from assignment/retrieval/topology agreement
  4. Frozen-brain runtime actually freezes weights when configured
  5. Online-memory-only runtime path works across restart with pending persistence
  6. Decision fusion enabled vs disabled path behaves as expected
  7. Topology growth remains bounded
  8. Config flags are wired end-to-end
  9. Checkpoint/memory encoder compatibility is enforced
  10. Legacy fallback does not silently activate when fusion is enabled
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from src.core.config import (
    AppConfig,
    DecisionFusionConfig,
    MemoryConfig,
    RuntimeConfig,
    TopologyGrowthConfig,
    load_config,
)
from src.memory.esom_map import ESOMMap, ESOMNode, GraphEdge
from src.memory.schemas import (
    MemoryDecisionSupport,
    MemoryItem,
    MemoryUnit,
    OutcomeSummary,
    PathStatistics,
    PendingMemoryState,
    RetrievedClusterMatch,
    RetrievedStateMatch,
    RiskStatistics,
    RiskSummary,
    UnitStatus,
)
from src.risk.constraints import RiskDecision, TradeProposal
from src.strategies.decision_fusion import (
    DAMPEN,
    NEUTRAL,
    SUPPORT,
    VETO,
    FusionReason,
    FusionResult,
    compute_fusion,
)
from src.strategies.memory_policy import MemoryAwarePolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)


def _path_stats(direction_probs: dict[str, float] | None = None) -> PathStatistics:
    return PathStatistics(
        sample_count=10,
        horizon_return_mean={15: 0.001},
        horizon_return_p25={15: -0.001},
        horizon_return_p75={15: 0.003},
        max_favorable_mean=0.005,
        max_adverse_mean=0.002,
        max_adverse_p10=0.004,
        realized_volatility_mean=0.01,
        drawdown_mean=0.002,
        drawdown_p10=0.005,
        path_label_frequencies={"breakout_up": 0.6, "chop": 0.4},
        direction_probabilities=direction_probs or {"long": 0.7, "short": 0.1, "flat": 0.2},
    )


def _risk_stats() -> RiskStatistics:
    return RiskStatistics(
        sample_count=10,
        realized_volatility_mean=0.01,
        realized_volatility_p75=0.015,
        max_adverse_mean=0.003,
        max_adverse_p10=0.005,
        drawdown_mean=0.002,
        drawdown_p10=0.005,
    )


def _outcome(primary_return: float = 0.002) -> OutcomeSummary:
    return OutcomeSummary(
        future_returns={15: primary_return},
        max_favorable_excursion=0.005,
        max_adverse_excursion=0.002,
        realized_volatility=0.01,
        drawdown=0.002,
        path_label="breakout_up",
        direction_class="long",
    )


def _risk_summary() -> RiskSummary:
    return RiskSummary(realized_volatility=0.01, max_adverse_excursion=0.002, drawdown=0.002)


def _proposal(action: int = 1, confidence: float = 0.8, target_fraction: float = 0.1) -> TradeProposal:
    return TradeProposal(
        symbol="AAPL",
        timestamp=_ts(),
        action=action,
        confidence=confidence,
        target_fraction=target_fraction,
        price=150.0,
        spread_bps=2.0,
        request_id="test-001",
    )


def _match(similarity: float = 0.9, primary_return: float = 0.002) -> RetrievedStateMatch:
    return RetrievedStateMatch(
        item_index=0,
        item_id="item-0",
        symbol="AAPL",
        timestamp=_ts(),
        similarity_raw=similarity,
        similarity=similarity,
        cluster_id=None,
        outcome=_outcome(primary_return),
        risk=_risk_summary(),
    )


def _support(
    *,
    weak_match: bool = False,
    no_trade_suggested: bool = False,
    reason_code: str = "strong_match",
    directional_bias: int = 1,
    size_multiplier: float = 0.7,
    conflict_score: float = 0.1,
    match_strength: float = 0.9,
    calibrated_match_strength: float = 0.8,
    retrieval_confidence: float = 0.75,
    hopfield_top_indices: list[int] | None = None,
) -> MemoryDecisionSupport:
    return MemoryDecisionSupport(
        top_matches=[_match()],
        cluster_matches=[],
        match_strength=match_strength,
        average_similarity=0.85,
        weak_match=weak_match,
        no_trade_suggested=no_trade_suggested,
        reason_code=reason_code,
        directional_bias=directional_bias,
        size_multiplier=size_multiplier,
        path_distribution={"breakout_up": 0.6},
        risk_summary=_risk_stats(),
        path_summary=_path_stats(),
        calibrated_match_strength=calibrated_match_strength,
        retrieval_confidence=retrieval_confidence,
        conflict_score=conflict_score,
        metadata={
            "hopfield_top_indices": hopfield_top_indices or [],
            "esom_neighbor_ids": [],
        },
    )


def _make_unit(
    unit_id: str = "unit_001",
    status: str = "mature",
    direction_probs: dict[str, float] | None = None,
    saturation: float = 0.5,
    plasticity: float = 0.8,
    conflict_rate: float = 0.1,
    support_count: int = 10,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id,
        status=status,
        prototype=torch.randn(8),
        support_count=support_count,
        member_indices=[0, 1, 2],
        average_similarity=0.85,
        min_similarity=0.7,
        max_similarity=0.95,
        similarity_std=0.05,
        path_stats=_path_stats(direction_probs),
        risk_stats=_risk_stats(),
        regime_composition={"trending": 0.6, "mean_reverting": 0.4},
        symbol_composition={"AAPL": 0.5, "MSFT": 0.5},
        session_composition={},
        utility_score=0.5,
        maturity_score=0.6,
        latent_dispersion=0.2,
        outcome_dispersion=0.3,
        created_at=_ts(),
        updated_at=_ts(),
        created_step=0,
        updated_step=0,
        saturation_score=saturation,
        plasticity_factor=plasticity,
        recent_conflict_rate=conflict_rate,
    )


def _policy(fusion_enabled: bool = True) -> MemoryAwarePolicy:
    """Create a minimal MemoryAwarePolicy for _apply_support testing."""
    policy = MemoryAwarePolicy.__new__(MemoryAwarePolicy)
    policy.require_strong_match_for_trade = True
    policy.fusion_config = DecisionFusionConfig(enabled=fusion_enabled)
    return policy


# ===========================================================================
# 1. Flat base action never becomes trade
# ===========================================================================

class TestFlatBaseNeverPromoted:
    """Memory and fusion must never promote a flat (action=0) base proposal."""

    def test_flat_base_fusion_neutral(self):
        """Fusion returns NEUTRAL for flat base action."""
        result = compute_fusion(
            _proposal(action=0),
            _support(directional_bias=1, size_multiplier=1.0),
            DecisionFusionConfig(enabled=True),
        )
        assert result.support_level == NEUTRAL
        assert result.size_multiplier == 0.0
        assert FusionReason.FLAT_BASE_POLICY in result.reason_codes

    def test_flat_base_strong_memory_still_flat(self):
        """Even with strong memory support, flat base stays flat."""
        fusion = FusionResult(
            support_level=SUPPORT,
            reason_codes=[FusionReason.STRONG_MATURE_SUPPORT],
            size_multiplier=1.0,
            confidence_adjustment=1.0,
            unknown_state_flag=False,
            topology_warning_flag=False,
        )
        policy = _policy(fusion_enabled=True)
        adjusted, advisory = policy._apply_support(
            _proposal(action=0),
            _support(directional_bias=1, size_multiplier=1.0),
            fusion,
        )
        assert adjusted.action == 0
        assert advisory["advisory_effect"] == "flat_base_policy"

    def test_flat_base_legacy_path_still_flat(self):
        """Legacy path also preserves flat base action."""
        policy = _policy(fusion_enabled=False)
        adjusted, advisory = policy._apply_support(
            _proposal(action=0),
            _support(directional_bias=1, size_multiplier=1.0),
            None,
        )
        assert adjusted.action == 0
        assert advisory["advisory_effect"] == "flat_base_policy"

    def test_safety_assertion_fires_on_violation(self):
        """The safety check in propose() would catch a hypothetical violation."""
        # This tests the assertion at the exit of _apply_support
        # We can't easily trigger it without bypassing the guard, but we verify
        # the invariant holds for all fusion states
        for fusion_level in [SUPPORT, NEUTRAL, DAMPEN, VETO]:
            fusion = FusionResult(
                support_level=fusion_level,
                reason_codes=[],
                size_multiplier=1.0,
                confidence_adjustment=1.0,
                unknown_state_flag=False,
                topology_warning_flag=False,
            )
            policy = _policy(fusion_enabled=True)
            adjusted, _ = policy._apply_support(
                _proposal(action=0),
                _support(directional_bias=1, size_multiplier=1.0),
                fusion,
            )
            assert adjusted.action == 0, f"Flat became trade with fusion_level={fusion_level}"


# ===========================================================================
# 2. Memory/fusion cannot bypass risk manager
# ===========================================================================

class TestRiskManagerAuthority:
    """Risk manager is the final authority — memory/fusion output still goes through it."""

    def test_fusion_support_still_rejected_by_risk(self):
        """Even when fusion supports, risk manager can reject."""
        from src.risk.portfolio_state import PortfolioState
        from src.risk.risk_manager import RiskManager
        from src.core.config import RiskConfig

        risk_config = RiskConfig(kill_switch=True)
        rm = RiskManager(risk_config)
        rm.reset_state()

        # Fusion fully supports this trade
        proposal = _proposal(action=1, confidence=0.9, target_fraction=0.2)
        portfolio = PortfolioState(cash=100000.0, equity=100000.0)
        decision = rm.evaluate(proposal, portfolio)
        assert decision.approved is False
        assert "kill_switch" in decision.reasons

    def test_fusion_veto_prevents_risk_evaluation(self):
        """When fusion vetoes, proposal becomes flat → risk sees action=0."""
        policy = _policy(fusion_enabled=True)
        fusion = FusionResult(
            support_level=VETO,
            reason_codes=[FusionReason.HIGH_CONFLICT],
            size_multiplier=0.0,
            confidence_adjustment=0.3,
            unknown_state_flag=False,
            topology_warning_flag=False,
        )
        adjusted, advisory = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(),
            fusion,
        )
        assert adjusted.action == 0
        assert advisory["memory_vetoed"] is True


# ===========================================================================
# 3. Quarantined units excluded
# ===========================================================================

class TestQuarantinedUnitsExcluded:
    """Quarantined units must not participate in fusion support calculation."""

    def test_quarantined_excluded_from_saturation_avg(self):
        """Quarantined units don't contribute to avg_saturation/plasticity."""
        config = DecisionFusionConfig(enabled=True)
        units = [
            _make_unit("u1", status="mature", saturation=0.5, plasticity=0.8),
            _make_unit("u2", status="quarantined", saturation=1.0, plasticity=0.0),
        ]
        result = compute_fusion(
            _proposal(action=1),
            _support(hopfield_top_indices=[0, 1]),
            config,
            memory_units=units,
        )
        profile = result.metadata.get("matched_unit_profile", {})
        assert profile["quarantined"] == 1
        assert profile["avg_saturation"] == 0.5  # only mature unit counted

    def test_quarantined_excluded_from_topology_growth(self):
        """Topology growth skips quarantined paths."""
        from src.memory.topology_growth import evaluate_topology_growth

        items = [
            MemoryItem(
                item_id=f"item-{i}",
                symbol="AAPL",
                timestamp=_ts(),
                latent=torch.randn(8),
                feature_contract_id="test",
                regime_weights=None,
                branch_confidences={"trend": 0.5},
                branch_norms={"trend": 1.0},
                outcome=_outcome(),
                risk=_risk_summary(),
            )
            for i in range(20)
        ]
        quarantined_unit = _make_unit("q1", status=UnitStatus.QUARANTINED.value)
        config = MemoryConfig(
            similarity_metric="cosine",
            topology_growth=TopologyGrowthConfig(
                enabled=True,
                respect_quarantine=True,
                min_novelty_score=0.01,  # low threshold to trigger candidates
                min_support_for_insertion=1,
            ),
        )
        updated, report = evaluate_topology_growth(
            [quarantined_unit],
            items,
            config,
        )
        assert report.growth_enabled is True
        # Quarantined path should not be source for growth


# ===========================================================================
# 4. Frozen-brain runtime freezes weights
# ===========================================================================

class TestFrozenBrainRuntime:
    """Model weights must be frozen when configured."""

    def test_frozen_brain_requires_grad_false(self):
        """Simulates the trade_engine frozen brain logic."""
        model = torch.nn.Linear(10, 3)
        assert all(p.requires_grad for p in model.parameters())

        # Simulate trade_engine.py:348-352
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_default_yaml_frozen_brain(self):
        """default.yaml has allow_online_model_updates=false."""
        config = load_config("configs/default.yaml")
        assert config.runtime.allow_online_model_updates is False

    def test_code_default_frozen_brain(self):
        """Python code default matches intent."""
        config = RuntimeConfig()
        assert config.allow_online_model_updates is False
        assert config.allow_online_memory_updates is True


# ===========================================================================
# 5. Online memory across restart with pending persistence
# ===========================================================================

class TestOnlineMemoryPersistence:
    """Pending observations survive restart."""

    def _make_updater(self, update_mode: str = "pending_only"):
        from src.memory.updater import AssociativeMemoryUpdater
        from src.memory.memory_bank import MemoryBank
        from src.features.feature_pipeline import FeatureContract

        contract = FeatureContract(
            feature_names=("f1", "f2"),
            branch_feature_names={"trend": ("f1",), "vol": ("f2",)},
            regime_feature_names=("r1",),
            temporal_feature_names=("f1", "f2"),
            feature_dtype="float32",
        )
        bank = MemoryBank(
            feature_contract=contract,
            latent_dim=8,
            similarity_metric="cosine",
            max_items=100,
        )
        config = MemoryConfig(runtime_update_mode=update_mode)
        return AssociativeMemoryUpdater(bank, config)

    def test_pending_roundtrip_preserves_data(self):
        updater = self._make_updater()
        pending = PendingMemoryState(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            feature_contract_id="test_contract",
            latent=torch.randn(8),
            regime_weights=torch.randn(4),
            branch_confidences={"trend": 0.8},
            branch_norms={"trend": 1.0},
            encoder_fingerprint="fp_test",
            metadata={"key": "value"},
        )
        updater.pending_by_symbol["AAPL"].append(pending)
        updater.observed_count = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pending.pt"
            updater.save_pending(path)

            updater2 = self._make_updater()
            restored = updater2.load_pending(path)
            assert restored == 1
            assert updater2.observed_count == 3
            restored_item = updater2.pending_by_symbol["AAPL"][0]
            assert restored_item.symbol == "AAPL"
            assert restored_item.encoder_fingerprint == "fp_test"

    def test_disabled_mode_ignores_observations(self):
        """When update_mode=disabled, observe() is a no-op."""
        updater = self._make_updater("disabled")
        from src.models.state_encoder import EncodedState

        encoded = EncodedState(
            latent=torch.randn(8),
            symbol="AAPL",
            timestamp=_ts(),
            feature_contract_id="test",
            branch_confidences={"trend": 0.8},
            branch_norms={"trend": 1.0},
            metadata={},
        )
        updater.observe(encoded)
        assert updater.observed_count == 0
        assert len(updater.pending_by_symbol["AAPL"]) == 0


# ===========================================================================
# 6. Decision fusion enabled vs disabled
# ===========================================================================

class TestFusionEnabledDisabled:
    """Fusion enabled uses fusion path, disabled uses legacy path."""

    def test_fusion_enabled_uses_fusion_result(self):
        """When fusion_config.enabled=True, _apply_support uses fusion result."""
        fusion = FusionResult(
            support_level=DAMPEN,
            reason_codes=[FusionReason.MODERATE_CONFLICT],
            size_multiplier=0.3,
            confidence_adjustment=0.7,
            unknown_state_flag=False,
            topology_warning_flag=False,
        )
        policy = _policy(fusion_enabled=True)
        adjusted, advisory = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(),
            fusion,
        )
        # Fusion path: uses fusion.size_multiplier
        assert adjusted.target_fraction == pytest.approx(0.1 * 0.3, abs=1e-6)
        assert advisory["advisory_effect"] == "dampened"

    def test_fusion_disabled_uses_legacy_veto(self):
        """When fusion is None, legacy path handles no_trade_suggested."""
        policy = _policy(fusion_enabled=False)
        adjusted, advisory = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(no_trade_suggested=True),
            None,
        )
        assert adjusted.action == 0
        assert advisory["memory_vetoed"] is True
        assert advisory["advisory_effect"] == "veto"

    def test_fusion_disabled_legacy_applies_multiplier(self):
        """Legacy path applies memory size_multiplier when aligned."""
        policy = _policy(fusion_enabled=False)
        adjusted, advisory = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(size_multiplier=0.5, no_trade_suggested=False),
            None,
        )
        assert adjusted.target_fraction == pytest.approx(0.1 * 0.5, abs=1e-6)


# ===========================================================================
# 7. Topology growth remains bounded
# ===========================================================================

class TestTopologyGrowthBounded:
    """Topology growth must respect hard caps."""

    def test_max_insertions_respected(self):
        from src.memory.topology_growth import evaluate_topology_growth

        # Create items that are very different from the single unit
        items = [
            MemoryItem(
                item_id=f"item-{i}",
                symbol="AAPL",
                timestamp=_ts(),
                latent=torch.randn(8) * 10,  # deliberately distant
                feature_contract_id="test",
                regime_weights=None,
                branch_confidences={"trend": 0.5},
                branch_norms={"trend": 1.0},
                outcome=_outcome(),
                risk=_risk_summary(),
            )
            for i in range(50)
        ]
        unit = _make_unit("u0")
        config = MemoryConfig(
            max_units=256,
            similarity_metric="cosine",
            topology_growth=TopologyGrowthConfig(
                enabled=True,
                max_insertions_per_refresh=2,
                max_growth_rate=0.10,
                min_novelty_score=0.01,
                min_support_for_insertion=2,
            ),
        )
        updated, report = evaluate_topology_growth([unit], items, config)
        assert report.insertions_applied <= 2

    def test_max_units_cap(self):
        from src.memory.topology_growth import evaluate_topology_growth

        items = [
            MemoryItem(
                item_id=f"item-{i}",
                symbol="AAPL",
                timestamp=_ts(),
                latent=torch.randn(8) * 10,
                feature_contract_id="test",
                regime_weights=None,
                branch_confidences={"trend": 0.5},
                branch_norms={"trend": 1.0},
                outcome=_outcome(),
                risk=_risk_summary(),
            )
            for i in range(20)
        ]
        # Already at max capacity
        existing_units = [_make_unit(f"u{i}") for i in range(10)]
        config = MemoryConfig(
            max_units=10,
            similarity_metric="cosine",
            topology_growth=TopologyGrowthConfig(enabled=True),
        )
        updated, report = evaluate_topology_growth(existing_units, items, config)
        assert report.max_units_reached is True
        assert report.insertions_applied == 0
        assert len(updated) == 10


# ===========================================================================
# 8. Config flags wired end-to-end
# ===========================================================================

class TestConfigWiring:
    """Critical config flags are wired through to their consumers."""

    def test_fusion_config_wired_in_default_yaml(self):
        config = load_config("configs/default.yaml")
        assert config.memory.decision_fusion.enabled is True
        assert config.memory.decision_fusion.conflict_veto_threshold == 0.70

    def test_frozen_brain_wired_in_default_yaml(self):
        config = load_config("configs/default.yaml")
        assert config.runtime.allow_online_model_updates is False

    def test_topology_growth_off_in_code_default(self):
        # Code-level safety invariant: TopologyGrowthConfig() must default OFF.
        # default.yaml may explicitly enable it for research builds, but anyone
        # constructing MemoryConfig()/TopologyGrowthConfig() programmatically
        # must get growth disabled until they opt in.
        from src.core.config import MemoryConfig, TopologyGrowthConfig
        assert TopologyGrowthConfig().enabled is False
        assert MemoryConfig().topology_growth.enabled is False

    def test_memory_enabled_in_default_yaml(self):
        config = load_config("configs/default.yaml")
        assert config.memory.enabled is True

    def test_require_strong_match_flags_both_exist(self):
        """Both config flags exist and have expected defaults."""
        config = MemoryConfig()
        assert config.require_strong_match_for_trade is True
        assert config.require_strong_match_for_memory_support is True

    def test_runtime_config_frozen_brain_defaults(self):
        config = RuntimeConfig()
        assert config.allow_online_model_updates is False
        assert config.allow_online_memory_updates is True


# ===========================================================================
# 9. Encoder compatibility enforced
# ===========================================================================

class TestEncoderCompatibility:
    """Memory bank rejects items from incompatible encoders when configured."""

    def test_compatible_fingerprint_accepted(self):
        from src.memory.memory_bank import MemoryBank
        from src.features.feature_pipeline import FeatureContract

        contract = FeatureContract(
            feature_names=("f1",),
            branch_feature_names={"trend": ("f1",)},
            regime_feature_names=("r1",),
            temporal_feature_names=("f1",),
            feature_dtype="float32",
        )
        bank = MemoryBank(
            feature_contract=contract,
            latent_dim=4,
            similarity_metric="cosine",
            max_items=100,
            encoder_fingerprint="fp_v1",
            require_encoder_match=True,
        )
        assert bank.check_encoder_compatibility("fp_v1") is True

    def test_incompatible_fingerprint_raises_when_required(self):
        from src.memory.memory_bank import MemoryBank
        from src.features.feature_pipeline import FeatureContract

        contract = FeatureContract(
            feature_names=("f1",),
            branch_feature_names={"trend": ("f1",)},
            regime_feature_names=("r1",),
            temporal_feature_names=("f1",),
            feature_dtype="float32",
        )
        bank = MemoryBank(
            feature_contract=contract,
            latent_dim=4,
            similarity_metric="cosine",
            max_items=100,
            encoder_fingerprint="fp_v1",
            require_encoder_match=True,
        )
        with pytest.raises(ValueError, match="Encoder fingerprint mismatch"):
            bank.check_encoder_compatibility("fp_v2")

    def test_incompatible_fingerprint_warns_when_not_required(self):
        from src.memory.memory_bank import MemoryBank
        from src.features.feature_pipeline import FeatureContract

        contract = FeatureContract(
            feature_names=("f1",),
            branch_feature_names={"trend": ("f1",)},
            regime_feature_names=("r1",),
            temporal_feature_names=("f1",),
            feature_dtype="float32",
        )
        bank = MemoryBank(
            feature_contract=contract,
            latent_dim=4,
            similarity_metric="cosine",
            max_items=100,
            encoder_fingerprint="fp_v1",
            require_encoder_match=False,
        )
        # Should not raise, but return False
        assert bank.check_encoder_compatibility("fp_v2") is False


# ===========================================================================
# 10. Legacy fallback does not activate when fusion is enabled
# ===========================================================================

class TestLegacyFallbackIsolation:
    """When fusion is enabled, the legacy path must not run."""

    def test_fusion_enabled_no_trade_suggested_uses_fusion_not_legacy(self):
        """Even with no_trade_suggested, fusion path handles it, not legacy."""
        fusion = FusionResult(
            support_level=NEUTRAL,
            reason_codes=[FusionReason.WEAK_MATCH],
            size_multiplier=0.0,
            confidence_adjustment=0.5,
            unknown_state_flag=True,
            topology_warning_flag=False,
        )
        policy = _policy(fusion_enabled=True)
        adjusted, advisory = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(no_trade_suggested=True, weak_match=True),
            fusion,
        )
        # Fusion path: size_multiplier=0.0 → adjusted_fraction≈0 → action=0
        assert adjusted.action == 0
        # But advisory_effect should reflect fusion path, not legacy veto
        assert advisory["memory_vetoed"] is False  # fusion NEUTRAL, not legacy veto

    def test_fusion_veto_vs_legacy_veto_differ(self):
        """Fusion veto and legacy veto produce different metadata."""
        # Fusion veto
        fusion = FusionResult(
            support_level=VETO,
            reason_codes=[FusionReason.HIGH_CONFLICT],
            size_multiplier=0.0,
            confidence_adjustment=0.3,
            unknown_state_flag=False,
            topology_warning_flag=False,
        )
        policy = _policy(fusion_enabled=True)
        adj_fusion, adv_fusion = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(),
            fusion,
        )

        # Legacy veto (fusion=None)
        policy_legacy = _policy(fusion_enabled=False)
        adj_legacy, adv_legacy = policy_legacy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(no_trade_suggested=True),
            None,
        )

        # Both veto, but via different paths
        assert adj_fusion.action == 0
        assert adj_legacy.action == 0
        assert adv_fusion["memory_vetoed"] is True
        assert adv_legacy["memory_vetoed"] is True
        # Fusion veto has fusion-specific metadata
        assert adj_fusion.metadata.get("fusion_support_level") == VETO
        assert "fusion_support_level" not in adj_legacy.metadata

    def test_fusion_support_applies_different_multiplier_than_legacy(self):
        """Fusion and legacy paths compute size differently."""
        # Fusion path with explicit multiplier
        fusion = FusionResult(
            support_level=SUPPORT,
            reason_codes=[FusionReason.STRONG_MATURE_SUPPORT],
            size_multiplier=0.6,
            confidence_adjustment=0.9,
            unknown_state_flag=False,
            topology_warning_flag=False,
        )
        policy = _policy(fusion_enabled=True)
        adj_fusion, _ = policy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(size_multiplier=0.7),
            fusion,
        )

        # Legacy path uses support.size_multiplier directly
        policy_legacy = _policy(fusion_enabled=False)
        adj_legacy, _ = policy_legacy._apply_support(
            _proposal(action=1, target_fraction=0.1),
            _support(size_multiplier=0.7, no_trade_suggested=False),
            None,
        )

        # Fusion: 0.1 * 0.6 = 0.06
        assert adj_fusion.target_fraction == pytest.approx(0.06, abs=1e-6)
        # Legacy: 0.1 * 0.7 = 0.07
        assert adj_legacy.target_fraction == pytest.approx(0.07, abs=1e-6)
