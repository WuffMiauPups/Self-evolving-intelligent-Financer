"""Tests for the SatSOM / GNG / latent-versioning memory evolution foundations.

These tests validate:
- Encoder fingerprint and latent-space versioning
- UnitStatus enum correctness
- Saturation / plasticity behavior
- Outcome-aware unit assignment
- Conflict history tracking and quarantine
- Graph edge topology (GNG foundations)
- Enhanced memory health diagnostics
- Advisory-only memory behavior unchanged
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.memory.dynamic_units import build_dynamic_units
from src.memory.esom_map import GraphEdge, build_esom_map
from src.memory.memory_bank import MemoryBank, compute_encoder_fingerprint, feature_contract_fingerprint
from src.memory.maintenance import maintain_memory_units
from src.memory.outcomes import build_risk_summary
from src.memory.retrieval import AssociativeRetriever, RetrievalReason
from src.memory.schemas import (
    MemoryItem,
    MemoryUnit,
    OutcomeSummary,
    UnitStatus,
)
from src.memory.path_stats import aggregate_path_statistics, aggregate_risk_statistics
from src.memory.storage import load_memory_bank, save_memory_bank
from src.models.state_encoder import EncodedState


def _load_config():
    return load_config("configs/default.yaml")


def _make_outcome(primary_return: float, *, label: str = "chop") -> OutcomeSummary:
    return OutcomeSummary(
        future_returns={1: primary_return, 5: primary_return},
        max_favorable_excursion=max(primary_return, 0.0),
        max_adverse_excursion=min(primary_return, 0.0),
        realized_volatility=abs(primary_return) * 0.5,
        drawdown=min(primary_return, 0.0),
        path_label=label,
        direction_class="up" if primary_return > 0 else ("down" if primary_return < 0 else "flat"),
        time_to_move=1,
    )


def _make_item(
    contract_id: str,
    *,
    latent: torch.Tensor,
    item_id: str,
    outcome: OutcomeSummary,
    encoder_fingerprint: str | None = None,
) -> MemoryItem:
    return MemoryItem(
        item_id=item_id,
        symbol="AAPL",
        timestamp=datetime(2026, 4, 2, 12, 0, tzinfo=UTC),
        latent=latent,
        feature_contract_id=contract_id,
        regime_weights=torch.tensor([0.8, 0.2], dtype=torch.float32),
        branch_confidences={"trend": 0.7},
        branch_norms={"trend": 1.0},
        outcome=outcome,
        risk=build_risk_summary(outcome),
        encoder_fingerprint=encoder_fingerprint,
    )


def _build_snapshot():
    config = _load_config()
    bars = generate_synthetic_bars("AAPL", 140)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    return config, snapshot


# ========== UnitStatus Enum ==========

class TestUnitStatusEnum:
    def test_unit_status_values(self):
        assert UnitStatus.CANDIDATE.value == "candidate"
        assert UnitStatus.MATURE.value == "mature"
        assert UnitStatus.ARCHIVED.value == "archived"
        assert UnitStatus.QUARANTINED.value == "quarantined"

    def test_unit_status_from_str(self):
        assert UnitStatus.from_str("candidate") == UnitStatus.CANDIDATE
        assert UnitStatus.from_str("mature") == UnitStatus.MATURE
        assert UnitStatus.from_str("archived") == UnitStatus.ARCHIVED
        assert UnitStatus.from_str("quarantined") == UnitStatus.QUARANTINED
        assert UnitStatus.from_str("unknown") == UnitStatus.CANDIDATE

    def test_unit_status_is_string_compatible(self):
        assert UnitStatus.MATURE == "mature"
        assert UnitStatus.CANDIDATE == "candidate"


# ========== Encoder Fingerprint / Latent-Space Versioning ==========

class TestEncoderFingerprint:
    def test_compute_encoder_fingerprint_deterministic(self):
        meta = {"checkpoint_version": "1.4", "model_kind": "brain", "model_config": {"dim": 24}}
        fp1 = compute_encoder_fingerprint(meta)
        fp2 = compute_encoder_fingerprint(meta)
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) == 16

    def test_different_meta_gives_different_fingerprint(self):
        meta_a = {"checkpoint_version": "1.4", "model_kind": "brain", "model_config": {"dim": 24}}
        meta_b = {"checkpoint_version": "1.4", "model_kind": "brain", "model_config": {"dim": 48}}
        assert compute_encoder_fingerprint(meta_a) != compute_encoder_fingerprint(meta_b)

    def test_memory_bank_stores_encoder_fingerprint(self):
        config, snapshot = _build_snapshot()
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
            encoder_fingerprint="abc123",
        )
        assert bank.encoder_fingerprint == "abc123"
        assert bank.summary()["encoder_fingerprint"] == "abc123"

    def test_memory_item_stores_encoder_fingerprint(self):
        config, snapshot = _build_snapshot()
        contract_id = feature_contract_fingerprint(snapshot.contract)
        item = _make_item(
            contract_id,
            latent=torch.tensor([1.0, 0.0]),
            item_id="a",
            outcome=_make_outcome(0.01),
            encoder_fingerprint="fp123",
        )
        assert item.encoder_fingerprint == "fp123"
        payload = item.to_payload()
        assert payload["encoder_fingerprint"] == "fp123"
        restored = MemoryItem.from_payload(payload)
        assert restored.encoder_fingerprint == "fp123"

    def test_memory_bank_warns_on_fingerprint_mismatch(self):
        config, snapshot = _build_snapshot()
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
            encoder_fingerprint="bank_fp",
            require_encoder_match=False,
        )
        item = _make_item(
            contract_id,
            latent=torch.tensor([1.0, 0.0]),
            item_id="a",
            outcome=_make_outcome(0.01),
            encoder_fingerprint="different_fp",
        )
        # Should not raise, just warn
        bank.append(item)
        assert len(bank) == 1

    def test_memory_bank_raises_on_fingerprint_mismatch_when_required(self):
        config, snapshot = _build_snapshot()
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
            encoder_fingerprint="bank_fp",
            require_encoder_match=True,
        )
        item = _make_item(
            contract_id,
            latent=torch.tensor([1.0, 0.0]),
            item_id="a",
            outcome=_make_outcome(0.01),
            encoder_fingerprint="different_fp",
        )
        with pytest.raises(ValueError, match="Encoder fingerprint mismatch"):
            bank.append(item)

    def test_memory_bank_allows_matching_fingerprint(self):
        config, snapshot = _build_snapshot()
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
            encoder_fingerprint="same_fp",
            require_encoder_match=True,
        )
        item = _make_item(
            contract_id,
            latent=torch.tensor([1.0, 0.0]),
            item_id="a",
            outcome=_make_outcome(0.01),
            encoder_fingerprint="same_fp",
        )
        bank.append(item)
        assert len(bank) == 1

    def test_encoder_fingerprint_survives_save_load(self, tmp_path):
        config, snapshot = _build_snapshot()
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
            encoder_fingerprint="test_fp",
        )
        bank.append(
            _make_item(
                contract_id,
                latent=torch.tensor([1.0, 0.0]),
                item_id="a",
                outcome=_make_outcome(0.01),
                encoder_fingerprint="test_fp",
            )
        )
        path = tmp_path / "bank.pt"
        save_memory_bank(path, bank)
        loaded = load_memory_bank(path)
        assert loaded.encoder_fingerprint == "test_fp"
        assert loaded.items[0].encoder_fingerprint == "test_fp"


# ========== SatSOM Saturation / Plasticity ==========

class TestSaturationPlasticity:
    def test_units_have_saturation_and_plasticity_fields(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        contract_id = feature_contract_fingerprint(snapshot.contract)
        items = [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.012)),
            _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="c", outcome=_make_outcome(0.011)),
        ]
        latent_matrix = torch.stack([item.latent for item in items], dim=0)
        units, report = build_dynamic_units(items, latent_matrix, config.memory)
        assert len(units) > 0
        for unit in units:
            assert hasattr(unit, "saturation_score")
            assert hasattr(unit, "plasticity_factor")
            assert hasattr(unit, "cumulative_updates")
            assert 0.0 <= unit.saturation_score <= 1.0
            assert 0.0 <= unit.plasticity_factor <= 1.0
            assert unit.cumulative_updates >= 0

    def test_saturation_enabled_attenuates_prototype_updates(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        config.memory.saturation_enabled = True
        config.memory.maturity_update_attenuation = 0.9
        contract_id = feature_contract_fingerprint(snapshot.contract)
        # Create many similar items to build mature units
        items = [
            _make_item(
                contract_id,
                latent=torch.tensor([1.0 + i * 0.001, 0.0]),
                item_id=f"item_{i}",
                outcome=_make_outcome(0.01),
            )
            for i in range(10)
        ]
        latent_matrix = torch.stack([item.latent for item in items], dim=0)
        units, report = build_dynamic_units(items, latent_matrix, config.memory)
        assert report.saturation_attenuated_updates > 0

    def test_saturation_and_plasticity_survive_serialization(self):
        unit = MemoryUnit(
            unit_id="test",
            status="mature",
            prototype=torch.tensor([1.0, 0.0]),
            support_count=5,
            member_indices=[0, 1, 2, 3, 4],
            average_similarity=0.9,
            min_similarity=0.8,
            max_similarity=1.0,
            similarity_std=0.05,
            path_stats=aggregate_path_statistics([_make_outcome(0.01)]),
            risk_stats=aggregate_risk_statistics([_make_outcome(0.01)]),
            regime_composition={"regime_0": 1.0},
            symbol_composition={"AAPL": 1.0},
            session_composition={"regular": 1.0},
            utility_score=0.5,
            maturity_score=1.0,
            latent_dispersion=0.1,
            outcome_dispersion=0.2,
            created_at=datetime(2026, 4, 2, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 4, 2, 13, 0, tzinfo=UTC),
            created_step=0,
            updated_step=4,
            saturation_score=0.85,
            plasticity_factor=0.3,
            cumulative_updates=15,
            conflict_history=[0.0, 0.1, 0.0, 0.5],
            recent_conflict_rate=0.25,
            local_error_accumulator=0.05,
            insertion_candidate_score=0.01,
        )
        payload = unit.to_payload()
        restored = MemoryUnit.from_payload(payload)
        assert restored.saturation_score == pytest.approx(0.85)
        assert restored.plasticity_factor == pytest.approx(0.3)
        assert restored.cumulative_updates == 15
        assert len(restored.conflict_history) == 4
        assert restored.recent_conflict_rate == pytest.approx(0.25)
        assert restored.local_error_accumulator == pytest.approx(0.05)
        assert restored.insertion_candidate_score == pytest.approx(0.01)


# ========== Outcome-Aware Assignment ==========

class TestOutcomeAwareAssignment:
    def test_conflicting_outcomes_create_separate_units_when_enabled(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        config.memory.require_outcome_alignment_for_strong_update = True
        contract_id = feature_contract_fingerprint(snapshot.contract)
        # Create items that look similar but have opposite outcomes
        items = [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.02)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.019)),
            _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="c", outcome=_make_outcome(0.018)),
            # Now a similar-looking item with opposite outcome
            _make_item(contract_id, latent=torch.tensor([0.97, 0.03]), item_id="d", outcome=_make_outcome(-0.02)),
            _make_item(contract_id, latent=torch.tensor([0.96, 0.04]), item_id="e", outcome=_make_outcome(-0.019)),
        ]
        latent_matrix = torch.stack([item.latent for item in items], dim=0)
        units, report = build_dynamic_units(items, latent_matrix, config.memory)
        assert report.conflict_rejections > 0
        # Should have created more units than a purely similarity-based approach
        assert len(units) >= 2


# ========== Conflict History Tracking ==========

class TestConflictHistory:
    def test_units_track_conflict_history(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        contract_id = feature_contract_fingerprint(snapshot.contract)
        items = [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(-0.01)),
            _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="c", outcome=_make_outcome(0.01)),
        ]
        latent_matrix = torch.stack([item.latent for item in items], dim=0)
        units, _ = build_dynamic_units(items, latent_matrix, config.memory)
        # At least one unit should have conflict history entries
        has_conflict = any(len(unit.conflict_history) > 0 for unit in units)
        assert has_conflict


# ========== GNG Graph Edge Topology ==========

class TestGraphEdgeTopology:
    def test_esom_map_has_edges(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        contract_id = feature_contract_fingerprint(snapshot.contract)
        items = [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.012)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.01)),
            _make_item(contract_id, latent=torch.tensor([-0.99, 0.01]), item_id="d", outcome=_make_outcome(-0.012)),
        ]
        latent_matrix = torch.stack([item.latent for item in items], dim=0)
        units, _ = build_dynamic_units(items, latent_matrix, config.memory)
        esom_map = build_esom_map(units, config.memory, current_step=3)
        assert esom_map is not None
        assert len(esom_map.nodes) == len(units)
        assert len(esom_map.edges) > 0
        # Edges should have correct structure
        for edge in esom_map.edges:
            assert isinstance(edge, GraphEdge)
            assert edge.age == 0
            assert edge.similarity > 0.0

    def test_graph_statistics(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        contract_id = feature_contract_fingerprint(snapshot.contract)
        items = [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.5, 0.5]), item_id="b", outcome=_make_outcome(0.005)),
            _make_item(contract_id, latent=torch.tensor([0.0, 1.0]), item_id="c", outcome=_make_outcome(-0.01)),
        ]
        latent_matrix = torch.stack([item.latent for item in items], dim=0)
        units, _ = build_dynamic_units(items, latent_matrix, config.memory)
        esom_map = build_esom_map(units, config.memory)
        assert esom_map is not None
        stats = esom_map.graph_statistics
        assert "node_count" in stats
        assert "edge_count" in stats
        assert "avg_density" in stats
        assert "avg_edge_age" in stats
        assert stats["node_count"] == len(units)

    def test_edge_aging_and_removal(self):
        edge1 = GraphEdge(source_id="a", target_id="b", similarity=0.9, age=99)
        edge2 = GraphEdge(source_id="b", target_id="c", similarity=0.7, age=0)
        from src.memory.esom_map import ESOMMap
        esom_map = ESOMMap(nodes=[], edges=[edge1, edge2], neighbor_k=2, metric="cosine")
        expired = esom_map.age_edges(max_age=100)
        assert len(expired) == 0
        assert esom_map.edges[0].age == 100
        assert esom_map.edges[1].age == 1
        # Age again - edge1 should expire (age 101 > max_age 100)
        expired = esom_map.age_edges(max_age=100)
        assert len(expired) == 1
        assert expired[0].source_id == "a"
        assert len(esom_map.edges) == 1

    def test_edge_activation_refreshes_or_creates(self):
        from src.memory.esom_map import ESOMMap
        esom_map = ESOMMap(nodes=[], edges=[], neighbor_k=2, metric="cosine")
        # Create new edge
        edge = esom_map.activate_edge("a", "b", similarity=0.8, step=10)
        assert edge.age == 0
        assert edge.similarity == 0.8
        assert len(esom_map.edges) == 1
        # Age it
        esom_map.age_edges(max_age=100)
        assert esom_map.edges[0].age == 1
        # Reactivate it
        edge = esom_map.activate_edge("a", "b", similarity=0.85, step=20)
        assert edge.age == 0
        assert edge.similarity == 0.85
        assert len(esom_map.edges) == 1

    def test_esom_map_edges_survive_serialization(self):
        from src.memory.esom_map import ESOMMap
        edges = [
            GraphEdge(source_id="a", target_id="b", similarity=0.9, age=5, creation_step=0, last_activated_step=3),
        ]
        esom_map = ESOMMap(nodes=[], edges=edges, neighbor_k=2, metric="cosine")
        payload = esom_map.to_payload()
        restored = ESOMMap.from_payload(payload)
        assert len(restored.edges) == 1
        assert restored.edges[0].source_id == "a"
        assert restored.edges[0].age == 5


# ========== Enhanced Health Diagnostics ==========

class TestEnhancedHealthDiagnostics:
    def test_health_report_includes_saturation_diagnostics(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
        )
        bank.extend([
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.012)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.01)),
        ])
        report = bank.refresh_organization(config.memory)
        health = report["health"]
        assert hasattr(health, "average_saturation")
        assert hasattr(health, "average_plasticity")
        assert hasattr(health, "saturated_unit_count")
        assert hasattr(health, "high_plasticity_unit_count")
        assert hasattr(health, "conflict_heavy_unit_count")
        assert hasattr(health, "average_conflict_rate")
        assert hasattr(health, "graph_edge_count")
        assert hasattr(health, "quarantined_units")

    def test_organization_report_includes_new_metrics(self):
        config, snapshot = _build_snapshot()
        config.memory.min_candidate_support = 2
        config.memory.min_similarity = 0.5
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
        )
        bank.extend([
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.012)),
        ])
        report = bank.refresh_organization(config.memory)
        assert "conflict_rejections" in report
        assert "saturation_attenuated_updates" in report
        assert "quarantined_units" in report
        assert "esom_edges" in report


# ========== Advisory-Only Memory Behavior Unchanged ==========

class TestAdvisoryBehaviorUnchanged:
    def test_memory_retrieval_still_advisory_only(self):
        config, snapshot = _build_snapshot()
        config.memory.min_similarity = 0.95
        config.memory.min_effective_matches = 1
        contract_id = feature_contract_fingerprint(snapshot.contract)
        bank = MemoryBank(
            feature_contract=snapshot.contract,
            latent_dim=2,
            similarity_metric="cosine",
            max_items=16,
        )
        bank.append(
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01))
        )
        retriever = AssociativeRetriever(bank, config.memory)
        support = retriever.retrieve(
            EncodedState(
                symbol="AAPL",
                timestamp=snapshot.timestamp,
                latent=torch.tensor([-1.0, 0.0]),
                feature_contract_id=contract_id,
            )
        )
        # Memory should still suggest no-trade for unknown states
        assert support.weak_match is True
        assert support.no_trade_suggested is True

    def test_risk_manager_authority_unaffected(self):
        """Verify the deterministic risk manager remains fully authoritative."""
        from src.risk.constraints import TradeProposal
        from src.risk.portfolio_state import PortfolioState
        from src.risk.risk_manager import RiskManager

        config = _load_config()
        risk_manager = RiskManager(config.risk)
        proposal = TradeProposal(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 2, 12, 0, tzinfo=UTC),
            action=0,
            confidence=0.0,
            target_fraction=0.0,
            price=150.0,
            spread_bps=1.0,
            request_id="test-1",
            metadata={},
        )
        decision = risk_manager.evaluate(proposal, PortfolioState(cash=100_000.0, equity=100_000.0))
        assert decision.approved is False
        assert "no_position_change" in decision.reasons
