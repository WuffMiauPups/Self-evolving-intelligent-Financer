from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
import torch

from src.backtest.engine import BacktestEngine
from src.core.config import load_config
from src.core.exceptions import ConfigurationError
from src.data.historical_data import generate_synthetic_bars
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.encoder_eval import evaluate_encoder_quality
from src.memory.memory_bank import MemoryBank, feature_contract_fingerprint
from src.memory.maintenance import maintain_memory_units
from src.memory.outcomes import build_risk_summary, compute_outcome_summary
from src.memory.retrieval import AssociativeRetriever, RetrievalReason
from src.memory.schemas import MemoryItem, MemoryUnit, OutcomeSummary
from src.memory.path_stats import aggregate_path_statistics, aggregate_risk_statistics
from src.memory.storage import load_memory_bank, save_memory_bank
from src.models.brain_model import BrainTradingModel
from src.models.state_encoder import EncodedState, ModelStateEncoder
from src.risk.constraints import TradeProposal
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.strategies.memory_policy import MemoryAwarePolicy


def _build_snapshot():
    config = load_config("configs/default.yaml")
    bars = generate_synthetic_bars("AAPL", 140)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    model = BrainTradingModel(
        branch_input_dims={name: tensor.numel() for name, tensor in snapshot.branch_inputs.items()},
        config=config.model.model_copy(update={"regime_input_dim": snapshot.regime_input.numel()}),
        temporal_input_dim=snapshot.temporal_tensor.shape[0],
    )
    model.eval()
    encoder = ModelStateEncoder(
        model=model,
        device=torch.device("cpu"),
        feature_contract=snapshot.contract,
    )
    return config, bars, snapshot, encoder


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


def _make_item(contract_id: str, *, latent: torch.Tensor, item_id: str, outcome: OutcomeSummary) -> MemoryItem:
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
    )


def _make_unit(unit_id: str, member_indices: list[int], items: list[MemoryItem], *, status: str = "mature") -> MemoryUnit:
    member_items = [items[index] for index in member_indices]
    latents = torch.stack([item.latent for item in member_items], dim=0)
    prototype = latents.mean(dim=0)
    similarities = 1.0 / (1.0 + torch.norm(latents - prototype, dim=-1))
    return MemoryUnit(
        unit_id=unit_id,
        status=status,
        prototype=prototype,
        support_count=len(member_items),
        member_indices=list(member_indices),
        average_similarity=float(similarities.mean().item()),
        min_similarity=float(similarities.min().item()),
        max_similarity=float(similarities.max().item()),
        similarity_std=float(similarities.std(unbiased=False).item()) if similarities.numel() > 1 else 0.0,
        path_stats=aggregate_path_statistics([item.outcome for item in member_items]),
        risk_stats=aggregate_risk_statistics([item.outcome for item in member_items]),
        regime_composition={"regime_0": 1.0},
        symbol_composition={"AAPL": 1.0},
        session_composition={"regular": 1.0},
        utility_score=0.1,
        maturity_score=1.0 if status == "mature" else 0.5,
        latent_dispersion=float(torch.norm(latents - prototype, dim=-1).mean().item()),
        outcome_dispersion=0.5,
        created_at=member_items[0].timestamp,
        updated_at=member_items[-1].timestamp,
        created_step=min(member_indices),
        updated_step=max(member_indices),
        calibration_confidence=0.5,
    )


def test_state_encoder_enforces_feature_contract_consistency() -> None:
    _, _, snapshot, encoder = _build_snapshot()
    encoded = encoder.encode(snapshot)
    assert encoded.latent.dim() == 1
    tampered = deepcopy(snapshot)
    tampered.feature_names = list(reversed(tampered.feature_names))
    with pytest.raises(ConfigurationError):
        encoder.encode_batch([snapshot, tampered])


def test_memory_bank_append_save_and_load_roundtrip(tmp_path) -> None:
    _, _, snapshot, encoder = _build_snapshot()
    encoded = encoder.encode(snapshot)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=encoded.latent.shape[0],
        similarity_metric="cosine",
        max_items=16,
    )
    outcome = _make_outcome(0.01, label="breakout_up")
    bank.append(
        _make_item(
            bank.feature_contract_id or feature_contract_fingerprint(snapshot.contract),
            latent=encoded.latent.clone(),
            item_id="item-1",
            outcome=outcome,
        )
    )
    bank.rebuild_prototypes(min_similarity=0.75, min_cluster_size=1, temperature=1.0)
    path = tmp_path / "memory_bank.pt"
    save_memory_bank(path, bank)
    loaded = load_memory_bank(path)
    assert len(loaded) == 1
    assert loaded.latent_dim == bank.latent_dim
    assert torch.allclose(loaded.items[0].latent, bank.items[0].latent)
    assert loaded.items[0].outcome.path_label == "breakout_up"


def test_memory_bank_similarity_query_returns_nearest_top_k() -> None:
    config, _, snapshot, _ = _build_snapshot()
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric=config.memory.similarity_metric,
        max_items=16,
    )
    bank.extend(
        [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.9, 0.1]), item_id="b", outcome=_make_outcome(0.008)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.01)),
        ]
    )
    matches = bank.query_top_k(torch.tensor([1.0, 0.0]), top_k=2, temperature=1.0)
    assert [match.item_id for match in matches] == ["a", "b"]


def test_cluster_assignment_respects_similarity_threshold() -> None:
    _, _, snapshot, _ = _build_snapshot()
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.extend(
        [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.012)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.02)),
        ]
    )
    clusters = bank.rebuild_prototypes(min_similarity=0.8, min_cluster_size=1, temperature=1.0)
    assert len(clusters) == 2
    assert sorted(cluster.member_count for cluster in clusters) == [1, 2]


def test_outcome_summary_computation_is_deterministic() -> None:
    bars = [
        MarketBar(symbol="AAPL", timestamp=datetime(2026, 4, 2, 12, idx, tzinfo=UTC), open=100.0 + idx, high=0.0, low=0.0, close=close, volume=1000.0)
        for idx, close in enumerate([100.0, 101.0, 103.0, 102.0])
    ]
    summary = compute_outcome_summary(
        bars,
        0,
        horizons=[1, 2],
        path_label_rules={
            "breakout_return_threshold": 0.01,
            "mean_reversion_return_threshold": 0.005,
            "flat_return_threshold": 0.001,
            "time_to_move_threshold": 0.005,
        },
    )
    assert summary.future_returns[1] == pytest.approx(0.01)
    assert summary.future_returns[2] == pytest.approx(0.03)
    assert summary.path_label == "breakout_up"
    assert summary.time_to_move == 1


def test_retrieval_handles_unknown_state_with_explicit_no_trade() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.min_similarity = 0.95
    config.memory.min_effective_matches = 1
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.append(_make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)))
    retriever = AssociativeRetriever(bank, config.memory)
    support = retriever.retrieve(
        EncodedState(
            symbol="AAPL",
            timestamp=snapshot.timestamp,
            latent=torch.tensor([-1.0, 0.0]),
            feature_contract_id=contract_id,
        )
    )
    assert support.weak_match is True
    assert support.no_trade_suggested is True
    assert support.reason_code == RetrievalReason.INSUFFICIENT_SIMILARITY


def test_dynamic_units_candidate_and_mature_lifecycle() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.min_candidate_support = 2
    config.memory.min_similarity = 0.9
    config.memory.novelty_quantile = 0.2
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.extend(
        [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(0.012)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.01)),
        ]
    )
    report = bank.refresh_organization(config.memory)
    assert len(bank.memory_units) == 2
    assert sum(1 for unit in bank.memory_units if unit.status == "mature") == 1
    assert sum(1 for unit in bank.memory_units if unit.status == "candidate") == 1
    assert report["created_units"] >= 1


def test_maintenance_merge_logic_merges_redundant_units() -> None:
    config, _, snapshot, _ = _build_snapshot()
    contract_id = feature_contract_fingerprint(snapshot.contract)
    items = [
        _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
        _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="b", outcome=_make_outcome(0.011)),
        _make_item(contract_id, latent=torch.tensor([0.96, 0.04]), item_id="c", outcome=_make_outcome(0.012)),
        _make_item(contract_id, latent=torch.tensor([0.94, 0.06]), item_id="d", outcome=_make_outcome(0.013)),
    ]
    units = [_make_unit("u1", [0, 1], items), _make_unit("u2", [2, 3], items)]
    merged_units, report = maintain_memory_units(units, items, config.memory, latest_step=3)
    assert report.merged_units >= 1
    assert len(merged_units) == 1


def test_maintenance_split_logic_splits_broad_units() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.split_variance_trigger = 0.1
    contract_id = feature_contract_fingerprint(snapshot.contract)
    items = [
        _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.02, label="breakout_up")),
        _make_item(contract_id, latent=torch.tensor([0.95, 0.05]), item_id="b", outcome=_make_outcome(0.018, label="breakout_up")),
        _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.02, label="breakout_down")),
        _make_item(contract_id, latent=torch.tensor([-0.95, -0.05]), item_id="d", outcome=_make_outcome(-0.018, label="breakout_down")),
    ]
    unit = _make_unit("broad", [0, 1, 2, 3], items)
    unit.outcome_dispersion = 1.0
    split_units, report = maintain_memory_units([unit], items, config.memory, latest_step=3)
    assert report.split_units >= 1
    assert len(split_units) == 2


def test_maintenance_prune_and_archive_logic() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.min_utility = 0.5
    config.memory.max_candidate_age = 1
    contract_id = feature_contract_fingerprint(snapshot.contract)
    items = [
        _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.001)),
        _make_item(contract_id, latent=torch.tensor([0.0, 1.0]), item_id="b", outcome=_make_outcome(0.001)),
    ]
    stale_candidate = _make_unit("candidate", [0], items, status="candidate")
    stale_candidate.updated_step = 0
    stale_candidate.utility_score = 0.0
    kept_units, report = maintain_memory_units([stale_candidate], items, config.memory, latest_step=5)
    assert report.pruned_units == 1
    assert kept_units == []
    assert len(report.archived_units) == 1


def test_retrieval_calibration_and_conflict_handling() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.min_similarity = 0.5
    config.memory.min_effective_matches = 2
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.extend(
        [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.02)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="b", outcome=_make_outcome(-0.02)),
            _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="c", outcome=_make_outcome(0.018)),
        ]
    )
    bank.refresh_organization(config.memory)
    assert bank.calibration_profile is not None
    support = AssociativeRetriever(bank, config.memory).retrieve(
        EncodedState(
            symbol="AAPL",
            timestamp=snapshot.timestamp,
            latent=torch.tensor([1.0, 0.0]),
            feature_contract_id=contract_id,
        )
    )
    assert support.calibrated_match_strength >= 0.0
    assert support.reason_code in {
        RetrievalReason.STRONG_MATCH,
        RetrievalReason.CONFLICTING_OUTCOMES,
        RetrievalReason.CALIBRATED_WEAK_MATCH,
    }


def test_symbol_specific_calibration_profile_is_selected_when_available() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.min_similarity = 0.5
    config.memory.min_effective_matches = 2
    config.memory.symbol_specific_profiles = True
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=32,
    )
    bank.extend(
        [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a1", outcome=_make_outcome(0.02)),
            _make_item(contract_id, latent=torch.tensor([0.99, 0.01]), item_id="a2", outcome=_make_outcome(0.019)),
            _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="a3", outcome=_make_outcome(0.018)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="m1", outcome=_make_outcome(-0.02)),
            _make_item(contract_id, latent=torch.tensor([-0.99, 0.01]), item_id="m2", outcome=_make_outcome(-0.019)),
            _make_item(contract_id, latent=torch.tensor([-0.98, 0.02]), item_id="m3", outcome=_make_outcome(-0.018)),
        ]
    )
    bank.items[3].symbol = "MSFT"
    bank.items[4].symbol = "MSFT"
    bank.items[5].symbol = "MSFT"
    bank.refresh_organization(config.memory)
    assert "AAPL" in bank.calibration_profiles_by_symbol
    support = AssociativeRetriever(bank, config.memory).retrieve(
        EncodedState(
            symbol="AAPL",
            timestamp=snapshot.timestamp,
            latent=torch.tensor([1.0, 0.0]),
            feature_contract_id=contract_id,
        )
    )
    assert support.metadata["calibration_scope"] == "symbol:AAPL"


def test_encoder_quality_report_runs_offline() -> None:
    config, _, snapshot, _ = _build_snapshot()
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.extend(
        [
            _make_item(contract_id, latent=torch.tensor([1.0, 0.0]), item_id="a", outcome=_make_outcome(0.01)),
            _make_item(contract_id, latent=torch.tensor([0.98, 0.02]), item_id="b", outcome=_make_outcome(0.009)),
            _make_item(contract_id, latent=torch.tensor([-1.0, 0.0]), item_id="c", outcome=_make_outcome(-0.01)),
        ]
    )
    bank.refresh_organization(config.memory)
    report = evaluate_encoder_quality(bank)
    assert report.item_count == 3
    assert report.nearest_neighbor_similarity >= 0.0
    assert report.unit_count >= 1


@dataclass
class DummyBasePolicy:
    def propose(self, snapshot) -> tuple[TradeProposal, dict[str, object]]:
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=1,
            confidence=0.9,
            target_fraction=0.1,
            price=snapshot.last_price,
            spread_bps=snapshot.spread_bps,
            request_id="dummy-1",
            metadata={},
        )
        return proposal, {}


@dataclass
class DummyEncoder:
    feature_contract_id: str

    def encode(self, snapshot) -> EncodedState:
        return EncodedState(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            latent=torch.tensor([-1.0, 0.0]),
            feature_contract_id=self.feature_contract_id,
        )


def test_memory_policy_no_trade_suggestion_still_flows_through_risk_manager() -> None:
    config, _, snapshot, _ = _build_snapshot()
    config.memory.min_effective_matches = 1
    config.memory.min_similarity = 0.95
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.append(
        _make_item(
            feature_contract_fingerprint(snapshot.contract),
            latent=torch.tensor([1.0, 0.0]),
            item_id="a",
            outcome=_make_outcome(0.01),
        )
    )
    policy = MemoryAwarePolicy(
        base_policy=DummyBasePolicy(),
        encoder=DummyEncoder(feature_contract_fingerprint(snapshot.contract)),
        retriever=AssociativeRetriever(bank, config.memory),
        require_strong_match_for_trade=True,
    )
    proposal, diagnostics = policy.propose(snapshot)
    assert proposal.action == 0
    assert proposal.target_fraction == 0.0
    assert diagnostics["memory_support"]["no_trade_suggested"] is True
    assert diagnostics["memory_support"]["memory_vetoed"] is True
    assert diagnostics["memory_support"]["advisory_effect"] == "veto"

    risk_manager = RiskManager(config.risk)
    decision = risk_manager.evaluate(proposal, PortfolioState(cash=100_000.0, equity=100_000.0))
    assert decision.approved is False
    assert "no_position_change" in decision.reasons


def test_memory_aware_backtest_reports_memory_metrics() -> None:
    config, bars, snapshot, _ = _build_snapshot()
    config.memory.min_effective_matches = 1
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=64,
    )
    policy = MemoryAwarePolicy(
        base_policy=DummyBasePolicy(),
        encoder=DummyEncoder(feature_contract_fingerprint(snapshot.contract)),
        retriever=AssociativeRetriever(bank, config.memory),
        require_strong_match_for_trade=True,
    )
    engine = BacktestEngine(config, FeaturePipeline(config.features, device=torch.device("cpu")), RiskManager(config.risk))
    report = engine.run({"AAPL": bars}, policy)
    assert "memory_average_similarity" in report.metrics
    assert "memory_weak_match_rate" in report.metrics
