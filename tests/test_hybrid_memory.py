from __future__ import annotations

from datetime import UTC, datetime

import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank, feature_contract_fingerprint
from src.memory.retrieval import AssociativeRetriever, RetrievalReason
from src.memory.schemas import MemoryItem, OutcomeSummary
from src.memory.outcomes import build_risk_summary
from src.models.brain_model import BrainTradingModel
from src.models.state_encoder import EncodedState, ModelStateEncoder


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
    encoder = ModelStateEncoder(model=model, device=torch.device("cpu"), feature_contract=snapshot.contract)
    return config, snapshot, encoder


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


def _make_item(contract_id: str, latent: torch.Tensor, item_id: str, outcome: OutcomeSummary) -> MemoryItem:
    return MemoryItem(
        item_id=item_id,
        symbol="AAPL",
        timestamp=datetime(2026, 4, 2, 12, 0, tzinfo=UTC),
        latent=latent,
        feature_contract_id=contract_id,
        regime_weights=torch.tensor([0.7, 0.3], dtype=torch.float32),
        branch_confidences={"trend": 0.6},
        branch_norms={"trend": 1.0},
        outcome=outcome,
        risk=build_risk_summary(outcome),
    )


def test_feature_contract_exposes_temporal_channels_for_cnn_encoder() -> None:
    _, snapshot, _ = _build_snapshot()
    assert snapshot.contract.temporal_dim == snapshot.temporal_tensor.shape[0]
    assert snapshot.contract.temporal_dim > 0


def test_state_encoder_is_reproducible_for_same_snapshot() -> None:
    _, snapshot, encoder = _build_snapshot()
    first = encoder.encode(snapshot)
    second = encoder.encode(snapshot)
    assert torch.allclose(first.latent, second.latent)


def test_hopfield_retrieval_returns_context_vector_and_neighbors() -> None:
    config, snapshot, _ = _build_snapshot()
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=32,
    )
    bank.extend(
        [
            _make_item(contract_id, torch.tensor([1.0, 0.0]), "a", _make_outcome(0.02, label="breakout_up")),
            _make_item(contract_id, torch.tensor([0.98, 0.02]), "b", _make_outcome(0.018, label="breakout_up")),
            _make_item(contract_id, torch.tensor([-1.0, 0.0]), "c", _make_outcome(-0.02, label="breakout_down")),
        ]
    )
    bank.refresh_organization(config.memory)
    support = AssociativeRetriever(bank, config.memory).retrieve(
        EncodedState(
            symbol="AAPL",
            timestamp=snapshot.timestamp,
            latent=torch.tensor([1.0, 0.0]),
            feature_contract_id=contract_id,
        )
    )
    assert support.context_vector is not None
    assert support.metadata["hopfield_top_indices"]
    assert "esom_neighbor_ids" in support.metadata


def test_esom_map_builds_nodes_for_dynamic_units() -> None:
    config, snapshot, _ = _build_snapshot()
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=32,
    )
    bank.extend(
        [
            _make_item(contract_id, torch.tensor([1.0, 0.0]), "a", _make_outcome(0.01)),
            _make_item(contract_id, torch.tensor([0.99, 0.01]), "b", _make_outcome(0.011)),
            _make_item(contract_id, torch.tensor([-1.0, 0.0]), "c", _make_outcome(-0.012)),
            _make_item(contract_id, torch.tensor([-0.99, -0.01]), "d", _make_outcome(-0.013)),
        ]
    )
    bank.refresh_organization(config.memory)
    assert bank.esom_map is not None
    assert len(bank.esom_map.nodes) == len(bank.memory_units)


def test_unknown_state_remains_explicit_under_hybrid_retrieval() -> None:
    config, snapshot, _ = _build_snapshot()
    config.memory.min_similarity = 0.99
    config.memory.min_effective_matches = 1
    contract_id = feature_contract_fingerprint(snapshot.contract)
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=16,
    )
    bank.append(_make_item(contract_id, torch.tensor([1.0, 0.0]), "a", _make_outcome(0.01)))
    bank.refresh_organization(config.memory)
    support = AssociativeRetriever(bank, config.memory).retrieve(
        EncodedState(
            symbol="AAPL",
            timestamp=snapshot.timestamp,
            latent=torch.tensor([-1.0, 0.0]),
            feature_contract_id=contract_id,
        )
    )
    assert support.weak_match is True
    assert support.no_trade_suggested is True
    assert support.reason_code in {
        RetrievalReason.INSUFFICIENT_SIMILARITY,
        RetrievalReason.CALIBRATED_WEAK_MATCH,
    }
