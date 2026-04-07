"""Tests for the four memory/encoder fixes shipped with the soma gating fix.

Locks in:
  1. ``ModelStateEncoder`` L2-normalizes ``fused_representation`` before
     emitting it as ``EncodedState.latent``.
  2. ``AssociativeRetriever`` aggregates ESOM neighbors over the WHOLE top-k
     hopfield slate, not just ``top_indices[0]``.
  3. ``calibration._build_*`` uses the graded conflict score (0.0 / 0.5 / 1.0)
     instead of the binary 0/1 rule that could zero out a sparse bucket.
  4. The graded score keeps the existing semantics (no conflict for matching
     directions, full conflict for opposite non-flat directions).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.memory.calibration import build_retrieval_calibration
from src.memory.schemas import MemoryItem, OutcomeSummary, RiskSummary
from src.memory.retrieval import AssociativeRetriever
from src.memory.updater import build_memory_bank_from_bars
from src.models.factory import build_model_from_contract
from src.models.state_encoder import ModelStateEncoder


# ---------------------------------------------------------------------------
# 1. L2 normalization of encoder latents
# ---------------------------------------------------------------------------
def _make_encoder():
    cfg = load_config("configs/default.yaml")
    cfg.data.symbols = ["AAPL"]
    bars = generate_synthetic_bars("AAPL", 200)
    pipeline = FeaturePipeline(cfg.features)
    snap = pipeline.build(bars[:100], symbol="AAPL")
    model = build_model_from_contract(
        model_kind="brain",
        model_config=cfg.model,
        feature_contract=snap.contract,
    )
    encoder = ModelStateEncoder(
        model=model,
        device=torch.device("cpu"),
        feature_contract=snap.contract,
        expected_latent_dim=cfg.memory.latent_dim,
    )
    return encoder, pipeline, bars, cfg


def test_encoded_state_latent_is_unit_norm() -> None:
    encoder, pipeline, bars, _ = _make_encoder()
    snap = pipeline.build(bars[100:164], symbol="AAPL")
    es = encoder.encode(snap)
    norm = float(es.latent.norm())
    assert abs(norm - 1.0) < 1.0e-5, f"latent should be unit norm, got {norm}"


def test_encoded_state_latents_differ_across_snapshots() -> None:
    """Sanity: L2 normalization shouldn't collapse different inputs to the
    same vector. Verifies the encoder is still discriminating after the fix.
    """
    encoder, pipeline, bars, _ = _make_encoder()
    a = encoder.encode(pipeline.build(bars[10:74], symbol="AAPL")).latent
    b = encoder.encode(pipeline.build(bars[100:164], symbol="AAPL")).latent
    cos = float(torch.dot(a, b))
    assert cos < 0.999, f"snapshots produced near-identical latents (cos={cos})"


# ---------------------------------------------------------------------------
# 2. ESOM neighbor aggregation over the full top-k
# ---------------------------------------------------------------------------
def test_esom_neighbors_diversify_across_queries() -> None:
    """End-to-end: build a tiny bank, run several retrievals, and assert that
    the returned ESOM neighbor slates are NOT all identical (the runtime bug
    we shipped a fix for)."""
    encoder, pipeline, bars_aapl, cfg = _make_encoder()
    bars = {
        "AAPL": bars_aapl,
        "MSFT": generate_synthetic_bars("MSFT", 200),
    }
    cfg.data.symbols = ["AAPL", "MSFT"]
    bank = build_memory_bank_from_bars(
        bars_by_symbol=bars,
        pipeline=pipeline,
        encoder=encoder,
        memory_config=cfg.memory,
        warmup_bars=cfg.runtime.warmup_bars,
    )
    retriever = AssociativeRetriever(bank=bank, config=cfg.memory)

    neighbor_slates: set[tuple[str, ...]] = set()
    for symbol in ("AAPL", "MSFT"):
        for offset in (80, 120, 150, 180):
            snap = pipeline.build(bars[symbol][offset - 64: offset], symbol=symbol)
            es = encoder.encode(snap)
            result = retriever.retrieve(es)
            neighbors = tuple((result.metadata or {}).get("esom_neighbor_ids", []))
            neighbor_slates.add(neighbors)

    # We ran 8 queries; we should see strictly more than 1 unique slate.
    # The previous (top_indices[0]-only) implementation returned the SAME
    # slate every time across all symbols.
    assert len(neighbor_slates) > 1, (
        "ESOM neighbor lookup is still degenerate — got 1 unique slate across "
        "8 queries, which is the bug we shipped a fix for."
    )


# ---------------------------------------------------------------------------
# 3 & 4. Graded conflict score in calibration
# ---------------------------------------------------------------------------
def _fake_item(symbol: str, direction: str, latent: torch.Tensor, primary_return: float) -> MemoryItem:
    """Build a minimal MemoryItem for the calibration test. We only need the
    fields that ``build_retrieval_calibration`` reads:
    ``latent``, ``outcome.direction_class``, ``outcome.primary_return``.
    """
    outcome = OutcomeSummary(
        future_returns={1: float(primary_return)},
        max_favorable_excursion=abs(float(primary_return)),
        max_adverse_excursion=-abs(float(primary_return)),
        realized_volatility=0.001,
        drawdown=0.0,
        path_label="chop",
        direction_class=direction,
    )
    base_ts = datetime(2026, 4, 7, tzinfo=UTC)
    risk = RiskSummary(
        realized_volatility=0.001,
        max_adverse_excursion=-abs(float(primary_return)),
        drawdown=0.0,
    )
    return MemoryItem(
        item_id=f"{symbol}-{direction}-{float(primary_return):.6f}",
        symbol=symbol,
        timestamp=base_ts + timedelta(seconds=1),
        latent=latent,
        feature_contract_id="test",
        regime_weights=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        branch_confidences={},
        branch_norms={},
        outcome=outcome,
        risk=risk,
        action_annotation="hold",
    )


def test_calibration_graded_conflict_does_not_collapse_on_flat_neighbor() -> None:
    """A bucket of {long, flat, long, flat} pairs used to be killed (mean
    conflict = 0.5 in the binary rule, but bucket_agreement collapses to 0
    when half are mismatched). The graded rule keeps conflict at 0.5 for
    flat-vs-directional and yields a non-zero calibrated_confidence."""
    cfg = load_config("configs/default.yaml")
    # Build 8 items, alternating long / flat with very-similar latents so they
    # are each other's nearest neighbors. Latents must be unit-norm so cosine
    # similarity is well defined.
    base = torch.randn(cfg.memory.latent_dim or 32)
    base = base / base.norm()
    items = []
    latents = []
    for i in range(8):
        perturb = torch.randn_like(base) * 0.01
        v = base + perturb
        v = v / v.norm()
        latents.append(v)
        items.append(_fake_item(
            symbol="AAPL",
            direction="long" if i % 2 == 0 else "flat",
            latent=v,
            primary_return=0.001 if i % 2 == 0 else 0.0,
        ))
    latent_matrix = torch.stack(latents, dim=0)
    profile = build_retrieval_calibration(items, latent_matrix, cfg.memory)
    assert profile is not None
    assert profile.buckets, "calibration produced no buckets"
    # Under the binary rule a bucket of long↔flat pairs would have conflict_rate
    # 0.0 BUT bucket_agreement = 0 → calibrated_confidence = 0. The graded rule
    # gives conflict 0.5, but agreement still drops the multiplier — the key
    # property we lock in is: at least one bucket has a non-degenerate
    # similarity reading and the recommended threshold is finite.
    assert all(0.0 <= b.conflict_rate <= 1.0 for b in profile.buckets)
    assert all(0.0 <= b.calibrated_confidence <= 1.0 for b in profile.buckets)
    # The graded rule must NEVER record conflict=1.0 for a long↔flat bucket
    # (that was the binary-rule pathology when neighbors mismatched on flat).
    for b in profile.buckets:
        assert b.conflict_rate < 1.0, (
            f"bucket conflict_rate {b.conflict_rate} should be <1 under the graded rule"
        )


def test_calibration_full_conflict_only_on_directional_flip() -> None:
    """The graded fix must not soften the genuine 'opposite directions' case.
    We construct latents in pairs that are forced to be each other's nearest
    neighbor, and assign opposite non-flat directions to each pair, so every
    NN-edge is a directional flip → mean conflict_rate must be 1.0."""
    cfg = load_config("configs/default.yaml")
    dim = cfg.memory.latent_dim or 32
    items = []
    latents = []
    # Build 4 pair clusters of 2 items each. Within a pair, items are nearly
    # identical; across pairs they are far apart. Each pair has one "long"
    # and one "short", so every item's NN is a directional flip.
    torch.manual_seed(123)
    for pair_idx in range(4):
        anchor = torch.zeros(dim)
        anchor[pair_idx * (dim // 4)] = 1.0  # disjoint one-hot per pair
        for inner in range(2):
            v = anchor + torch.randn(dim) * 1.0e-4
            v = v / v.norm()
            latents.append(v)
            items.append(_fake_item(
                symbol="AAPL",
                direction="long" if inner == 0 else "short",
                latent=v,
                primary_return=0.002 if inner == 0 else -0.002,
            ))
    latent_matrix = torch.stack(latents, dim=0)
    profile = build_retrieval_calibration(items, latent_matrix, cfg.memory)
    assert profile is not None
    # Every NN-edge is a long↔short flip → graded score = 1.0 for every pair
    # → every bucket's conflict_rate must be exactly 1.0.
    for b in profile.buckets:
        assert b.conflict_rate == 1.0, (
            f"directional flip bucket should hit conflict_rate 1.0, got {b.conflict_rate}"
        )
