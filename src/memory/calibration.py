from __future__ import annotations

from collections import defaultdict

import torch

from src.core.config import MemoryConfig
from src.memory.schemas import MemoryItem, RetrievalCalibrationBucket, RetrievalCalibrationProfile
from src.memory.similarity import pairwise_similarity


def build_retrieval_calibration(
    items: list[MemoryItem],
    latent_matrix: torch.Tensor,
    config: MemoryConfig,
) -> RetrievalCalibrationProfile | None:
    if len(items) < 3 or latent_matrix.shape[0] < 3:
        return None
    similarities: list[float] = []
    agreements: list[float] = []
    dispersions: list[float] = []
    conflicts: list[float] = []
    for index, item in enumerate(items):
        query = latent_matrix[index]
        raw_scores, normalized_scores = pairwise_similarity(
            query,
            latent_matrix,
            metric=config.similarity_metric,
            temperature=config.attention_temperature,
        )
        del raw_scores
        normalized_scores = normalized_scores.clone()
        normalized_scores[index] = -1.0
        neighbor_index = int(torch.argmax(normalized_scores).item())
        similarity = float(normalized_scores[neighbor_index].item())
        neighbor = items[neighbor_index]
        agreement = 1.0 if item.outcome.direction_class == neighbor.outcome.direction_class else 0.0
        dispersion = abs(item.outcome.primary_return - neighbor.outcome.primary_return)
        # --- Graded conflict score -------------------------------------------
        # Previously this was binary: 1.0 iff item and neighbor had opposite
        # *non-flat* direction labels, else 0.0. On small per-symbol calibration
        # buckets, a single mismatched pair could push the bucket's mean
        # conflict to 1.0 and zero out calibrated_confidence entirely
        # (calibrated_confidence is a 4-way product). The graded version below
        # gives a 0.5 weight to flat-vs-directional disagreements (real but
        # weaker), keeps directional flips at 1.0, and grants 0.0 only to
        # exact agreement, so a sparse bucket with one mismatch no longer
        # collapses the whole calibration multiplier.
        if item.outcome.direction_class == neighbor.outcome.direction_class:
            conflict = 0.0
        elif "flat" in {item.outcome.direction_class, neighbor.outcome.direction_class}:
            conflict = 0.5
        else:
            conflict = 1.0
        similarities.append(similarity)
        agreements.append(agreement)
        dispersions.append(dispersion)
        conflicts.append(conflict)

    similarity_tensor = torch.tensor(similarities, dtype=torch.float32)
    dispersion_tensor = torch.tensor(dispersions, dtype=torch.float32)
    dispersion_scale = max(float(torch.quantile(dispersion_tensor, 0.75).item()), 1.0e-6)
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    bucket_edges = torch.quantile(similarity_tensor, quantiles).tolist()
    bucket_payloads: dict[int, list[int]] = defaultdict(list)
    for sample_index, similarity in enumerate(similarities):
        bucket_index = resolve_bucket_index(similarity, bucket_edges)
        bucket_payloads[bucket_index].append(sample_index)

    buckets: list[RetrievalCalibrationBucket] = []
    confidence_values: list[float] = []
    for bucket_index in sorted(bucket_payloads):
        indices = bucket_payloads[bucket_index]
        bucket_similarity = torch.tensor([similarities[idx] for idx in indices], dtype=torch.float32)
        bucket_agreement = torch.tensor([agreements[idx] for idx in indices], dtype=torch.float32)
        bucket_dispersion = torch.tensor([dispersions[idx] for idx in indices], dtype=torch.float32)
        bucket_conflict = torch.tensor([conflicts[idx] for idx in indices], dtype=torch.float32)
        normalized_dispersion = float(torch.clamp(bucket_dispersion.mean() / dispersion_scale, min=0.0, max=1.0).item())
        calibrated_confidence = float(
            (
                bucket_similarity.mean()
                * bucket_agreement.mean()
                * (1.0 - normalized_dispersion)
                * (1.0 - bucket_conflict.mean())
            ).item()
        )
        confidence_values.append(calibrated_confidence)
        buckets.append(
            RetrievalCalibrationBucket(
                lower_bound=bucket_edges[bucket_index],
                upper_bound=bucket_edges[min(bucket_index + 1, len(bucket_edges) - 1)],
                sample_count=len(indices),
                mean_similarity=float(bucket_similarity.mean().item()),
                direction_agreement=float(bucket_agreement.mean().item()),
                outcome_dispersion=float(bucket_dispersion.mean().item()),
                conflict_rate=float(bucket_conflict.mean().item()),
                calibrated_confidence=calibrated_confidence,
            )
        )
    recommended = float(torch.tensor(confidence_values, dtype=torch.float32).median().item()) if confidence_values else 0.0
    return RetrievalCalibrationProfile(
        metric=config.similarity_metric,
        bucket_edges=[float(value) for value in bucket_edges],
        buckets=buckets,
        sample_count=len(similarities),
        recommended_strong_match_threshold=recommended,
    )


def build_retrieval_calibration_profiles(
    items: list[MemoryItem],
    latent_matrix: torch.Tensor,
    config: MemoryConfig,
) -> tuple[
    RetrievalCalibrationProfile | None,
    dict[str, RetrievalCalibrationProfile],
    dict[str, RetrievalCalibrationProfile],
]:
    global_profile = build_retrieval_calibration(items, latent_matrix, config)
    symbol_profiles: dict[str, RetrievalCalibrationProfile] = {}
    regime_profiles: dict[str, RetrievalCalibrationProfile] = {}
    if config.symbol_specific_profiles:
        symbol_groups = _group_indices_by_symbol(items)
        for symbol, indices in symbol_groups.items():
            profile = _build_group_profile(indices, items, latent_matrix, config)
            if profile is not None:
                symbol_profiles[symbol] = profile
    if config.regime_specific_profiles:
        regime_groups = _group_indices_by_regime(items)
        for regime, indices in regime_groups.items():
            profile = _build_group_profile(indices, items, latent_matrix, config)
            if profile is not None:
                regime_profiles[regime] = profile
    return global_profile, symbol_profiles, regime_profiles


def resolve_bucket_index(similarity: float, bucket_edges: list[float]) -> int:
    for index in range(len(bucket_edges) - 1):
        upper = bucket_edges[index + 1]
        if similarity <= upper or index == len(bucket_edges) - 2:
            return index
    return max(len(bucket_edges) - 2, 0)


def calibrate_match_strength(
    profile: RetrievalCalibrationProfile | None,
    similarity: float,
) -> tuple[float, float, float]:
    if profile is None or not profile.buckets:
        return similarity, similarity, 0.0
    bucket_index = resolve_bucket_index(similarity, profile.bucket_edges)
    bucket = profile.buckets[min(bucket_index, len(profile.buckets) - 1)]
    confidence = bucket.calibrated_confidence
    conflict_score = min(max(bucket.conflict_rate + bucket.outcome_dispersion, 0.0), 1.0)
    calibrated = max(min(confidence * similarity, 1.0), 0.0)
    return calibrated, confidence, conflict_score


def _build_group_profile(
    indices: list[int],
    items: list[MemoryItem],
    latent_matrix: torch.Tensor,
    config: MemoryConfig,
) -> RetrievalCalibrationProfile | None:
    if len(indices) < max(config.min_effective_matches + 1, 3):
        return None
    group_items = [items[index] for index in indices]
    group_latents = latent_matrix[torch.tensor(indices, dtype=torch.long)]
    return build_retrieval_calibration(group_items, group_latents, config)


def _group_indices_by_symbol(items: list[MemoryItem]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[item.symbol].append(index)
    return groups


def _group_indices_by_regime(items: list[MemoryItem]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        regime_key = dominant_regime_key(item.regime_weights)
        if regime_key is not None:
            groups[regime_key].append(index)
    return groups


def dominant_regime_key(regime_weights: torch.Tensor | None) -> str | None:
    if regime_weights is None or regime_weights.numel() == 0:
        return None
    return f"regime_{int(torch.argmax(regime_weights).item())}"
