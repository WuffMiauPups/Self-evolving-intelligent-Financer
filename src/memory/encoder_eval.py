from __future__ import annotations

from collections import defaultdict

import torch

from src.memory.calibration import calibrate_match_strength
from src.memory.memory_bank import MemoryBank
from src.memory.schemas import EncoderQualityReport, MemoryUnit
from src.memory.similarity import pairwise_similarity


def evaluate_encoder_quality(bank: MemoryBank, *, min_samples: int = 3) -> EncoderQualityReport:
    if len(bank.items) < max(min_samples, 2) or bank.latent_matrix.shape[0] < max(min_samples, 2):
        return EncoderQualityReport(
            item_count=len(bank.items),
            unit_count=len(bank.memory_units),
            nearest_neighbor_similarity=0.0,
            nearest_neighbor_directional_consistency=0.0,
            average_outcome_dispersion=0.0,
            unit_compactness=0.0,
            unit_separation=0.0,
            symbol_stability={},
            similarity_bucket_consistency={},
            retrieval_coverage=0.0,
            weak_match_rate=0.0,
            conflict_rate=0.0,
            average_calibrated_confidence=0.0,
        )
    similarities: list[float] = []
    consistencies: list[float] = []
    dispersions: list[float] = []
    conflicts: list[float] = []
    calibrated_confidences: list[float] = []
    symbol_consistency: dict[str, list[float]] = defaultdict(list)
    regime_consistency: dict[str, list[float]] = defaultdict(list)
    profile = bank.calibration_profile
    for index, item in enumerate(bank.items):
        _, normalized_scores = pairwise_similarity(
            bank.latent_matrix[index],
            bank.latent_matrix,
            metric=bank.similarity_metric,
            temperature=float(bank.meta.get("attention_temperature", 1.0)),
        )
        normalized_scores = normalized_scores.clone()
        normalized_scores[index] = -1.0
        neighbor_index = int(torch.argmax(normalized_scores).item())
        similarity = float(normalized_scores[neighbor_index].item())
        neighbor = bank.items[neighbor_index]
        agreement = 1.0 if item.outcome.direction_class == neighbor.outcome.direction_class else 0.0
        dispersion = abs(item.outcome.primary_return - neighbor.outcome.primary_return)
        conflict = 1.0 if (
            item.outcome.direction_class not in {"flat"} and neighbor.outcome.direction_class not in {"flat"}
            and item.outcome.direction_class != neighbor.outcome.direction_class
        ) else 0.0
        _, calibrated_confidence, _ = calibrate_match_strength(profile, similarity)
        similarities.append(similarity)
        consistencies.append(agreement)
        dispersions.append(dispersion)
        conflicts.append(conflict)
        calibrated_confidences.append(calibrated_confidence)
        symbol_consistency[item.symbol].append(agreement)
        if item.regime_weights is not None and item.regime_weights.numel() > 0:
            regime_key = f"regime_{int(torch.argmax(item.regime_weights).item())}"
            regime_consistency[regime_key].append(agreement)

    bucket_consistency: dict[str, float] = {}
    if bank.calibration_profile is not None:
        for bucket in bank.calibration_profile.buckets:
            label = f"{bucket.lower_bound:.3f}-{bucket.upper_bound:.3f}"
            bucket_consistency[label] = bucket.direction_agreement
    compactness = _average_unit_compactness(bank.memory_units)
    separation = _average_unit_separation(bank.memory_units)
    recommended = 0.0 if profile is None else profile.recommended_strong_match_threshold
    weak_match_rate = (
        sum(1 for similarity in similarities if similarity < recommended) / len(similarities)
        if similarities and recommended > 0.0
        else 0.0
    )
    return EncoderQualityReport(
        item_count=len(bank.items),
        unit_count=len(bank.memory_units),
        nearest_neighbor_similarity=float(torch.tensor(similarities, dtype=torch.float32).mean().item()),
        nearest_neighbor_directional_consistency=float(torch.tensor(consistencies, dtype=torch.float32).mean().item()),
        average_outcome_dispersion=float(torch.tensor(dispersions, dtype=torch.float32).mean().item()),
        unit_compactness=compactness,
        unit_separation=separation,
        symbol_stability={symbol: sum(values) / len(values) for symbol, values in sorted(symbol_consistency.items())},
        similarity_bucket_consistency=bucket_consistency,
        regime_stability={name: sum(values) / len(values) for name, values in sorted(regime_consistency.items())},
        retrieval_coverage=1.0 - weak_match_rate,
        weak_match_rate=weak_match_rate,
        conflict_rate=float(torch.tensor(conflicts, dtype=torch.float32).mean().item()),
        average_calibrated_confidence=float(torch.tensor(calibrated_confidences, dtype=torch.float32).mean().item()),
    )


def _average_unit_compactness(units: list[MemoryUnit]) -> float:
    if not units:
        return 0.0
    return sum(max(0.0, 1.0 - unit.latent_dispersion) for unit in units) / len(units)


def _average_unit_separation(units: list[MemoryUnit]) -> float:
    mature_units = [unit for unit in units if unit.status == "mature"]
    if len(mature_units) < 2:
        return 0.0
    centroids = torch.stack([unit.prototype for unit in mature_units], dim=0)
    distances = torch.cdist(centroids, centroids, p=2)
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool)
    if not mask.any():
        return 0.0
    return float((1.0 / (1.0 + distances[mask])).mean().item())
