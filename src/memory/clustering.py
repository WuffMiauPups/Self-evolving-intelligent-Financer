from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from src.memory.path_stats import aggregate_path_statistics, aggregate_risk_statistics
from src.memory.schemas import MemoryItem, PrototypeCluster
from src.memory.similarity import pairwise_similarity


@dataclass
class _ClusterAccumulator:
    cluster_id: str
    centroid: torch.Tensor
    member_indices: list[int] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)

    def add_member(self, index: int, latent: torch.Tensor, similarity: float) -> None:
        member_count = len(self.member_indices)
        self.centroid = ((self.centroid * member_count) + latent) / float(member_count + 1)
        self.member_indices.append(index)
        self.similarities.append(float(similarity))


def build_prototypes(
    items: list[MemoryItem],
    latents: torch.Tensor,
    *,
    similarity_metric: str,
    min_similarity: float,
    min_cluster_size: int,
    temperature: float,
) -> list[PrototypeCluster]:
    if not items:
        return []
    if latents.shape[0] != len(items):
        raise ValueError(f"Expected {len(items)} latents, got {latents.shape[0]}")
    accumulators: list[_ClusterAccumulator] = []
    for index, latent in enumerate(latents):
        latent = latent.detach().cpu().to(dtype=torch.float32)
        if not accumulators:
            accumulators.append(
                _ClusterAccumulator(
                    cluster_id=f"cluster_{len(accumulators):04d}",
                    centroid=latent.clone(),
                    member_indices=[index],
                    similarities=[1.0],
                )
            )
            continue
        centroids = torch.stack([cluster.centroid for cluster in accumulators], dim=0)
        _, normalized_scores = pairwise_similarity(
            latent,
            centroids,
            metric=similarity_metric,
            temperature=temperature,
        )
        best_index = int(torch.argmax(normalized_scores).item())
        best_similarity = float(normalized_scores[best_index].item())
        if best_similarity >= min_similarity:
            accumulators[best_index].add_member(index, latent, best_similarity)
            continue
        accumulators.append(
            _ClusterAccumulator(
                cluster_id=f"cluster_{len(accumulators):04d}",
                centroid=latent.clone(),
                member_indices=[index],
                similarities=[1.0],
            )
        )

    clusters: list[PrototypeCluster] = []
    for cluster in accumulators:
        member_outcomes = [items[index].outcome for index in cluster.member_indices]
        regime_labels = Counter()
        for index in cluster.member_indices:
            regime_weights = items[index].regime_weights
            if regime_weights is None or regime_weights.numel() == 0:
                continue
            regime_label = f"regime_{int(torch.argmax(regime_weights).item())}"
            regime_labels[regime_label] += 1
        member_count = len(cluster.member_indices)
        regime_composition = (
            {}
            if member_count < min_cluster_size
            else {
                label: count / member_count
                for label, count in sorted(regime_labels.items())
            }
        )
        similarities = torch.tensor(cluster.similarities, dtype=torch.float32)
        clusters.append(
            PrototypeCluster(
                cluster_id=cluster.cluster_id,
                centroid=cluster.centroid.detach().cpu(),
                member_indices=list(cluster.member_indices),
                member_count=member_count,
                average_similarity=float(similarities.mean().item()),
                min_similarity=float(similarities.min().item()),
                max_similarity=float(similarities.max().item()),
                similarity_std=float(similarities.std(unbiased=False).item()) if similarities.numel() > 1 else 0.0,
                path_stats=aggregate_path_statistics(member_outcomes),
                risk_stats=aggregate_risk_statistics(member_outcomes),
                regime_composition=regime_composition,
            )
        )
    return clusters
