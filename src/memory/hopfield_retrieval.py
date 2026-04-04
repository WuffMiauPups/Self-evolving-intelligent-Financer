from __future__ import annotations

from dataclasses import dataclass

import torch

from src.memory.similarity import pairwise_similarity


@dataclass(frozen=True)
class HopfieldRetrievalResult:
    context_vector: torch.Tensor
    attention_weights: torch.Tensor
    top_indices: list[int]
    top_similarities: list[float]
    average_similarity: float
    entropy: float
    weak_match: bool
    conflict_score: float


def hopfield_retrieve(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor | None = None,
    top_k: int,
    metric: str,
    temperature: float,
    min_similarity: float,
    conflict_values: torch.Tensor | None = None,
) -> HopfieldRetrievalResult:
    if query.dim() != 1:
        query = query.reshape(-1)
    if keys.dim() != 2:
        raise ValueError(f"Expected keys shape (items, dim), got {tuple(keys.shape)}")
    if values is None:
        values = keys
    if values.shape != keys.shape:
        raise ValueError(f"Expected values shape {tuple(keys.shape)}, got {tuple(values.shape)}")
    raw_scores, normalized_scores = pairwise_similarity(
        query,
        keys,
        metric=metric,
        temperature=temperature,
    )
    if raw_scores.dim() != 1:
        raw_scores = raw_scores.reshape(-1)
    if normalized_scores.dim() != 1:
        normalized_scores = normalized_scores.reshape(-1)
    resolved_top_k = min(max(top_k, 1), keys.shape[0])
    top_values, top_indices = torch.topk(normalized_scores, k=resolved_top_k)
    attention_logits = raw_scores[top_indices] / max(temperature, 1.0e-6)
    attention_weights = torch.softmax(attention_logits, dim=0)
    context_vector = torch.sum(values[top_indices] * attention_weights.unsqueeze(-1), dim=0)
    normalized_attention = torch.clamp(attention_weights, min=1.0e-9)
    entropy = float((-(normalized_attention * torch.log(normalized_attention))).sum().item())
    average_similarity = float(top_values.mean().item())
    weak_match = bool(top_values[0].item() < min_similarity)
    conflict_score = 0.0
    if conflict_values is not None and conflict_values.numel() == keys.shape[0]:
        selected = conflict_values[top_indices].to(dtype=torch.float32)
        weighted_conflict = torch.sum(attention_weights * selected.sign()).abs()
        conflict_score = float(torch.clamp(1.0 - weighted_conflict, min=0.0, max=1.0).item())
    return HopfieldRetrievalResult(
        context_vector=context_vector.detach().cpu(),
        attention_weights=attention_weights.detach().cpu(),
        top_indices=top_indices.tolist(),
        top_similarities=[float(value.item()) for value in top_values],
        average_similarity=average_similarity,
        entropy=entropy,
        weak_match=weak_match,
        conflict_score=conflict_score,
    )
