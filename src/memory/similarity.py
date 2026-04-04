from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_similarity(
    query: torch.Tensor,
    candidates: torch.Tensor,
    *,
    metric: str = "cosine",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.dim() == 1:
        query = query.unsqueeze(0)
    if candidates.dim() == 1:
        candidates = candidates.unsqueeze(0)
    if query.shape[-1] != candidates.shape[-1]:
        raise ValueError(
            f"Similarity dimension mismatch: query {query.shape[-1]} vs candidates {candidates.shape[-1]}"
        )
    query = query.to(dtype=torch.float32)
    candidates = candidates.to(device=query.device, dtype=torch.float32)

    if metric == "cosine":
        normalized_query = F.normalize(query, dim=-1)
        normalized_candidates = F.normalize(candidates, dim=-1)
        raw = torch.matmul(normalized_query, normalized_candidates.transpose(0, 1))
        normalized = (raw + 1.0) * 0.5
        return raw.squeeze(0), normalized.squeeze(0)

    if metric == "euclidean":
        distances = torch.cdist(query, candidates, p=2)
        raw = -distances
        normalized = 1.0 / (1.0 + distances)
        return raw.squeeze(0), normalized.squeeze(0)

    if metric == "attention":
        normalized_query = F.normalize(query, dim=-1)
        normalized_candidates = F.normalize(candidates, dim=-1)
        raw = torch.matmul(normalized_query, normalized_candidates.transpose(0, 1)) / temperature
        normalized = torch.softmax(raw, dim=-1)
        return raw.squeeze(0), normalized.squeeze(0)

    raise ValueError(f"Unsupported similarity metric: {metric}")
