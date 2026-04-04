from __future__ import annotations

import torch


def safe_zscore(value: torch.Tensor, reference: torch.Tensor, epsilon: float) -> torch.Tensor:
    std = reference.std(unbiased=False)
    return (value - reference.mean()) / (std + epsilon)

