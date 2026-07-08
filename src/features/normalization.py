from __future__ import annotations

import torch


def safe_zscore(value: torch.Tensor, reference: torch.Tensor, epsilon: float) -> torch.Tensor:
    std = reference.std(unbiased=False)
    return (value - reference.mean()) / (std + epsilon)


class RunningStandardizer:
    """Exponential moving average standardizer for online feature normalization.

    Tracks running mean and variance using EMA updates. After a warmup
    period, transforms features to approximately zero-mean, unit-variance.
    """

    def __init__(self, feature_dim: int, alpha: float = 0.01, warmup: int = 50) -> None:
        self.mean = torch.zeros(feature_dim)
        self.var = torch.ones(feature_dim)
        self.alpha = alpha
        self.warmup = warmup
        self.count = 0

    def update(self, features: torch.Tensor) -> None:
        """Update running statistics with a new observation."""
        self.count += 1
        if self.count <= self.warmup:
            # During warmup, use simple cumulative average
            weight = 1.0 / self.count
            delta = features - self.mean
            self.mean = self.mean + weight * delta
            self.var = self.var + weight * (delta * (features - self.mean) - self.var)
        else:
            delta = features - self.mean
            self.mean = self.mean + self.alpha * delta
            self.var = (1.0 - self.alpha) * self.var + self.alpha * delta * (features - self.mean)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize features using current running statistics."""
        if self.count < self.warmup:
            return features
        std = torch.sqrt(self.var.clamp(min=1e-8))
        return (features - self.mean) / std

    def update_and_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Update statistics and return standardized features."""
        self.update(features)
        return self.transform(features)

