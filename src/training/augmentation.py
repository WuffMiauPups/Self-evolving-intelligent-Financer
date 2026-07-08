"""Training data augmentation for market feature tensors.

Provides Gaussian noise injection and temporal jitter to improve
generalization on sparse trading data. Applied only during training,
never during validation or inference.
"""
from __future__ import annotations

import torch


class TrainingAugmenter:
    """Augments feature tensors for training with noise and jitter."""

    def __init__(
        self,
        noise_std: float = 0.01,
        temporal_jitter_prob: float = 0.1,
    ) -> None:
        self.noise_std = noise_std
        self.temporal_jitter_prob = temporal_jitter_prob

    def augment(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to feature tensor.

        Args:
            features: 1-D feature tensor of shape (F,)

        Returns:
            Noisy features with same shape.
        """
        if self.noise_std > 0:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        return features

    def temporal_jitter(self, temporal_tensor: torch.Tensor) -> torch.Tensor:
        """Randomly roll temporal features by +/-1 step with probability.

        Args:
            temporal_tensor: 2-D tensor of shape (T_features, T_steps)

        Returns:
            Jittered tensor with same shape.
        """
        if temporal_tensor.dim() < 2 or temporal_tensor.shape[1] < 3:
            return temporal_tensor
        if torch.rand(1).item() < self.temporal_jitter_prob:
            shift = 1 if torch.rand(1).item() < 0.5 else -1
            temporal_tensor = torch.roll(temporal_tensor, shifts=shift, dims=1)
            # Zero out the wrapped edge to avoid look-ahead/look-behind
            if shift > 0:
                temporal_tensor[:, :shift] = 0.0
            else:
                temporal_tensor[:, shift:] = 0.0
        return temporal_tensor
