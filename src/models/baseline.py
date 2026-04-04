from __future__ import annotations

import torch
from torch import nn

from src.models.base import ModelOutput


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.action = nn.Linear(hidden_dim, 3)
        self.confidence = nn.Linear(hidden_dim, 1)
        self.target_fraction = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        *,
        branch_inputs: dict[str, torch.Tensor],
        regime_input: torch.Tensor,
        memory_state: torch.Tensor | None = None,
        full_features: torch.Tensor | None = None,
        temporal_tensor: torch.Tensor | None = None,
    ) -> ModelOutput:
        del branch_inputs, regime_input, memory_state, temporal_tensor
        if full_features is None:
            raise ValueError("BaselineMLP requires full_features")
        if full_features.dim() == 1:
            full_features = full_features.unsqueeze(0)
        if full_features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected full_features dim {self.input_dim}, got {full_features.shape[-1]}")
        hidden = self.network(full_features)
        confidence = torch.sigmoid(self.confidence(hidden))
        return ModelOutput(
            action_logits=self.action(hidden),
            confidence=confidence,
            target_fraction=torch.tanh(self.target_fraction(hidden)) * confidence,
            fused_representation=hidden,
            branch_embeddings={},
            branch_confidences={},
            soma_weights=torch.ones(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype),
            soma_activation=torch.ones(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype),
            regime_weights=None,
            value=self.value(hidden),
            memory_state=None,
            temporal_embedding=None,
        )
