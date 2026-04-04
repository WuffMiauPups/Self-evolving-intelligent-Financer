from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.encoders import FeedForwardBlock


@dataclass
class SomaOutput:
    fused: torch.Tensor
    weights: torch.Tensor
    activation: torch.Tensor


class SomaAggregator(nn.Module):
    def __init__(
        self,
        branch_dim: int,
        regime_dim: int,
        memory_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gate_layer = nn.Linear(branch_dim + regime_dim + 1, 1)
        self.fuser = FeedForwardBlock(
            input_dim=branch_dim + regime_dim + memory_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=2,
            dropout=dropout,
        )
        self.activation_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        branch_embeddings: torch.Tensor,
        branch_confidences: torch.Tensor,
        regime_embedding: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> SomaOutput:
        batch_size, branch_count, _ = branch_embeddings.shape
        regime_context = regime_embedding.unsqueeze(1).expand(batch_size, branch_count, -1)
        gate_input = torch.cat([branch_embeddings, regime_context, branch_confidences], dim=-1)
        raw_weights = self.gate_layer(gate_input).squeeze(-1)
        weights = torch.softmax(raw_weights, dim=-1)
        weighted = (weights.unsqueeze(-1) * branch_embeddings).sum(dim=1)
        fused = self.fuser(torch.cat([weighted, regime_embedding, memory_state], dim=-1))
        mean_confidence = branch_confidences.mean(dim=1)
        activation = torch.sigmoid(self.activation_head(torch.cat([fused, mean_confidence], dim=-1)))
        return SomaOutput(fused=fused, weights=weights, activation=activation)

