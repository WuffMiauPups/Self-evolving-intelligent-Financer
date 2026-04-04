from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.encoders import FeedForwardBlock


@dataclass
class RegimeOutput:
    embedding: torch.Tensor
    weights: torch.Tensor
    logits: torch.Tensor


class RegimeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        regime_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = FeedForwardBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=regime_count,
            depth=2,
            dropout=dropout,
        )
        self.prototypes = nn.Parameter(torch.randn(regime_count, embedding_dim) * 0.02)

    def forward(self, regime_input: torch.Tensor) -> RegimeOutput:
        logits = self.encoder(regime_input)
        weights = torch.softmax(logits, dim=-1)
        embedding = weights @ self.prototypes
        return RegimeOutput(embedding=embedding, weights=weights, logits=logits)

