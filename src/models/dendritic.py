from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.encoders import FeedForwardBlock


@dataclass
class BranchOutput:
    embedding: torch.Tensor
    confidence: torch.Tensor


class DendriticBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        regime_dim: int,
        *,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = FeedForwardBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            dropout=dropout,
        )
        self.regime_gate = nn.Linear(regime_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, branch_input: torch.Tensor, regime_embedding: torch.Tensor) -> BranchOutput:
        encoded = self.encoder(branch_input)
        gate = torch.sigmoid(self.regime_gate(regime_embedding))
        gated = self.norm(encoded * gate)
        confidence = torch.sigmoid(self.confidence_head(gated))
        return BranchOutput(embedding=gated, confidence=confidence)

