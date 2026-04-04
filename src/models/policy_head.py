from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PolicyOutput:
    action_logits: torch.Tensor
    confidence: torch.Tensor
    target_fraction: torch.Tensor


class PolicyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action = nn.Linear(hidden_dim, 3)
        self.confidence = nn.Linear(hidden_dim, 1)
        self.target_fraction = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, soma_activation: torch.Tensor) -> PolicyOutput:
        hidden = self.backbone(inputs)
        action_logits = self.action(hidden)
        confidence = torch.sigmoid(self.confidence(hidden)) * soma_activation
        target_fraction = torch.tanh(self.target_fraction(hidden)) * soma_activation
        return PolicyOutput(
            action_logits=action_logits,
            confidence=confidence,
            target_fraction=target_fraction,
        )

