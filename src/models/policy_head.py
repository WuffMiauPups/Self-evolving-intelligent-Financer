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
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, decouple_confidence: bool = False, depth: int = 1) -> None:
        super().__init__()
        self.decouple_confidence = decouple_confidence
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.action = nn.Linear(hidden_dim, 3)
        self.confidence = nn.Linear(hidden_dim, 1)
        self.target_fraction = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, soma_activation: torch.Tensor | None = None) -> PolicyOutput:
        # ``soma_activation`` parameter kept for backward compatibility but
        # ignored: the soma aggregator was removed in the Great Simplification
        # and multiplying by it caused FP16 underflow in inference. Confidence
        # is now always decoupled from that (dead) signal.
        del soma_activation
        hidden = self.backbone(inputs)
        action_logits = self.action(hidden)
        confidence = torch.sigmoid(self.confidence(hidden))
        target_fraction = torch.tanh(self.target_fraction(hidden))
        return PolicyOutput(
            action_logits=action_logits,
            confidence=confidence,
            target_fraction=target_fraction,
        )

