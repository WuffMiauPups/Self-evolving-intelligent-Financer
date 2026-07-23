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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        decouple_confidence: bool = False,
        depth: int = 1,
        decoupled: bool = False,
    ) -> None:
        super().__init__()
        self.decouple_confidence = decouple_confidence
        # Phase 2 §6.1 disentanglement: with ``decoupled=True`` there is no
        # learned confidence or sizing head. Confidence is the softmax
        # probability of the chosen class (so it MEANS "P(this direction)"
        # and nothing else can train it toward a different target), and
        # target_fraction is emitted as zeros — sizing belongs to the risk
        # layer, not the classifier.
        self.decoupled = decoupled
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
        if not decoupled:
            self.confidence = nn.Linear(hidden_dim, 1)
            self.target_fraction = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> PolicyOutput:
        hidden = self.backbone(inputs)
        action_logits = self.action(hidden)
        if self.decoupled:
            confidence = torch.softmax(action_logits, dim=-1).amax(dim=-1, keepdim=True)
            target_fraction = action_logits.new_zeros(action_logits.shape[0], 1)
        else:
            confidence = torch.sigmoid(self.confidence(hidden))
            target_fraction = torch.tanh(self.target_fraction(hidden))
        return PolicyOutput(
            action_logits=action_logits,
            confidence=confidence,
            target_fraction=target_fraction,
        )

