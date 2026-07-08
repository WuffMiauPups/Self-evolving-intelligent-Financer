from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class ModelOutput:
    action_logits: torch.Tensor
    confidence: torch.Tensor
    target_fraction: torch.Tensor
    fused_representation: torch.Tensor
    branch_embeddings: dict[str, torch.Tensor]
    branch_confidences: dict[str, torch.Tensor]
    soma_weights: torch.Tensor
    soma_activation: torch.Tensor
    regime_weights: torch.Tensor | None = None
    value: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None
    temporal_embedding: torch.Tensor | None = None
    risk_predictions: dict[str, torch.Tensor] | None = None


class TradingModelProtocol(Protocol):
    def forward(
        self,
        *,
        branch_inputs: dict[str, torch.Tensor],
        regime_input: torch.Tensor,
        memory_state: torch.Tensor | None = None,
        full_features: torch.Tensor | None = None,
        temporal_tensor: torch.Tensor | None = None,
    ) -> ModelOutput:
        ...
