from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F

from src.features.feature_pipeline import FeatureContract, FeatureSnapshot
from src.memory.memory_bank import feature_contract_fingerprint
from src.models.base import TradingModelProtocol
from src.utils.torch_utils import ensure_batch_dimension, to_tensor


@dataclass
class EncodedState:
    symbol: str
    timestamp: datetime
    latent: torch.Tensor
    feature_contract_id: str
    regime_weights: torch.Tensor | None = None
    branch_confidences: dict[str, float] = field(default_factory=dict)
    branch_norms: dict[str, float] = field(default_factory=dict)
    encoder_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelStateEncoder:
    model: TradingModelProtocol
    device: torch.device
    feature_contract: FeatureContract | None = None
    expected_latent_dim: int | None = None
    encoder_fingerprint: str | None = None

    def encode(self, snapshot: FeatureSnapshot) -> EncodedState:
        return self.encode_batch([snapshot])[0]

    def encode_batch(self, snapshots: Sequence[FeatureSnapshot]) -> list[EncodedState]:
        if not snapshots:
            raise ValueError("encode_batch requires at least one snapshot")
        reference_contract = snapshots[0].contract
        if self.feature_contract is not None:
            self.feature_contract.assert_matches(reference_contract, context="state_encoder[batch]")
        for snapshot in snapshots[1:]:
            reference_contract.assert_matches(snapshot.contract, context="state_encoder[batch]")

        branch_inputs = {
            branch_name: torch.stack(
                [
                    ensure_batch_dimension(
                        to_tensor(snapshot.branch_inputs[branch_name], device=self.device, dtype=torch.float32)
                    ).squeeze(0)
                    for snapshot in snapshots
                ],
                dim=0,
            )
            for branch_name in reference_contract.branch_feature_names
        }
        regime_input = torch.stack(
            [
                ensure_batch_dimension(to_tensor(snapshot.regime_input, device=self.device, dtype=torch.float32)).squeeze(0)
                for snapshot in snapshots
            ],
            dim=0,
        )
        full_features = torch.stack(
            [
                ensure_batch_dimension(to_tensor(snapshot.features, device=self.device, dtype=torch.float32)).squeeze(0)
                for snapshot in snapshots
            ],
            dim=0,
        )
        temporal_slices = [
            ensure_batch_dimension(to_tensor(snapshot.temporal_tensor, device=self.device, dtype=torch.float32)).squeeze(0)
            for snapshot in snapshots
        ]
        max_time = max(temporal_slice.shape[-1] for temporal_slice in temporal_slices)
        temporal_tensor = torch.stack(
            [
                F.pad(temporal_slice, (max_time - temporal_slice.shape[-1], 0))
                for temporal_slice in temporal_slices
            ],
            dim=0,
        )
        with torch.no_grad():
            output = self.model(
                branch_inputs=branch_inputs,
                regime_input=regime_input,
                memory_state=None,
                full_features=full_features,
                temporal_tensor=temporal_tensor,
            )
        if output.fused_representation.dim() != 2 or output.fused_representation.shape[0] != len(snapshots):
            raise ValueError(
                f"Expected fused_representation shape (batch, latent), got {tuple(output.fused_representation.shape)}"
            )
        latent_dim = output.fused_representation.shape[-1]
        if self.expected_latent_dim is not None and latent_dim != self.expected_latent_dim:
            raise ValueError(f"Expected latent dim {self.expected_latent_dim}, got {latent_dim}")

        contract_id = feature_contract_fingerprint(reference_contract)
        encoded_states: list[EncodedState] = []
        for index, snapshot in enumerate(snapshots):
            branch_confidences = {
                name: float(value[index].reshape(-1)[0].item())
                for name, value in output.branch_confidences.items()
            }
            branch_norms = {
                name: float(output.branch_embeddings[name][index].norm().item())
                for name in output.branch_embeddings
            }
            regime_weights = None
            if output.regime_weights is not None:
                regime_weights = output.regime_weights[index].detach().cpu().to(dtype=torch.float32)
            encoded_states.append(
                EncodedState(
                    symbol=snapshot.symbol,
                    timestamp=snapshot.timestamp,
                    latent=output.fused_representation[index].detach().cpu().to(dtype=torch.float32),
                    feature_contract_id=contract_id,
                    regime_weights=regime_weights,
                    branch_confidences=branch_confidences,
                    branch_norms=branch_norms,
                    encoder_fingerprint=self.encoder_fingerprint,
                    metadata={
                        "soma_activation": float(output.soma_activation[index].reshape(-1)[0].item()),
                        "feature_dim": snapshot.features.numel(),
                        "temporal_feature_dim": snapshot.temporal_tensor.shape[0],
                        "temporal_embedding_norm": 0.0
                        if output.temporal_embedding is None
                        else float(output.temporal_embedding[index].norm().item()),
                    },
                )
            )
        return encoded_states
