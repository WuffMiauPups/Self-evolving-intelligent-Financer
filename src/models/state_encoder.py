from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F

from src.core.device import maybe_autocast
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
    # FP16 autocast enable for the encoder forward. ``None`` auto-enables on
    # CUDA and skips on CPU; ``maybe_autocast`` further guards against a
    # non-CUDA device. Latents are explicitly cast back to fp32 below, so
    # memory writes stay in fp32 regardless of this flag.
    use_amp: bool | None = None

    def _amp_enabled(self) -> bool:
        if self.use_amp is None:
            return self.device.type == "cuda"
        return bool(self.use_amp)

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
        with torch.no_grad(), maybe_autocast(self.device, self._amp_enabled()):
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
        # --- L2-normalize fused representations before they enter memory -----
        # Memory similarity is cosine-based but the brain's fused_representation
        # has no norm constraint, so latents can drift onto an arbitrarily small
        # manifold and saturate cosine similarity to ~1.0 against everything
        # (the observed retrieval pathology). Normalizing at the encoder
        # boundary makes every latent live on the unit sphere, which is what
        # cosine similarity actually expects. This is a bank-side change only
        # — model training is unaffected because fused_representation is still
        # used unnormalized inside the model heads.
        normalized_fused = F.normalize(output.fused_representation, p=2.0, dim=-1, eps=1.0e-8)

        # --- Batched scalar extraction: collapse N*M per-bar .item() calls -----
        # The per-sample loop below previously issued one `.item()` per branch
        # per sample for confidences/norms plus soma/temporal, each of which
        # forces a GPU->CPU sync. We stack everything on-device first and then
        # transfer as a single block per logical group so the whole bar costs
        # at most four GPU->CPU syncs (confidences, norms, soma, temporal)
        # instead of ~2 * n_branches * batch + 2 per bar.
        batch_size = len(snapshots)

        branch_conf_names = list(output.branch_confidences.keys())
        if branch_conf_names:
            branch_conf_stack = torch.stack(
                [
                    output.branch_confidences[name].reshape(batch_size, -1)[:, 0]
                    for name in branch_conf_names
                ],
                dim=0,
            )
            branch_conf_cpu = branch_conf_stack.detach().cpu().numpy()
        else:
            branch_conf_cpu = None

        branch_embed_names = list(output.branch_embeddings.keys())
        if branch_embed_names:
            branch_norm_stack = torch.stack(
                [
                    output.branch_embeddings[name].reshape(batch_size, -1).norm(dim=-1)
                    for name in branch_embed_names
                ],
                dim=0,
            )
            branch_norm_cpu = branch_norm_stack.detach().cpu().numpy()
        else:
            branch_norm_cpu = None

        if output.soma_activation is not None:
            soma_cpu = (
                output.soma_activation.reshape(batch_size, -1)[:, 0]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            soma_cpu = None

        if output.temporal_embedding is not None:
            temporal_norm_cpu = (
                output.temporal_embedding.reshape(batch_size, -1)
                .norm(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            temporal_norm_cpu = None

        encoded_states: list[EncodedState] = []
        for index, snapshot in enumerate(snapshots):
            branch_confidences: dict[str, float] = {}
            if branch_conf_cpu is not None:
                for j, name in enumerate(branch_conf_names):
                    branch_confidences[name] = float(branch_conf_cpu[j, index])
            branch_norms: dict[str, float] = {}
            if branch_norm_cpu is not None:
                for j, name in enumerate(branch_embed_names):
                    branch_norms[name] = float(branch_norm_cpu[j, index])
            regime_weights = None
            if output.regime_weights is not None:
                regime_weights = output.regime_weights[index].detach().cpu().to(dtype=torch.float32)
            soma_value = 0.0 if soma_cpu is None else float(soma_cpu[index])
            temporal_norm_value = (
                0.0 if temporal_norm_cpu is None else float(temporal_norm_cpu[index])
            )
            encoded_states.append(
                EncodedState(
                    symbol=snapshot.symbol,
                    timestamp=snapshot.timestamp,
                    latent=normalized_fused[index].detach().cpu().to(dtype=torch.float32),
                    feature_contract_id=contract_id,
                    regime_weights=regime_weights,
                    branch_confidences=branch_confidences,
                    branch_norms=branch_norms,
                    encoder_fingerprint=self.encoder_fingerprint,
                    metadata={
                        "soma_activation": soma_value,
                        "feature_dim": snapshot.features.numel(),
                        "temporal_feature_dim": snapshot.temporal_tensor.shape[0],
                        "temporal_embedding_norm": temporal_norm_value,
                    },
                )
            )
        return encoded_states
