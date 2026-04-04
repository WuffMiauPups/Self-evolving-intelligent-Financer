from __future__ import annotations

import torch
from torch import nn

from src.core.config import ModelConfig
from src.models.base import ModelOutput
from src.models.cnn_encoder import MarketTemporalCNN
from src.models.dendritic import DendriticBranch
from src.models.memory import LatentMemory
from src.models.policy_head import PolicyHead
from src.models.regime_map import RegimeEncoder
from src.models.soma import SomaAggregator
from src.models.value_head import ValueHead
from src.utils.torch_utils import ensure_batch_dimension


class BrainTradingModel(nn.Module):
    def __init__(self, branch_input_dims: dict[str, int], config: ModelConfig, *, temporal_input_dim: int = 0) -> None:
        super().__init__()
        self.model_config = config.model_copy(deep=True)
        self.branch_input_dims = dict(branch_input_dims)
        self.branch_names = list(branch_input_dims.keys())
        self.expected_regime_input_dim = config.regime_input_dim
        self.memory_dim = config.memory_dim
        self.temporal_input_dim = temporal_input_dim
        self.use_cnn_encoder = config.use_cnn_encoder and temporal_input_dim > 0
        self.regime_encoder = RegimeEncoder(
            input_dim=config.regime_input_dim,
            hidden_dim=config.regime_hidden_dim,
            embedding_dim=config.regime_embedding_dim,
            regime_count=config.regime_count,
            dropout=config.dropout,
        )
        self.temporal_encoder = None
        self.temporal_embedding_dim = 0
        if self.use_cnn_encoder:
            self.temporal_encoder = MarketTemporalCNN(
                input_channels=temporal_input_dim,
                channels=config.cnn_channels,
                kernel_sizes=config.cnn_kernel_sizes,
                temporal_blocks=config.temporal_blocks,
                dropout=config.dropout,
            )
            self.temporal_embedding_dim = self.temporal_encoder.output_dim
        self.branches = nn.ModuleDict(
            {
                branch_name: DendriticBranch(
                    input_dim=max(input_dim, 1),
                    hidden_dim=config.branch_hidden_dim,
                    output_dim=config.branch_output_dim,
                    regime_dim=config.regime_embedding_dim,
                    depth=config.branch_depth,
                    dropout=config.dropout,
                )
                for branch_name, input_dim in branch_input_dims.items()
            }
        )
        self.memory = LatentMemory(input_dim=config.soma_hidden_dim, memory_dim=config.memory_dim)
        self.soma = SomaAggregator(
            branch_dim=config.branch_output_dim,
            regime_dim=config.regime_embedding_dim,
            memory_dim=config.memory_dim,
            hidden_dim=config.soma_hidden_dim,
            dropout=config.dropout,
        )
        head_input_dim = config.soma_hidden_dim + config.memory_dim + self.temporal_embedding_dim
        self.policy_head = PolicyHead(
            input_dim=head_input_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.value_head = ValueHead(input_dim=head_input_dim, hidden_dim=config.head_hidden_dim)

    def initial_memory_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.memory.initial_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        *,
        branch_inputs: dict[str, torch.Tensor],
        regime_input: torch.Tensor,
        memory_state: torch.Tensor | None = None,
        full_features: torch.Tensor | None = None,
        temporal_tensor: torch.Tensor | None = None,
    ) -> ModelOutput:
        del full_features
        regime_input = ensure_batch_dimension(regime_input)
        if regime_input.shape[-1] != self.expected_regime_input_dim:
            raise ValueError(
                f"Expected regime_input dim {self.expected_regime_input_dim}, got {regime_input.shape[-1]}"
            )
        regime_output = self.regime_encoder(regime_input)
        batch_size = regime_input.shape[0]
        missing_branches = [name for name in self.branch_names if name not in branch_inputs]
        if missing_branches:
            raise ValueError(f"Missing branch inputs: {missing_branches}")
        if memory_state is None:
            memory_state = self.initial_memory_state(
                batch_size,
                device=regime_input.device,
                dtype=regime_input.dtype,
            )
        else:
            if memory_state.shape != (batch_size, self.memory_dim):
                raise ValueError(
                    f"Expected memory_state shape {(batch_size, self.memory_dim)}, got {tuple(memory_state.shape)}"
                )
            memory_state = memory_state.to(device=regime_input.device, dtype=regime_input.dtype)

        branch_embeddings: dict[str, torch.Tensor] = {}
        branch_confidences: dict[str, torch.Tensor] = {}
        ordered_embeddings: list[torch.Tensor] = []
        ordered_confidences: list[torch.Tensor] = []
        for branch_name in self.branch_names:
            branch_tensor = ensure_batch_dimension(branch_inputs[branch_name])
            expected_input_dim = max(self.branch_input_dims[branch_name], 1)
            if branch_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Branch {branch_name} batch size {branch_tensor.shape[0]} does not match regime batch {batch_size}"
                )
            if branch_tensor.shape[-1] != expected_input_dim:
                raise ValueError(
                    f"Branch {branch_name} expected dim {expected_input_dim}, got {branch_tensor.shape[-1]}"
                )
            branch_output = self.branches[branch_name](branch_tensor, regime_output.embedding)
            branch_embeddings[branch_name] = branch_output.embedding
            branch_confidences[branch_name] = branch_output.confidence
            ordered_embeddings.append(branch_output.embedding)
            ordered_confidences.append(branch_output.confidence)

        stacked_embeddings = torch.stack(ordered_embeddings, dim=1)
        stacked_confidences = torch.stack(ordered_confidences, dim=1)
        soma_output = self.soma(
            branch_embeddings=stacked_embeddings,
            branch_confidences=stacked_confidences,
            regime_embedding=regime_output.embedding,
            memory_state=memory_state,
        )
        next_memory = self.memory(soma_output.fused, memory_state)
        temporal_embedding = None
        if self.use_cnn_encoder:
            if temporal_tensor is None:
                raise ValueError("BrainTradingModel requires temporal_tensor when use_cnn_encoder=True")
            if temporal_tensor.dim() == 2:
                temporal_tensor = temporal_tensor.unsqueeze(0)
            if temporal_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Temporal tensor batch size {temporal_tensor.shape[0]} does not match regime batch {batch_size}"
                )
            temporal_output = self.temporal_encoder(temporal_tensor.to(device=regime_input.device, dtype=regime_input.dtype))
            temporal_embedding = temporal_output.embedding
        if temporal_embedding is None:
            head_input = torch.cat([soma_output.fused, next_memory], dim=-1)
        else:
            head_input = torch.cat([soma_output.fused, next_memory, temporal_embedding], dim=-1)
        policy_output = self.policy_head(head_input, soma_output.activation)
        value = self.value_head(head_input)
        return ModelOutput(
            action_logits=policy_output.action_logits,
            confidence=policy_output.confidence,
            target_fraction=policy_output.target_fraction,
            fused_representation=head_input,
            branch_embeddings=branch_embeddings,
            branch_confidences=branch_confidences,
            soma_weights=soma_output.weights,
            soma_activation=soma_output.activation,
            regime_weights=regime_output.weights,
            value=value,
            memory_state=next_memory,
            temporal_embedding=temporal_embedding,
        )
