"""Simplified brain trading model (post-Great-Simplification).

Architecture: optional Temporal CNN on raw OHLCV history → concat with flat
feature vector → 2-layer MLP → policy head + value head + optional risk head.

Dendritic branches, soma aggregator, regime encoder, and GRU latent memory
have all been removed — they were adding complexity without measurable
benefit (baseline MLP scored 0.412 val_acc vs brain 0.430, 1.8pt gap across
5000+ LOC of specialized machinery).

ModelOutput fields are populated defensively so existing callers (brain_policy,
state_encoder, trainer) don't crash when they access legacy fields. Removed
components contribute None / empty / ones-like placeholders.
"""
from __future__ import annotations

import torch
from torch import nn

from src.core.config import ModelConfig
from src.models.base import ModelOutput
from src.models.cnn_encoder import MarketTemporalCNN
from src.models.policy_head import PolicyHead
from src.models.value_head import RiskHead, ValueHead
from src.utils.torch_utils import ensure_batch_dimension


class BrainTradingModel(nn.Module):
    """Single-block CNN on OHLCV + MLP on flat features → policy/value/risk heads."""

    def __init__(
        self,
        branch_input_dims: dict[str, int],
        config: ModelConfig,
        *,
        temporal_input_dim: int = 0,
    ) -> None:
        super().__init__()
        self.model_config = config.model_copy(deep=True)
        # ``branch_input_dims`` is retained as a constructor arg for backward
        # compatibility with the factory; the dendritic branches are gone so
        # we only use the sum of input dims as the flat feature width.
        self.branch_input_dims = dict(branch_input_dims)
        self.branch_names = list(branch_input_dims.keys())
        self.expected_regime_input_dim = config.regime_input_dim
        self.memory_dim = 0  # GRU memory removed
        self.temporal_input_dim = temporal_input_dim
        self.use_cnn_encoder = config.use_cnn_encoder and temporal_input_dim > 0

        self.temporal_encoder: MarketTemporalCNN | None = None
        self.temporal_embedding_dim = 0
        if self.use_cnn_encoder:
            self.temporal_encoder = MarketTemporalCNN(
                input_channels=temporal_input_dim,
                channels=config.cnn_channels,
                kernel_sizes=config.cnn_kernel_sizes,
                temporal_blocks=config.temporal_blocks,
                dropout=config.dropout,
                dilations=getattr(config, "cnn_dilations", None),
            )
            self.temporal_embedding_dim = self.temporal_encoder.output_dim

        # Compute flat-feature width from the sum of branch input dims.
        # (``full_features`` is the concatenation of all branch inputs at the
        # pipeline level — the ``feature_contract.feature_dim`` is the sum.)
        self.flat_feature_dim = max(sum(int(d) for d in branch_input_dims.values()), 1)

        backbone_input_dim = self.flat_feature_dim + self.temporal_embedding_dim
        hidden_dim = config.soma_hidden_dim  # reuse config field for MLP width
        self.backbone = nn.Sequential(
            nn.LayerNorm(backbone_input_dim),
            nn.Linear(backbone_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        head_input_dim = hidden_dim
        self.policy_head = PolicyHead(
            input_dim=head_input_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
            decouple_confidence=True,  # confidence no longer multiplied by soma
            depth=getattr(config, "head_depth", 1),
        )
        self.value_head = ValueHead(input_dim=head_input_dim, hidden_dim=config.head_hidden_dim)
        self.risk_head: RiskHead | None = None
        if getattr(config, "separate_risk_heads", False):
            self.risk_head = RiskHead(head_input_dim)

    def initial_memory_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return an empty memory state (kept for backward compat — GRU removed)."""
        return torch.empty(batch_size, 0, device=device, dtype=dtype)

    def forward(
        self,
        *,
        branch_inputs: dict[str, torch.Tensor],
        regime_input: torch.Tensor,
        memory_state: torch.Tensor | None = None,
        full_features: torch.Tensor | None = None,
        temporal_tensor: torch.Tensor | None = None,
    ) -> ModelOutput:
        # ``branch_inputs``, ``regime_input``, ``memory_state`` are accepted for
        # backward compatibility with the protocol but not used directly — the
        # MLP sees only ``full_features`` (+ optional CNN embedding).
        del memory_state

        if full_features is None:
            # §4.3.4: the old fallback rebuilt the flat vector by
            # concatenating ``branch_inputs`` in branch order, which is a
            # silent permutation of the contract's global (sorted) feature
            # ordering — the model would train/serve on scrambled inputs
            # without any error.  Every real caller passes the
            # contract-ordered ``full_features``; refuse anything else.
            raise ValueError(
                "BrainTradingModel requires full_features (contract-ordered flat "
                "feature vector); reconstructing it from branch_inputs would "
                "silently permute features against the feature contract"
            )
        full_features = ensure_batch_dimension(full_features)

        batch_size = full_features.shape[0]
        if full_features.shape[-1] != self.flat_feature_dim:
            raise ValueError(
                f"Expected full_features dim {self.flat_feature_dim}, got {full_features.shape[-1]}"
            )

        temporal_embedding: torch.Tensor | None = None
        if self.use_cnn_encoder:
            if temporal_tensor is None:
                raise ValueError("BrainTradingModel requires temporal_tensor when use_cnn_encoder=True")
            if temporal_tensor.dim() == 2:
                temporal_tensor = temporal_tensor.unsqueeze(0)
            if temporal_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Temporal tensor batch size {temporal_tensor.shape[0]} does not match flat feature batch {batch_size}"
                )
            temporal_output = self.temporal_encoder(
                temporal_tensor.to(device=full_features.device, dtype=full_features.dtype)
            )
            temporal_embedding = temporal_output.embedding

        if temporal_embedding is not None:
            flat = torch.cat([full_features, temporal_embedding], dim=-1)
        else:
            flat = full_features

        hidden = self.backbone(flat)
        policy_output = self.policy_head(hidden, torch.ones(batch_size, 1, device=hidden.device, dtype=hidden.dtype))
        value = self.value_head(hidden)
        risk_predictions = self.risk_head(hidden) if self.risk_head is not None else None

        ones = torch.ones(batch_size, 1, device=hidden.device, dtype=hidden.dtype)
        return ModelOutput(
            action_logits=policy_output.action_logits,
            confidence=policy_output.confidence,
            target_fraction=policy_output.target_fraction,
            fused_representation=hidden,
            branch_embeddings={},  # no branches
            branch_confidences={},  # no branches
            soma_weights=ones,  # placeholder
            soma_activation=ones,  # placeholder (no damping)
            regime_weights=None,
            value=value,
            memory_state=None,  # GRU removed
            temporal_embedding=temporal_embedding,
            risk_predictions=risk_predictions,
        )
