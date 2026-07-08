from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    feature_dim_hint: int = 128
    branch_hidden_dim: int = 128
    branch_output_dim: int = 96
    branch_depth: int = 2
    regime_input_dim: int = 4
    regime_hidden_dim: int = 64
    regime_embedding_dim: int = 64
    regime_count: int = 6
    soma_hidden_dim: int = 192
    memory_dim: int = 128
    head_hidden_dim: int = 192
    head_depth: int = 2
    dropout: float = 0.10
    baseline_hidden_dim: int = 192
    use_cnn_encoder: bool = True
    cnn_channels: list[int] = Field(default_factory=lambda: [64, 128])
    cnn_kernel_sizes: list[int] = Field(default_factory=lambda: [3, 5])
    cnn_dilations: list[int] = Field(default_factory=lambda: [1, 2, 4])
    temporal_blocks: int = 3
    checkpoint_path: str | None = None
    # --- Ablation flags (all True = full model, set False to disable) ---
    regime_encoder_enabled: bool = True
    latent_gru_memory_enabled: bool = True
    branch_confidence_gating_enabled: bool = True
    soma_confidence_weighting_enabled: bool = True
    decouple_confidence_from_soma: bool = False
    regime_prototype_init_scale: float = 0.02
    separate_risk_heads: bool = True  # RiskHead emits MFE/MAE for EV gate (Phase 5)
