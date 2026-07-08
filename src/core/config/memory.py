from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError


class DecisionFusionConfig(BaseModel):
    """Inert stub — the decision-fusion layer was removed in the Great
    Simplification. Kept as a ``config.memory.decision_fusion`` attribute so
    GUI and legacy log-filter code that read ``.enabled`` keep working.

    INTENTIONAL EXCEPTION to the package-wide ``extra='forbid'`` policy:
    old YAMLs and presets still carry the deleted fusion layer's sub-keys
    (conflict_veto_threshold, dampen weights, ...). This stub deliberately
    swallows them so legacy configs keep loading. Do NOT tighten to
    ``forbid`` unless every preset under configs/ has been scrubbed first.
    """

    enabled: bool = False

    model_config = {"extra": "ignore"}


class TopologyGrowthConfig(BaseModel):
    """Inert stub — topology growth removed in the Great Simplification.

    INTENTIONAL EXCEPTION to the package-wide ``extra='forbid'`` policy:
    deliberately swallows the deleted growth layer's legacy sub-keys
    (insertion/split thresholds and weights) so old YAMLs keep loading.
    """

    enabled: bool = False

    model_config = {"extra": "ignore"}


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    build_from_real_checkpoint: bool = False
    latent_dim: int | None = None
    top_k: int = 5
    similarity_metric: str = "cosine"
    min_similarity: float = 0.75
    allow_unknown_state: bool = True
    cluster_method: str = "incremental_prototypes"
    max_items: int = 5000
    rebuild_on_startup: bool = False
    persist_path: str = "artifacts/market_memory.pt"
    forward_horizons: list[int] = Field(default_factory=lambda: [1, 5, 15])
    path_label_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "breakout_return_threshold": 0.003,
            "mean_reversion_return_threshold": 0.0015,
            "flat_return_threshold": 0.00075,
            "time_to_move_threshold": 0.002,
        }
    )
    min_cluster_size: int = 3
    runtime_update_mode: str = "pending_only"
    require_strong_match_for_trade: bool = True
    build_history_bars: int = 320
    min_effective_matches: int = 2
    attention_temperature: float = 1.0
    use_hopfield_retrieval: bool = True
    use_esom_map: bool = True
    dynamic_growth_enabled: bool = True
    novelty_mode: str = "quantile"
    novelty_quantile: float = 0.2
    match_calibration_enabled: bool = True
    min_calibrated_match: float = 0.15
    require_strong_match_for_memory_support: bool = True
    candidate_promotion_mode: str = "support_and_similarity"
    min_candidate_support: int = 5
    max_candidate_age: int = 128
    merge_enabled: bool = True
    merge_similarity_mode: str = "adaptive"
    merge_outcome_consistency_threshold: float = 0.65
    split_enabled: bool = True
    split_variance_trigger: float = 0.75
    prune_enabled: bool = True
    utility_decay: float = 0.995
    min_utility: float = 0.025
    max_units: int = 256
    archive_pruned: bool = True
    calibration_enabled: bool = True
    match_conflict_penalty: float = 0.5
    require_calibrated_strong_match_for_trade_support: bool = True
    topology_neighbor_k: int = 3
    adaptive_topology_k: bool = True
    topology_k_min: int = 2
    topology_k_max: int = 5
    health_report_enabled: bool = True
    symbol_specific_profiles: bool = False
    regime_specific_profiles: bool = False
    # --- Encoder versioning ---
    require_encoder_match: bool = False
    # --- SatSOM-inspired saturation / plasticity ---
    plasticity_floor: float = 0.05
    saturation_enabled: bool = False
    maturity_update_attenuation: float = 0.95
    # --- Outcome-aware updates ---
    require_outcome_alignment_for_strong_update: bool = False
    # --- Conflict tracking ---
    conflict_tracking_enabled: bool = False
    # --- GNG-inspired topology (RESERVED — not yet wired into runtime).
    # The Hebbian edge-aging path (`ESOMMap.activate_edge` / `age_edges`) is
    # implemented but is currently never invoked.  These flags are kept for
    # future activation; today they have no runtime effect and the validators
    # below only enforce range sanity.
    graph_growth_enabled: bool = False  # RESERVED
    graph_max_edge_age: int = 100  # RESERVED
    # --- Pending observation persistence ---
    persist_pending_observations: bool = True
    pending_store_path: str = "artifacts/pending_observations.pt"
    # --- Adaptive homeostatic gating ---
    # When True, replace fixed similarity/calibration/coherence thresholds
    # with self-regulating EMA-based gating that adapts to the bank's own
    # retrieval distribution.  No magic numbers — the system tracks its own
    # similarity baseline and adjusts automatically.
    adaptive_gating: bool = True
    adaptation_rate: float = 0.01  # EMA smoothing constant (lower = slower, more stable)
    # --- Dynamic unit lifecycle ---
    conflict_history_max_size: int = 100
    conflict_history_keep_size: int = 50
    conflict_rate_window: int = 20
    conflict_rate_threshold: float = 0.5
    saturation_support_multiplier: float = 2.0
    saturation_min_support_floor: int = 6
    quarantine_conflict_threshold: float = 0.6
    min_outcome_samples: int = 3
    direction_alignment_threshold: float = 0.5
    # --- Retrieval thresholds ---
    adaptive_coherence_floor: float = 0.1
    adaptive_weak_quality_threshold: float = 0.3
    cluster_top_k_cap: int = 3
    size_averaging_weight: float = 0.5
    weak_support_size_cap: float = 0.25
    legacy_coherence_threshold: float = 0.35
    # --- Calibration ---
    calibration_num_buckets: int = 5
    calibration_graded_conflict_mid: float = 0.5
    # --- ESOM ---
    esom_neighbor_agreement_threshold: float = 0.5
    esom_regime_labeling: bool = False
    regime_strong_threshold: float = 0.010
    regime_trend_threshold: float = 0.002
    regime_volatility_threshold: float = 0.025
    # --- Async memory refresh ---
    # INERT post-Great-Simplification. The minimal memory bank has no
    # prototype clustering, ESOM map, dynamic-unit maintenance or
    # calibration profiles to refresh, so there is nothing a background
    # worker would do. Field is retained for YAML backward-compat only.
    async_memory_refresh: bool = True
    # --- RAN merge (online prototype blending for near-duplicate items) ---
    ran_merge_enabled: bool = False
    ran_merge_alpha: float = 0.3
    # --- LOO-CV temperature auto-tuning ---
    auto_tune_temperature: bool = False
    temperature_candidates: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    )
    # --- Modern Hopfield Network (Ramsauer et al. 2020) ---
    modern_hopfield_enabled: bool = False
    modern_hopfield_iterations: int = 3
    # --- Deep Hopfield learnable projections ---
    deep_projections_enabled: bool = False
    deep_projections_hidden_dim: int = 64
    deep_projections_train_epochs: int = 100
    deep_projections_initial_beta: float = 8.0
    deep_projections_mse_weight: float = 0.7
    deep_projections_lr: float = 1e-3
    # --- Multi-timescale retrieval ---
    multi_timescale_retrieval: bool = False
    retrieval_temperature_multipliers: list[float] = Field(
        default_factory=lambda: [2.0, 1.0, 0.5]
    )
    retrieval_temperature_weights: list[float] = Field(
        default_factory=lambda: [0.5, 0.3, 0.2]
    )
    # --- Health tracker thresholds ---
    retrieval_health_warmup: int = 100
    outcome_similarity_adjustment: float = -0.1
    fusion_cold_start_veto: float = 0.15
    fusion_cold_start_support: float = 0.65
    fusion_health_warmup: int = 50
    veto_profit_threshold: float = 0.01
    veto_relax_factor: float = 0.9
    veto_tighten_factor: float = 1.1
    veto_loss_threshold: float = -0.005
    topology_health_warmup: int = 20
    # --- Counterfactual self-calibration ---
    counterfactual_calibration_enabled: bool = False
    # --- Value-aware decay / outlier filtering ---
    value_aware_decay: bool = False
    value_protection_threshold: float = 0.02
    value_protection_cap: float = 0.5
    outlier_filter_enabled: bool = False
    outlier_sigma: float = 3.0
    # --- Decision fusion layer ---
    decision_fusion: DecisionFusionConfig = Field(default_factory=DecisionFusionConfig)
    # --- Topology growth ---
    topology_growth: TopologyGrowthConfig = Field(default_factory=TopologyGrowthConfig)

    @model_validator(mode="after")
    def validate_memory_config(self) -> "MemoryConfig":
        if self.top_k <= 0:
            raise ConfigurationError("memory.top_k must be positive")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigurationError("memory.min_similarity must be within [0, 1]")
        if self.max_items <= 0:
            raise ConfigurationError("memory.max_items must be positive")
        if not self.forward_horizons or any(horizon <= 0 for horizon in self.forward_horizons):
            raise ConfigurationError("memory.forward_horizons must contain positive integers")
        if self.min_cluster_size <= 0:
            raise ConfigurationError("memory.min_cluster_size must be positive")
        if self.min_effective_matches <= 0:
            raise ConfigurationError("memory.min_effective_matches must be positive")
        if self.runtime_update_mode not in {"disabled", "pending_only", "append_when_ready"}:
            raise ConfigurationError(
                "memory.runtime_update_mode must be one of: disabled, pending_only, append_when_ready"
            )
        if self.cluster_method not in {"incremental_prototypes"}:
            raise ConfigurationError("memory.cluster_method currently supports only incremental_prototypes")
        if self.similarity_metric not in {"cosine", "euclidean", "attention"}:
            raise ConfigurationError("memory.similarity_metric must be one of: cosine, euclidean, attention")
        if self.attention_temperature <= 0.0:
            raise ConfigurationError("memory.attention_temperature must be positive")
        if self.min_calibrated_match < 0.0 or self.min_calibrated_match > 1.0:
            raise ConfigurationError("memory.min_calibrated_match must be within [0, 1]")
        if self.novelty_mode not in {"quantile"}:
            raise ConfigurationError("memory.novelty_mode currently supports only quantile")
        if not 0.0 < self.novelty_quantile < 1.0:
            raise ConfigurationError("memory.novelty_quantile must be within (0, 1)")
        if self.candidate_promotion_mode not in {"support_and_similarity", "support_only"}:
            raise ConfigurationError(
                "memory.candidate_promotion_mode must be one of: support_and_similarity, support_only"
            )
        if self.min_candidate_support <= 0:
            raise ConfigurationError("memory.min_candidate_support must be positive")
        if self.max_candidate_age <= 0:
            raise ConfigurationError("memory.max_candidate_age must be positive")
        if self.merge_similarity_mode not in {"adaptive"}:
            raise ConfigurationError("memory.merge_similarity_mode currently supports only adaptive")
        if not 0.0 <= self.merge_outcome_consistency_threshold <= 1.0:
            raise ConfigurationError("memory.merge_outcome_consistency_threshold must be within [0, 1]")
        if self.split_variance_trigger <= 0.0:
            raise ConfigurationError("memory.split_variance_trigger must be positive")
        if not 0.0 < self.utility_decay <= 1.0:
            raise ConfigurationError("memory.utility_decay must be within (0, 1]")
        if self.min_utility < 0.0:
            raise ConfigurationError("memory.min_utility must be non-negative")
        if self.max_units <= 0:
            raise ConfigurationError("memory.max_units must be positive")
        if self.match_conflict_penalty < 0.0:
            raise ConfigurationError("memory.match_conflict_penalty must be non-negative")
        if self.topology_neighbor_k <= 0:
            raise ConfigurationError("memory.topology_neighbor_k must be positive")
        if not 0.0 < self.maturity_update_attenuation < 1.0:
            raise ConfigurationError("memory.maturity_update_attenuation must be within (0, 1)")
        if self.graph_max_edge_age <= 0:
            raise ConfigurationError("memory.graph_max_edge_age must be positive")
        if not 0.0 < self.ran_merge_alpha <= 1.0:
            raise ConfigurationError("memory.ran_merge_alpha must be within (0, 1]")
        if not self.temperature_candidates:
            raise ConfigurationError("memory.temperature_candidates must be non-empty")
        if self.outlier_sigma <= 0.0:
            raise ConfigurationError("memory.outlier_sigma must be positive")
        return self
