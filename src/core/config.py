from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")
    file_enabled: bool = True
    file_path: str = "logs/seif_runtime.log"


class DeviceConfig(BaseModel):
    preferred: str = "auto"
    use_amp: bool = False
    matmul_precision: str = "high"


class BrokerConfig(BaseModel):
    provider: str = "alpaca"
    paper: bool = True
    live_enabled: bool = False
    confirm_live: bool = False
    require_account_match: bool = True
    base_url: str | None = None
    data_feed: str = "iex"
    polling_interval_sec: float = 2.0
    mock_data_on_missing_credentials: bool = True

    @model_validator(mode="after")
    def validate_live_mode(self) -> "BrokerConfig":
        if self.live_enabled and self.paper:
            raise ConfigurationError("paper=true and live_enabled=true cannot be enabled together")
        if not self.paper and not self.live_enabled:
            raise ConfigurationError("paper=false requires live_enabled=true")
        return self


class DataConfig(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    timeframe: str = "1Min"
    start_date: str | date | datetime | None = None
    end_date: str | date | datetime | None = None
    cache_dir: str = "artifacts/historical"
    force_refresh: bool = False
    max_missing_gap_handling: str = "warn"
    timezone_normalization: str = "UTC"

    @model_validator(mode="after")
    def validate_data_config(self) -> "DataConfig":
        if self.max_missing_gap_handling not in {"ignore", "warn", "error"}:
            raise ConfigurationError("data.max_missing_gap_handling must be one of: ignore, warn, error")
        if not self.timezone_normalization:
            raise ConfigurationError("data.timezone_normalization must not be empty")
        return self


class RuntimeConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Min"
    warmup_bars: int = 64
    decision_interval_sec: float = 2.0
    max_runtime_steps: int = 50
    state_reset_gap_sec: int | None = 300
    require_startup_reconciliation: bool = True
    block_on_reconciliation_mismatch: bool = True
    allowed_sessions: list[str] = Field(default_factory=lambda: ["regular"])
    allow_prepost: bool = False
    max_data_staleness_sec: int = 30
    block_on_backwards_time: bool = True
    block_on_data_gap: bool = True
    block_on_symbol_ineligible: bool = False
    require_market_open: bool = True


class FeatureConfig(BaseModel):
    windows: dict[str, int] = Field(default_factory=lambda: {"short": 5, "medium": 20, "long": 60})
    volatility_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    momentum_windows: list[int] = Field(default_factory=lambda: [3, 5, 10])
    volume_windows: list[int] = Field(default_factory=lambda: [5, 20])
    epsilon: float = 1.0e-6
    branch_feature_prefixes: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "trend": ["return_", "log_return_", "momentum_"],
            "volatility": ["volatility_", "range_", "body_", "upper_wick_", "lower_wick_"],
            "volume": ["volume_", "volume_z_", "volume_change_"],
            "regime": ["regime_"],
        }
    )


class ModelConfig(BaseModel):
    feature_dim_hint: int = 32
    branch_hidden_dim: int = 32
    branch_output_dim: int = 24
    branch_depth: int = 2
    regime_input_dim: int = 4
    regime_hidden_dim: int = 16
    regime_embedding_dim: int = 12
    regime_count: int = 4
    soma_hidden_dim: int = 32
    memory_dim: int = 24
    head_hidden_dim: int = 32
    dropout: float = 0.05
    baseline_hidden_dim: int = 48
    use_cnn_encoder: bool = True
    cnn_channels: list[int] = Field(default_factory=lambda: [16, 32])
    cnn_kernel_sizes: list[int] = Field(default_factory=lambda: [3, 5])
    temporal_blocks: int = 3
    checkpoint_path: str | None = None


class RiskConfig(BaseModel):
    max_daily_loss: float = 500.0
    max_position_notional: float = 5000.0
    max_position_per_symbol: int = 1
    max_concurrent_positions: int = 3
    cooldown_sec: int = 60
    stale_data_sec: int = 30
    max_spread_bps: float = 15.0
    duplicate_order_window_sec: int = 15
    min_trade_confidence: float = 0.55
    min_trade_fraction: float = 0.02
    kill_switch: bool = False
    allowed_sessions: list[str] = Field(default_factory=list)


class BacktestConfig(BaseModel):
    initial_cash: float = 100000.0
    fee_bps: float = 1.0
    slippage_bps: float = 2.0


class TrainingConfig(BaseModel):
    use_real_historical_data: bool = False
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    label_epsilon: float = 5.0e-4
    checkpoint_dir: str = "checkpoints"
    supervised_horizon: int = 1
    sequence_length: int = 64
    use_amp: bool = False
    forward_horizons: list[int] = Field(default_factory=lambda: [1, 5, 15])
    path_label_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "breakout_return_threshold": 0.003,
            "mean_reversion_return_threshold": 0.0015,
            "flat_return_threshold": 0.00075,
            "time_to_move_threshold": 0.002,
        }
    )
    class_weight_mode: str = "balanced"
    class_weight_cap: float | None = 3.0
    use_weighted_sampler: bool = False
    weighted_sampler_mode: str = "inverse_frequency"
    minority_class_floor_warning: float = 0.1
    report_target_stats: bool = True
    report_logit_stats: bool = True
    report_prediction_distribution: bool = True
    report_per_class_metrics: bool = True
    inspect_target_boundaries: bool = True
    multitask_enabled: bool = True
    contrastive_enabled: bool = False
    classification_loss: str = "cross_entropy"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "classification": 1.0,
            "regression": 0.35,
            "latent": 0.10,
        }
    )
    collapse_warning_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "long_prediction_floor": 0.08,
            "minority_recall_floor": 0.15,
        }
    )
    regression_targets: list[str] = Field(
        default_factory=lambda: ["primary_return", "drawdown", "stability"]
    )
    save_memory_after_training: bool = False
    memory_build_split: str = "train"

    @model_validator(mode="after")
    def validate_training_config(self) -> "TrainingConfig":
        if self.batch_size <= 0:
            raise ConfigurationError("training.batch_size must be positive")
        split_total = self.train_split + self.val_split + self.test_split
        if abs(split_total - 1.0) > 1.0e-6:
            raise ConfigurationError("training.train_split + val_split + test_split must equal 1.0")
        if not self.forward_horizons or any(horizon <= 0 for horizon in self.forward_horizons):
            raise ConfigurationError("training.forward_horizons must contain positive integers")
        if self.class_weight_mode not in {"none", "balanced", "effective_number"}:
            raise ConfigurationError("training.class_weight_mode must be one of: none, balanced, effective_number")
        if self.class_weight_cap is not None and self.class_weight_cap <= 0.0:
            raise ConfigurationError("training.class_weight_cap must be positive when provided")
        if self.weighted_sampler_mode not in {"inverse_frequency", "effective_number"}:
            raise ConfigurationError(
                "training.weighted_sampler_mode must be one of: inverse_frequency, effective_number"
            )
        if not 0.0 <= self.minority_class_floor_warning <= 1.0:
            raise ConfigurationError("training.minority_class_floor_warning must be within [0, 1]")
        if self.classification_loss not in {"cross_entropy", "focal"}:
            raise ConfigurationError("training.classification_loss must be one of: cross_entropy, focal")
        if self.focal_gamma < 0.0:
            raise ConfigurationError("training.focal_gamma must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigurationError("training.label_smoothing must be within [0, 1)")
        required_loss_keys = {"classification", "regression", "latent"}
        if set(self.loss_weights) != required_loss_keys:
            raise ConfigurationError(
                "training.loss_weights must define exactly: classification, regression, latent"
            )
        if any(value < 0.0 for value in self.loss_weights.values()):
            raise ConfigurationError("training.loss_weights values must be non-negative")
        required_warning_keys = {"long_prediction_floor", "minority_recall_floor"}
        if set(self.collapse_warning_thresholds) != required_warning_keys:
            raise ConfigurationError(
                "training.collapse_warning_thresholds must define exactly: long_prediction_floor, minority_recall_floor"
            )
        if not 0.0 <= self.collapse_warning_thresholds["long_prediction_floor"] <= 1.0:
            raise ConfigurationError("training.collapse_warning_thresholds.long_prediction_floor must be within [0, 1]")
        if not 0.0 <= self.collapse_warning_thresholds["minority_recall_floor"] <= 1.0:
            raise ConfigurationError("training.collapse_warning_thresholds.minority_recall_floor must be within [0, 1]")
        if self.memory_build_split not in {"train", "val", "test", "all"}:
            raise ConfigurationError("training.memory_build_split must be one of: train, val, test, all")
        return self


class ExecutionConfig(BaseModel):
    max_submit_retries: int = 2
    retry_backoff_sec: float = 0.5


class MemoryConfig(BaseModel):
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
    min_candidate_support: int = 3
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
    health_report_enabled: bool = True
    symbol_specific_profiles: bool = False
    regime_specific_profiles: bool = False
    # --- Encoder versioning ---
    require_encoder_match: bool = False
    # --- SatSOM-inspired saturation / plasticity ---
    saturation_enabled: bool = False
    maturity_update_attenuation: float = 0.95
    # --- Outcome-aware updates ---
    require_outcome_alignment_for_strong_update: bool = False
    # --- Conflict tracking ---
    conflict_tracking_enabled: bool = False
    # --- GNG-inspired topology ---
    graph_growth_enabled: bool = False
    graph_max_edge_age: int = 100

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
        return self


class AppConfig(BaseModel):
    app_name: str = "seif_trading_brain"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @model_validator(mode="after")
    def normalize_data_defaults(self) -> "AppConfig":
        if not self.data.symbols:
            self.data.symbols = list(self.runtime.symbols)
        if not self.data.timeframe:
            self.data.timeframe = self.runtime.timeframe
        return self


def load_config(*paths: str | Path) -> AppConfig:
    load_dotenv()
    merged: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        merged = _deep_merge(merged, payload)
    return AppConfig.model_validate(merged)
