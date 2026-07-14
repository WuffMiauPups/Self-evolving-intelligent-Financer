from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError


class FastModeConfig(BaseModel):
    """Explicit fast experiment mode: reduced data/epochs for rapid research iteration.

    Fast mode is:
    - explicitly opt-in (enabled=False by default)
    - clearly logged and recorded in run summaries
    - deterministic when sample_seed is fixed
    - NOT suitable for final production checkpoints

    Controls:
        enabled: must be True for any fast-mode behaviour to activate
        fast_months: restrict data to last N months (overrides data.start_date)
        fast_start_date: explicit start date override (ISO string, e.g. "2025-01-01")
        fast_end_date: explicit end date override (ISO string)
        symbols: train on this symbol subset only (empty = use data.symbols)
        max_train_samples: cap training samples (deterministic with sample_seed)
        max_val_samples: cap validation samples
        max_test_samples: cap test samples
        sample_seed: RNG seed for deterministic subsampling
        max_epochs: cap training epochs (min of this and training.epochs)
        disable_future_return: zero future-return loss weight
        disable_risk_path: zero risk-path loss weight
        model_profile: "default" (unchanged arch) or "small" (reduced dims)
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Date range
    fast_months: int | None = 6
    fast_start_date: str | None = None
    fast_end_date: str | None = None
    # Symbol subset
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    # Sample caps
    max_train_samples: int | None = 50000
    max_val_samples: int | None = 15000
    max_test_samples: int | None = 10000
    sample_seed: int = 42
    # Epoch cap
    max_epochs: int | None = None
    # Target/loss simplification
    disable_future_return: bool = False
    disable_risk_path: bool = False
    # Model profile
    model_profile: str = "default"

    @model_validator(mode="after")
    def validate_fast_mode_config(self) -> "FastModeConfig":
        if self.fast_months is not None and self.fast_months <= 0:
            raise ConfigurationError("training.fast_mode.fast_months must be positive")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ConfigurationError("training.fast_mode.max_train_samples must be positive")
        if self.max_val_samples is not None and self.max_val_samples <= 0:
            raise ConfigurationError("training.fast_mode.max_val_samples must be positive")
        if self.max_test_samples is not None and self.max_test_samples <= 0:
            raise ConfigurationError("training.fast_mode.max_test_samples must be positive")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ConfigurationError("training.fast_mode.max_epochs must be positive")
        if self.model_profile not in {"default", "small"}:
            raise ConfigurationError("training.fast_mode.model_profile must be one of: default, small")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_real_historical_data: bool = False
    # Config-driven training seed (§4.7): the seed was hardcoded to 7 in
    # build_training_components, which silently zeroed seed variance for
    # any sweep routed through it — every "different" run trained
    # identically. YAML is canonical; 7 preserves historical behavior.
    seed: int = 7
    batch_size: int = 384
    epochs: int = 5
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    # --- Learning rate scheduling ---
    scheduler_type: str = "cosine"  # "cosine", "plateau", or "none"
    scheduler_warmup_epochs: int = 0
    scheduler_patience: int = 2  # for ReduceLROnPlateau
    scheduler_factor: float = 0.5  # for ReduceLROnPlateau
    scheduler_min_lr: float = 1.0e-6
    scheduler_t_max: int | None = None  # for CosineAnnealingLR; defaults to epochs if None
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    label_epsilon: float = 5.0e-4
    transaction_cost_bps: float = 6.0
    # --- Adaptive label epsilon (per-symbol volatility scaling) ---
    adaptive_label_epsilon: bool = True
    adaptive_epsilon_multiplier: float = 1.0
    adaptive_epsilon_min: float = 1.0e-4
    adaptive_epsilon_max: float = 5.0e-3
    checkpoint_dir: str = "checkpoints"
    sequence_length: int = 48
    use_amp: bool = True
    forward_horizons: list[int] = Field(default_factory=lambda: [1, 3, 6])
    # --- State + future-path target design ---
    primary_direction_horizon: int | None = None  # defaults to max(forward_horizons) if None
    auxiliary_horizons: list[int] = Field(default_factory=list)  # additional horizons for future-path targets
    enable_future_return_targets: bool = True  # per-horizon return regression
    enable_risk_path_targets: bool = True  # MFE/MAE/volatility/drawdown regression
    path_label_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "breakout_return_threshold": 0.003,
            "mean_reversion_return_threshold": 0.0015,
            "flat_return_threshold": 0.00075,
            "time_to_move_threshold": 0.002,
        }
    )
    # Class-imbalance policy: exactly ONE primary correction (balanced
    # class weights).  The sampler and focal loss stack multiplicatively
    # with the weights when enabled together, over-correcting the prior
    # 2-3x and pushing predicted distributions far from base rates —
    # they are opt-in experiment knobs now, not defaults.
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
    # --- Diagnostic thresholds (class_balance / metrics) ---
    label_epsilon_min_bars: int = 20
    near_flat_multiplier: float = 1.25
    epsilon_reduction_factor: float = 0.5
    flat_dominance_threshold: float = 0.80
    metrics_flat_dominance_threshold: float = 0.75
    multitask_enabled: bool = True
    contrastive_enabled: bool = False  # RESERVED
    contrastive_margin: float = 0.1
    classification_loss: str = "cross_entropy"
    focal_gamma: float = 2.0
    adaptive_focal_gamma: bool = False
    focal_gamma_base: float = 1.5
    focal_gamma_boost: float = 2.0
    label_smoothing: float = 0.0
    # Best-epoch selection / early-stopping metric.  Raw multi-task val
    # loss rewards flat-hedging; balanced accuracy is the deployment-
    # aligned default (it is what the scorecard grades).
    model_selection_metric: str = "val_balanced_accuracy"
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "classification": 1.0,
            "regression": 0.35,
            "latent": 0.10,
            "path_label": 0.10,
            "future_return": 0.20,
            "risk_path": 0.15,
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
    disable_regression: bool = False
    disable_latent: bool = False
    save_memory_after_training: bool = False
    memory_build_split: str = "train"
    runs_dir: str = "runs"
    # --- Retrain trigger (self-evolution loop, MASTER_PLAN §4.5.3) ---
    # Per-metric drift thresholds: a 0.5 Sharpe drop and a 5pt win-rate
    # drop are comparable alarms; a shared threshold conflates noise
    # (Sharpe) with catastrophe (win rate).
    retrain_triggers: dict[str, float] = Field(
        default_factory=lambda: {
            "sharpe_drop": 0.5,
            "win_rate_drop": 0.05,
            "min_observations": 200,
        }
    )
    # --- Anchored negatives + adaptive curriculum (§6.5, roadmap §4.8-9) ---
    # Defaults are OFF/neutral: enabling either is an explicit experiment.
    curriculum_enabled: bool = False
    curriculum_start_quantile: float = 0.5  # epoch-0 inclusion quantile, in (0, 1]
    curriculum_full_epoch: int = 3  # epoch at which the ramp reaches 1.0 (all samples)
    anchored_negatives_enabled: bool = False
    anchored_negatives_path: str = "runs/anchored_negatives.jsonl"
    anchored_negatives_max: int = 5000
    anchored_noise_buffer_bps: float = 5.0  # added to transaction_cost_bps for the mistake margin
    # --- Forgetting-safe retraining (§4.5.6): anchor to the incumbent ---
    anchor_checkpoint_path: str | None = None
    distill_lambda_kd: float = 0.0  # KL(incumbent || student) weight; 0 = off
    proximal_lambda_l2: float = 0.0  # L2(student - anchor) weight; 0 = off
    # --- RL fine-tuning (reserved scaffold) ---
    rl_finetune_enabled: bool = False
    rl_finetune_episodes: int = 100
    rl_replay_buffer_capacity: int = 10000
    rl_gamma: float = 0.99
    rl_batch_size: int = 32
    rl_learning_rate: float = 1.0e-4
    # --- Self-supervised pretraining (reserved scaffold) ---
    pretrain_enabled: bool = False
    pretrain_epochs: int = 5
    pretrain_learning_rate: float = 5.0e-4
    # --- Training augmentation (noise on features; temporal jitter reserved) ---
    augmentation_enabled: bool = True
    noise_std: float = 0.01
    temporal_jitter_prob: float = 0.0  # reserved; 0.0 = off until batched jitter lands
    # --- Early stopping ---
    early_stopping_patience: int = 2
    restore_best_model: bool = True
    # --- Platt calibration ---
    platt_calibration_enabled: bool = True
    # --- Fast experiment mode ---
    fast_mode: FastModeConfig = Field(default_factory=FastModeConfig)

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
        if self.model_selection_metric not in {"val_loss", "val_balanced_accuracy", "val_macro_f1"}:
            raise ConfigurationError(
                "training.model_selection_metric must be one of: val_loss, val_balanced_accuracy, val_macro_f1"
            )
        if self.focal_gamma < 0.0:
            raise ConfigurationError("training.focal_gamma must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigurationError("training.label_smoothing must be within [0, 1)")
        # Allow both old-style {classification, regression, latent} and
        # new-style {classification, regression, latent, path_label, future_return, risk_path}.
        # Migrate old-style to new-style by adding defaults for missing keys.
        # path_label dropped post-Great-Simplification: head was killed (val acc
        # below majority baseline + 7.7pt train→val overfit). Old configs with
        # ``path_label`` set are silently ignored (popped before further validation).
        self.loss_weights.pop("path_label", None)
        _required_base = {"classification", "regression", "latent"}
        _new_keys = {"future_return", "risk_path"}
        all_keys = set(self.loss_weights)
        if all_keys == _required_base:
            # Backwards-compat: old 3-key config → add new keys with defaults
            self.loss_weights.setdefault("future_return", 0.20)
            self.loss_weights.setdefault("risk_path", 0.15)
        elif all_keys != (_required_base | _new_keys):
            raise ConfigurationError(
                "training.loss_weights must define: classification, regression, latent, "
                "future_return, risk_path"
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
        allowed_trigger_keys = {"sharpe_drop", "win_rate_drop", "min_observations"}
        unknown_triggers = set(self.retrain_triggers) - allowed_trigger_keys
        if unknown_triggers:
            raise ConfigurationError(
                f"training.retrain_triggers has unknown keys: {sorted(unknown_triggers)} "
                f"(allowed: {sorted(allowed_trigger_keys)})"
            )
        if any(value < 0.0 for value in self.retrain_triggers.values()):
            raise ConfigurationError("training.retrain_triggers values must be non-negative")
        if not 0.0 < self.curriculum_start_quantile <= 1.0:
            raise ConfigurationError("training.curriculum_start_quantile must be within (0, 1]")
        if self.curriculum_full_epoch < 1:
            raise ConfigurationError("training.curriculum_full_epoch must be >= 1")
        if self.anchored_negatives_max <= 0:
            raise ConfigurationError("training.anchored_negatives_max must be positive")
        if self.anchored_noise_buffer_bps < 0.0:
            raise ConfigurationError("training.anchored_noise_buffer_bps must be non-negative")
        if self.distill_lambda_kd < 0.0:
            raise ConfigurationError("training.distill_lambda_kd must be non-negative")
        if self.proximal_lambda_l2 < 0.0:
            raise ConfigurationError("training.proximal_lambda_l2 must be non-negative")
        return self

    @property
    def effective_primary_horizon(self) -> int:
        """Primary direction/path horizon: explicit setting or max(forward_horizons)."""
        if self.primary_direction_horizon is not None:
            return self.primary_direction_horizon
        return max(self.forward_horizons)

    @property
    def all_horizons(self) -> list[int]:
        """All unique horizons (forward + auxiliary), sorted."""
        combined = set(self.forward_horizons) | set(self.auxiliary_horizons)
        return sorted(combined)
