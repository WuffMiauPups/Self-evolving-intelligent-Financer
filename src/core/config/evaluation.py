from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError


class WalkForwardConfig(BaseModel):
    """Walk-forward cross-validation configuration.

    Defines how data is split into sequential train/val/test folds
    for out-of-sample evaluation. All durations are in bars.
    """

    model_config = ConfigDict(extra="forbid")

    train_bars: int = 5000
    val_bars: int = 1000
    test_bars: int = 1000
    step_bars: int = 1000
    window_type: str = "rolling"  # "rolling" or "expanding"
    min_train_bars: int = 2000
    # reset_model_per_fold was DELETED (2026-07-14 audit): it was defined
    # but read nowhere — per-fold retraining does not exist (§4.4.3 [L]).
    # Every fold already gets a freshly built policy from the checkpoint;
    # extra="forbid" makes any config still setting the old key fail loudly.
    reset_memory_per_fold: bool = True

    @model_validator(mode="after")
    def validate_walk_forward(self) -> "WalkForwardConfig":
        if self.window_type not in {"rolling", "expanding"}:
            raise ConfigurationError("walk_forward.window_type must be rolling or expanding")
        if self.train_bars <= 0:
            raise ConfigurationError("walk_forward.train_bars must be positive")
        if self.val_bars < 0:
            raise ConfigurationError("walk_forward.val_bars must be non-negative")
        if self.test_bars <= 0:
            raise ConfigurationError("walk_forward.test_bars must be positive")
        if self.step_bars <= 0:
            raise ConfigurationError("walk_forward.step_bars must be positive")
        if self.min_train_bars <= 0:
            raise ConfigurationError("walk_forward.min_train_bars must be positive")
        return self


class PromotionConfig(BaseModel):
    """Scorecard-judged promotion criteria (MASTER_PLAN §4.5.5/§4.5.7).

    A candidate must beat the incumbent's benchmark row by more than the
    configured deltas AND beat buy-and-hold, and may not degrade any
    frozen held-out slice by more than ``backward_transfer_floor``.
    """

    model_config = ConfigDict(extra="forbid")

    min_balanced_accuracy_delta: float = 0.0  # candidate must exceed incumbent by more than this
    min_expectancy_delta: float = 0.0
    min_excess_vs_buy_hold: float = 0.0  # candidate return must exceed buy&hold by more than this
    backward_transfer_floor: float = 0.05  # max per-slice balanced-accuracy degradation
    # §4.5.7 slice producer: the shared OOS window is cut into this many
    # chronological regime slices, each graded for candidate-vs-incumbent
    # balanced accuracy. 0 disables slice production (floor gate then idle).
    backward_transfer_slices: int = 4
    # Slices with fewer decisions than this (either checkpoint) are skipped
    # — a handful of trades cannot evidence forgetting.
    backward_transfer_min_decisions: int = 50

    @model_validator(mode="after")
    def validate_promotion(self) -> "PromotionConfig":
        if self.backward_transfer_floor < 0.0:
            raise ConfigurationError("evaluation.promotion.backward_transfer_floor must be non-negative")
        if self.backward_transfer_slices < 0:
            raise ConfigurationError("evaluation.promotion.backward_transfer_slices must be non-negative")
        if self.backward_transfer_min_decisions < 1:
            raise ConfigurationError("evaluation.promotion.backward_transfer_min_decisions must be positive")
        return self


class EvaluationConfig(BaseModel):
    """Top-level evaluation / experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    seed: int = 42
    output_dir: str = "experiments"
