from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: str = "monthly"
    max_fetch_workers: int = 4
    retry_count: int = 3
    retry_backoff_sec: float = 1.0
    validation_strictness: str = "warn"
    max_prep_workers: int = 4
    manifest_path: str = "artifacts/historical/manifest.json"

    @model_validator(mode="after")
    def validate_pipeline_config(self) -> "PipelineConfig":
        if self.chunk_size not in {"monthly", "quarterly", "yearly"}:
            raise ConfigurationError("pipeline.chunk_size must be one of: monthly, quarterly, yearly")
        if self.max_fetch_workers <= 0:
            raise ConfigurationError("pipeline.max_fetch_workers must be positive")
        if self.retry_count < 0:
            raise ConfigurationError("pipeline.retry_count must be non-negative")
        if self.retry_backoff_sec < 0.0:
            raise ConfigurationError("pipeline.retry_backoff_sec must be non-negative")
        if self.validation_strictness not in {"ignore", "warn", "error"}:
            raise ConfigurationError("pipeline.validation_strictness must be one of: ignore, warn, error")
        if self.max_prep_workers <= 0:
            raise ConfigurationError("pipeline.max_prep_workers must be positive")
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(default_factory=list)
    timeframe: str = "5Min"
    start_date: str | date | datetime | None = date(2022, 1, 1)
    end_date: str | date | datetime | None = None
    cache_dir: str = "artifacts/historical"
    # Corporate-action adjustment for historical bar fetches.  "all"
    # (split + dividend) is the only value that keeps splits from showing
    # up as fake crashes in training data.
    bar_adjustment: str = "all"
    force_refresh: bool = False
    max_missing_gap_handling: str = "warn"
    timezone_normalization: str = "UTC"
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    bars_per_trading_day: int = 390
    min_lookback_days_minute: int = 3
    min_lookback_days_hourly: int = 7
    min_lookback_multiplier_daily: int = 3
    min_lookback_days_daily: int = 30

    @model_validator(mode="after")
    def validate_data_config(self) -> "DataConfig":
        if self.max_missing_gap_handling not in {"ignore", "warn", "error"}:
            raise ConfigurationError("data.max_missing_gap_handling must be one of: ignore, warn, error")
        if self.bar_adjustment not in {"raw", "split", "dividend", "all"}:
            raise ConfigurationError("data.bar_adjustment must be one of: raw, split, dividend, all")
        if not self.timezone_normalization:
            raise ConfigurationError("data.timezone_normalization must not be empty")
        return self
