from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.config.arbitration import ArbitrationConfig
from src.core.config.broker import BrokerConfig
from src.core.config.common import (
    BacktestConfig,
    DeviceConfig,
    ExecutionConfig,
    FeatureConfig,
    LoggingConfig,
)
from src.core.config.data import DataConfig
from src.core.config.evaluation import EvaluationConfig
from src.core.config.memory import MemoryConfig
from src.core.config.model import ModelConfig
from src.core.config.risk import RiskConfig
from src.core.config.runtime import RuntimeConfig
from src.core.config.training import TrainingConfig
from src.core.exceptions import ConfigurationError


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class AppConfig(BaseModel):
    # extra='forbid' package-wide: a YAML key without a model field is a
    # hard error, never a silent no-op.
    model_config = ConfigDict(extra="forbid")

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
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    arbitration: ArbitrationConfig = Field(default_factory=ArbitrationConfig)

    @model_validator(mode="after")
    def normalize_data_defaults(self) -> "AppConfig":
        if not self.data.symbols:
            self.data.symbols = list(self.runtime.symbols)
        if not self.data.timeframe:
            self.data.timeframe = self.runtime.timeframe
        # Forward horizons and path_label_rules are defined on TrainingConfig and
        # must match on MemoryConfig so the offline memory build labels match
        # what the brain was trained against. Sync from training -> memory unless
        # the user has *explicitly* overridden them on the memory side.
        mem_set = self.memory.model_fields_set
        if "forward_horizons" not in mem_set:
            self.memory.forward_horizons = list(self.training.forward_horizons)
        if "path_label_rules" not in mem_set:
            self.memory.path_label_rules = dict(self.training.path_label_rules)
        return self

    @model_validator(mode="after")
    def validate_staleness_covers_bar(self) -> "AppConfig":
        """``risk.stale_data_sec`` must exceed one bar period.

        A bar's timestamp is its bucket START, so by the time the bucket
        closes the newest bar is a full period old — that is normal data.
        A threshold below the period silently blocks the back of every bar
        on stale_data: the 2026-07-15 paper run lost 56% of its decisions
        that way at 180s on 5Min bars (and default.yaml's 30s would have
        lost ~90%). Fail loudly at load instead (workstream 1F contract).
        """
        # Local import: src.core.config must not import src.data at module
        # scope (src.data.schemas imports back into core).
        from src.data.bar_aggregator import timeframe_delta

        try:
            bar_seconds = int(timeframe_delta(self.runtime.timeframe).total_seconds())
        except ValueError:
            return self  # timeframe itself is validated elsewhere
        if self.risk.stale_data_sec <= bar_seconds:
            raise ConfigurationError(
                f"risk.stale_data_sec={self.risk.stale_data_sec} must exceed one "
                f"runtime.timeframe={self.runtime.timeframe} bar ({bar_seconds}s): a bar is "
                f"already {bar_seconds}s old when its bucket closes, so the runtime would "
                f"reject roughly "
                f"{max(0, round(100 * (1 - self.risk.stale_data_sec / bar_seconds)))}% of "
                f"decisions as stale_data. Use bar period + slack (e.g. {bar_seconds + 120})."
            )
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
