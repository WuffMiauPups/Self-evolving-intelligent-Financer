"""SETAI configuration — split into sub-modules for navigability.

All public names are re-exported here so existing imports continue to work:
    from src.core.config import AppConfig, load_config
"""

from src.core.config.app import AppConfig, load_config
from src.core.config.broker import BrokerConfig
from src.core.config.common import (
    BacktestConfig,
    DeviceConfig,
    ExecutionConfig,
    FeatureConfig,
    LoggingConfig,
)
from src.core.config.data import DataConfig, PipelineConfig
from src.core.config.evaluation import EvaluationConfig, PromotionConfig, WalkForwardConfig
from src.core.config.memory import DecisionFusionConfig, MemoryConfig, TopologyGrowthConfig
from src.core.config.model import ModelConfig
from src.core.config.risk import RiskConfig
from src.core.config.runtime import RuntimeConfig
from src.core.config.training import FastModeConfig, TrainingConfig

__all__ = [
    "AppConfig",
    "BacktestConfig",
    "BrokerConfig",
    "DataConfig",
    "DecisionFusionConfig",
    "DeviceConfig",
    "EvaluationConfig",
    "ExecutionConfig",
    "FastModeConfig",
    "FeatureConfig",
    "load_config",
    "LoggingConfig",
    "MemoryConfig",
    "ModelConfig",
    "PipelineConfig",
    "PromotionConfig",
    "RiskConfig",
    "RuntimeConfig",
    "TopologyGrowthConfig",
    "TrainingConfig",
    "WalkForwardConfig",
]
