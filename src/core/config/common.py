from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    torch_compile_enabled: bool = False  # opt-in: requires MSVC cl.exe on Windows or a compiler toolchain; set True once your env is verified


class FeatureConfig(BaseModel):
    windows: dict[str, int] = Field(default_factory=lambda: {"short": 5, "medium": 20, "long": 60})
    volatility_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    momentum_windows: list[int] = Field(default_factory=lambda: [3, 5, 10])
    volume_windows: list[int] = Field(default_factory=lambda: [5, 20])
    epsilon: float = 1.0e-6
    branch_feature_prefixes: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "trend": ["return_", "log_return_", "momentum_", "rsi_", "macd_"],
            "volatility": ["volatility_", "range_", "body_", "upper_wick_", "lower_wick_", "atr_"],
            "volume": ["volume_", "volume_z_", "volume_change_"],
            "regime": ["regime_", "calendar_"],
        }
    )


class BacktestConfig(BaseModel):
    initial_cash: float = 100000.0
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    force_directional_flip_enabled: bool = True
    bars_per_year: int = 19656  # 252 trading days * 78 five-min bars
    seconds_per_bar: float = 300.0


class ExecutionConfig(BaseModel):
    max_submit_retries: int = 2
    retry_backoff_sec: float = 0.5
    order_journal_enabled: bool = True
    order_journal_path: str = "runs/order_journal.jsonl"
    order_journal_max_bytes: int = 50_000_000
    order_journal_max_backups: int = 3
