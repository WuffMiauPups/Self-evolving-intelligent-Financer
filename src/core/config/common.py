from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.exceptions import ConfigurationError


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")
    file_enabled: bool = True
    file_path: str = "logs/seif_runtime.log"


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred: str = "auto"
    # YAML is canonical (§4.6.5); autocast is still guarded off on
    # non-CUDA devices, so True is safe everywhere.
    use_amp: bool = True
    matmul_precision: str = "high"
    torch_compile_enabled: bool = False  # opt-in: requires MSVC cl.exe on Windows or a compiler toolchain; set True once your env is verified


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    windows: dict[str, int] = Field(default_factory=lambda: {"short": 5, "medium": 20, "long": 60})
    volatility_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    momentum_windows: list[int] = Field(default_factory=lambda: [3, 5, 10])
    volume_windows: list[int] = Field(default_factory=lambda: [5, 20])
    # Denominator floor for return/ratio features. 5e-4 (the long-standing
    # YAML value) is canonical (§4.6.5) — the old 1e-6 code default was
    # never effective on any load_config path but silently diverged for
    # code constructing FeatureConfig() directly.
    epsilon: float = 5.0e-4
    # Cross-sectional market context (RESEARCH_ROADMAP Phase 3): SPY/QQQ
    # returns + realized vol, universe breadth, symbol-vs-SPY relative
    # strength, emitted as ``market_*`` features.  Flipping this changes
    # the feature contract (names/dims), so it requires a full retrain
    # (⟲3); checkpoints trained with the other setting refuse to load.
    market_context_enabled: bool = False
    branch_feature_prefixes: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "trend": ["return_", "log_return_", "momentum_", "rsi_", "macd_"],
            "volatility": ["volatility_", "range_", "body_", "upper_wick_", "lower_wick_", "atr_"],
            "volume": ["volume_", "volume_z_", "volume_change_"],
            "regime": ["regime_", "calendar_"],
        }
    )


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_cash: float = 100000.0
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    # Fill timing: a decision made on bar t cannot execute at the very
    # close it just observed. "next_open" (default) fills at bar t+1's
    # open; "same_close" keeps the legacy optimistic fill for comparisons.
    fill_timing: str = "next_open"
    bars_per_year: int = 19656  # 252 trading days * 78 five-min bars
    seconds_per_bar: float = 300.0

    @field_validator("fill_timing")
    @classmethod
    def _validate_fill_timing(cls, value: str) -> str:
        if value not in {"next_open", "same_close"}:
            raise ConfigurationError("backtest.fill_timing must be one of: next_open, same_close")
        return value


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_submit_retries: int = 2
    retry_backoff_sec: float = 0.5
    order_journal_enabled: bool = True
    order_journal_path: str = "runs/order_journal.jsonl"
    order_journal_max_bytes: int = 50_000_000
    order_journal_max_backups: int = 3
    # Durable outcome dataset (self-evolution raw material): realized
    # fills and per-decision events persist as append-only JSONL.
    ledger_persist_enabled: bool = True
    ledger_persist_path: str = "runs/fills.jsonl"
    decision_log_enabled: bool = True
    decision_log_path: str = "runs/decisions.jsonl"
    # Per-decision outcome stream (§4.5.3): one DecisionObservation per
    # executed fill, appended live — the retrain drift trigger's data
    # source (scripts/run_train.py --check-retrain-trigger reads it).
    # Backtests deliberately do NOT write here: simulated outcomes would
    # contaminate live drift detection.
    observation_stream_enabled: bool = True
    observation_stream_path: str = "runs/observations.jsonl"
