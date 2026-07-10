from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.exceptions import ConfigurationError
from src.core.time_utils import SESSION_NAMES


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Min"
    warmup_bars: int = 64
    decision_interval_sec: float = 2.0
    # 0 (or any non-positive value) means UNBOUNDED — the paper/live trade loop
    # runs until manually stopped, fatal error, kill switch, or stream exhaustion.
    # Set a positive value only for explicit bounded debug/test runs.
    max_runtime_steps: int = 0
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
    # --- Regime hysteresis / market priors: removed in Great Simplification.
    # Top-level flags kept as inert bools so GUI status widgets render "off"
    # without crashing; sub-fields (thresholds, cooldowns, etc.) stripped.
    regime_hysteresis_enabled: bool = False
    market_priors_enabled: bool = False
    # --- Frozen-brain runtime policy ---
    allow_online_model_updates: bool = False  # model weights NEVER change at runtime
    allow_online_memory_updates: bool = True  # memory bank can learn online
    # Retired symptom patch (was backtest-only, ON by default, breaking
    # backtest/live parity). Labels are cost/horizon-aware now; this single
    # flag is consumed identically by policy_factory and runtime_bootstrap
    # and exists only for explicit experiments.
    force_directional_flip: bool = False
    # --- Alerting ---
    alert_webhook_url: str | None = None
    alert_timeout_sec: float = 5.0
    # --- Log rotation ---
    log_rotation_max_bytes: int = 50_000_000
    log_rotation_max_backups: int = 3
    # --- Stream manager ---
    stream_circuit_breaker_limit: int = 60
    stream_empty_poll_warn_frequency: int = 10
    # --- Brain policy ---
    default_target_fraction: float = 0.05
    # --- Parallelism ---
    max_parallel_workers: int = 8

    @field_validator("allowed_sessions")
    @classmethod
    def validate_allowed_sessions(cls, value: list[str]) -> list[str]:
        unknown = sorted({name for name in value if name.lower() not in SESSION_NAMES})
        if unknown:
            raise ConfigurationError(
                f"Unknown session name(s) {unknown} in runtime.allowed_sessions; "
                f"known sessions: {sorted(SESSION_NAMES)}"
            )
        return value
