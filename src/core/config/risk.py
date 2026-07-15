from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.exceptions import ConfigurationError
from src.core.time_utils import SESSION_NAMES


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Limits are YAML-aligned (§4.6.5): default.yaml is canonical and these
    # defaults must match it so code building RiskConfig() directly trades
    # under the same limits the deployed config states.
    max_daily_loss: float = 350.0
    max_position_notional: float = 12000.0
    max_position_per_symbol: int = 3
    max_concurrent_positions: int = 15
    cooldown_sec: int = 60
    # Must exceed the bar period; see AppConfig.validate_staleness_covers_bar.
    # 420 = one 5Min bar + 120s slack for late delivery.
    stale_data_sec: int = 420
    max_spread_bps: float = 100.0
    duplicate_order_window_sec: int = 15
    # Churn deadband (2026-07-15 paper run): with no floor on order size,
    # re-deciding every tick against an unchanged 5-min bar produced dust
    # rebalances (0.008 NVDA ≈ $1.69) and 60-second oscillations — spread
    # and fees paid for sub-dollar position adjustments. Orders below this
    # notional are rejected UNLESS they fully flatten a position (dust
    # must always be closeable). 0 disables the gate.
    min_trade_notional: float = 100.0
    # Trading costs (physics, not decisions): fed into the EV gate so the
    # risk manager can reject trades whose expected value doesn't cover costs.
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    # §4.4.4 fail-fast: a model without a risk head feeds (0, 0) into the
    # EV gate, which then rejects every trade — a silent zero-trade
    # session. True (default) refuses to CONSTRUCT a brain policy around
    # such a model; flip to false only for explicit experiments that
    # accept an inert EV gate.
    ev_gate_requires_signals: bool = True
    # Brokers cannot short fractional shares (Alpaca rejects with
    # 42210000 "fractional orders cannot be sold short"). Found in the
    # 2026-07-15 paper run: sub-share short targets were submitted, hard-
    # rejected, and resubmitted every duplicate_order_window forever.
    # False (default = Alpaca reality) quantizes short targets to whole
    # shares in BOTH the backtest and the live path, so simulated shorts
    # stay executable and scorecard rows do not bank fills the broker
    # would refuse. Set True only for a venue that supports them.
    fractional_shorts_supported: bool = False
    kill_switch: bool = False
    allowed_sessions: list[str] = Field(default_factory=list)
    max_aggregate_notional: float = 50000.0
    max_leverage_ratio: float = 2.0
    max_daily_loss_pct: float | None = None  # percentage of equity; overrides max_daily_loss when set
    max_rolling_drawdown: float = -0.10  # -10% default
    rolling_drawdown_window: int = 100
    quantity_precision: float = 0.001
    max_processed_request_ids: int = 100_000
    # --- Outcome-adaptive risk budget (Phase 2 §6.4) ---
    # When enabled, risk-INCREASING trades are sized down as realized rolling
    # drawdown deepens: multiplier 1.0 at zero drawdown, linear to
    # budget_min_multiplier at budget_full_at_drawdown. Deterministic, bounded,
    # logged on every decision (RiskDecision.budget_multiplier). Risk-reducing
    # trades are never throttled — de-risking must stay full-size.
    adaptive_budget_enabled: bool = False
    budget_min_multiplier: float = 0.25
    budget_full_at_drawdown: float = -0.05

    @field_validator("allowed_sessions")
    @classmethod
    def validate_allowed_sessions(cls, value: list[str]) -> list[str]:
        unknown = sorted({name for name in value if name.lower() not in SESSION_NAMES})
        if unknown:
            raise ConfigurationError(
                f"Unknown session name(s) {unknown} in risk.allowed_sessions; "
                f"known sessions: {sorted(SESSION_NAMES)}"
            )
        return value

    @field_validator("max_daily_loss")
    @classmethod
    def validate_max_daily_loss(cls, value: float) -> float:
        # Backward compat: some configs state the loss limit as a negative
        # number ("-350 means down 350"). Normalize to the positive magnitude
        # at load time; the risk manager's abs()-at-read stays harmless.
        if value == 0.0:
            raise ConfigurationError("risk.max_daily_loss must be non-zero (a zero budget blocks all trading)")
        return abs(value)

    @field_validator("fee_bps", "slippage_bps", "max_spread_bps")
    @classmethod
    def validate_non_negative_bps(cls, value: float, info) -> float:
        if value < 0.0:
            raise ConfigurationError(f"risk.{info.field_name} must be >= 0")
        return value

    @field_validator("cooldown_sec", "stale_data_sec", "duplicate_order_window_sec")
    @classmethod
    def validate_non_negative_seconds(cls, value: int, info) -> int:
        if value < 0:
            raise ConfigurationError(f"risk.{info.field_name} must be >= 0")
        return value

    @field_validator("max_concurrent_positions")
    @classmethod
    def validate_max_concurrent_positions(cls, value: int) -> int:
        if value < 1:
            raise ConfigurationError("risk.max_concurrent_positions must be >= 1")
        return value

    @field_validator("quantity_precision")
    @classmethod
    def validate_quantity_precision(cls, value: float) -> float:
        if value <= 0.0:
            raise ConfigurationError("risk.quantity_precision must be > 0")
        return value

    @field_validator("max_daily_loss_pct")
    @classmethod
    def validate_max_daily_loss_pct(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 < value <= 1.0:
            raise ConfigurationError("risk.max_daily_loss_pct must be within (0, 1] when set")
        return value

    @field_validator("max_position_notional", "max_aggregate_notional", "max_leverage_ratio")
    @classmethod
    def validate_positive_caps(cls, value: float, info) -> float:
        # A zero or negative cap silently zeroes every trade — nonsense that
        # should fail at load (found by the §6.4 cap-sweep harness).
        if value <= 0.0:
            raise ConfigurationError(f"risk.{info.field_name} must be > 0")
        return value

    @field_validator("min_trade_notional")
    @classmethod
    def validate_min_trade_notional(cls, value: float) -> float:
        if value < 0.0:
            raise ConfigurationError("risk.min_trade_notional must be >= 0 (0 disables)")
        return value

    @model_validator(mode="after")
    def validate_notional_caps(self) -> "RiskConfig":
        if self.max_aggregate_notional < self.max_position_notional:
            raise ConfigurationError(
                "risk.max_aggregate_notional must be >= risk.max_position_notional "
                f"(got {self.max_aggregate_notional} < {self.max_position_notional})"
            )
        if not 0.0 < self.budget_min_multiplier <= 1.0:
            raise ConfigurationError("risk.budget_min_multiplier must be within (0, 1]")
        if self.budget_full_at_drawdown >= 0.0:
            raise ConfigurationError(
                "risk.budget_full_at_drawdown must be negative (a drawdown level)"
            )
        return self
