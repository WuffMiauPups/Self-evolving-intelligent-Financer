from __future__ import annotations

from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    max_daily_loss: float = 500.0
    max_position_notional: float = 5000.0
    max_position_per_symbol: int = 1
    max_concurrent_positions: int = 3
    cooldown_sec: int = 60
    stale_data_sec: int = 30
    max_spread_bps: float = 15.0
    duplicate_order_window_sec: int = 15
    # Trading costs (physics, not decisions): fed into the EV gate so the
    # risk manager can reject trades whose expected value doesn't cover costs.
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    kill_switch: bool = False
    allowed_sessions: list[str] = Field(default_factory=list)
    max_aggregate_notional: float = 50000.0
    max_leverage_ratio: float = 2.0
    max_daily_loss_pct: float | None = None  # percentage of equity; overrides max_daily_loss when set
    max_rolling_drawdown: float = -0.10  # -10% default
    rolling_drawdown_window: int = 100
    quantity_precision: float = 0.001
    max_processed_request_ids: int = 100_000
