"""Risk manager status — read-only view of the configured risk envelope.

The GUI cannot mutate a *running* RiskManager directly (it lives inside the
paper-trading subprocess). What it can do, safely:

  - read the current risk *configuration* from the loaded AppConfig
  - request a kill-switch by writing a marker file that the runtime checks
    OR by stopping the runtime via RuntimeService.stop()

For v1 we only expose the config view + a hard stop via the runtime service.
The runtime's RiskManager remains the final authority on every order.
"""
from __future__ import annotations

from src.gui_services.config_service import load_app_config
from src.gui_services.state import RiskStatus, ServiceError


def get_risk_status(*, config_paths: list[str] | None = None) -> RiskStatus:
    config_paths = config_paths or ["configs/default.yaml"]
    try:
        cfg = load_app_config(*config_paths)
    except ServiceError as exc:
        return RiskStatus(
            kill_switch=True,
            max_position_notional=0.0,
            max_daily_loss=0.0,
            cooldown_seconds=0,
            notes=f"Config error: {exc}",
        )
    risk = cfg.risk
    return RiskStatus(
        kill_switch=bool(getattr(risk, "kill_switch", False)),
        max_position_notional=float(getattr(risk, "max_position_notional", 0.0)),
        max_daily_loss=float(getattr(risk, "max_daily_loss", 0.0)),
        cooldown_seconds=int(getattr(risk, "cooldown_sec", 0)),
        max_aggregate_notional=float(getattr(risk, "max_aggregate_notional", 50000.0)),
        max_leverage_ratio=float(getattr(risk, "max_leverage_ratio", 2.0)),
        max_daily_loss_pct=getattr(risk, "max_daily_loss_pct", None),
        max_rolling_drawdown=float(getattr(risk, "max_rolling_drawdown", -0.10)),
        rolling_drawdown_window=int(getattr(risk, "rolling_drawdown_window", 100)),
        max_concurrent_positions=int(getattr(risk, "max_concurrent_positions", 15)),
        fee_bps=float(getattr(risk, "fee_bps", 1.0)),
        slippage_bps=float(getattr(risk, "slippage_bps", 2.0)),
        notes="Risk manager is final authority during execution.",
    )
