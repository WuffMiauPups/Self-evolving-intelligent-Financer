"""Self-monitor configuration (docs/analysis/SELF_MONITOR_DESIGN.md).

Lives under ``AppConfig.monitoring.self_monitor``. Default OFF — the L0
monitor records nothing and writes nothing until explicitly enabled, so a
disabled runtime/backtest is byte-identical to one without the monitor.
The L1 damping ladder is a second, independent gate that stays off until a
scorecard row justifies it.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SelfMonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False                 # master gate: records + writes nothing when off
    window: int = 50                      # resolved trades in the rolling outcome window
    baseline_win_rate: float = 0.40       # S1 expectation to be surprised against
    surprise_z: float = 2.0               # S1: |z| beyond which win-rate flags
    calibration_gap: float = 0.25         # S2: |claimed - realized| beyond which it flags
    min_bucket: int = 15                  # S2: resolved trades per conf bucket before judging
    churn_per_hour: float = 6.0           # S4: close->reopen cycles/hour beyond which it flags
    # S5 realized-vs-assumed cost
    assumed_slippage_bps: float = 2.0     # the EV gate's assumed slippage (risk.slippage_bps)
    cost_surprise_bps: float = 3.0        # S5: realized median slippage over assumed by this flags
    min_cost_samples: int = 15            # S5: slippage samples before judging
    # reporting
    report_interval_decisions: int = 25   # emit one HealthReport every N decisions
    # L1 damping ladder (shrink-only), gated independently of `enabled`
    damping_enabled: bool = False
    damping_floor: float = Field(default=0.5, ge=0.0, le=1.0)


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    self_monitor: SelfMonitorConfig = Field(default_factory=SelfMonitorConfig)
