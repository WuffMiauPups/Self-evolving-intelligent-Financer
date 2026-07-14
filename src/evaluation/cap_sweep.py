"""Cap-parameter sweep harness (MASTER_PLAN §6.4, RESEARCH_ROADMAP §2.8).

Risk caps (max_position_notional, max_aggregate_notional, ...) are
operating points, not opinions. This harness backtests identical bars
with identical seeds across a range of values for ONE risk parameter,
reports return / drawdown / CVaR95 per setting, and marks the knee —
the setting with the best risk-adjusted return, ties resolved toward
the smaller (more conservative) cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from src.backtest.engine import BacktestEngine
from src.core.config import AppConfig, RiskConfig
from src.core.logging import get_logger
from src.core.seeds import seed_everything
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager

logger = get_logger("evaluation.cap_sweep")

PolicyBuilder = Callable[[AppConfig, dict[str, list], FeaturePipeline], object]

# Guard against near-zero CVaR blowing the ratio up to infinity.
_CVAR_EPSILON = 1.0e-6


@dataclass(frozen=True)
class CapSweepPoint:
    value: float
    metrics: dict[str, float]
    risk_adjusted_return: float


@dataclass(frozen=True)
class CapSweepReport:
    parameter: str
    points: list[CapSweepPoint]
    knee_value: float
    current_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "knee_value": self.knee_value,
            "current_value": self.current_value,
            "points": [
                {
                    "value": p.value,
                    "risk_adjusted_return": p.risk_adjusted_return,
                    "metrics": p.metrics,
                }
                for p in self.points
            ],
        }


def risk_adjusted_return(metrics: dict[str, float]) -> float:
    """Total return per unit of CVaR95 tail risk.

    CVaR95 is the mean of the worst 5% of per-bar returns (negative in
    any realistic run); its magnitude is the denominator. A run with no
    tail (|CVaR| ~ 0) scores by raw return so degenerate flat runs don't
    rank above real ones on a 0/0 artifact.
    """
    total_return = float(metrics.get("total_return", 0.0))
    tail = abs(float(metrics.get("cvar_95", 0.0)))
    if tail < _CVAR_EPSILON:
        return total_return
    return total_return / tail


def pick_knee(points: list[CapSweepPoint]) -> float:
    """The knee: best risk-adjusted return; ties go to the smaller cap."""
    if not points:
        raise ValueError("cap sweep produced no points")
    best = max(points, key=lambda p: (p.risk_adjusted_return, -p.value))
    return best.value


def run_cap_sweep(
    config: AppConfig,
    bars_by_symbol: dict[str, list],
    policy_builder: PolicyBuilder,
    *,
    parameter: str,
    values: list[float],
    seed: int | None = None,
) -> CapSweepReport:
    """Backtest identical bars at each value of one RiskConfig parameter."""
    if parameter not in RiskConfig.model_fields:
        raise ValueError(f"unknown risk parameter: {parameter!r}")
    if not values:
        raise ValueError("values must be non-empty")
    current_value = getattr(config.risk, parameter)
    if not isinstance(current_value, (int, float)) or isinstance(current_value, bool):
        raise ValueError(f"risk parameter {parameter!r} is not numeric (got {current_value!r})")

    base_seed = int(config.evaluation.seed if seed is None else seed)
    points: list[CapSweepPoint] = []
    for value in values:
        arm_config = config.model_copy(deep=True)
        setattr(arm_config.risk, parameter, float(value))
        # Re-validate so a sweep can never smuggle in an invalid risk config.
        arm_config.risk = RiskConfig.model_validate(arm_config.risk.model_dump())
        seed_everything(base_seed)
        pipeline = FeaturePipeline(arm_config.features, device=torch.device("cpu"))
        policy = policy_builder(arm_config, bars_by_symbol, pipeline)
        report = BacktestEngine(
            arm_config, pipeline, RiskManager(arm_config.risk.model_copy())
        ).run(bars_by_symbol, policy)
        metrics = {
            key: float(report.metrics.get(key, 0.0))
            for key in (
                "total_return",
                "sharpe_ratio",
                "expectancy",
                "max_drawdown",
                "cvar_95",
                "fill_count",
            )
        }
        point = CapSweepPoint(
            value=float(value),
            metrics=metrics,
            risk_adjusted_return=risk_adjusted_return(metrics),
        )
        points.append(point)
        logger.info(
            "cap_sweep_point",
            parameter=parameter,
            value=value,
            risk_adjusted_return=point.risk_adjusted_return,
            **metrics,
        )

    knee = pick_knee(points)
    logger.info(
        "cap_sweep_knee",
        parameter=parameter,
        knee_value=knee,
        current_value=float(current_value),
        note="apply the knee to the risk YAML manually — the sweep never mutates config",
    )
    return CapSweepReport(
        parameter=parameter,
        points=points,
        knee_value=knee,
        current_value=float(current_value),
    )
