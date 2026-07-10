"""Training target generation from outcome summaries.

Supports the state + future-path training design:
- direction classification (short/flat/long) from a primary horizon
- path label classification (breakout_up/down, mean_reversion_up/down, chop)
- future-return regression targets (per-horizon returns)
- risk/path regression targets (MFE, MAE, volatility, drawdown, stability)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.memory.schemas import OutcomeSummary


PATH_LABEL_VOCABULARY: tuple[str, ...] = (
    "breakout_down",
    "mean_reversion_down",
    "chop",
    "mean_reversion_up",
    "breakout_up",
)


@dataclass(frozen=True)
class TrainingTargets:
    direction_label: int
    direction_name: str
    path_label: int
    path_name: str
    primary_return: float
    regression_targets: dict[str, float]
    future_returns: dict[int, float]
    max_favorable_excursion: float
    max_adverse_excursion: float
    realized_volatility: float
    drawdown: float


def compute_symbol_epsilon(
    bars: list,
    *,
    multiplier: float = 1.0,
    min_epsilon: float = 1.0e-4,
    max_epsilon: float = 5.0e-3,
    min_bars: int = 20,
) -> float:
    """Compute adaptive label epsilon from per-bar log-return standard deviation.

    Returns ``multiplier * std(log_returns)`` clamped to ``[min_epsilon, max_epsilon]``.
    Falls back to ``min_epsilon`` if fewer than ``min_bars`` bars are available.

    This replaces the static ``label_epsilon`` with a per-symbol threshold
    that scales with the instrument's realized volatility.  Volatile symbols
    (e.g. TSLA, MSTR) get a wider flat band; calm symbols (e.g. SPY, VOO)
    get a tighter one.
    """
    if len(bars) < min_bars:
        return min_epsilon
    closes = []
    for bar in bars:
        c = getattr(bar, "close", None)
        if c is not None and c > 0:
            closes.append(float(c))
    if len(closes) < min_bars:
        return min_epsilon
    log_returns = []
    for i in range(1, len(closes)):
        log_returns.append(math.log(closes[i] / closes[i - 1]))
    if len(log_returns) < 10:
        return min_epsilon
    mean_ret = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_ret) ** 2 for r in log_returns) / len(log_returns)
    std = variance ** 0.5
    epsilon = multiplier * std
    return max(min_epsilon, min(epsilon, max_epsilon))


def build_training_targets(
    outcome: OutcomeSummary,
    *,
    label_epsilon: float,
    regression_targets: list[str],
    primary_horizon: int | None = None,
    enable_future_return_targets: bool = False,
    enable_risk_path_targets: bool = False,
    transaction_cost_bps: float = 0.0,
) -> TrainingTargets:
    """Build training targets from an outcome summary.

    Args:
        outcome: Realized future outcome for a decision point.
        label_epsilon: Threshold for short/flat/long direction label.
        regression_targets: Explicit list of regression target names to include.
        primary_horizon: Primary direction horizon. If None, uses max horizon.
        enable_future_return_targets: If True, add per-horizon return regression targets.
        enable_risk_path_targets: If True, add MFE/MAE/volatility/drawdown regression targets.
        transaction_cost_bps: Transaction cost in basis points.  Widens the
            flat-band epsilon so moves smaller than the cost are labeled FLAT.
    """
    if primary_horizon is not None and primary_horizon in outcome.future_returns:
        effective_horizon = primary_horizon
    else:
        effective_horizon = max(outcome.future_returns)
    primary_return = float(outcome.future_returns[effective_horizon])
    # The volatility-derived epsilon is a 1-bar return scale, but it
    # thresholds an H-bar return whose noise std grows ~sqrt(H); without
    # the scaling the flat band is ~sqrt(H) too narrow and noise gets
    # labeled as signal.  The cost floor is horizon-independent: a move
    # smaller than the round trip can never be worth trading.
    effective_epsilon = label_epsilon * math.sqrt(max(effective_horizon, 1)) + (
        transaction_cost_bps / 10_000.0
    )
    if primary_return > effective_epsilon:
        direction_label = 2
        direction_name = "long"
    elif primary_return < -effective_epsilon:
        direction_label = 0
        direction_name = "short"
    else:
        direction_label = 1
        direction_name = "flat"

    path_name = outcome.path_label
    if path_name not in PATH_LABEL_VOCABULARY:
        path_name = "chop"
    path_label = PATH_LABEL_VOCABULARY.index(path_name)

    # Build the full regression target payload
    regression_payload = _build_regression_targets(outcome)

    # Add per-horizon return targets
    if enable_future_return_targets:
        for horizon, ret in outcome.future_returns.items():
            regression_payload[f"return_h{horizon}"] = float(ret)

    # Add risk/path targets
    if enable_risk_path_targets:
        regression_payload["mfe"] = float(outcome.max_favorable_excursion)
        regression_payload["mae"] = float(outcome.max_adverse_excursion)
        regression_payload["volatility"] = float(outcome.realized_volatility)
        regression_payload["path_drawdown"] = float(outcome.drawdown)

    return TrainingTargets(
        direction_label=direction_label,
        direction_name=direction_name,
        path_label=path_label,
        path_name=path_name,
        primary_return=primary_return,
        regression_targets={
            name: float(regression_payload[name])
            for name in set(regression_targets) | set(
                k for k in regression_payload
                if (enable_future_return_targets and k.startswith("return_h"))
                or (enable_risk_path_targets and k in ("mfe", "mae", "volatility", "path_drawdown"))
            )
            if name in regression_payload
        },
        future_returns={int(horizon): float(value) for horizon, value in outcome.future_returns.items()},
        max_favorable_excursion=float(outcome.max_favorable_excursion),
        max_adverse_excursion=float(outcome.max_adverse_excursion),
        realized_volatility=float(outcome.realized_volatility),
        drawdown=float(outcome.drawdown),
    )


def _build_regression_targets(outcome: OutcomeSummary) -> dict[str, float]:
    primary_return = float(outcome.future_returns[max(outcome.future_returns)])
    drawdown = float(outcome.drawdown)
    realized_volatility = float(outcome.realized_volatility)
    denominator = abs(primary_return) + abs(drawdown) + realized_volatility + 1.0e-6
    # Stability = how little of the total movement comes from adverse
    # components (drawdown + volatility).  High return with low drawdown
    # and low volatility → stability near 1.  Previously this was
    # |return| / denom which just measured return dominance.
    adverse = abs(drawdown) + realized_volatility
    stability = max(1.0 - adverse / denominator, 0.0)
    return {
        "primary_return": primary_return,
        "max_favorable_excursion": float(outcome.max_favorable_excursion),
        "max_adverse_excursion": float(outcome.max_adverse_excursion),
        "realized_volatility": realized_volatility,
        "drawdown": drawdown,
        "stability": stability,
    }
