from __future__ import annotations

from dataclasses import dataclass

from src.memory.outcomes import OutcomeSummary


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


def build_training_targets(
    outcome: OutcomeSummary,
    *,
    label_epsilon: float,
    regression_targets: list[str],
) -> TrainingTargets:
    primary_horizon = max(outcome.future_returns)
    primary_return = float(outcome.future_returns[primary_horizon])
    if primary_return > label_epsilon:
        direction_label = 2
        direction_name = "long"
    elif primary_return < -label_epsilon:
        direction_label = 0
        direction_name = "short"
    else:
        direction_label = 1
        direction_name = "flat"

    path_name = outcome.path_label
    if path_name not in PATH_LABEL_VOCABULARY:
        path_name = "chop"
    path_label = PATH_LABEL_VOCABULARY.index(path_name)
    regression_payload = _build_regression_targets(outcome)
    return TrainingTargets(
        direction_label=direction_label,
        direction_name=direction_name,
        path_label=path_label,
        path_name=path_name,
        primary_return=primary_return,
        regression_targets={
            name: float(regression_payload[name])
            for name in regression_targets
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
    stability = abs(primary_return) / denominator
    return {
        "primary_return": primary_return,
        "max_favorable_excursion": float(outcome.max_favorable_excursion),
        "max_adverse_excursion": float(outcome.max_adverse_excursion),
        "realized_volatility": realized_volatility,
        "drawdown": drawdown,
        "stability": stability,
    }
