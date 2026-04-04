from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.schemas import MarketBar
from src.memory.schemas import OutcomeSummary, RiskSummary


def compute_outcome_summary(
    bars: Sequence[MarketBar],
    index: int,
    *,
    horizons: Sequence[int],
    path_label_rules: dict[str, float],
) -> OutcomeSummary:
    if index < 0 or index >= len(bars):
        raise IndexError(f"Outcome index out of bounds: {index}")
    if not horizons:
        raise ValueError("At least one forward horizon is required")
    max_horizon = max(horizons)
    if index + max_horizon >= len(bars):
        raise ValueError(
            f"Not enough future bars to compute outcome at index {index} for horizon {max_horizon}"
        )

    base_price = float(bars[index].close)
    future_closes = torch.tensor(
        [float(bars[index + step].close) for step in range(1, max_horizon + 1)],
        dtype=torch.float32,
    )
    path_prices = torch.cat([torch.tensor([base_price], dtype=torch.float32), future_closes], dim=0)
    relative_path = future_closes / max(base_price, 1.0e-8) - 1.0
    log_returns = torch.log(path_prices[1:] / path_prices[:-1])
    cumulative_path = future_closes / max(base_price, 1.0e-8)
    running_peaks = torch.cummax(cumulative_path, dim=0).values
    drawdowns = cumulative_path / torch.clamp(running_peaks, min=1.0e-8) - 1.0
    future_returns = {
        int(horizon): float((float(bars[index + horizon].close) / max(base_price, 1.0e-8)) - 1.0)
        for horizon in horizons
    }
    primary_horizon = max(future_returns)
    primary_return = future_returns[primary_horizon]
    time_to_move_threshold = float(path_label_rules.get("time_to_move_threshold", 0.002))
    time_to_move = next(
        (
            step
            for step, value in enumerate(relative_path.tolist(), start=1)
            if abs(float(value)) >= time_to_move_threshold
        ),
        None,
    )
    path_label = classify_path_label(
        primary_return=primary_return,
        max_favorable=float(relative_path.max().item()),
        max_adverse=float(relative_path.min().item()),
        rules=path_label_rules,
    )
    direction_class = classify_direction(
        primary_return=primary_return,
        flat_threshold=float(path_label_rules.get("flat_return_threshold", 0.00075)),
    )
    return OutcomeSummary(
        future_returns=future_returns,
        max_favorable_excursion=float(relative_path.max().item()),
        max_adverse_excursion=float(relative_path.min().item()),
        realized_volatility=float(log_returns.std(unbiased=False).item()) if log_returns.numel() > 0 else 0.0,
        drawdown=float(drawdowns.min().item()) if drawdowns.numel() > 0 else 0.0,
        path_label=path_label,
        direction_class=direction_class,
        time_to_move=time_to_move,
    )


def classify_direction(*, primary_return: float, flat_threshold: float) -> str:
    if primary_return >= flat_threshold:
        return "up"
    if primary_return <= -flat_threshold:
        return "down"
    return "flat"


def classify_path_label(
    *,
    primary_return: float,
    max_favorable: float,
    max_adverse: float,
    rules: dict[str, float],
) -> str:
    breakout_threshold = float(rules.get("breakout_return_threshold", 0.003))
    mean_reversion_threshold = float(rules.get("mean_reversion_return_threshold", 0.0015))
    if primary_return >= breakout_threshold:
        return "breakout_up"
    if primary_return <= -breakout_threshold:
        return "breakout_down"
    if primary_return > 0.0 and max_adverse <= -mean_reversion_threshold:
        return "mean_reversion_up"
    if primary_return < 0.0 and max_favorable >= mean_reversion_threshold:
        return "mean_reversion_down"
    return "chop"


def build_risk_summary(outcome: OutcomeSummary) -> RiskSummary:
    return RiskSummary(
        realized_volatility=outcome.realized_volatility,
        max_adverse_excursion=outcome.max_adverse_excursion,
        drawdown=outcome.drawdown,
    )
