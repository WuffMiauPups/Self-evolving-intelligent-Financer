from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch

from src.memory.schemas import OutcomeSummary, PathStatistics, RiskStatistics


def aggregate_path_statistics(outcomes: Sequence[OutcomeSummary]) -> PathStatistics:
    if not outcomes:
        raise ValueError("Cannot aggregate zero outcomes")
    horizons = sorted(outcomes[0].future_returns)
    return_matrix = torch.tensor(
        [[outcome.future_returns[horizon] for horizon in horizons] for outcome in outcomes],
        dtype=torch.float32,
    )
    max_favorable = torch.tensor([outcome.max_favorable_excursion for outcome in outcomes], dtype=torch.float32)
    max_adverse = torch.tensor([outcome.max_adverse_excursion for outcome in outcomes], dtype=torch.float32)
    realized_vol = torch.tensor([outcome.realized_volatility for outcome in outcomes], dtype=torch.float32)
    drawdowns = torch.tensor([outcome.drawdown for outcome in outcomes], dtype=torch.float32)
    label_counts = Counter(outcome.path_label for outcome in outcomes)
    direction_counts = Counter(outcome.direction_class for outcome in outcomes)
    sample_count = len(outcomes)
    return PathStatistics(
        sample_count=sample_count,
        horizon_return_mean={horizon: float(return_matrix[:, idx].mean().item()) for idx, horizon in enumerate(horizons)},
        horizon_return_p25={
            horizon: float(torch.quantile(return_matrix[:, idx], 0.25).item()) for idx, horizon in enumerate(horizons)
        },
        horizon_return_p75={
            horizon: float(torch.quantile(return_matrix[:, idx], 0.75).item()) for idx, horizon in enumerate(horizons)
        },
        max_favorable_mean=float(max_favorable.mean().item()),
        max_adverse_mean=float(max_adverse.mean().item()),
        max_adverse_p10=float(torch.quantile(max_adverse, 0.10).item()),
        realized_volatility_mean=float(realized_vol.mean().item()),
        drawdown_mean=float(drawdowns.mean().item()),
        drawdown_p10=float(torch.quantile(drawdowns, 0.10).item()),
        path_label_frequencies={label: count / sample_count for label, count in sorted(label_counts.items())},
        direction_probabilities={label: count / sample_count for label, count in sorted(direction_counts.items())},
    )


def aggregate_risk_statistics(outcomes: Sequence[OutcomeSummary]) -> RiskStatistics:
    if not outcomes:
        raise ValueError("Cannot aggregate zero outcomes")
    realized_vol = torch.tensor([outcome.realized_volatility for outcome in outcomes], dtype=torch.float32)
    max_adverse = torch.tensor([outcome.max_adverse_excursion for outcome in outcomes], dtype=torch.float32)
    drawdowns = torch.tensor([outcome.drawdown for outcome in outcomes], dtype=torch.float32)
    return RiskStatistics(
        sample_count=len(outcomes),
        realized_volatility_mean=float(realized_vol.mean().item()),
        realized_volatility_p75=float(torch.quantile(realized_vol, 0.75).item()),
        max_adverse_mean=float(max_adverse.mean().item()),
        max_adverse_p10=float(torch.quantile(max_adverse, 0.10).item()),
        drawdown_mean=float(drawdowns.mean().item()),
        drawdown_p10=float(torch.quantile(drawdowns, 0.10).item()),
    )
