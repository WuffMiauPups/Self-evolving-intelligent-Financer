"""Inspect how label_epsilon and horizon choices affect class balance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabelProfile:
    """Result of labeling samples under a specific epsilon/horizon config."""
    label_epsilon: float
    primary_horizon: int
    total_samples: int
    class_counts: dict[str, int]
    class_ratios: dict[str, float]
    flat_rate: float
    near_flat_rate: float
    near_boundary_rate: float
    primary_return_quantiles: dict[str, float]


def inspect_label_profiles(
    samples_with_returns: list[dict[str, Any]],
    *,
    epsilon_values: list[float],
    horizons: list[int],
) -> list[LabelProfile]:
    """Sweep epsilon and horizon values, returning class balance profiles.

    Args:
        samples_with_returns: Each dict must have 'future_returns' mapping
            horizon (int) → return (float).
        epsilon_values: Epsilon thresholds to test.
        horizons: Horizons to use as primary return horizon.

    Returns:
        A LabelProfile for each (epsilon, horizon) combination.
    """
    import torch

    profiles: list[LabelProfile] = []
    for horizon in sorted(horizons):
        # Collect primary returns for this horizon
        returns: list[float] = []
        for sample in samples_with_returns:
            fr = sample.get("future_returns", {})
            if horizon in fr:
                returns.append(float(fr[horizon]))

        if not returns:
            continue

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        quantiles = torch.quantile(
            returns_tensor,
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float32),
        )
        return_quantiles = {
            "p10": float(quantiles[0].item()),
            "p25": float(quantiles[1].item()),
            "p50": float(quantiles[2].item()),
            "p75": float(quantiles[3].item()),
            "p90": float(quantiles[4].item()),
        }

        for epsilon in epsilon_values:
            long_count = 0
            short_count = 0
            flat_count = 0
            near_flat_count = 0
            near_boundary_count = 0
            for ret in returns:
                if ret > epsilon:
                    long_count += 1
                elif ret < -epsilon:
                    short_count += 1
                else:
                    flat_count += 1
                if abs(ret) <= epsilon * 1.25:
                    near_flat_count += 1
                if abs(abs(ret) - epsilon) <= max(epsilon * 0.5, 1.0e-6):
                    near_boundary_count += 1

            total = len(returns)
            profiles.append(LabelProfile(
                label_epsilon=epsilon,
                primary_horizon=horizon,
                total_samples=total,
                class_counts={"short": short_count, "flat": flat_count, "long": long_count},
                class_ratios={
                    "short": short_count / max(total, 1),
                    "flat": flat_count / max(total, 1),
                    "long": long_count / max(total, 1),
                },
                flat_rate=flat_count / max(total, 1),
                near_flat_rate=near_flat_count / max(total, 1),
                near_boundary_rate=near_boundary_count / max(total, 1),
                primary_return_quantiles=return_quantiles,
            ))

    return profiles


def format_label_profiles(profiles: list[LabelProfile]) -> str:
    """Format label profiles as a human-readable table."""
    if not profiles:
        return "No label profiles to display."

    lines: list[str] = []
    header = (
        f"{'horizon':>8} {'epsilon':>10} {'total':>8} "
        f"{'short%':>8} {'flat%':>8} {'long%':>8} "
        f"{'near_flat%':>10} {'boundary%':>10} "
        f"{'p10':>10} {'p50':>10} {'p90':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for p in profiles:
        line = (
            f"{p.primary_horizon:>8} {p.label_epsilon:>10.5f} {p.total_samples:>8} "
            f"{p.class_ratios['short']:>8.3f} {p.class_ratios['flat']:>8.3f} {p.class_ratios['long']:>8.3f} "
            f"{p.near_flat_rate:>10.3f} {p.near_boundary_rate:>10.3f} "
            f"{p.primary_return_quantiles.get('p10', 0.0):>10.5f} "
            f"{p.primary_return_quantiles.get('p50', 0.0):>10.5f} "
            f"{p.primary_return_quantiles.get('p90', 0.0):>10.5f}"
        )
        lines.append(line)

    return "\n".join(lines)
