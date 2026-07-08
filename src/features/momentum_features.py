from __future__ import annotations

import torch

from src.features.rolling import rolling_mean, rolling_slope


def compute_momentum_summaries(bars: torch.Tensor, windows: list[int], epsilon: float) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    features: dict[str, torch.Tensor] = {}
    for window in windows:
        if close.numel() >= window:
            window_slice = close[-window:]
            mean_price = window_slice.mean()
            slope = (window_slice[-1] - window_slice[0]) / (window + epsilon)
            features[f"momentum_slope_{window}"] = slope / (mean_price + epsilon)
        else:
            features[f"momentum_slope_{window}"] = torch.zeros((), dtype=bars.dtype, device=bars.device)
    return features


def compute_momentum_summaries_rolling(
    bars: torch.Tensor, windows: list[int], epsilon: float
) -> dict[str, torch.Tensor]:
    """Vectorized rolling variant of ``compute_momentum_summaries``."""
    close = bars[:, 3]
    n = close.numel()
    zero_full = torch.zeros(n, dtype=bars.dtype, device=bars.device)
    features: dict[str, torch.Tensor] = {}
    for window in windows:
        if n >= window:
            slopes = rolling_slope(close, window, epsilon=epsilon)
            means = rolling_mean(close, window)
            normalized = slopes / (means + epsilon)
            if window > 1:
                normalized[: window - 1] = 0.0
            features[f"momentum_slope_{window}"] = normalized
        else:
            features[f"momentum_slope_{window}"] = zero_full
    return features

