from __future__ import annotations

import torch

from src.features.normalization import safe_zscore
from src.features.rolling import rolling_mean, rolling_std


def compute_multi_timeframe_features(
    bars: torch.Tensor,
    windows: dict[str, int],
    epsilon: float,
) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    features: dict[str, torch.Tensor] = {}
    for label, window in windows.items():
        if close.numel() >= window:
            window_close = close[-window:]
            moving_average = window_close.mean()
            features[f"regime_close_to_ma_{label}"] = (close[-1] / (moving_average + epsilon)) - 1.0
            features[f"regime_price_z_{label}"] = safe_zscore(close[-1], window_close, epsilon)
        else:
            zero = torch.zeros((), dtype=bars.dtype, device=bars.device)
            features[f"regime_close_to_ma_{label}"] = zero
            features[f"regime_price_z_{label}"] = zero
    return features


def compute_multi_timeframe_features_rolling(
    bars: torch.Tensor,
    windows: dict[str, int],
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Vectorized rolling variant of ``compute_multi_timeframe_features``."""
    close = bars[:, 3]
    n = close.numel()
    zero_full = torch.zeros(n, dtype=bars.dtype, device=bars.device)
    features: dict[str, torch.Tensor] = {}
    for label, window in windows.items():
        if n >= window:
            means = rolling_mean(close, window)
            stds = rolling_std(close, window, unbiased=False)
            close_to_ma = (close / (means + epsilon)) - 1.0
            price_z = (close - means) / (stds + epsilon)
            if window > 1:
                close_to_ma[: window - 1] = 0.0
                price_z[: window - 1] = 0.0
            features[f"regime_close_to_ma_{label}"] = close_to_ma
            features[f"regime_price_z_{label}"] = price_z
        else:
            features[f"regime_close_to_ma_{label}"] = zero_full
            features[f"regime_price_z_{label}"] = zero_full
    return features

