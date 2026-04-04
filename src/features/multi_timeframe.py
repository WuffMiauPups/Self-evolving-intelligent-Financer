from __future__ import annotations

import torch

from src.features.normalization import safe_zscore


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

