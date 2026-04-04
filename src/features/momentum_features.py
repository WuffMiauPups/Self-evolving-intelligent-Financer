from __future__ import annotations

import torch


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

