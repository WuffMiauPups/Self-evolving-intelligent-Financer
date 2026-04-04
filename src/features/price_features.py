from __future__ import annotations

import torch


def compute_price_features(bars: torch.Tensor, momentum_windows: list[int], epsilon: float) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    open_price = bars[:, 0]
    high = bars[:, 1]
    low = bars[:, 2]
    last_close = close[-1]
    features: dict[str, torch.Tensor] = {}

    if close.numel() > 1:
        one_step_return = (last_close / close[-2]) - 1.0
        features["return_1"] = one_step_return
        features["log_return_1"] = torch.log((last_close / close[-2]).clamp_min(epsilon))
    else:
        zero = torch.zeros((), dtype=bars.dtype, device=bars.device)
        features["return_1"] = zero
        features["log_return_1"] = zero

    latest_range = (high[-1] - low[-1]) / (last_close + epsilon)
    body = (bars[-1, 3] - open_price[-1]) / (open_price[-1] + epsilon)
    upper_wick = (high[-1] - torch.maximum(open_price[-1], last_close)) / (last_close + epsilon)
    lower_wick = (torch.minimum(open_price[-1], last_close) - low[-1]) / (last_close + epsilon)
    features["range_1"] = latest_range
    features["body_1"] = body
    features["upper_wick_1"] = upper_wick
    features["lower_wick_1"] = lower_wick

    for window in momentum_windows:
        if close.numel() > window:
            past_close = close[-(window + 1)]
            window_mean = close[-window:].mean()
            features[f"return_{window}"] = (last_close / past_close) - 1.0
            features[f"log_return_{window}"] = torch.log((last_close / past_close).clamp_min(epsilon))
            features[f"momentum_{window}"] = (last_close / (window_mean + epsilon)) - 1.0
        else:
            zero = torch.zeros((), dtype=bars.dtype, device=bars.device)
            features[f"return_{window}"] = zero
            features[f"log_return_{window}"] = zero
            features[f"momentum_{window}"] = zero
    return features
