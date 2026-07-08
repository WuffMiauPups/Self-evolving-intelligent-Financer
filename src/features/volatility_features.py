from __future__ import annotations

import torch

from src.features.rolling import rolling_std


def compute_volatility_features(bars: torch.Tensor, windows: list[int], epsilon: float) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    features: dict[str, torch.Tensor] = {}
    if close.numel() < 2:
        return {f"volatility_{window}": torch.zeros((), dtype=bars.dtype, device=bars.device) for window in windows}

    log_returns = torch.log((close[1:] / close[:-1]).clamp_min(epsilon))
    for window in windows:
        if log_returns.numel() >= window:
            features[f"volatility_{window}"] = log_returns[-window:].std(unbiased=False)
        else:
            features[f"volatility_{window}"] = torch.zeros((), dtype=bars.dtype, device=bars.device)
    return features


def compute_volatility_features_rolling(
    bars: torch.Tensor, windows: list[int], epsilon: float
) -> dict[str, torch.Tensor]:
    """Vectorized rolling variant of ``compute_volatility_features``.

    Returns a dict of ``(N_bars,)`` tensors.  Each row ``i`` equals what
    the scalar path would return for ``compute_volatility_features(bars[: i + 1], ...)``.
    The one-step log-return sequence is computed once and shared across
    all windows.
    """
    close = bars[:, 3]
    n = close.numel()
    zero_full = torch.zeros(n, dtype=bars.dtype, device=bars.device)
    if n < 2:
        return {f"volatility_{window}": zero_full for window in windows}

    # Per-bar log-return.  log_returns[i] (for i >= 1) = log(close[i]/close[i-1]).
    # At i = 0 the scalar path has no prior bar, so we set it to 0.
    ratios = (close / torch.cat([close[:1], close[:-1]], dim=0)).clamp_min(epsilon)
    log_returns_full = torch.log(ratios)
    log_returns_full[0] = 0.0

    features: dict[str, torch.Tensor] = {}
    for window in windows:
        # Scalar path: ``log_returns[-window:].std()`` over bars 1..n-1
        # (log_returns has length n - 1 because the first bar has no prior).
        # In the rolling formulation: std over a window-sized slice of
        # the one-step log-return series.  The ``>= window`` guard in
        # the scalar path compares against ``log_returns.numel()``
        # (which is n - 1), so the first valid bar for ``volatility_{W}``
        # is at index ``W`` (not ``W - 1``) in the full bars tensor.
        if n - 1 >= window:
            # rolling_std over log_returns_full[1:] with window=W
            # gives a tensor of length (n - 1) - W + 1.  Padded to
            # (n - 1,) on the left, then prepend one zero for the
            # anchor bar (index 0 in the full history).
            rolled = rolling_std(log_returns_full[1:], window, unbiased=False)
            # Prepend one zero for the anchor bar (no prior return).
            full = torch.cat(
                [torch.zeros(1, dtype=bars.dtype, device=bars.device), rolled],
                dim=0,
            )
            features[f"volatility_{window}"] = full
        else:
            features[f"volatility_{window}"] = zero_full
    return features

