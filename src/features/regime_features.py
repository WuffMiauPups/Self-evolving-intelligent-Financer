from __future__ import annotations

import torch

from src.features.rolling import (
    rolling_first_last_return,
    rolling_first_last_return_clipped,
    rolling_mean,
    rolling_mean_clipped,
    rolling_std,
    rolling_std_clipped,
)


def compute_regime_features(
    bars: torch.Tensor,
    *,
    short_window: int,
    medium_window: int,
    long_window: int,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    volume = bars[:, 4]
    short_window = min(short_window, close.numel())
    medium_window = min(medium_window, close.numel())
    long_window = min(long_window, close.numel())
    short_close = close[-short_window:]
    medium_close = close[-medium_window:]
    long_close = close[-long_window:]
    short_return = (short_close[-1] / (short_close[0] + epsilon)) - 1.0 if short_close.numel() > 1 else torch.zeros((), dtype=bars.dtype, device=bars.device)
    medium_return = (medium_close[-1] / (medium_close[0] + epsilon)) - 1.0 if medium_close.numel() > 1 else torch.zeros((), dtype=bars.dtype, device=bars.device)
    long_return = (long_close[-1] / (long_close[0] + epsilon)) - 1.0 if long_close.numel() > 1 else torch.zeros((), dtype=bars.dtype, device=bars.device)
    short_vol = short_close.std(unbiased=False)
    long_vol = long_close.std(unbiased=False)
    short_volume = volume[-short_window:].mean() if volume.numel() >= short_window else volume.mean()
    long_volume = volume[-long_window:].mean() if volume.numel() >= long_window else volume.mean()
    return {
        "regime_trend_strength": torch.abs(medium_return) / (short_vol + epsilon),
        "regime_volatility_ratio": short_vol / (long_vol + epsilon),
        "regime_volume_ratio": short_volume / (long_volume + epsilon) - 1.0,
        "regime_return_alignment": torch.tanh((short_return + medium_return + long_return) / 3.0),
    }


def compute_regime_features_rolling(
    bars: torch.Tensor,
    *,
    short_window: int,
    medium_window: int,
    long_window: int,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Vectorized rolling variant of ``compute_regime_features``.

    Matches the scalar path's **clipped-window** semantics: at bar ``i``
    with ``i + 1 < W`` the scalar path clips the window to the
    available prefix (``short_close = close[-short_window:]`` after
    ``short_window = min(short_window, close.numel())``) and computes
    the reduction on the clipped slice.  The clipped rolling helpers
    (``rolling_*_clipped`` in ``src/features/rolling.py``) mirror that
    behaviour, so the vectorized output matches the scalar path at
    every bar index — including the partial prefix.

    The only observable difference is floating-point reduction order
    (``torch.unfold`` sums vs per-window ``.std()`` sums): observed
    <= ~2.6e-5 for full windows (pinned at rtol=1e-4 / atol=5e-5 by
    tests/test_feature_parity.py).  In the clipped-prefix region
    (bar index < window - 1) the cumsum-based running std is less
    stable and the drift can reach ~5e-3 on ``regime_price_z_*`` /
    near-constant prices — still fp32 reduction noise, not a semantic
    divergence.
    """
    close = bars[:, 3]
    volume = bars[:, 4]
    n = close.numel()
    zero_scalar_fallback = torch.zeros(n, dtype=bars.dtype, device=bars.device)
    if n == 0:
        return {
            "regime_trend_strength": zero_scalar_fallback,
            "regime_volatility_ratio": zero_scalar_fallback,
            "regime_volume_ratio": zero_scalar_fallback,
            "regime_return_alignment": zero_scalar_fallback,
        }

    short_return = rolling_first_last_return_clipped(close, short_window, epsilon=epsilon)
    medium_return = rolling_first_last_return_clipped(close, medium_window, epsilon=epsilon)
    long_return = rolling_first_last_return_clipped(close, long_window, epsilon=epsilon)

    short_vol = rolling_std_clipped(close, short_window, unbiased=False)
    long_vol = rolling_std_clipped(close, long_window, unbiased=False)

    short_volume = rolling_mean_clipped(volume, short_window)
    long_volume = rolling_mean_clipped(volume, long_window)

    trend_strength = torch.abs(medium_return) / (short_vol + epsilon)
    volatility_ratio = short_vol / (long_vol + epsilon)
    volume_ratio = short_volume / (long_volume + epsilon) - 1.0
    return_alignment = torch.tanh((short_return + medium_return + long_return) / 3.0)

    # Scalar regime path returns raw zero at bar 0 (single-bar case:
    # ``short_close.numel() > 1`` is False → zeroed).  Match that
    # explicitly so the two paths agree at index 0.
    if n > 0:
        trend_strength[0] = 0.0
        volatility_ratio[0] = 0.0
        return_alignment[0] = 0.0
        # volume_ratio at bar 0: scalar path uses short_volume = volume[-1]
        # and long_volume = volume[-1], so ratio - 1 = 0.
        volume_ratio[0] = 0.0

    return {
        "regime_trend_strength": trend_strength,
        "regime_volatility_ratio": volatility_ratio,
        "regime_volume_ratio": volume_ratio,
        "regime_return_alignment": return_alignment,
    }

