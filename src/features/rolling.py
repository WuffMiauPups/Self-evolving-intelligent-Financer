"""Vectorized rolling-window reductions for the feature pipeline.

The per-call feature functions (``compute_volatility_features`` etc.)
each compute a single scalar via ``tensor[-window:].reduction()``.
That is fine for the live runtime path where we only need one bar's
features per tick.  It is *wasteful* when we need features for every
bar in a multi-year history (backtest / training / memory build) —
each bar redoes ``O(window)`` of work that 99% overlaps with its
neighbours.

This module provides rolling helpers that compute the full
``(N_bars,)`` output in one vectorized pass via ``torch.unfold``.
Each helper matches the scalar-path semantics exactly:

* For indices ``i < window - 1`` (partial window) the output is 0,
  matching the ``if close.numel() >= window: ...; else: zero`` guard
  in the per-call functions.
* For indices ``i >= window - 1`` the output is ``reduction(x[i - window + 1 : i + 1])``.

These helpers are pure and allocation-light.  They accept a contiguous
1-D input and return a contiguous 1-D output of the same length.  The
callers assemble per-bar ``FeatureSnapshot`` objects out of these
precomputed tensors in the batched pipeline path.
"""
from __future__ import annotations

import torch


def _pad_left(values: torch.Tensor, pad_length: int) -> torch.Tensor:
    """Pad ``values`` on the left with ``pad_length`` zeros."""
    if pad_length <= 0:
        return values
    head = torch.zeros(pad_length, dtype=values.dtype, device=values.device)
    return torch.cat([head, values], dim=0)


def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling mean over ``window``.  Output has the same length as ``x``.

    Partial-window positions (``i < window - 1``) are 0, matching the
    scalar path's "insufficient bars" fallback.  When the whole input
    is shorter than the window, returns all-zero.
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return x.clone()
    if n < window:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    rolled = x.unfold(0, window, 1).mean(dim=-1)
    return _pad_left(rolled, window - 1)


def rolling_std(x: torch.Tensor, window: int, *, unbiased: bool = False) -> torch.Tensor:
    """Rolling std over ``window``.  Same conventions as ``rolling_mean``.

    ``unbiased=False`` matches the default used throughout the feature
    pipeline (``tensor.std(unbiased=False)``).
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    if n < window:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    rolled = x.unfold(0, window, 1).std(dim=-1, unbiased=unbiased)
    return _pad_left(rolled, window - 1)


def rolling_first_last_return(
    x: torch.Tensor, window: int, *, epsilon: float
) -> torch.Tensor:
    """``(x[i] / x[i - window + 1]) - 1``, rolling over the full history.

    Matches the scalar path's per-window ``return_{window}`` feature
    which compares the last bar's close to the first bar's close of the
    window.  Partial-window positions are zero.
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    if n < window:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    # ``first = x[i - window + 1]`` → start at x[0], end at x[n - window].
    first = x[: n - window + 1]
    last = x[window - 1 :]
    ratio = last / (first + epsilon)
    rolled = ratio - 1.0
    return _pad_left(rolled, window - 1)


def rolling_log_return(
    x: torch.Tensor, window: int, *, epsilon: float
) -> torch.Tensor:
    """``log(x[i] / x[i - window + 1])`` rolling, with ``clamp_min(epsilon)``.

    Matches the scalar path's per-window ``log_return_{window}`` feature.
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    if n < window:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    first = x[: n - window + 1]
    last = x[window - 1 :]
    rolled = torch.log((last / first).clamp_min(epsilon))
    return _pad_left(rolled, window - 1)


def rolling_close_over_mean(
    x: torch.Tensor, window: int, *, epsilon: float
) -> torch.Tensor:
    """``(x[i] / mean(x[i - window + 1 : i + 1])) - 1`` rolling.

    Matches the per-window ``momentum_{window}`` feature.
    """
    means = rolling_mean(x, window)
    if x.numel() == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    output = (x / (means + epsilon)) - 1.0
    # Partial-window positions must be zero (the scalar path's
    # fallback).  rolling_mean already returns 0 there, so the ratio
    # is ``x / epsilon - 1`` which is not zero — we force-zero those
    # positions to match the scalar semantics.
    if x.numel() >= window > 1:
        output[: window - 1] = 0.0
    elif window > 1:
        output.zero_()
    return output


def rolling_slope(
    x: torch.Tensor, window: int, *, epsilon: float
) -> torch.Tensor:
    """``(x[i] - x[i - window + 1]) / (window + epsilon)`` rolling.

    Matches the per-window ``momentum_slope_{window}`` feature's raw
    slope term (the normalization by the window mean is applied by
    the caller).
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    if n < window:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    first = x[: n - window + 1]
    last = x[window - 1 :]
    rolled = (last - first) / (window + epsilon)
    return _pad_left(rolled, window - 1)


def rolling_first_last_return_clipped(
    x: torch.Tensor, window: int, *, epsilon: float
) -> torch.Tensor:
    """Clipped-window variant of ``rolling_first_last_return``.

    At bar ``i`` the value is ``(x[i] / x[max(0, i - window + 1)]) - 1``.
    For ``i < window - 1`` this uses whatever prior bars exist (the
    full available prefix), matching the scalar regime path's
    ``short_window = min(short_window, close.numel())`` clip instead
    of the zero-fill used by ``rolling_first_last_return``.
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    indices = torch.arange(n, device=x.device)
    start_indices = torch.clamp(indices - (window - 1), min=0)
    starts = x[start_indices]
    result = (x / (starts + epsilon)) - 1.0
    return result


def rolling_std_clipped(
    x: torch.Tensor, window: int, *, unbiased: bool = False
) -> torch.Tensor:
    """Clipped-window variant of ``rolling_std``.

    At bar ``i`` the value is ``std(x[max(0, i - window + 1) : i + 1])``.
    For ``i >= window - 1`` this equals ``rolling_std``; for earlier
    indices it equals the cumulative std over the available prefix,
    matching the scalar regime path's clipped-window semantics.
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1 or n == 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    # Cumulative mean + cumulative E[X^2] → cumulative variance → std.
    # ``unbiased=False`` matches the scalar path throughout the feature
    # pipeline.  torch.std_mean returns (std, mean) but only for the
    # whole tensor, so we derive cumulative values manually with cumsum.
    csum = torch.cumsum(x, dim=0)
    csum_sq = torch.cumsum(x * x, dim=0)
    ns = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    means = csum / ns
    var_cum = (csum_sq / ns) - means * means
    std_cum = torch.sqrt(var_cum.clamp(min=0))
    if n >= window:
        rolled = rolling_std(x, window, unbiased=unbiased)  # already padded to (n,)
        # Replace the padded prefix with cumulative std so bars < window - 1
        # match the scalar clipped-window behaviour.
        result = rolled
        prefix = min(window - 1, n)
        result[:prefix] = std_cum[:prefix]
        return result
    # Fewer than ``window`` total bars: cumulative std across the whole
    # history, matching the scalar path's clip to ``close.numel()``.
    return std_cum


def rolling_mean_clipped(x: torch.Tensor, window: int) -> torch.Tensor:
    """Clipped-window variant of ``rolling_mean``.

    At bar ``i`` the value is ``mean(x[max(0, i - window + 1) : i + 1])``.
    """
    n = x.numel()
    if n == 0:
        return torch.zeros(0, dtype=x.dtype, device=x.device)
    if window <= 1:
        return x.clone()
    csum = torch.cumsum(x, dim=0)
    ns = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    cumulative_means = csum / ns
    if n >= window:
        rolled = rolling_mean(x, window)
        prefix = min(window - 1, n)
        rolled[:prefix] = cumulative_means[:prefix]
        return rolled
    return cumulative_means


def rolling_zscore(
    value: torch.Tensor, window: int, *, epsilon: float
) -> torch.Tensor:
    """Rolling z-score of ``value[i]`` against its own ``window``-length
    history.  ``(value[i] - rolling_mean[i]) / (rolling_std[i] + eps)``.
    Partial-window positions are 0.
    """
    means = rolling_mean(value, window)
    stds = rolling_std(value, window, unbiased=False)
    n = value.numel()
    if n == 0:
        return torch.zeros(0, dtype=value.dtype, device=value.device)
    output = (value - means) / (stds + epsilon)
    if window > 1:
        output[: min(window - 1, n)] = 0.0
    return output
