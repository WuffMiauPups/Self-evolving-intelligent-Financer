from __future__ import annotations

import torch

from src.features.rolling import (
    rolling_close_over_mean,
    rolling_first_last_return,
    rolling_log_return,
)


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


def compute_price_features_rolling(
    bars: torch.Tensor, momentum_windows: list[int], epsilon: float
) -> dict[str, torch.Tensor]:
    """Vectorized rolling variant of ``compute_price_features``.

    Returns a dict of ``(N_bars,)`` tensors.  Row ``i`` of each tensor
    equals what ``compute_price_features(bars[: i + 1], ...)`` would
    have returned for the corresponding feature — modulo the
    ``+1`` offset quirk of the scalar path where ``return_{window}``
    compares ``close[-1]`` against ``close[-(window+1)]`` (so the
    window parameter refers to bars back, not window size).

    This is called once per symbol by ``FeaturePipeline.build_batch``
    on the full bar history and the results are then sliced per-bar
    to construct individual ``FeatureSnapshot`` objects.  The wall-
    time win vs the per-bar scalar path is roughly linear in the
    number of bars per window (10–50x for typical configs).
    """
    close = bars[:, 3]
    open_price = bars[:, 0]
    high = bars[:, 1]
    low = bars[:, 2]
    n = close.numel()
    zero_full = torch.zeros(n, dtype=bars.dtype, device=bars.device)
    features: dict[str, torch.Tensor] = {}

    # --- one-step return / log-return ---
    if n > 1:
        prev = torch.cat([close[:1], close[:-1]], dim=0)
        # Match scalar path: element 0 is 0 (no prior bar).
        step_return = (close / prev) - 1.0
        step_return[0] = 0.0
        step_log_return = torch.log((close / prev).clamp_min(epsilon))
        step_log_return[0] = 0.0
        features["return_1"] = step_return
        features["log_return_1"] = step_log_return
    else:
        features["return_1"] = zero_full
        features["log_return_1"] = zero_full

    # --- per-bar range / body / wick (pointwise, no rolling) ---
    features["range_1"] = (high - low) / (close + epsilon)
    features["body_1"] = (close - open_price) / (open_price + epsilon)
    features["upper_wick_1"] = (high - torch.maximum(open_price, close)) / (close + epsilon)
    features["lower_wick_1"] = (torch.minimum(open_price, close) - low) / (close + epsilon)

    # --- per-window return / log-return / momentum ---
    # Scalar-path quirk: ``return_{W}`` uses ``close[-(W+1)]`` as the
    # anchor, so the window effectively compares the current close
    # against the close ``W`` bars ago.  Our rolling helpers produce
    # ``x[i] / x[i - W + 1]`` which compares ``W - 1`` bars.  To
    # exactly match the scalar path we use ``W + 1`` bars as the
    # window for the return/log_return helpers.
    for window in momentum_windows:
        anchor_span = window + 1  # number of bars between compare points, inclusive
        if n > window:
            features[f"return_{window}"] = rolling_first_last_return(
                close, anchor_span, epsilon=epsilon
            )
            features[f"log_return_{window}"] = rolling_log_return(
                close, anchor_span, epsilon=epsilon
            )
            features[f"momentum_{window}"] = rolling_close_over_mean(
                close, window, epsilon=epsilon
            )
        else:
            features[f"return_{window}"] = zero_full
            features[f"log_return_{window}"] = zero_full
            features[f"momentum_{window}"] = zero_full
    return features


# ---------------------------------------------------------------------------
# Technical indicators: RSI(14), ATR(14), MACD(12,26,9)
#
# All smoothing uses SMA (simple moving average) rather than EMA so that
# the value at any bar depends only on the most recent ``period`` bars.
# This is critical for batch/scalar-path equivalence: the batch rolling
# path runs on the full history while the scalar path sees only a
# ``warmup_bars``-sized window.  SMA is window-local so both paths
# produce identical results for any bar with enough lookback.
# ---------------------------------------------------------------------------

_RSI_PERIOD = 14
_ATR_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9

# Minimum bars required by MACD: 26 (slow EMA seed) + 9 (signal seed)
_MACD_MIN_BARS = _MACD_SLOW + _MACD_SIGNAL


def _rolling_sma(x: torch.Tensor, period: int) -> torch.Tensor:
    """Rolling SMA over a 1-D tensor using ``unfold``.

    Returns ``(N,)`` tensor.  Positions ``0 .. period-2`` are set to
    ``x[0]`` (same seed-then-expand convention as the old EMA helper).
    """
    n = x.numel()
    if n < period:
        return x.clone()
    # cumsum trick for O(N) rolling mean
    cs = torch.zeros(n + 1, dtype=x.dtype, device=x.device)
    cs[1:] = x.cumsum(0)
    sma = (cs[period:] - cs[:-period]) / period  # length = n - period + 1
    # Pad the leading positions with the first valid SMA value
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    out[period - 1:] = sma
    out[:period - 1] = sma[0]
    return out


def compute_technical_indicators(
    bars: torch.Tensor, epsilon: float
) -> dict[str, torch.Tensor]:
    """Compute RSI(14), ATR(14), MACD(12,26,9) from OHLCV bars tensor.

    Args:
        bars: tensor of shape (N, 6) with columns [open, high, low, close, volume, spread]
        epsilon: numerical stability constant

    Returns:
        dict of feature_name -> scalar tensor (values for the last bar)
    """
    close = bars[:, 3]
    high = bars[:, 1]
    low = bars[:, 2]
    n = close.numel()
    device = bars.device
    dtype = bars.dtype
    zero = torch.zeros((), dtype=dtype, device=device)

    # ---- RSI(14) ----
    if n > _RSI_PERIOD:
        deltas = close[1:] - close[:-1]
        gains = deltas.clamp(min=0.0)
        losses = (-deltas).clamp(min=0.0)
        avg_gain = _rolling_sma(gains, _RSI_PERIOD)[-1]
        avg_loss = _rolling_sma(losses, _RSI_PERIOD)[-1]
        rs = avg_gain / (avg_loss + epsilon)
        rsi = 1.0 - 1.0 / (1.0 + rs)  # already in [0, 1]
    else:
        rsi = zero

    # ---- ATR(14) ----
    if n > _ATR_PERIOD:
        prev_close = torch.cat([close[:1], close[:-1]])
        tr = torch.maximum(
            high - low,
            torch.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )
        atr_raw = _rolling_sma(tr, _ATR_PERIOD)[-1]
        atr_norm = atr_raw / (close[-1] + epsilon)
    else:
        atr_norm = zero

    # ---- MACD(12, 26, 9) ----
    if n > _MACD_MIN_BARS:
        sma_fast = _rolling_sma(close, _MACD_FAST)
        sma_slow = _rolling_sma(close, _MACD_SLOW)
        macd_line_raw = sma_fast - sma_slow
        signal_line_raw = _rolling_sma(macd_line_raw, _MACD_SIGNAL)
        histogram_raw = macd_line_raw - signal_line_raw
        last_close = close[-1] + epsilon
        macd_line = macd_line_raw[-1] / last_close
        macd_signal = signal_line_raw[-1] / last_close
        macd_histogram = histogram_raw[-1] / last_close
    else:
        macd_line = zero
        macd_signal = zero
        macd_histogram = zero

    return {
        "rsi_14": rsi,
        "atr_14": atr_norm,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
    }


def compute_technical_indicators_rolling(
    bars: torch.Tensor, epsilon: float
) -> dict[str, torch.Tensor]:
    """Vectorized version of ``compute_technical_indicators``.

    Returns dict of feature_name -> ``(N,)`` tensors.  Row ``i``
    equals what ``compute_technical_indicators(bars[:i+1], ...)``
    would return for the last bar.
    """
    close = bars[:, 3]
    high = bars[:, 1]
    low = bars[:, 2]
    n = close.numel()
    device = bars.device
    dtype = bars.dtype
    zero_full = torch.zeros(n, dtype=dtype, device=device)

    # ---- RSI(14) rolling ----
    if n > _RSI_PERIOD:
        deltas = close[1:] - close[:-1]
        gains = deltas.clamp(min=0.0)
        losses = (-deltas).clamp(min=0.0)
        avg_gain = _rolling_sma(gains, _RSI_PERIOD)   # (n-1,)
        avg_loss = _rolling_sma(losses, _RSI_PERIOD)   # (n-1,)
        rs = avg_gain / (avg_loss + epsilon)
        rsi_vals = 1.0 - 1.0 / (1.0 + rs)     # (n-1,)
        rsi_out = torch.zeros(n, dtype=dtype, device=device)
        rsi_out[1:] = rsi_vals
        # Zero out positions without enough history
        rsi_out[: _RSI_PERIOD + 1] = 0.0
    else:
        rsi_out = zero_full

    # ---- ATR(14) rolling ----
    if n > _ATR_PERIOD:
        prev_close = torch.cat([close[:1], close[:-1]])
        tr = torch.maximum(
            high - low,
            torch.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )
        atr_sma = _rolling_sma(tr, _ATR_PERIOD)       # (n,)
        atr_out = atr_sma / (close + epsilon)
        atr_out[: _ATR_PERIOD] = 0.0
    else:
        atr_out = zero_full

    # ---- MACD(12, 26, 9) rolling ----
    if n > _MACD_MIN_BARS:
        sma_fast = _rolling_sma(close, _MACD_FAST)
        sma_slow = _rolling_sma(close, _MACD_SLOW)
        macd_line_raw = sma_fast - sma_slow
        signal_line_raw = _rolling_sma(macd_line_raw, _MACD_SIGNAL)
        histogram_raw = macd_line_raw - signal_line_raw
        close_safe = close + epsilon
        macd_line_out = macd_line_raw / close_safe
        macd_signal_out = signal_line_raw / close_safe
        macd_hist_out = histogram_raw / close_safe
        # Zero out warmup region
        macd_line_out[:_MACD_MIN_BARS] = 0.0
        macd_signal_out[:_MACD_MIN_BARS] = 0.0
        macd_hist_out[:_MACD_MIN_BARS] = 0.0
    else:
        macd_line_out = zero_full
        macd_signal_out = zero_full
        macd_hist_out = zero_full

    return {
        "rsi_14": rsi_out,
        "atr_14": atr_out,
        "macd_line": macd_line_out,
        "macd_signal": macd_signal_out,
        "macd_histogram": macd_hist_out,
    }
