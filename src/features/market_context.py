"""Cross-sectional market-context features (RESEARCH_ROADMAP Phase 3).

Per-symbol features only see one symbol's bars; signals and regime
detection were blind to what the MARKET was doing.  This module computes
market-wide state once per tick / bar-index from the multi-symbol bar
set and broadcasts it into every symbol's ``FeatureSnapshot``:

* ``market_spy_return_short`` / ``market_spy_return_medium`` — SPY return
  over the configured short / medium feature windows.
* ``market_spy_vol_medium`` — SPY realized volatility (std of one-step
  log returns) over the medium window.
* ``market_qqq_return_short`` / ``market_qqq_return_medium`` /
  ``market_qqq_vol_medium`` — same for QQQ.
* ``market_breadth_medium`` — fraction of universe symbols whose
  medium-window return is positive.
* ``market_rel_strength_medium`` — the symbol's own medium-window return
  minus SPY's medium-window return (the only non-broadcast market
  feature; computed per symbol inside the pipeline).

Formulas mirror the existing feature-window conventions exactly
(``src/features/price_features.py`` returns, ``volatility_features.py``
realized vol) so the scalar (live) and rolling (batch) paths stay
parity-equivalent — pinned by tests/test_feature_parity.py.

Alignment: the batched context is keyed by exact bar timestamp (union
timeline across symbols), so symbols whose histories start later or have
gaps never receive another bar's market state.  Rows without a match
emit 0.0.

Graceful degradation: when SPY / QQQ are absent from the bar set
(single-symbol tests, walk-forward slices) the derived features emit 0.0
and a ONE-TIME warning — never a crash, never NaN.  Same when the
pipeline is asked to build with ``market_context_enabled`` but no
context object was supplied.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import torch

from src.core.logging import get_logger
from src.data.schemas import MarketBar
from src.features.rolling import rolling_first_last_return, rolling_std

SPY_SYMBOL = "SPY"
QQQ_SYMBOL = "QQQ"

# Broadcast features: identical value for every symbol at a given bar.
MARKET_BROADCAST_FEATURE_NAMES: tuple[str, ...] = (
    "market_breadth_medium",
    "market_qqq_return_medium",
    "market_qqq_return_short",
    "market_qqq_vol_medium",
    "market_spy_return_medium",
    "market_spy_return_short",
    "market_spy_vol_medium",
)
# Per-symbol feature (symbol medium return minus SPY medium return).
MARKET_REL_STRENGTH_FEATURE = "market_rel_strength_medium"

MARKET_FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted((*MARKET_BROADCAST_FEATURE_NAMES, MARKET_REL_STRENGTH_FEATURE))
)

_LOGGER = get_logger("market_context")

# One-time warning registry: degradation warnings (missing SPY/QQQ,
# missing context) fire once per process, not once per bar.
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str, **kwargs: Any) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    _LOGGER.warning(message, **kwargs)


def reset_market_context_warnings() -> None:
    """Clear the one-time warning registry (tests only)."""
    _WARNED_KEYS.clear()


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _closes_from(bars: torch.Tensor | Sequence[MarketBar]) -> torch.Tensor:
    """Extract the 1-D close series from a ``(T, 6+)`` bars tensor or a
    ``MarketBar`` sequence.  Always returns a float32 CPU-or-source-device
    tensor; empty inputs return a zero-length tensor."""
    if isinstance(bars, torch.Tensor):
        if bars.dim() != 2 or bars.shape[-1] < 4:
            raise ValueError(
                f"Expected bars tensor shape (time, 6+), got {tuple(bars.shape)}"
            )
        return bars[:, 3]
    return torch.tensor([float(bar.close) for bar in bars], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Scalar (per-tick) formulas — mirror price_features / volatility_features
# ---------------------------------------------------------------------------


def _window_return(close: torch.Tensor, window: int, epsilon: float) -> float:
    """``(close[-1] / (close[-(window+1)] + eps)) - 1`` under the scalar
    path's ``numel > window`` guard (see compute_price_features)."""
    if close.numel() > window:
        value = (close[-1] / (close[-(window + 1)] + epsilon)) - 1.0
        return float(value.item())
    return 0.0


def _window_vol(close: torch.Tensor, window: int, epsilon: float) -> float:
    """Realized vol: std of one-step log returns over the last ``window``
    returns, under the scalar path's ``log_returns.numel() >= window``
    guard (see compute_volatility_features)."""
    if close.numel() < 2:
        return 0.0
    log_returns = torch.log((close[1:] / close[:-1]).clamp_min(epsilon))
    if log_returns.numel() >= window:
        return float(log_returns[-window:].std(unbiased=False).item())
    return 0.0


# ---------------------------------------------------------------------------
# Rolling (full-history) formulas — mirror the *_rolling feature helpers
# ---------------------------------------------------------------------------


def _rolling_return(close: torch.Tensor, window: int, epsilon: float) -> torch.Tensor:
    """Rolling variant of ``_window_return``: row ``i`` equals the scalar
    value on ``close[: i + 1]``.  Uses the same ``window + 1`` anchor span
    as compute_price_features_rolling."""
    n = close.numel()
    if n > window:
        return rolling_first_last_return(close, window + 1, epsilon=epsilon)
    return torch.zeros(n, dtype=close.dtype, device=close.device)


def _rolling_vol(close: torch.Tensor, window: int, epsilon: float) -> torch.Tensor:
    """Rolling variant of ``_window_vol`` (same construction as
    compute_volatility_features_rolling for a single window)."""
    n = close.numel()
    zero_full = torch.zeros(n, dtype=close.dtype, device=close.device)
    if n < 2:
        return zero_full
    ratios = (close / torch.cat([close[:1], close[:-1]], dim=0)).clamp_min(epsilon)
    log_returns_full = torch.log(ratios)
    log_returns_full[0] = 0.0
    if n - 1 >= window:
        rolled = rolling_std(log_returns_full[1:], window, unbiased=False)
        return torch.cat(
            [torch.zeros(1, dtype=close.dtype, device=close.device), rolled], dim=0
        )
    return zero_full


# ---------------------------------------------------------------------------
# Context containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketContextSnapshot:
    """Market-wide values at a single tick, broadcast to every symbol.

    ``spy_return_medium`` is carried separately as the anchor for the
    per-symbol relative-strength feature.  ``spy_available`` gates that
    feature: when SPY is absent the pipeline emits 0.0 instead of
    treating "no market" as "flat market"."""

    values: dict[str, float]
    spy_return_medium: float
    spy_available: bool


@dataclass
class MarketContextBatch:
    """Market-wide feature columns over the union bar timeline.

    Built once per multi-symbol bar set by ``compute_market_context_batch``
    and passed to ``FeaturePipeline.build_batch`` for every symbol.  Row
    lookup is by exact bar timestamp so per-symbol gaps / late starts
    never smear another row's market state."""

    timeline: list[datetime]
    columns: dict[str, torch.Tensor]  # each (M,), float32
    spy_return_medium: torch.Tensor  # (M,), float32
    spy_row_mask: torch.Tensor  # (M,), bool — rows where SPY has a bar
    spy_available: bool
    index_of: dict[datetime, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index_of:
            self.index_of = {ts: i for i, ts in enumerate(self.timeline)}

    def rows_for(self, timestamps: Sequence[datetime]) -> tuple[torch.Tensor, torch.Tensor]:
        """Map bar timestamps to context rows.

        Returns ``(indices, present)`` where ``indices`` is a clamped
        ``(N,)`` long tensor safe for gathering and ``present`` is the
        ``(N,)`` bool mask of exact-match rows (missing rows must emit
        0.0 — the caller applies the mask)."""
        raw = torch.tensor(
            [self.index_of.get(ts, -1) for ts in timestamps], dtype=torch.long
        )
        present = raw >= 0
        return raw.clamp_min(0), present

    def signature(self) -> dict[str, Any]:
        """Deterministic content signature for cache fingerprinting.

        The batched feature cache key must change whenever the market
        context data changes (e.g. re-fetched SPY bars with the same
        endpoints), so we hash the actual column bytes."""
        hasher = hashlib.sha256()
        for name in sorted(self.columns):
            hasher.update(name.encode("utf-8"))
            hasher.update(self.columns[name].contiguous().cpu().numpy().tobytes())
        hasher.update(self.spy_return_medium.contiguous().cpu().numpy().tobytes())
        hasher.update(self.spy_row_mask.contiguous().cpu().numpy().tobytes())
        return {
            "rows": len(self.timeline),
            "first_ts": self.timeline[0].isoformat() if self.timeline else None,
            "last_ts": self.timeline[-1].isoformat() if self.timeline else None,
            "spy_available": bool(self.spy_available),
            "digest": hasher.hexdigest()[:32],
        }


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def compute_market_context_snapshot(
    bars_by_symbol: Mapping[str, torch.Tensor | Sequence[MarketBar]],
    *,
    short_window: int,
    medium_window: int,
    epsilon: float,
) -> MarketContextSnapshot:
    """Compute the per-tick market context from the current bar set.

    Called ONCE per tick (live) or per index (scalar test paths) with
    every symbol's bar window; the result is broadcast to each symbol's
    ``FeaturePipeline.build`` call.  Missing SPY / QQQ degrade to 0.0
    with a one-time warning."""
    closes_by_symbol = {
        symbol: _closes_from(bars) for symbol, bars in bars_by_symbol.items()
    }

    values: dict[str, float] = {name: 0.0 for name in MARKET_BROADCAST_FEATURE_NAMES}

    spy_close = closes_by_symbol.get(SPY_SYMBOL)
    spy_available = spy_close is not None and spy_close.numel() > 0
    if spy_available:
        values["market_spy_return_short"] = _window_return(spy_close, short_window, epsilon)
        values["market_spy_return_medium"] = _window_return(spy_close, medium_window, epsilon)
        values["market_spy_vol_medium"] = _window_vol(spy_close, medium_window, epsilon)
    else:
        _warn_once(
            "spy_missing",
            "market_context_spy_missing — SPY absent from the bar set; "
            "SPY-derived market features emit 0.0",
        )

    qqq_close = closes_by_symbol.get(QQQ_SYMBOL)
    if qqq_close is not None and qqq_close.numel() > 0:
        values["market_qqq_return_short"] = _window_return(qqq_close, short_window, epsilon)
        values["market_qqq_return_medium"] = _window_return(qqq_close, medium_window, epsilon)
        values["market_qqq_vol_medium"] = _window_vol(qqq_close, medium_window, epsilon)
    else:
        _warn_once(
            "qqq_missing",
            "market_context_qqq_missing — QQQ absent from the bar set; "
            "QQQ-derived market features emit 0.0",
        )

    populated = [close for close in closes_by_symbol.values() if close.numel() > 0]
    if populated:
        positive = sum(
            1 for close in populated if _window_return(close, medium_window, epsilon) > 0.0
        )
        values["market_breadth_medium"] = positive / len(populated)

    return MarketContextSnapshot(
        values=values,
        spy_return_medium=values["market_spy_return_medium"],
        spy_available=bool(spy_available),
    )


def compute_market_context_batch(
    bars_by_symbol: Mapping[str, Sequence[MarketBar]],
    *,
    short_window: int,
    medium_window: int,
    epsilon: float,
) -> MarketContextBatch:
    """Compute market-context columns over the union bar timeline.

    Called ONCE per multi-symbol bar set (backtest / training / memory
    build); every symbol's ``build_batch`` call then gathers its rows by
    timestamp.  Row ``i`` of each column equals what
    ``compute_market_context_snapshot`` returns at that timestamp on the
    same bar histories (window-local formulas), which is what keeps the
    scalar/batch parity invariant intact for market features."""
    sorted_bars_by_symbol = {
        symbol: sorted(bars, key=lambda bar: bar.timestamp)
        for symbol, bars in bars_by_symbol.items()
        if bars
    }
    timeline = sorted(
        {bar.timestamp for bars in sorted_bars_by_symbol.values() for bar in bars}
    )
    index_of = {ts: i for i, ts in enumerate(timeline)}
    m = len(timeline)

    def _zeros() -> torch.Tensor:
        return torch.zeros(m, dtype=torch.float32)

    columns: dict[str, torch.Tensor] = {
        name: _zeros() for name in MARKET_BROADCAST_FEATURE_NAMES
    }
    spy_row_mask = torch.zeros(m, dtype=torch.bool)

    rows_by_symbol: dict[str, torch.Tensor] = {}
    closes_by_symbol: dict[str, torch.Tensor] = {}
    for symbol, bars in sorted_bars_by_symbol.items():
        rows_by_symbol[symbol] = torch.tensor(
            [index_of[bar.timestamp] for bar in bars], dtype=torch.long
        )
        closes_by_symbol[symbol] = torch.tensor(
            [float(bar.close) for bar in bars], dtype=torch.float32
        )

    def _fill_proxy(symbol: str, prefix: str) -> None:
        close = closes_by_symbol[symbol]
        rows = rows_by_symbol[symbol]
        columns[f"{prefix}_return_short"][rows] = _rolling_return(close, short_window, epsilon)
        columns[f"{prefix}_return_medium"][rows] = _rolling_return(close, medium_window, epsilon)
        columns[f"{prefix}_vol_medium"][rows] = _rolling_vol(close, medium_window, epsilon)

    spy_available = SPY_SYMBOL in sorted_bars_by_symbol
    if spy_available:
        _fill_proxy(SPY_SYMBOL, "market_spy")
        spy_row_mask[rows_by_symbol[SPY_SYMBOL]] = True
    else:
        _warn_once(
            "spy_missing",
            "market_context_spy_missing — SPY absent from the bar set; "
            "SPY-derived market features emit 0.0",
        )
    if QQQ_SYMBOL in sorted_bars_by_symbol:
        _fill_proxy(QQQ_SYMBOL, "market_qqq")
    else:
        _warn_once(
            "qqq_missing",
            "market_context_qqq_missing — QQQ absent from the bar set; "
            "QQQ-derived market features emit 0.0",
        )

    # Breadth: fraction of symbols (present at each row) whose
    # medium-window return is positive.  Warmup rows count as
    # non-positive, matching the scalar guard's 0.0.
    positives = torch.zeros(m, dtype=torch.float32)
    present_counts = torch.zeros(m, dtype=torch.float32)
    for symbol in sorted_bars_by_symbol:
        rows = rows_by_symbol[symbol]
        returns = _rolling_return(closes_by_symbol[symbol], medium_window, epsilon)
        positives[rows] += (returns > 0.0).to(torch.float32)
        present_counts[rows] += 1.0
    columns["market_breadth_medium"] = torch.where(
        present_counts > 0,
        positives / present_counts.clamp_min(1.0),
        torch.zeros(m, dtype=torch.float32),
    )

    return MarketContextBatch(
        timeline=timeline,
        columns=columns,
        spy_return_medium=columns["market_spy_return_medium"].clone(),
        spy_row_mask=spy_row_mask,
        spy_available=bool(spy_available),
        index_of=index_of,
    )


# ---------------------------------------------------------------------------
# Pipeline-side feature assembly (called by FeaturePipeline when
# ``features.market_context_enabled`` is true)
# ---------------------------------------------------------------------------


def compute_market_context_features(
    bars_tensor: torch.Tensor,
    *,
    context: MarketContextSnapshot | None,
    medium_window: int,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Scalar-path market features for one symbol's snapshot.

    Always emits every ``market_*`` name (the feature contract must not
    depend on context availability).  Missing context degrades every
    value to 0.0 with a one-time warning."""
    dtype = bars_tensor.dtype
    device = bars_tensor.device

    def _scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=dtype, device=device)

    if context is None:
        _warn_once(
            "context_missing_scalar",
            "market_context_unavailable — market_context_enabled but no "
            "MarketContextSnapshot was supplied; market features emit 0.0",
        )
        return {name: _scalar(0.0) for name in MARKET_FEATURE_NAMES}

    features = {
        name: _scalar(float(context.values.get(name, 0.0)))
        for name in MARKET_BROADCAST_FEATURE_NAMES
    }
    close = bars_tensor[:, 3]
    if context.spy_available and close.numel() > medium_window:
        symbol_return = (close[-1] / (close[-(medium_window + 1)] + epsilon)) - 1.0
        features[MARKET_REL_STRENGTH_FEATURE] = (
            symbol_return - _scalar(float(context.spy_return_medium))
        ).to(dtype)
    else:
        features[MARKET_REL_STRENGTH_FEATURE] = _scalar(0.0)
    return features


def compute_market_context_features_rolling(
    bars_tensor: torch.Tensor,
    timestamps: Sequence[datetime],
    *,
    context: MarketContextBatch | None,
    medium_window: int,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Batch-path market features for one symbol's full bar history.

    Row ``i`` equals what ``compute_market_context_features`` returns for
    the scalar path at bar ``i`` (given a scalar context computed from
    the same bar histories) — the standard rolling-parity invariant."""
    n = bars_tensor.shape[0]
    dtype = bars_tensor.dtype
    device = bars_tensor.device
    zeros = torch.zeros(n, dtype=dtype, device=device)

    if context is None:
        _warn_once(
            "context_missing_batch",
            "market_context_unavailable — market_context_enabled but no "
            "MarketContextBatch was supplied; market features emit 0.0",
        )
        return {name: zeros.clone() for name in MARKET_FEATURE_NAMES}

    indices, present = context.rows_for(timestamps)
    present_f = present.to(device=device)
    features: dict[str, torch.Tensor] = {}
    for name in MARKET_BROADCAST_FEATURE_NAMES:
        gathered = context.columns[name][indices].to(dtype=dtype, device=device)
        features[name] = torch.where(present_f, gathered, zeros)

    close = bars_tensor[:, 3]
    if context.spy_available:
        symbol_return = _rolling_return(close, medium_window, epsilon)
        # Scalar guard: rel strength needs ``numel > medium_window`` bars
        # of symbol history AND a SPY bar at the same timestamp.
        warm = torch.arange(n, device=device) >= medium_window
        spy_present = context.spy_row_mask[indices].to(device=device) & present_f
        spy_return = context.spy_return_medium[indices].to(dtype=dtype, device=device)
        features[MARKET_REL_STRENGTH_FEATURE] = torch.where(
            warm & spy_present, symbol_return - spy_return, zeros
        )
    else:
        features[MARKET_REL_STRENGTH_FEATURE] = zeros.clone()
    return features
