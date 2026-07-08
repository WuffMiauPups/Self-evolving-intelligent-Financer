"""Stylized-facts data-quality checks for historical bar data.

Real equity return series exhibit a small set of robust statistical
regularities ("stylized facts"): near-zero return autocorrelation,
positive slow-decaying |return| autocorrelation (volatility clustering),
excess kurtosis (fat tails), and negative leverage correlation
(corr(r_t, |r_{t+k}|)).  Data that violates them wholesale — or that
contains single-bar moves no equity produces — is usually corrupted:
the canonical failure is an unadjusted stock split showing up as a fake
crash (NVDA's 10:1 split reads as a -90% bar in raw prices).

This module computes those statistics per symbol plus a cross-symbol
return-correlation summary and flags suspected corporate actions / data
errors.  It is advisory: callers log the report, they do not hard-fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import torch

from src.data.schemas import MarketBar

# Any single-bar |return| above this is treated as a suspected
# unadjusted split or data error.  No large-cap equity moves 20% in one
# intraday bar under normal market function.
MAX_ABS_RETURN_THRESHOLD = 0.20

RETURN_ACF_LAGS = 10
LEVERAGE_LAGS = 5

# Statistics need a minimum number of returns to mean anything.
_MIN_RETURNS = 32


@dataclass(frozen=True)
class SymbolFactsReport:
    """Stylized-facts statistics for a single symbol."""

    symbol: str
    bar_count: int
    return_acf: list[float]  # lags 1..RETURN_ACF_LAGS, expected ~0
    abs_return_acf: list[float]  # lags 1..RETURN_ACF_LAGS, expected > 0, slow decay
    excess_kurtosis: float  # expected > 0 (fat tails)
    leverage_corr: list[float]  # corr(r_t, |r_{t+k}|), k=1..LEVERAGE_LAGS, typically < 0
    max_abs_return: float
    flagged_bars: list[tuple[datetime | None, float]] = field(default_factory=list)
    suspicious: bool = False


@dataclass(frozen=True)
class StylizedFactsReport:
    """Aggregate report across all checked symbols."""

    symbols: dict[str, SymbolFactsReport]
    # Off-diagonal cross-symbol return correlation summary (NaN-free;
    # zeros when fewer than two symbols have enough overlapping bars).
    cross_corr_mean: float
    cross_corr_min: float
    cross_corr_max: float
    # (symbol, timestamp, return) triples for every suspected
    # unadjusted split / data error.
    flagged: list[tuple[str, datetime | None, float]]
    suspicious: bool


def _log_returns(closes: torch.Tensor) -> torch.Tensor:
    closes = closes.to(torch.float64).flatten()
    closes = closes.clamp_min(1.0e-12)
    return torch.log(closes[1:] / closes[:-1])


def _autocorrelation(series: torch.Tensor, max_lag: int) -> list[float]:
    """Sample autocorrelation of ``series`` at lags 1..max_lag."""
    n = series.numel()
    centered = series - series.mean()
    denom = float((centered * centered).sum())
    acf: list[float] = []
    for lag in range(1, max_lag + 1):
        if lag >= n or denom <= 0.0:
            acf.append(0.0)
            continue
        acf.append(float((centered[:-lag] * centered[lag:]).sum()) / denom)
    return acf


def _excess_kurtosis(series: torch.Tensor) -> float:
    n = series.numel()
    if n < 4:
        return 0.0
    centered = series - series.mean()
    variance = float((centered * centered).mean())
    if variance <= 0.0:
        return 0.0
    fourth = float((centered**4).mean())
    return fourth / (variance * variance) - 3.0


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or a.numel() != b.numel():
        return 0.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(torch.sqrt((a_c * a_c).sum() * (b_c * b_c).sum()))
    if denom <= 0.0:
        return 0.0
    return float((a_c * b_c).sum()) / denom


def _leverage_correlation(returns: torch.Tensor, max_lag: int) -> list[float]:
    """corr(r_t, |r_{t+k}|) for k=1..max_lag (typically negative)."""
    out: list[float] = []
    abs_returns = returns.abs()
    for lag in range(1, max_lag + 1):
        if lag >= returns.numel():
            out.append(0.0)
            continue
        out.append(_corr(returns[:-lag], abs_returns[lag:]))
    return out


def compute_symbol_facts(
    symbol: str,
    closes: torch.Tensor,
    *,
    timestamps: list[datetime] | None = None,
    max_return_threshold: float = MAX_ABS_RETURN_THRESHOLD,
) -> SymbolFactsReport:
    """Compute stylized-facts statistics for one symbol's close series."""
    returns = _log_returns(closes)
    n = returns.numel()
    if n == 0:
        return SymbolFactsReport(
            symbol=symbol,
            bar_count=int(closes.numel()),
            return_acf=[0.0] * RETURN_ACF_LAGS,
            abs_return_acf=[0.0] * RETURN_ACF_LAGS,
            excess_kurtosis=0.0,
            leverage_corr=[0.0] * LEVERAGE_LAGS,
            max_abs_return=0.0,
        )

    # Corporate-action detector: flag on simple (not log) returns so the
    # threshold reads as the familiar percentage move.
    simple_returns = torch.expm1(returns)
    abs_simple = simple_returns.abs()
    flagged_bars: list[tuple[datetime | None, float]] = []
    for idx in torch.nonzero(abs_simple > max_return_threshold).flatten().tolist():
        # Return index i corresponds to the bar at closes index i + 1.
        ts = timestamps[idx + 1] if timestamps is not None and idx + 1 < len(timestamps) else None
        flagged_bars.append((ts, float(simple_returns[idx])))

    stats_valid = n >= _MIN_RETURNS
    return SymbolFactsReport(
        symbol=symbol,
        bar_count=int(closes.numel()),
        return_acf=_autocorrelation(returns, RETURN_ACF_LAGS) if stats_valid else [0.0] * RETURN_ACF_LAGS,
        abs_return_acf=_autocorrelation(returns.abs(), RETURN_ACF_LAGS) if stats_valid else [0.0] * RETURN_ACF_LAGS,
        excess_kurtosis=_excess_kurtosis(returns) if stats_valid else 0.0,
        leverage_corr=_leverage_correlation(returns, LEVERAGE_LAGS) if stats_valid else [0.0] * LEVERAGE_LAGS,
        max_abs_return=float(abs_simple.max()),
        flagged_bars=flagged_bars,
        suspicious=bool(flagged_bars),
    )


def _cross_symbol_correlation(returns_by_symbol: dict[str, torch.Tensor]) -> tuple[float, float, float]:
    """Mean/min/max off-diagonal correlation over trailing-aligned returns."""
    usable = {s: r for s, r in returns_by_symbol.items() if r.numel() >= _MIN_RETURNS}
    if len(usable) < 2:
        return 0.0, 0.0, 0.0
    min_len = min(r.numel() for r in usable.values())
    aligned = [r[-min_len:] for r in usable.values()]
    corrs: list[float] = []
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            corrs.append(_corr(aligned[i], aligned[j]))
    if not corrs:
        return 0.0, 0.0, 0.0
    return sum(corrs) / len(corrs), min(corrs), max(corrs)


def compute_stylized_facts(
    bars_by_symbol: dict[str, list[MarketBar]],
    *,
    max_return_threshold: float = MAX_ABS_RETURN_THRESHOLD,
) -> StylizedFactsReport:
    """Run the full stylized-facts check over per-symbol bar lists."""
    closes_by_symbol = {
        symbol: torch.tensor([bar.close for bar in bars], dtype=torch.float64)
        for symbol, bars in bars_by_symbol.items()
    }
    timestamps_by_symbol = {
        symbol: [bar.timestamp for bar in bars] for symbol, bars in bars_by_symbol.items()
    }
    return compute_stylized_facts_from_closes(
        closes_by_symbol,
        timestamps_by_symbol=timestamps_by_symbol,
        max_return_threshold=max_return_threshold,
    )


def compute_stylized_facts_from_closes(
    closes_by_symbol: dict[str, torch.Tensor],
    *,
    timestamps_by_symbol: dict[str, list[datetime]] | None = None,
    max_return_threshold: float = MAX_ABS_RETURN_THRESHOLD,
) -> StylizedFactsReport:
    """Tensor-input variant of :func:`compute_stylized_facts`."""
    symbol_reports: dict[str, SymbolFactsReport] = {}
    returns_by_symbol: dict[str, torch.Tensor] = {}
    flagged: list[tuple[str, datetime | None, float]] = []
    for symbol, closes in closes_by_symbol.items():
        timestamps = timestamps_by_symbol.get(symbol) if timestamps_by_symbol else None
        report = compute_symbol_facts(
            symbol,
            closes,
            timestamps=timestamps,
            max_return_threshold=max_return_threshold,
        )
        symbol_reports[symbol] = report
        returns_by_symbol[symbol] = _log_returns(closes)
        flagged.extend((symbol, ts, ret) for ts, ret in report.flagged_bars)
    corr_mean, corr_min, corr_max = _cross_symbol_correlation(returns_by_symbol)
    return StylizedFactsReport(
        symbols=symbol_reports,
        cross_corr_mean=corr_mean,
        cross_corr_min=corr_min,
        cross_corr_max=corr_max,
        flagged=flagged,
        suspicious=bool(flagged),
    )
