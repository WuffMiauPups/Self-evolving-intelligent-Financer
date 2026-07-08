"""Tests for the stylized-facts data-quality module.

A fabricated 10:1 split discontinuity must be flagged; clean geometric
random-walk bars must not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from src.data.schemas import MarketBar
from src.evaluation.stylized_facts import (
    compute_stylized_facts,
    compute_stylized_facts_from_closes,
    compute_symbol_facts,
)
from src.training.data_loading import _advisory_stylized_facts_check

T0 = datetime(2023, 1, 2, 14, 30, tzinfo=UTC)


def _random_walk_closes(count: int, *, seed: int = 7, vol: float = 0.005) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    returns = torch.randn(count, generator=generator, dtype=torch.float64) * vol
    return 100.0 * torch.exp(torch.cumsum(returns, dim=0))


def _bars_from_closes(symbol: str, closes: torch.Tensor) -> list[MarketBar]:
    bars: list[MarketBar] = []
    for i, close in enumerate(closes.tolist()):
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=T0 + timedelta(minutes=i),
                open=close,
                high=close * 1.001,
                low=close * 0.999,
                close=close,
                volume=1000.0,
                spread=0.01,
            )
        )
    return bars


class TestCorporateActionDetector:

    def test_fabricated_split_discontinuity_is_flagged(self):
        closes = _random_walk_closes(2000)
        closes[1000:] = closes[1000:] / 10.0  # unadjusted 10:1 split
        bars = {"NVDA": _bars_from_closes("NVDA", closes)}

        report = compute_stylized_facts(bars)
        assert report.suspicious
        assert len(report.flagged) == 1
        symbol, timestamp, ret = report.flagged[0]
        assert symbol == "NVDA"
        assert timestamp == T0 + timedelta(minutes=1000)
        assert ret == pytest.approx(-0.9, abs=0.02)
        assert report.symbols["NVDA"].max_abs_return == pytest.approx(0.9, abs=0.02)

    def test_clean_random_walk_is_not_flagged(self):
        bars = {"AAPL": _bars_from_closes("AAPL", _random_walk_closes(2000))}
        report = compute_stylized_facts(bars)
        assert not report.suspicious
        assert report.flagged == []
        facts = report.symbols["AAPL"]
        assert facts.max_abs_return < 0.05
        # IID returns: autocorrelation ~0 at every lag.
        assert all(abs(value) < 0.15 for value in facts.return_acf)


class TestStylizedStatistics:

    def test_volatility_clustering_yields_positive_abs_return_acf(self):
        generator = torch.Generator().manual_seed(11)
        # Alternating calm/volatile regimes of 100 bars each.
        vol = torch.where((torch.arange(2000) // 100) % 2 == 0, 0.002, 0.02).to(torch.float64)
        returns = torch.randn(2000, generator=generator, dtype=torch.float64) * vol
        closes = 100.0 * torch.exp(torch.cumsum(returns, dim=0))

        facts = compute_symbol_facts("SPY", closes)
        assert facts.abs_return_acf[0] > 0.05
        # Regime-mixture returns are fat-tailed.
        assert facts.excess_kurtosis > 0.5

    def test_cross_symbol_correlation_of_identical_series(self):
        closes = _random_walk_closes(500)
        report = compute_stylized_facts_from_closes({"A": closes, "B": closes.clone()})
        assert report.cross_corr_mean == pytest.approx(1.0, abs=1e-9)

    def test_single_symbol_has_zero_cross_correlation_summary(self):
        report = compute_stylized_facts_from_closes({"A": _random_walk_closes(500)})
        assert report.cross_corr_mean == 0.0
        assert report.cross_corr_min == 0.0
        assert report.cross_corr_max == 0.0

    def test_short_series_does_not_crash(self):
        facts = compute_symbol_facts("X", torch.tensor([100.0, 101.0, 100.5]))
        assert facts.bar_count == 3
        assert not facts.suspicious

    def test_empty_series_does_not_crash(self):
        facts = compute_symbol_facts("X", torch.empty(0))
        assert facts.bar_count == 0
        assert facts.max_abs_return == 0.0


class _CapturingLogger:
    def __init__(self):
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs):
        self.warnings.append((event, kwargs))


class TestAdvisoryHook:

    def test_hook_warns_on_split_and_never_raises(self):
        closes = _random_walk_closes(500)
        closes[250:] = closes[250:] / 10.0
        logger = _CapturingLogger()
        _advisory_stylized_facts_check({"NVDA": _bars_from_closes("NVDA", closes)}, logger)
        events = [event for event, _ in logger.warnings]
        assert "stylized_facts_suspicious" in events
        _, payload = logger.warnings[0]
        assert payload["flagged"][0][0] == "NVDA"

    def test_hook_is_silent_on_clean_data(self):
        logger = _CapturingLogger()
        _advisory_stylized_facts_check({"AAPL": _bars_from_closes("AAPL", _random_walk_closes(500))}, logger)
        assert logger.warnings == []
