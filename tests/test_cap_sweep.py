"""Phase 2 §6.4-B — cap-sweep harness with knee selection."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.core.config import AppConfig
from src.evaluation.cap_sweep import (
    CapSweepPoint,
    pick_knee,
    risk_adjusted_return,
    run_cap_sweep,
)

_T0 = datetime(2026, 6, 1, 14, 30, tzinfo=UTC)


def _bars(n: int = 160, symbol: str = "AAPL"):
    import math

    from src.data.schemas import MarketBar

    return [
        MarketBar(
            symbol=symbol,
            timestamp=_T0 + timedelta(minutes=5 * i),
            open=100.0 + 3.0 * math.sin(i / 9.0),
            high=101.0 + 3.0 * math.sin(i / 9.0),
            low=99.0 + 3.0 * math.sin(i / 9.0),
            close=100.5 + 3.0 * math.sin(i / 9.0),
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(n)
    ]


class TestKneeMath:
    def _point(self, value: float, total_return: float, cvar: float) -> CapSweepPoint:
        metrics = {"total_return": total_return, "cvar_95": cvar}
        return CapSweepPoint(
            value=value, metrics=metrics, risk_adjusted_return=risk_adjusted_return(metrics)
        )

    def test_knee_is_best_risk_adjusted_return(self) -> None:
        points = [
            self._point(3000, 0.010, -0.0010),   # 10.0
            self._point(6000, 0.018, -0.0012),   # 15.0  <-- knee
            self._point(12000, 0.020, -0.0020),  # 10.0
            self._point(24000, 0.021, -0.0040),  # 5.25
        ]
        assert pick_knee(points) == 6000

    def test_tie_goes_to_smaller_cap(self) -> None:
        points = [
            self._point(6000, 0.010, -0.001),
            self._point(12000, 0.020, -0.002),  # same ratio, larger cap
        ]
        assert pick_knee(points) == 6000

    def test_zero_tail_falls_back_to_raw_return(self) -> None:
        assert risk_adjusted_return({"total_return": 0.5, "cvar_95": 0.0}) == 0.5

    def test_empty_points_raise(self) -> None:
        with pytest.raises(ValueError):
            pick_knee([])


class TestSweepValidation:
    def test_unknown_parameter_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown risk parameter"):
            run_cap_sweep(
                AppConfig(), {"AAPL": _bars(40)}, lambda *a: None,
                parameter="not_a_field", values=[1.0],
            )

    def test_non_numeric_parameter_raises(self) -> None:
        with pytest.raises(ValueError, match="not numeric"):
            run_cap_sweep(
                AppConfig(), {"AAPL": _bars(40)}, lambda *a: None,
                parameter="kill_switch", values=[1.0],
            )

    def test_empty_values_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            run_cap_sweep(
                AppConfig(), {"AAPL": _bars(40)}, lambda *a: None,
                parameter="max_position_notional", values=[],
            )

    def test_invalid_swept_value_fails_validation(self) -> None:
        # RiskConfig validators run per arm: a nonsense cap must raise, not
        # silently produce a fictional operating point.
        from src.backtest.policy_factory import build_buy_and_hold_policy

        with pytest.raises(Exception):
            run_cap_sweep(
                AppConfig(),
                {"AAPL": _bars(40)},
                lambda c, b, p: build_buy_and_hold_policy(),
                parameter="max_position_notional",
                values=[-5.0],
            )


def test_sweep_end_to_end_caps_actually_bind() -> None:
    """Real engine, real caps: a tiny position cap must produce smaller
    fills/returns than a huge one, the report carries one point per value,
    and the base config object is never mutated."""
    from src.backtest.policy_factory import build_buy_and_hold_policy

    config = AppConfig()
    config.backtest.fill_timing = "next_open"
    original_cap = config.risk.max_position_notional
    bars = {"AAPL": _bars(160)}

    report = run_cap_sweep(
        config,
        bars,
        lambda arm_config, fold_bars, pipeline: build_buy_and_hold_policy(),
        parameter="max_position_notional",
        values=[500.0, 50000.0],
        seed=7,
    )
    assert [p.value for p in report.points] == [500.0, 50000.0]
    assert config.risk.max_position_notional == original_cap  # untouched
    small, large = report.points
    # Buy-and-hold under a 500-notional cap cannot deploy as much capital.
    assert abs(small.metrics["total_return"]) < abs(large.metrics["total_return"])
    assert report.knee_value in (500.0, 50000.0)
