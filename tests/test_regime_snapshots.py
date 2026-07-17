"""Regime snapshots, offline half (§7.2.2).

Per-session K-means over the regime feature family + Mahalanobis novelty,
consumed by the §4.5.7 backward-transfer slice gate as real regime labels
(replacing the chronological cut). Rails pinned here: determinism for a
given seed, bear/bull separation on synthetic regimes, novelty firing on
a shocked held-out session, identical bucketing for paired logs, and the
chronological fallback when clustering is impossible.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.config import AppConfig, load_config
from src.core.exceptions import ConfigurationError
from src.data.schemas import MarketBar
from src.evaluation.regime_snapshots import (
    SESSION_FEATURE_NAMES,
    RegimeSnapshot,
    build_regime_snapshot,
    build_session_features,
    slice_decision_log_by_regime,
)

_WINDOWS = {"short_window": 5, "medium_window": 20, "long_window": 60, "epsilon": 0.0005}


def _session_bars(
    symbol: str,
    day: datetime,
    *,
    drift_per_bar: float,
    vol: float,
    volume: float,
    n_bars: int = 78,
    seed: int = 0,
) -> list[MarketBar]:
    rng = np.random.default_rng(seed + day.toordinal())
    closes = 100.0 * np.cumprod(1.0 + drift_per_bar + rng.normal(0.0, vol, n_bars))
    bars = []
    for i in range(n_bars):
        close = float(closes[i])
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=day + timedelta(minutes=5 * i),
                open=close,
                high=close * 1.001,
                low=close * 0.999,
                close=close,
                volume=volume,
            )
        )
    return bars


def _two_regime_bars(n_days_each: int = 30) -> dict[str, list[MarketBar]]:
    """Synthetic bear (down-drift, high vol) then bull (up-drift, low vol)."""
    bars: list[MarketBar] = []
    day0 = datetime(2022, 3, 1, 14, 0, tzinfo=UTC)
    for d in range(n_days_each):
        bars += _session_bars(
            "TEST", day0 + timedelta(days=d), drift_per_bar=-0.0012, vol=0.004, volume=5000.0
        )
    day1 = datetime(2024, 5, 1, 14, 0, tzinfo=UTC)
    for d in range(n_days_each):
        bars += _session_bars(
            "TEST", day1 + timedelta(days=d), drift_per_bar=0.0008, vol=0.001, volume=2000.0
        )
    return {"TEST": bars}


def test_session_features_shape_and_order() -> None:
    dates, matrix = build_session_features(_two_regime_bars(5), **_WINDOWS)
    assert len(dates) == 10
    assert matrix.shape == (10, len(SESSION_FEATURE_NAMES))
    assert dates == sorted(dates)
    assert np.isfinite(matrix).all()


def test_kmeans_separates_bear_from_bull_and_is_deterministic() -> None:
    bars = _two_regime_bars(30)
    snapshot_a = build_regime_snapshot(bars, k=2, seed=42, **_WINDOWS)
    snapshot_b = build_regime_snapshot(bars, k=2, seed=42, **_WINDOWS)
    assert snapshot_a is not None and snapshot_b is not None
    assert snapshot_a.labels_by_date == snapshot_b.labels_by_date  # deterministic
    bear_labels = {v for k, v in snapshot_a.labels_by_date.items() if k.startswith("2022")}
    bull_labels = {v for k, v in snapshot_a.labels_by_date.items() if k.startswith("2024")}
    assert bear_labels == {0}  # first-occurrence relabeling: earliest regime is 0
    assert bull_labels == {1}


def test_novelty_fires_on_shocked_heldout_session() -> None:
    snapshot = build_regime_snapshot(_two_regime_bars(30), k=2, seed=42, **_WINDOWS)
    assert snapshot is not None
    # A crash session far outside both regimes: violent drift + 10x vol.
    shock_day = datetime(2025, 8, 5, 14, 0, tzinfo=UTC)
    shocked = {
        "TEST": _session_bars(
            "TEST", shock_day, drift_per_bar=-0.01, vol=0.03, volume=50000.0
        )
    }
    dates, features = build_session_features(shocked, **_WINDOWS)
    assert len(dates) == 1
    assert snapshot.score(features[0]).novel is True
    # An in-distribution bull session is NOT novel.
    calm_day = datetime(2024, 6, 20, 14, 0, tzinfo=UTC)
    calm = {
        "TEST": _session_bars(
            "TEST", calm_day, drift_per_bar=0.0008, vol=0.001, volume=2000.0
        )
    }
    _, calm_features = build_session_features(calm, **_WINDOWS)
    assert snapshot.score(calm_features[0]).novel is False


def test_snapshot_round_trips_through_dict() -> None:
    snapshot = build_regime_snapshot(_two_regime_bars(10), k=2, seed=42, **_WINDOWS)
    assert snapshot is not None
    restored = RegimeSnapshot.from_dict(snapshot.to_dict())
    assert restored.labels_by_date == snapshot.labels_by_date
    assert restored.novelty_threshold == pytest.approx(snapshot.novelty_threshold)
    probe = np.array([0.5, 1.0, 0.0, -0.5])
    assert restored.score(probe).distance == pytest.approx(snapshot.score(probe).distance)


def test_too_few_sessions_returns_none() -> None:
    assert build_regime_snapshot(_two_regime_bars(2), k=6, seed=42, **_WINDOWS) is None


def test_slice_by_regime_pairs_identically_and_drops_unlabeled() -> None:
    labels = {"2026-07-01": 0, "2026-07-02": 1, "2026-07-03": 0}

    def _log(offset_min: int) -> list[dict[str, object]]:
        return [
            {"timestamp": datetime(2026, 7, day, 14, offset_min, tzinfo=UTC), "id": day}
            for day in (1, 2, 3, 4)  # day 4 is unlabeled
        ]

    slices_a, dropped_a = slice_decision_log_by_regime(_log(0), labels)
    slices_b, dropped_b = slice_decision_log_by_regime(_log(30), labels)
    assert dropped_a == dropped_b == 1
    assert [name for name, _ in slices_a] == [name for name, _ in slices_b] == [
        "regime0",
        "regime1",
    ]
    assert [len(records) for _, records in slices_a] == [2, 1]


def test_config_pair_and_validators() -> None:
    config = load_config("configs/default.yaml")
    assert config.evaluation.promotion.backward_transfer_slicing == "regime"
    assert config.evaluation.regime_snapshot_k == 6
    with pytest.raises(ConfigurationError, match="backward_transfer_slicing"):
        AppConfig.model_validate(
            {"evaluation": {"promotion": {"backward_transfer_slicing": "alpha_shape"}}}
        )
    with pytest.raises(ConfigurationError, match="regime_snapshot_k"):
        AppConfig.model_validate({"evaluation": {"regime_snapshot_k": 1}})


def test_gate_uses_regime_slicing_source_contract() -> None:
    """§7.2.2's first consumer: benchmark_backward_transfer_slices must
    consult backward_transfer_slicing and bucket via the shared labels."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "evaluation" / "benchmark.py"
    ).read_text(encoding="utf-8")
    gate_region = source.split("def benchmark_backward_transfer_slices", 1)[1]
    assert "backward_transfer_slicing" in gate_region
    assert "slice_decision_log_by_regime" in gate_region
    assert "backward_transfer_regime_fallback_chronological" in gate_region
