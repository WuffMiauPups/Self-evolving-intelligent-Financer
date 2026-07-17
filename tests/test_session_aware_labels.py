"""Session-aware label masking (§9.4 arm-3 precondition).

An H-bar outcome window that crosses a session close contains the
overnight gap — the labeled "move" is mostly gap, not tradeable path.
Samples whose max-horizon window spans more than H×bar_period + 30 min
are dropped from eligibility; the flag is label-affecting and lives in
the sample-cache fingerprint.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from src.core.config import AppConfig, load_config
from src.training.data_module import _filter_session_boundary_indices
from src.training.sample_cache import compute_fingerprint

T0 = datetime(2026, 7, 16, 19, 0, tzinfo=UTC)  # late in session 1


def _bar(ts: datetime) -> SimpleNamespace:
    return SimpleNamespace(timestamp=ts)


def test_cross_session_windows_are_dropped_intra_session_kept() -> None:
    # Session 1: bars every 5 min until 19:55; session 2 resumes 13:30 next day.
    bars = [_bar(T0 + timedelta(minutes=5 * i)) for i in range(12)]  # 19:00..19:55
    bars += [
        _bar(datetime(2026, 7, 17, 13, 30, tzinfo=UTC) + timedelta(minutes=5 * i))
        for i in range(12)
    ]
    max_horizon = 6
    eligible = list(range(0, len(bars) - max_horizon))
    kept, dropped = _filter_session_boundary_indices(
        bars, eligible, max_horizon=max_horizon, bar_seconds=300.0
    )
    # Indices 0..5 have their full 6-bar window inside session 1 (end <= 19:55).
    # Indices 6..11 read across the overnight gap -> dropped. Session-2 heads
    # (12..17) are intra-session again.
    assert kept == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    assert dropped == 6


def test_slack_tolerates_late_bars_but_not_overnight() -> None:
    bars = [_bar(T0 + timedelta(minutes=5 * i)) for i in range(6)]
    # One bar delivered 20 minutes late — within the 30-min slack.
    bars.append(_bar(T0 + timedelta(minutes=5 * 6 + 20)))
    kept, dropped = _filter_session_boundary_indices(
        bars, [0], max_horizon=6, bar_seconds=300.0
    )
    assert kept == [0] and dropped == 0


def test_flag_is_label_affecting_in_the_cache_fingerprint() -> None:
    config_on = AppConfig()
    config_off = AppConfig.model_validate({"training": {"session_aware_labels": False}})
    bars: dict = {}
    assert compute_fingerprint(config=config_on, bars_by_symbol=bars) != compute_fingerprint(
        config=config_off, bars_by_symbol=bars
    )


def test_config_pair_defaults_on() -> None:
    assert load_config("configs/default.yaml").training.session_aware_labels is True
    assert AppConfig().training.session_aware_labels is True


def test_dataset_build_consults_the_flag_source_contract() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "training" / "data_module.py"
    ).read_text(encoding="utf-8")
    prepare_region = source.split('eligible_indices = list(range(eligible_start, eligible_stop))', 2)[-1]
    assert "config.training.session_aware_labels" in source
    assert "_filter_session_boundary_indices(" in prepare_region
