from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.config import RiskConfig, RuntimeConfig
from src.core.exceptions import ConfigurationError
from src.core.time_utils import SESSION_NAMES, session_bounds, within_any_session, within_session

# ---- DST correctness: 'regular' is 09:30-16:00 America/New_York ----------


def test_winter_est_final_hour_is_regular() -> None:
    # 2026-01-15 is EST (UTC-5): 15:30 ET == 20:30 UTC. The old fixed-UTC
    # bounds (13:30-20:00) wrongly excluded the final trading hour in winter.
    assert within_session(datetime(2026, 1, 15, 20, 30, tzinfo=UTC), "regular") is True


def test_winter_est_premarket_is_not_regular() -> None:
    # 09:00 ET == 14:00 UTC in winter — premarket, not regular.
    ts = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
    assert within_session(ts, "regular") is False
    assert within_session(ts, "premarket") is True


def test_summer_edt_final_hour_is_regular() -> None:
    # 2026-07-15 is EDT (UTC-4): 15:30 ET == 19:30 UTC.
    assert within_session(datetime(2026, 7, 15, 19, 30, tzinfo=UTC), "regular") is True


def test_summer_edt_premarket_is_not_regular() -> None:
    # 09:00 ET == 13:00 UTC in summer.
    ts = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
    assert within_session(ts, "regular") is False
    assert within_session(ts, "premarket") is True


# ---- Half-open boundaries: each instant belongs to exactly one session ----


def test_open_boundary_belongs_to_regular_only() -> None:
    # Exactly 09:30 ET (13:30 UTC in summer).
    ts = datetime(2026, 7, 15, 13, 30, tzinfo=UTC)
    assert within_session(ts, "regular") is True
    assert within_session(ts, "premarket") is False


def test_close_boundary_belongs_to_afterhours_only() -> None:
    # Exactly 16:00 ET (20:00 UTC in summer).
    ts = datetime(2026, 7, 15, 20, 0, tzinfo=UTC)
    assert within_session(ts, "regular") is False
    assert within_session(ts, "afterhours") is True


def test_crypto_session_is_always_open() -> None:
    assert within_session(datetime(2026, 1, 15, 3, 0, tzinfo=UTC), "crypto") is True
    assert within_session(datetime(2026, 7, 15, 23, 59, tzinfo=UTC), "crypto") is True


# ---- Fail closed on unknown session names ---------------------------------


def test_unknown_session_name_fails_closed() -> None:
    ts = datetime(2026, 7, 15, 14, 0, tzinfo=UTC)
    with pytest.raises(ConfigurationError):
        within_session(ts, "regulars")
    with pytest.raises(ConfigurationError):
        within_any_session(ts, ["regulars", "regular"])


def test_session_names_matches_session_bounds() -> None:
    assert SESSION_NAMES == frozenset(session_bounds())


# ---- Config validators reject typos at load time ---------------------------


def test_runtime_config_rejects_unknown_session_name() -> None:
    with pytest.raises(ConfigurationError):
        RuntimeConfig(allowed_sessions=["regulars"])


def test_risk_config_rejects_unknown_session_name() -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(allowed_sessions=["regular", "afterhous"])


def test_configs_accept_known_session_names() -> None:
    runtime = RuntimeConfig(allowed_sessions=["regular", "premarket", "afterhours", "crypto"])
    assert runtime.allowed_sessions == ["regular", "premarket", "afterhours", "crypto"]
    risk = RiskConfig(allowed_sessions=["regular"])
    assert risk.allowed_sessions == ["regular"]
