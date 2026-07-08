from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from src.core.exceptions import ConfigurationError

# US equity sessions are defined by the exchange in local New York time and
# shift against UTC with DST, so all session comparisons convert to this zone.
MARKET_TZ = ZoneInfo("America/New_York")


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_same_trading_day(left: datetime, right: datetime) -> bool:
    return left.astimezone(UTC).date() == right.astimezone(UTC).date()


def within_session(timestamp: datetime, session_name: str) -> bool:
    """Return True when ``timestamp`` falls inside the named session window.

    Windows are half-open ``[start, end)`` in America/New_York local time so
    every instant belongs to exactly one equity session. Unknown session names
    fail CLOSED by raising ConfigurationError (a typo must never widen the
    trading window; config validators catch this at load time).
    """
    session_name = session_name.lower()
    sessions = session_bounds()
    if session_name not in sessions:
        raise ConfigurationError(
            f"Unknown session name {session_name!r}; known sessions: {sorted(sessions)}"
        )
    start, end = sessions[session_name]
    ts = timestamp.astimezone(MARKET_TZ).time()
    if start < end:
        return start <= ts < end
    # Wrapping window (start >= end): [start, midnight) U [midnight, end).
    # start == end covers the full 24h day (used by 'crypto').
    return ts >= start or ts < end


def session_bounds() -> dict[str, tuple[time, time]]:
    """Session windows in America/New_York local time, half-open [start, end)."""
    return {
        "regular": (time(9, 30), time(16, 0)),
        "premarket": (time(4, 0), time(9, 30)),
        "afterhours": (time(16, 0), time(20, 0)),
        "crypto": (time(0, 0), time(0, 0)),  # start == end -> 24/7
    }


# Canonical session names for config validation (allowed_sessions fields).
SESSION_NAMES: frozenset[str] = frozenset(session_bounds())


def within_any_session(timestamp: datetime, sessions: list[str]) -> bool:
    return any(within_session(timestamp, session_name) for session_name in sessions)


def truncate_timestamp_to_minute(timestamp: datetime) -> datetime:
    return timestamp.astimezone(UTC).replace(second=0, microsecond=0)
