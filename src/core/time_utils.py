from __future__ import annotations

from datetime import UTC, datetime, time


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_same_trading_day(left: datetime, right: datetime) -> bool:
    return left.astimezone(UTC).date() == right.astimezone(UTC).date()


def within_session(timestamp: datetime, session_name: str) -> bool:
    session_name = session_name.lower()
    ts = timestamp.astimezone(UTC).time()
    sessions = session_bounds()
    start, end = sessions.get(session_name, sessions["crypto"])
    if start <= end:
        return start <= ts <= end
    return ts >= start or ts <= end


def session_bounds() -> dict[str, tuple[time, time]]:
    return {
        "regular": (time(13, 30), time(20, 0)),
        "premarket": (time(8, 0), time(13, 30)),
        "afterhours": (time(20, 0), time(0, 0)),
        "crypto": (time(0, 0), time(23, 59, 59)),
    }


def within_any_session(timestamp: datetime, sessions: list[str]) -> bool:
    return any(within_session(timestamp, session_name) for session_name in sessions)


def truncate_timestamp_to_minute(timestamp: datetime) -> datetime:
    return timestamp.astimezone(UTC).replace(second=0, microsecond=0)
