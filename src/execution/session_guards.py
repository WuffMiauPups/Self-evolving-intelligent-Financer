from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.config import RuntimeConfig
from src.core.time_utils import utc_now, within_any_session, within_session
from src.data.schemas import MarketBar
from src.execution.reason_codes import ReasonCode


@dataclass(frozen=True)
class SessionGuardResult:
    allowed: bool
    reason_code: str
    details: dict[str, Any] = field(default_factory=dict)


class SessionGuard:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.last_timestamp_by_symbol: dict[str, datetime] = {}

    def evaluate(
        self,
        *,
        symbol: str,
        bar: MarketBar,
        now: datetime | None = None,
        broker_clock: dict[str, Any] | None = None,
    ) -> SessionGuardResult:
        now_ts = now or utc_now()
        active_sessions = list(self.config.allowed_sessions)
        if self.config.allow_prepost:
            for session_name in ("premarket", "afterhours"):
                if session_name not in active_sessions:
                    active_sessions.append(session_name)

        if active_sessions and not within_any_session(now_ts, active_sessions):
            return SessionGuardResult(
                allowed=False,
                reason_code=ReasonCode.SESSION_DISALLOWED,
                details={"sessions": active_sessions, "timestamp": now_ts.isoformat()},
            )

        if self.config.require_market_open and broker_clock is not None and not self.config.allow_prepost:
            if not bool(broker_clock.get("is_open", False)) and not within_session(now_ts, "regular"):
                return SessionGuardResult(
                    allowed=False,
                    reason_code=ReasonCode.MARKET_CLOSED,
                    details={"clock_open": broker_clock.get("is_open", False)},
                )

        previous_timestamp = self.last_timestamp_by_symbol.get(symbol)
        if previous_timestamp is not None and bar.timestamp < previous_timestamp and self.config.block_on_backwards_time:
            return SessionGuardResult(
                allowed=False,
                reason_code=ReasonCode.BACKWARDS_TIMESTAMP,
                details={
                    "previous_timestamp": previous_timestamp.isoformat(),
                    "bar_timestamp": bar.timestamp.isoformat(),
                },
            )

        data_age = (now_ts - bar.timestamp).total_seconds()
        if data_age > self.config.max_data_staleness_sec:
            self.last_timestamp_by_symbol[symbol] = bar.timestamp
            return SessionGuardResult(
                allowed=False,
                reason_code=ReasonCode.STALE_DATA,
                details={"data_age_sec": data_age},
            )

        if (
            previous_timestamp is not None
            and self.config.state_reset_gap_sec is not None
            and self.config.block_on_data_gap
        ):
            gap_sec = (bar.timestamp - previous_timestamp).total_seconds()
            if gap_sec > self.config.state_reset_gap_sec:
                self.last_timestamp_by_symbol[symbol] = bar.timestamp
                return SessionGuardResult(
                    allowed=False,
                    reason_code=ReasonCode.DATA_GAP_BLOCK,
                    details={"gap_sec": gap_sec, "reset_required": True},
                )

        self.last_timestamp_by_symbol[symbol] = bar.timestamp
        return SessionGuardResult(
            allowed=True,
            reason_code=ReasonCode.STARTUP_OK,
            details={"clock_open": None if broker_clock is None else broker_clock.get("is_open")},
        )

