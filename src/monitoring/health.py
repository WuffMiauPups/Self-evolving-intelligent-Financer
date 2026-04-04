from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HealthState:
    started_at: datetime
    last_tick_at: datetime
    error_count: int = 0
    safe_to_trade: bool = True
    blocked_reason_code: str | None = None
    blocked_at: datetime | None = None
    blocked_details: dict[str, object] = field(default_factory=dict)

    def mark_tick(self, timestamp: datetime) -> None:
        self.last_tick_at = timestamp

    def mark_error(self) -> None:
        self.error_count += 1

    def block_trading(self, reason_code: str, *, timestamp: datetime, details: dict[str, object] | None = None) -> None:
        self.safe_to_trade = False
        self.blocked_reason_code = reason_code
        self.blocked_at = timestamp
        self.blocked_details = details or {}

    def restore_trading(self) -> None:
        self.safe_to_trade = True
        self.blocked_reason_code = None
        self.blocked_at = None
        self.blocked_details = {}
