"""Best-effort webhook alerting for critical events."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class AlertEvent:
    severity: str  # "info", "warning", "critical"
    event_type: str  # "kill_switch", "circuit_breaker", "account_blocked"
    message: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class WebhookAlerter:
    """Best-effort HTTP POST to configured webhook URL."""

    def __init__(self, webhook_url: str | None = None, timeout_sec: float = 5.0):
        self.webhook_url = webhook_url
        self.timeout_sec = timeout_sec

    def send(self, event: AlertEvent) -> bool:
        if not self.webhook_url:
            return False
        try:
            payload = json.dumps({
                "severity": event.severity,
                "event_type": event.event_type,
                "message": event.message,
                "timestamp": event.timestamp,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=self.timeout_sec)
            return True
        except Exception:
            return False
