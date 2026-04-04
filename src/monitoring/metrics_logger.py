from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricsLogger:
    counters: dict[str, int] = field(default_factory=dict)

    def increment(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def increment_reason(self, prefix: str, reason_code: str) -> None:
        self.increment(f"{prefix}.{reason_code}")

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)
