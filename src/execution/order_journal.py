"""Crash-safe JSONL order audit trail."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class OrderJournalEntry:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    client_order_id: str
    broker_order_id: str | None = None
    fee: float = 0.0
    reason: str = ""


class OrderJournal:
    """Append-only JSONL order log with rotation."""

    def __init__(self, path: str | Path = "runs/order_journal.jsonl", max_bytes: int = 50_000_000, max_backups: int = 3):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.max_backups = max_backups
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: OrderJournalEntry) -> None:
        """Best-effort append. Never raises into trade loop."""
        try:
            self._maybe_rotate()
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(asdict(entry), separators=(",", ":"), default=str) + "\n")
        except Exception:
            pass

    def read_all(self) -> list[dict]:
        """Read all entries for analysis."""
        if not self.path.exists():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _maybe_rotate(self) -> None:
        try:
            if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
                return
            backup = self.path.with_suffix(f".{int(time.time())}.bak")
            self.path.rename(backup)
            backups = sorted(self.path.parent.glob(f"{self.path.stem}.*.bak"), reverse=True)
            for old in backups[self.max_backups:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass
