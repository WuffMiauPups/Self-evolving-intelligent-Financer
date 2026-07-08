"""Structured per-tick runtime status writer.

The trade loop writes one JSON-line per tick to ``runs/runtime_status.jsonl``
(default). The GUI's RuntimeService reads the *last* line to display real,
structured operator state instead of grepping the log file for keywords.

Schema (one line; all fields optional except ``ts`` and ``step``):

    {
        "ts":              ISO-8601 string,
        "step":            int,
        "session_id":      str,
        "symbol":          str,
        "equity":          float,
        "fills":           int,
        "safe_to_trade":   bool,
        "last_decision":   {action, confidence, target_fraction, fusion_level} | null,
        "last_risk":       {approved, reasons[], side, quantity} | null,
        "fusion_enabled":  bool,
        "memory_enabled":  bool,
        "frozen_brain":    bool,
    }

Design rules:
- Append-only, one-line-per-tick.
- Best-effort: write failures NEVER raise into the trade loop.
- No Qt / no GUI / no service layer imports.
- Stable schema: new fields are additive; readers must tolerate missing keys.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_RUNTIME_STATUS_PATH = Path("runs/runtime_status.jsonl")


@dataclass
class RuntimeStatusSnapshot:
    ts: datetime
    step: int
    session_id: str = ""
    symbol: str = ""
    equity: float = 0.0
    fills: int = 0
    safe_to_trade: bool = True
    last_decision: dict[str, Any] | None = None
    last_risk: dict[str, Any] | None = None
    fusion_enabled: bool = False
    memory_enabled: bool = False
    frozen_brain: bool = True

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "ts": self.ts.isoformat(),
                "step": int(self.step),
                "session_id": self.session_id,
                "symbol": self.symbol,
                "equity": float(self.equity),
                "fills": int(self.fills),
                "safe_to_trade": bool(self.safe_to_trade),
                "last_decision": self.last_decision,
                "last_risk": self.last_risk,
                "fusion_enabled": bool(self.fusion_enabled),
                "memory_enabled": bool(self.memory_enabled),
                "frozen_brain": bool(self.frozen_brain),
            },
            separators=(",", ":"),
            default=str,
        )


class RuntimeStatusWriter:
    """Best-effort append writer. Failures never raise into the trade loop."""

    def __init__(
        self,
        path: Path | str = DEFAULT_RUNTIME_STATUS_PATH,
        max_bytes: int = 50_000_000,
        max_backups: int = 3,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.max_backups = max_backups
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

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

    def write(self, snap: RuntimeStatusSnapshot) -> None:
        self._maybe_rotate()
        try:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(snap.to_json_line() + "\n")
        except Exception:
            # Observability writer must never block trading.
            pass

    @staticmethod
    def parse_line(line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @classmethod
    def read_latest(cls, path: Path | str = DEFAULT_RUNTIME_STATUS_PATH) -> dict[str, Any] | None:
        """Return the most recent snapshot dict, or None if no readable line."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            # Read the last ~16 KiB and parse the final non-empty line
            with p.open("rb") as fp:
                fp.seek(0, 2)
                size = fp.tell()
                read = min(size, 16 * 1024)
                fp.seek(size - read)
                tail = fp.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        for ln in reversed(tail.splitlines()):
            parsed = cls.parse_line(ln)
            if parsed is not None:
                return parsed
        return None
