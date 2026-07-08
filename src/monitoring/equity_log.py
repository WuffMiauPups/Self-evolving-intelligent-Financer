"""Append-only equity / PnL snapshot writer for the runtime loop.

Writes one JSON-line per tick to ``runs/equity_live.jsonl`` (default). The
GUI's equity_service tails this file to build the live equity chart. The
schema is intentionally tiny and stable so the chart can be back-compatible.

Schema (one line):
    {
        "ts":            ISO-8601 string,
        "equity":        float,
        "realized_pnl":  float,
        "unrealized_pnl":float,
        "fills":         int,
        "session_id":    str,
    }

This module has no dependency on Qt, the GUI, or any service layer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_EQUITY_LOG_PATH = Path("runs/equity_live.jsonl")


@dataclass
class EquitySnapshot:
    ts: datetime
    equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fills: int = 0
    session_id: str = ""

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "ts": self.ts.isoformat(),
                "equity": float(self.equity),
                "realized_pnl": float(self.realized_pnl),
                "unrealized_pnl": float(self.unrealized_pnl),
                "fills": int(self.fills),
                "session_id": self.session_id,
            },
            separators=(",", ":"),
        )


class EquityLogWriter:
    """Best-effort append writer. Failures never raise into the trade loop."""

    def __init__(
        self,
        path: Path | str = DEFAULT_EQUITY_LOG_PATH,
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

    def write(self, snap: EquitySnapshot) -> None:
        self._maybe_rotate()
        try:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(snap.to_json_line() + "\n")
        except Exception:
            # Equity logging is observability, never block trading
            pass

    @staticmethod
    def parse_line(line: str) -> EquitySnapshot | None:
        line = line.strip()
        if not line:
            return None
        try:
            d: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return None
        try:
            return EquitySnapshot(
                ts=datetime.fromisoformat(d["ts"]),
                equity=float(d.get("equity", 0.0)),
                realized_pnl=float(d.get("realized_pnl", 0.0)),
                unrealized_pnl=float(d.get("unrealized_pnl", 0.0)),
                fills=int(d.get("fills", 0)),
                session_id=str(d.get("session_id", "")),
            )
        except (KeyError, ValueError, TypeError):
            return None
