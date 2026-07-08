"""Log file tailer for the GUI Logs tab.

Reads the structured log file produced by the backend (no subprocess parsing).
Filtering is done in-memory by simple keyword match — sufficient for v1.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.gui_services.config_service import load_app_config


@dataclass
class LogEntry:
    timestamp: datetime | None
    level: str
    message: str
    raw: str
    category: str = ""

    @property
    def is_error(self) -> bool:
        return self.level.lower() in {"error", "critical"}

    @property
    def is_warning(self) -> bool:
        return self.level.lower() == "warning"


_CATEGORY_KEYWORDS = {
    "runtime": ("trade_engine", "run_trade_loop", "runtime", "policy"),
    "memory": ("memory", "associative", "esom", "topology"),
    "fusion": ("fusion", "decision_fusion"),
    "risk": ("risk", "risk_manager", "risk_block"),
    "execution": ("execution", "order", "fill", "router"),
    "errors": ("error", "exception", "traceback"),
}


def _classify(line: str) -> str:
    lc = line.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lc:
                return category
    return ""


_ANSI_RX = re.compile(r"\x1b\[[0-9;]*m")

# structlog dev renderer: "2026-04-02T14:04:57Z [info     ] event_name  key=val key2=val2"
_STRUCTLOG_DEV_RX = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+"    # timestamp
    r"\[(\w+)\s*\]\s+"                        # [level]
    r"(\S+)\s*"                               # event name
    r"(.*)?$"                                 # key=value pairs (rest of line)
)


def _format_kv_pairs(kv_str: str) -> str:
    """Format 'key1=val1 key2=val2' into readable lines."""
    if not kv_str or not kv_str.strip():
        return ""
    parts = []
    # Simple split on spaces that precede key= patterns
    for pair in re.split(r"\s+(?=\w+=)", kv_str.strip()):
        pair = pair.strip()
        if pair:
            parts.append(f"  {pair}")
    if not parts:
        return ""
    return "\n" + "\n".join(parts)


def _parse_line(line: str) -> LogEntry:
    line = _ANSI_RX.sub("", line.rstrip("\n\r"))
    # Try JSON first.
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            ts = data.get("timestamp") or data.get("time") or data.get("ts")
            ts_dt: datetime | None = None
            if ts:
                try:
                    ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except ValueError:
                    ts_dt = None
            level = str(data.get("level") or data.get("severity") or "INFO")
            message = str(data.get("message") or data.get("msg") or data.get("event") or line)
            return LogEntry(
                timestamp=ts_dt, level=level, message=message, raw=line, category=_classify(line)
            )
    except (json.JSONDecodeError, ValueError):
        pass
    # Try structlog dev renderer format
    m = _STRUCTLOG_DEV_RX.match(line)
    if m:
        ts_str, level, event, kv_rest = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        ts_dt = None
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            pass
        formatted = event + _format_kv_pairs(kv_rest)
        return LogEntry(
            timestamp=ts_dt,
            level=level.strip(),
            message=formatted,
            raw=line,
            category=_classify(line),
        )
    return LogEntry(timestamp=None, level="INFO", message=line, raw=line, category=_classify(line))


def get_log_path(*, config_paths: list[str] | None = None) -> Path:
    config_paths = config_paths or ["configs/default.yaml"]
    try:
        cfg = load_app_config(*config_paths)
        return Path(getattr(cfg.logging, "file_path", "logs/seif_runtime.log"))
    except Exception:  # noqa: BLE001
        return Path("logs/seif_runtime.log")


_last_log_size: int = 0
_last_log_mtime: float = 0.0
_cached_entries: list = []


def tail(
    *,
    log_path: Path | str | None = None,
    max_bytes: int = 64 * 1024,
    categories: Iterable[str] | None = None,
) -> list[LogEntry]:
    path = Path(log_path) if log_path else get_log_path()
    if not path.exists():
        return []
    global _last_log_size, _last_log_mtime, _cached_entries
    try:
        stat = path.stat()
        if stat.st_size == _last_log_size and stat.st_mtime == _last_log_mtime:
            # File unchanged — return cached entries with filter applied
            if categories:
                wanted = set(categories)
                return [e for e in _cached_entries if not e.category or e.category in wanted]
            return list(_cached_entries)
    except OSError:
        return []
    try:
        size = stat.st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                _ = f.readline()  # discard partial first line
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    entries = [_parse_line(ln) for ln in data.splitlines() if ln.strip()]
    _last_log_size = stat.st_size
    _last_log_mtime = stat.st_mtime
    _cached_entries = list(entries)
    if categories:
        wanted = set(categories)
        entries = [e for e in entries if not e.category or e.category in wanted]
    return entries
