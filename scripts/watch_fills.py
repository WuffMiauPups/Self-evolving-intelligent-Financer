"""Emit newly-appended fills exactly once (paper-week fill watcher).

Idempotent by construction: a high-water-mark timestamp is persisted in
``logs/.fill_watch_state`` and only ever advances, so re-running (or a
duplicated poll) prints nothing new.

The cursor MUST be max(timestamp), never the last row in file order:
``runs/fills.jsonl`` is NOT timestamp-sorted (measured 2026-07-20: 41
out-of-order transitions in 352 fills — the tick processor appends per
symbol within a bar, so a later-stamped fill can be written first). A
last-row cursor regresses below the true max and re-emits the rows in
between on every poll, forever. A line-count cursor is also unsafe (it
re-prints the tail it re-reads).

Also surfaces the two conditions that must never look like silence: a stale
runtime status (trader dead/stalled) and an active halt.

Usage:
    python scripts/watch_fills.py          # print new fills, update cursor
    python scripts/watch_fills.py --init   # set cursor to now, print nothing
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

STATE = Path("logs/.fill_watch_state")
FILLS = Path("runs/fills.jsonl")
STATUS = Path("runs/runtime_status.jsonl")
STALE_SEC = 360


def _rows() -> list[dict]:
    if not FILLS.exists():
        return []
    out = []
    for line in FILLS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("timestamp"):
            out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit new fills once")
    parser.add_argument("--init", action="store_true", help="Set cursor to latest, emit nothing")
    args = parser.parse_args()

    rows = _rows()
    cursor = STATE.read_text(encoding="utf-8").strip() if STATE.exists() else ""

    # High-water mark over ALL rows — never the last row in file order.
    high_water = max((str(row["timestamp"]) for row in rows), default="")

    if args.init:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(high_water, encoding="utf-8")
        print(f"cursor initialised at {high_water or '(empty)'}")
        return 0

    fresh = [row for row in rows if str(row["timestamp"]) > cursor]
    fresh.sort(key=lambda row: str(row["timestamp"]))  # report chronologically
    for row in fresh:
        pnl = float(row.get("realized_pnl", 0.0) or 0.0)
        tag = f"CLOSE pnl={pnl:+.2f}" if abs(pnl) > 0 else "open"
        print(
            f"FILL {row.get('symbol')} {row.get('side')} "
            f"{float(row.get('quantity', 0)):.4f} @ {float(row.get('price', 0)):.2f}  {tag}"
        )
    if fresh and high_water > cursor:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(high_water, encoding="utf-8")

    # Liveness + halt: silence must never be mistaken for health.
    if STATUS.exists():
        age = time.time() - STATUS.stat().st_mtime
        if age > STALE_SEC:
            print(f"ALERT: runtime_status stale {age:.0f}s - trader DEAD or stalled")
        else:
            last = None
            for line in reversed(STATUS.read_text(encoding="utf-8").splitlines()):
                if line.strip():
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    break
            if last is not None and last.get("safe_to_trade") is False:
                print("ALERT: safe_to_trade=false - halt state active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
