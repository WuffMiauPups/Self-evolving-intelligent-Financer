"""Alert on genuine position OPENINGS in either direction.

Generalises watch_shorts.py: a fill is an "open" when the running net
position crosses zero -- >=0 to <0 is a SHORT open, <=0 to >0 is a LONG
open. Plain closes and size adjustments are ignored.

Longs matter today (2026-07-21) because the post-EV-leg-fix book opened
eight shorts and zero longs; the first long is the signal that the gate
is not stuck one-sided.

Replays the day's fills in TIMESTAMP order (runs/fills.jsonl is not
time-ordered); cursor in logs/.open_watch_state keeps it idempotent.

The replay MUST be seeded with the start-of-day book (logs/.open_watch_seed,
symbol -> qty): positions carry overnight, and a flat-start replay
misclassifies a sell that trims/closes a carried long as a "SHORT OPENED"
(exactly what happened 2026-07-21 13:31-14:00 -- nine false short alerts;
broker truth: zero shorts, the sells were closes of Friday's carried longs).
Rebuild the seed with: broker position qty minus the day's net fill qty.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

FILLS = Path("runs/fills.jsonl")
STATE = Path("logs/.open_watch_state")
SEED = Path("logs/.open_watch_seed")


def main() -> int:
    if not FILLS.exists():
        return 0
    today = datetime.now(UTC).date().isoformat()
    rows = []
    for line in FILLS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("timestamp", "")).startswith(today):
            rows.append(row)
    rows.sort(key=lambda r: r["timestamp"])

    cursor = STATE.read_text(encoding="utf-8").strip() if STATE.exists() else ""
    position: dict[str, float] = defaultdict(float)
    if SEED.exists():
        for symbol, qty in json.loads(SEED.read_text(encoding="utf-8")).items():
            position[symbol] = float(qty)
    events = []
    for row in rows:
        qty = float(row["quantity"]) * (1.0 if row["side"] == "buy" else -1.0)
        before = position[row["symbol"]]
        after = before + qty
        position[row["symbol"]] = after
        if before >= -1e-9 and after < -1e-9:
            events.append((row["timestamp"], "SHORT", row["symbol"], abs(after), float(row["price"])))
        elif before <= 1e-9 and after > 1e-9:
            events.append((row["timestamp"], "LONG", row["symbol"], after, float(row["price"])))

    fresh = [e for e in events if e[0] > cursor]
    for timestamp, side, symbol, size, price in fresh:
        print(
            f"*** {side} OPENED: {symbol} {size:.4f} sh @ {price:.2f} "
            f"({timestamp[11:19]} UTC) ***"
        )
    if fresh:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(max(e[0] for e in fresh), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
