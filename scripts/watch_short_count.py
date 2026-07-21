"""Emit once when post-fix SHORT proposals reach a decision threshold.

The confidence fix's short-side prediction (P1: EV-pass 48% -> 79%) cannot
be judged on a handful of samples -- check_postfix_signature.py holds its
verdict until MIN_SIDE shorts exist. This announces the moment that
sample is available, with the short-side profile that makes it
interpretable.

Post-fix rows are identified by the ``confidence_semantics`` tag; pre-fix
rows lack the key entirely.

Fires ONCE (marker in logs/.short_count_state), so it is safe to poll.

Usage:
    python scripts/watch_short_count.py [--threshold 25]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

DECISIONS = Path("runs/decisions.jsonl")
STATE = Path("logs/.short_count_state")


def main() -> int:
    parser = argparse.ArgumentParser(description="Announce the post-fix short-sample milestone")
    parser.add_argument("--threshold", type=int, default=25)
    args = parser.parse_args()

    if not DECISIONS.exists() or STATE.exists():
        return 0

    shorts: list[dict] = []
    for line in DECISIONS.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("confidence_semantics") and row.get("proposed_action") == -1:
            shorts.append(row)

    if len(shorts) < args.threshold:
        return 0

    conf = [float(r["confidence"]) for r in shorts if r.get("confidence") is not None]
    approved = sum(1 for r in shorts if r.get("approved"))
    blocked = sum(
        1 for r in shorts if "fractional_short_unsupported" in (r.get("reasons") or [])
    )
    neg_ev = sum(1 for r in shorts if "negative_ev" in (r.get("reasons") or []))
    print(
        f"*** {len(shorts)} POST-FIX SHORTS REACHED -- P1 short-side is now judgeable. "
        f"conf median {st.median(conf):.3f} (min {min(conf):.3f} max {max(conf):.3f}); "
        f"negative_ev {neg_ev}/{len(shorts)} ({neg_ev / len(shorts):.0%}); "
        f"whole-share blocked {blocked}/{len(shorts)}; approved {approved} ***"
    )
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(str(len(shorts)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
