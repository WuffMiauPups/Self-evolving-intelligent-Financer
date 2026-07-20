"""Compact paper-week status line (MASTER_PLAN §9.2 daily review checklist).

Prints one short block: equity vs broker, fills, resolved-trade progress
toward the §9.4 arm-1 gate (300), reason-code distribution (the checklist's
fail signal is negative_ev / stale_data dominance), deadband-reject ratio,
and liveness. Designed to be cheap enough to run on a 15-minute cadence —
the multi-MB decision ledger is tail-read, never fully parsed.

Usage:
    python scripts/paper_status.py            # today's session
    python scripts/paper_status.py --tail-mb 8
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ARM1_GATE = 300


def _last_json_line(path: Path) -> dict | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _tail_rows(path: Path, tail_mb: float) -> list[dict]:
    """Parse only the last tail_mb of a JSONL ledger (torn head line skipped)."""
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fp:
        if size > tail_mb * 1_000_000:
            fp.seek(int(size - tail_mb * 1_000_000))
            fp.readline()  # discard the partial line
        raw = fp.read().decode("utf-8", errors="ignore")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-week status (§9.2)")
    parser.add_argument("--tail-mb", type=float, default=6.0)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    today = datetime.now(UTC).date().isoformat()

    # --- resolved trades: |realized_pnl| > 0 (opening legs are unresolved) ---
    resolved = wins = 0
    for row in _tail_rows(runs / "observations.jsonl", 50.0):
        pnl = float(row.get("realized_pnl", 0.0) or 0.0)
        if abs(pnl) > 0.0:
            resolved += 1
            wins += pnl > 0

    status = _last_json_line(runs / "runtime_status.jsonl") or {}
    equity_row = _last_json_line(runs / "equity_live.jsonl") or {}

    # --- today's decisions: reason mix + deadband ratio ---
    decisions = [
        row for row in _tail_rows(runs / "decisions.jsonl", args.tail_mb)
        if str(row.get("ts", "")).startswith(today)
    ]
    reasons = Counter()
    approved = 0
    for row in decisions:
        if row.get("approved"):
            approved += 1
        for reason in row.get("reasons") or []:
            reasons[str(reason)] += 1
    top = ", ".join(f"{name}:{count}" for name, count in reasons.most_common(4)) or "-"
    deadband = reasons.get("below_min_trade_notional", 0)

    stamp = str(status.get("ts") or equity_row.get("ts") or "?")[:19]
    fills = int(equity_row.get("fills", status.get("fills", 0)) or 0)
    print(
        f"[{stamp}] equity={float(equity_row.get('equity', 0.0)):,.2f} "
        f"realized={float(equity_row.get('realized_pnl', 0.0)):+.2f} "
        f"unreal={float(equity_row.get('unrealized_pnl', 0.0)):+.2f} | "
        f"fills={fills} decisions_today={len(decisions)} approved={approved} "
        f"deadband={deadband} | resolved={resolved}/{ARM1_GATE} "
        f"(arm1 {'READY' if resolved >= ARM1_GATE else f'needs {ARM1_GATE - resolved}'}"
        f", win {wins / resolved:.1%})" if resolved else ""
    )
    print(f"   reasons: {top}")
    print(
        f"   safe_to_trade={status.get('safe_to_trade')} "
        f"step={status.get('step')} session={status.get('session_id')} "
        f"intervention_rate={float(status.get('intervention_rate', 0.0)):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
