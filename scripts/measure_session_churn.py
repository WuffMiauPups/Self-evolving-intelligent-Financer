"""End-of-session churn + overnight-carry measurement (paper-week ledgers).

Motivated by day 1 (2026-07-20): TSLA fully exited at -$6.09 at 19:41:05,
re-opened LARGER at a worse price at 19:46:03, carried overnight; the final
30 minutes rebuilt the book to ~$10.6k gross. Measurement only - no gate or
knob changes (MASTER_PLAN 9 step-5 stopping rule).

Reports, per session date (from runs/fills.jsonl, timestamp-sorted - the
file is NOT time-ordered, per-symbol append within a bar):

(a) final-30-minute window (19:30-20:00 UTC): realized PnL of closing legs
    whose most recent same-symbol opening leg is also inside the window
    (avg-price mixing caveat: closes may include shares opened earlier);
    plus notional OPENED in the window and still open at the bell (late-day
    exposure carried overnight).
(b) exit-then-reopen pairs, any time of day: a closing leg (|realized|>0)
    followed within 15 min by a re-entry in the SAME direction just exited.
    Cost of the pattern = crystallized exit PnL + observed re-entry price
    disadvantage + the modeled 6 bps round trip the extra exit/entry paid.
(c) overnight/unmanaged carry (from runs/equity_live.jsonl): first equity
    mark of session N+1 minus last mark of session N. Nothing trades while
    the runtime is down, so the gap is the carried book's unmanaged move
    (plus any pre-launch drift on the next open - stated, not hidden).

Usage:
    python scripts/measure_session_churn.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

FILLS = Path("runs/fills.jsonl")
EQUITY = Path("runs/equity_live.jsonl")
REPORT = Path("runs/session_churn_audit.json")

WINDOW_START = "19:30:00"
SESSION_END = "20:00:00"
REOPEN_WINDOW_SEC = 900.0
ROUND_TRIP_COST_BPS = 6.0  # training.transaction_cost_bps, the modeled round trip
FIRST_PAPER_DAY = "2026-07-01"  # April sessions are pre-C2 relics (frozen equity)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(row)
    return out


def _final_window_stats(day_fills: list[dict]) -> dict:
    window = [f for f in day_fills if WINDOW_START <= f["timestamp"][11:19] < SESSION_END]
    closes = [f for f in window if abs(float(f["realized_pnl"])) > 0.0]
    # Closing legs whose most recent same-symbol OPENING leg is also in-window.
    intra_realized = 0.0
    intra_count = 0
    for close in closes:
        prior_opens = [
            f
            for f in day_fills
            if f["symbol"] == close["symbol"]
            and abs(float(f["realized_pnl"])) == 0.0
            and f["timestamp"] < close["timestamp"]
        ]
        if prior_opens and prior_opens[-1]["timestamp"][11:19] >= WINDOW_START:
            intra_realized += float(close["realized_pnl"])
            intra_count += 1
    # Net shares opened in-window and still open at the bell, valued at fill price.
    carried_notional = 0.0
    for symbol in {f["symbol"] for f in window}:
        legs = [f for f in window if f["symbol"] == symbol]
        net_qty = sum(
            float(f["quantity"]) * (1 if f["side"] == "buy" else -1) for f in legs
        )
        if abs(net_qty) > 1e-9:
            last_price = legs[-1]["price"]
            carried_notional += abs(net_qty) * float(last_price)
    return {
        "window_fills": len(window),
        "window_closes": len(closes),
        "window_realized_total": round(sum(float(f["realized_pnl"]) for f in closes), 4),
        "intra_window_round_trips": intra_count,
        "intra_window_realized": round(intra_realized, 4),
        "opened_in_window_carried_notional": round(carried_notional, 2),
    }


def _exit_reopen_pairs(day_fills: list[dict]) -> list[dict]:
    pairs = []
    by_symbol: dict[str, list[dict]] = {}
    for fill in day_fills:
        by_symbol.setdefault(fill["symbol"], []).append(fill)
    for symbol, legs in by_symbol.items():
        for i, exit_leg in enumerate(legs):
            if abs(float(exit_leg["realized_pnl"])) == 0.0:
                continue  # not a closing leg
            exit_time = datetime.fromisoformat(exit_leg["timestamp"])
            # Re-entry = next leg in the direction just exited (a long exit is
            # a sell; re-entry is a buy) that OPENS (realized == 0).
            reentry_side = "buy" if exit_leg["side"] == "sell" else "sell"
            for later in legs[i + 1 :]:
                gap = (datetime.fromisoformat(later["timestamp"]) - exit_time).total_seconds()
                if gap > REOPEN_WINDOW_SEC:
                    break
                if later["side"] != reentry_side or abs(float(later["realized_pnl"])) > 0.0:
                    continue
                qty = min(float(exit_leg["quantity"]), float(later["quantity"]))
                sign = 1.0 if reentry_side == "buy" else -1.0
                disadvantage = sign * (float(later["price"]) - float(exit_leg["price"])) * qty
                notional = qty * float(exit_leg["price"])
                pairs.append(
                    {
                        "symbol": symbol,
                        "exit_ts": exit_leg["timestamp"][11:19],
                        "reopen_ts": later["timestamp"][11:19],
                        "gap_sec": round(gap, 1),
                        "exit_realized": round(float(exit_leg["realized_pnl"]), 4),
                        "reopen_qty": round(float(later["quantity"]), 4),
                        "reentry_disadvantage": round(disadvantage, 4),
                        "modeled_round_trip_cost": round(
                            notional * ROUND_TRIP_COST_BPS / 10_000.0, 4
                        ),
                    }
                )
                break
    return sorted(pairs, key=lambda p: p["exit_ts"])


def main() -> int:
    fills = sorted(
        (f for f in _rows(FILLS) if f["timestamp"][:10] >= FIRST_PAPER_DAY),
        key=lambda f: f["timestamp"],
    )
    days = sorted({f["timestamp"][:10] for f in fills})

    per_day: dict[str, dict] = {}
    for day in days:
        day_fills = [f for f in fills if f["timestamp"][:10] == day]
        pairs = _exit_reopen_pairs(day_fills)
        per_day[day] = {
            "fills": len(day_fills),
            "final_30min": _final_window_stats(day_fills),
            "exit_reopen_pairs": pairs,
            "exit_reopen_summary": {
                "n": len(pairs),
                "exit_realized_total": round(sum(p["exit_realized"] for p in pairs), 4),
                "reentry_disadvantage_total": round(
                    sum(p["reentry_disadvantage"] for p in pairs), 4
                ),
                "modeled_cost_total": round(
                    sum(p["modeled_round_trip_cost"] for p in pairs), 4
                ),
            },
        }

    # (c) unmanaged carry between consecutive July sessions, from equity marks.
    equity = sorted(
        (r for r in _rows(EQUITY) if r["ts"][:10] >= FIRST_PAPER_DAY),
        key=lambda r: r["ts"],
    )
    eq_days = sorted({r["ts"][:10] for r in equity})
    carries = []
    for prev_day, next_day in zip(eq_days, eq_days[1:]):
        prev_rows = [r for r in equity if r["ts"][:10] == prev_day]
        next_rows = [r for r in equity if r["ts"][:10] == next_day]
        carries.append(
            {
                "from": f"{prev_day} {prev_rows[-1]['ts'][11:19]}",
                "to": f"{next_day} {next_rows[0]['ts'][11:19]}",
                "close_equity": round(float(prev_rows[-1]["equity"]), 2),
                "open_equity": round(float(next_rows[0]["equity"]), 2),
                "unmanaged_carry_pnl": round(
                    float(next_rows[0]["equity"]) - float(prev_rows[-1]["equity"]), 2
                ),
            }
        )

    result = {
        "definition_notes": {
            "final_30min": "19:30-20:00 UTC; intra-window round trips pair each in-window close with its most recent same-symbol opening leg (avg-price mixing caveat: closes may include earlier shares)",
            "exit_reopen": f"closing leg then same-direction re-entry within {REOPEN_WINDOW_SEC:.0f}s; disadvantage = signed re-entry price slip x overlapping qty; modeled cost = {ROUND_TRIP_COST_BPS} bps round trip on that notional",
            "carry": "first equity mark of next session minus last mark of prior session; the runtime is down in between so nothing trades - includes any pre-launch drift on the next open",
            "coverage": "sessions before 2026-07-20 ended before 19:30 UTC (console/machine deaths), so final-30min data exists only where the session reached the close",
        },
        "per_day": per_day,
        "unmanaged_carry": carries,
    }
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n  SESSION CHURN + CARRY AUDIT")
    for day, stats in per_day.items():
        fw = stats["final_30min"]
        er = stats["exit_reopen_summary"]
        print(
            f"  {day}: fills={stats['fills']:>3}  final30: closes={fw['window_closes']:>2} "
            f"realized={fw['window_realized_total']:+8.2f} intraRT={fw['intra_window_round_trips']} "
            f"({fw['intra_window_realized']:+.2f}) carriedNewNotional={fw['opened_in_window_carried_notional']:>9,.0f}"
        )
        print(
            f"            exit->reopen(<=15min): n={er['n']:>2}  exitRealized={er['exit_realized_total']:+8.2f}  "
            f"reentrySlip={er['reentry_disadvantage_total']:+7.2f}  modeledCost={er['modeled_cost_total']:+6.2f}"
        )
    print("  unmanaged carry between sessions:")
    for carry in carries:
        print(
            f"    {carry['from']} -> {carry['to']}: {carry['close_equity']:,.2f} -> "
            f"{carry['open_equity']:,.2f}  ({carry['unmanaged_carry_pnl']:+.2f})"
        )
    print(f"  Report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
