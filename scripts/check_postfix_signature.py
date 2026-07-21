"""Flag live divergence from the confidence-fix backtest predictions.

The paired OOS measurement (runs/conf_fix_impact_post.json vs
runs/ev_leg_fix_post.json, incumbent, EV-leg fix constant) predicted a
specific live signature when the fix deploys:

  P1  EV-pass share RISES (long 76.9% -> 83.8%, short 48.4% -> 78.5%)
      => `negative_ev` should fall as a share of decisions
  P2  More shorts survive the gate and reach the broker constraint
      => `fractional_short_unsupported` share RISES
  P3  Confidence stops pinning (pre-fix 0.4850-0.5370, std 0.0095)
      => post-fix spread is far wider
  P4  Fills per decision RISE (backtest: 5,573 -> 6,716, +20.5%)

Pre/post split needs no session bookkeeping: post-fix ledger rows carry
``confidence_semantics``; pre-fix rows lack the key entirely.

Prints NOTHING when the live signature matches. Prints DIVERGENCE lines
only when a prediction is contradicted -- it is wired into a monitor, so
every printed line becomes a notification.

Usage:
    python scripts/check_postfix_signature.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DECISIONS = Path("runs/decisions.jsonl")
# Findings already reported; a standing condition must not re-notify.
SEEN = Path("logs/.postfix_signature_seen")
# Rates are noisy on small samples; judge nothing below this many post-fix rows.
MIN_DECISIONS = 60
# Per-direction rates need their own floor -- 3 shorts prove nothing.
MIN_SIDE = 25
# Rough per-share prices for the whole-share arithmetic (order of magnitude
# is all that matters: the test is "can this size cover one share at all").
PRICE_HINT = {
    "NFLX": 67.0, "MSTR": 102.0, "PLTR": 134.0, "NVDA": 207.0, "AMZN": 248.0,
    "AAPL": 325.0, "GOOGL": 349.0, "TSLA": 375.0, "MSFT": 399.0, "META": 647.0,
    "QQQ": 705.0, "SPY": 745.0,
}
EQUITY_HINT = 97_700.0
# Pre-fix confidence band, measured over 81,099 rows (CONFIDENCE_AUDIT.md).
PREFIX_CONF_MAX = 0.5370
PREFIX_CONF_STD = 0.0095


def _load() -> tuple[list[dict], list[dict]]:
    if not DECISIONS.exists():
        return [], []
    pre, post = [], []
    for line in DECISIONS.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        (post if row.get("confidence_semantics") else pre).append(row)
    return pre, post


def _reason_share(rows: list[dict], reason: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if reason in (r.get("reasons") or [])) / len(rows)


def main() -> int:
    pre, post = _load()
    if len(post) < MIN_DECISIONS:
        return 0

    findings: list[str] = []
    conf = [float(r["confidence"]) for r in post if r.get("confidence") is not None]

    # P3 -- confidence must no longer pin, and must stay a valid probability.
    if conf:
        spread = st.pstdev(conf)
        if max(conf) <= PREFIX_CONF_MAX and spread <= PREFIX_CONF_STD * 2:
            findings.append(
                f"P3 confidence STILL PINNED: max {max(conf):.4f} std {spread:.4f} "
                f"(pre-fix max {PREFIX_CONF_MAX} std {PREFIX_CONF_STD}) "
                "-- fix may not be live in this process"
            )
        bad = [c for c in conf if not (0.0 <= c <= 1.0)]
        if bad:
            findings.append(f"P3 {len(bad)} confidences OUTSIDE [0,1] e.g. {bad[0]}")
        zeros = sum(1 for c in conf if c == 0.0)
        if zeros:
            findings.append(
                f"P3 {zeros} ZERO confidences -- the non-finite-logit fail-closed "
                "guard is firing (fp16 overflow?)"
            )

    # P1/P2 need a pre-fix baseline to compare against. Rates are measured
    # WITHIN a proposal direction, never as a share of all decisions: the
    # per-direction proposal MIX is set by argmax(logits), which this fix
    # does not touch, so it drifts with market regime and would otherwise
    # masquerade as a gate change (false alarm, 2026-07-21 16:30).
    if len(pre) >= MIN_DECISIONS:
        pre_short = [r for r in pre if r.get("proposed_action") == -1]
        post_short = [r for r in post if r.get("proposed_action") == -1]
        pre_long = [r for r in pre if r.get("proposed_action") == 1]
        post_long = [r for r in post if r.get("proposed_action") == 1]

        # P1 -- EV-pass share within each direction should RISE.
        for name, rows_pre, rows_post in (
            ("long", pre_long, post_long),
            ("short", pre_short, post_short),
        ):
            if len(rows_pre) < MIN_SIDE and len(rows_post) < MIN_SIDE:
                continue
            if len(rows_post) < MIN_SIDE:
                continue
            ev_pre = _reason_share(rows_pre, "negative_ev")
            ev_post = _reason_share(rows_post, "negative_ev")
            if ev_post > ev_pre * 1.5 and ev_post > 0.05:
                findings.append(
                    f"P1 {name} negative_ev share ROSE {ev_pre:.1%} -> {ev_post:.1%} "
                    f"(n={len(rows_post)}; backtest predicted EV-pass UP, so this should FALL)"
                )

        # P2 -- a whole-share block on an EXPENSIVE symbol is arithmetic, not
        # a gate change: the model's median short is ~0.13% of equity (~$131,
        # measured 2026-07-21), below one share of all but the cheapest names,
        # so the block rate just tracks which symbols got proposed. Only a
        # block on a symbol whose price the size CAN cover is anomalous.
        affordable_blocked = [
            r for r in post_short
            if "fractional_short_unsupported" in (r.get("reasons") or [])
            and abs(float(r.get("target_fraction", 0.0))) * EQUITY_HINT
            >= float(PRICE_HINT.get(r.get("symbol", ""), 1e9))
        ]
        if affordable_blocked:
            symbols = sorted({r["symbol"] for r in affordable_blocked})
            findings.append(
                f"P2 whole-share block on AFFORDABLE shorts {symbols} "
                f"(x{len(affordable_blocked)}) -- size covers a full share, so "
                "this is not the price floor; check the sizing/flatten path"
            )
        ap_pre = sum(1 for r in pre if r.get("approved")) / len(pre)
        ap_post = sum(1 for r in post if r.get("approved")) / len(post)
        if ap_post < ap_pre * 0.6:
            findings.append(
                f"P4 approval rate FELL {ap_pre:.1%} -> {ap_post:.1%} "
                "(backtest predicted +20% fills)"
            )

    # Always-on safety signals, independent of the predictions.
    halted = Counter()
    for row in post[-200:]:
        for reason in row.get("reasons") or []:
            if reason in ("kill_switch", "daily_loss_limit", "daily_loss_pct_limit",
                          "session_disallowed", "stale_data"):
                halted[reason] += 1
    for reason, count in halted.items():
        findings.append(f"HALT-CLASS reason '{reason}' x{count} in last 200 decisions")

    # A standing condition must announce itself ONCE. Dedupe on the finding's
    # stable prefix (counts and rates drift every poll, the claim does not).
    seen = set()
    if SEEN.exists():
        seen = {ln for ln in SEEN.read_text(encoding="utf-8").splitlines() if ln}
    fresh = [f for f in findings if f.split("(")[0].strip() not in seen]
    for finding in fresh:
        print(f"DIVERGENCE: {finding}")
    if fresh:
        SEEN.parent.mkdir(parents=True, exist_ok=True)
        with SEEN.open("a", encoding="utf-8") as fp:
            for finding in fresh:
                fp.write(finding.split("(")[0].strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
