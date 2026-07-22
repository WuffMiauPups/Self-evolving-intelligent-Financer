# Self-Monitor Design — live feedback on the bot's own trading

Written 2026-07-22. Scaffold in `src/monitoring/self_monitor.py` +
`tests/test_self_monitor.py`; implement on the weekend (trading offline).

## Why this and not online learning

The train/live distribution gap is real: the model is a frozen function of
2022-2026 history trading a market that isn't in that data. But updating
WEIGHTS from live P&L is gradient descent on ~40 noisy trades/day on a
model with no measured edge — it fits noise with money, and it deletes the
one safety property this repo is built on (train offline -> gate -> promote
only if measurably better).

The version of "learn as it trades" that is actually sound:

> **The bot continuously measures its own live performance against its own
> stated expectations, writes the verdict down, and is allowed to respond
> only by becoming MORE conservative.**

Self-awareness without self-modification. Weights never move intraday;
behavior can only shrink toward zero, never grow, flip, or override a
veto. That monotone-conservative contract is what makes this shippable
without a new promotion gate (same §6.2 language as memory damping).

## What it watches (each grounded in something measured this week)

| # | Signal | Question it answers | Grounding |
|---|--------|--------------------|-----------|
| S1 | Rolling win rate / expectancy vs baseline | "Am I doing worse than I should expect?" | Day-3 live: 34.9% vs ~40% expectation — nothing noticed |
| S2 | Live calibration (realized win rate per confidence bucket) | "When I say 0.86, do I win 86%?" | Post-fix conf IS P(resolved class) — finally a checkable claim |
| S3 | Confidence-distribution drift | "Do my own outputs look like they did in validation/backtest?" | Pre-fix pinning went unnoticed for 81k decisions |
| S4 | Churn rate (close→reopen cycles) | "Am I fidgeting instead of deciding?" | §7.1.6: churn measured, deadband fixed it once already |
| S5 | Realized cost vs assumed cost_bps | "Is the EV gate's cost input honest per symbol?" | §7.1.5 per-symbol EV costs (planned, never built) |

All five are computable from streams that ALREADY exist
(`runs/decisions.jsonl` with the semantics tag, `runs/fills.jsonl`).
That is deliberate: **ledger-driven means replayable** — the weekend
implementation can be validated against this week's banked data before it
ever runs live ("would it have flagged day 3?").

## What it may do (bounded response ladder — monotone-conservative)

- **L0 observe** — append a `HealthReport` line to `runs/self_report.jsonl`
  every N decisions + at session end. Diagnostics only. *This is the
  weekend deliverable; ship L0 alone first.*
- **L1 damp** — multiply `target_fraction` by `m ∈ [floor, 1.0]` derived
  from health (mirror of memory damping's contract: shrink only).
- **L2 widen the cost floor** — add a health penalty to the EV gate's
  cost side (more conservative only).
- **L3 stand down** — stop OPENING positions (reuse the existing
  `safe_to_trade` machinery; closes still allowed).

L1–L3 are config-gated OFF and stay off until L0's reports justify them —
per house rules, turning them on is a measured decision, not a default.
The ladder can never make the bot trade more, bigger, or in a flipped
direction. Position-guard exits and kill-switch always win.

## Where it lives

- `src/monitoring/self_monitor.py` — pure state machine, no I/O in the
  metric path, fed by `record_decision()` / `record_fill()`.
- Live wiring (weekend, ~10 lines): call the two record methods where
  `tick_processor` already writes the decision event and the fill; emit
  the report where runtime status is written. SHARED PATH: the backtest
  engine gets the same calls so the monitor's behavior is backtestable
  (§10.4 — same rule that caught the EV-leg bug).
- Config: `SelfMonitorConfig` lives in the module for now (zero surface
  touched while the trader runs); weekend step moves it under `AppConfig`
  as `monitoring.self_monitor`, `enabled: false` default.
- Output: `runs/self_report.jsonl`, append-only, best-effort writer
  (never raises into the trade loop — same contract as DecisionLogWriter).

## Weekend implementation checklist (each step has a verify)

1. **Finish `SelfMonitor.health()`** — S1–S4 are implemented in the
   scaffold; S5 (realized-vs-assumed cost) is a TODO with the fill fields
   named. → verify: `tests/test_self_monitor.py` all green, including the
   currently-xfailed S5 test.
2. **Replay validation** — run `replay_from_ledgers()` over the banked
   week; the report for 2026-07-22 must flag S1 (win-rate surprise) and
   the report for pre-fix days must flag S3 (conf pinned). → verify:
   `scripts/selfmon_replay.py` (write it) prints both flags; numbers go
   in the doc.
3. **Wire L0 live** — two record calls + report emission in
   tick_processor AND backtest engine; config under AppConfig gated off.
   → verify: paper session writes `runs/self_report.jsonl`; full suite
   green; backtest decision_log unchanged byte-for-byte with monitor
   disabled.
4. **Only then** decide about L1 with a scorecard row (backtest with
   damping on vs off, same window) — it is an experiment like any other.

## What this is NOT

- Not a §9.4 arm (no edge claim; stopping rule untouched).
- Not online weight updates — `allow_online_model_updates` stays false.
- Not a replacement for retraining: when S2 says "my 0.86 is really a
  0.52," the fix is the pivot retrain; the monitor's job is to make the
  bot SMALL while it's wrong, and to make the wrongness visible in one
  line instead of a day of ledger archaeology.
