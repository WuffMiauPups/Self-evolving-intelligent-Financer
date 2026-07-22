# Open Fixes + Day-3 Runbook — written 2026-07-21 evening

Everything outstanding after the 2026-07-21 audit/fix day (see
`CONFIDENCE_AUDIT.md` for the day's full findings), plus the exact
morning sequence for Wednesday 2026-07-22. Trading starts 15:30 local
(13:30 UTC market open).

---

## A. Fixes that need to be done

Ordered by when they bite, not by size.

### A1. MSTR bars — RESOLVED 2026-07-22: NO ACTION (was a wrong diagnosis)
`stylized_facts_suspicious` flags two single-bar moves >20% on every
historical load: 2022-06-13 (−27.5%) and 2024-08-05 (−23.9%). The original
A1 hypothesis (unadjusted split → refetch adjusted) is **REFUTED** by the
2026-07-22 pivot repo-sweep:
- `bar_adjustment` already defaults to `"all"` (split+dividend) in both the
  client (`alpaca_client.py:20`) and config (`data.py:49`), threaded
  through every fetch path — the cache **is** split-adjusted.
- Proof: MSTR's 2024-08-08 10:1 split shows **no** price step
  (2024-08-07 close 124.60 → 08-08 close 132.68, continuous).
- The two flagged moves are **real Friday→Monday weekend crash gaps**
  (2022-06-13 the bitcoin-crash Monday; 2024-08-05 the Aug-2024 selloff).
  Adjustment cannot and should not remove them.
- `session_aware_labels` already excludes any label window spanning them,
  so they never enter a training target.
**Fix: none.** No refetch, no cache invalidation. (Also: MSTR's 5-min
thin-print sparsity — 64k rows vs ~90k peers — largely repairs at 30-min
bars, ~13 buckets/day; confirm at the pivot fetch.)

### A2. Train-time Platt fit calibrates against the wrong horizon (at pivot retrain)
`supervised.py` fits the calibrator using
`regression_targets["primary_return"]`, which `targets.py:169` builds
from `max(future_returns)` = **R60** — while the direction head it
calibrates is labeled at H=15. Any future calibrator is fitted against
5-hour outcomes for a 75-minute head. **Fix:** fit against the
primary-horizon return. Touches training machinery → do it as part of the
pivot retrain change-set, not before (it would silently change the next
fitted artifact).

### A3. Calibrator save is non-atomic (hardening, cheap, anytime)
`PlattCalibrator.save()` writes directly to the target path; a killed
training process mid-save leaves a half-written canonical sidecar. The
loader now degrades gracefully (never-raises honored, 2026-07-21), so
this is no longer a crash — but the artifact is still silently lost.
**Fix:** tmp+rename, same as `run_train.py`'s `_promote_platt_sidecar`.

### A4. GUI `platt_fitted` badge lies (ops signal, cheap, anytime)
`runtime_service.py` reports fitted=true if the sidecar file merely
EXISTS — right now it shows the refused stale artifact as active while
the runtime runs uncalibrated. **Fix:** mirror
`load_calibrator_for_checkpoint`'s verdict (semantics + fingerprint).

### A5. Only if ever revived (recorded, zero urgency)
- Seed-ensemble `_merge` averages members' own-class probabilities —
  violates p_resolved_class if arm 2 is ever revived. Correct rule:
  P(winner class) from mean logits.
- `force_directional_flip=true` runtimes would apply calibration
  out-of-support (fit excludes FLAT-argmax rows). Flag is false
  everywhere.

### Done today, listed so nobody re-opens them
Confidence = P(resolved class) + semantics boundary (ledger tag, fit
attestation, loader refusal, refit filter, NaN fail-closed at policy AND
gate) — `4caf1d9`. Sizing asymmetry root-caused as the same one-sided
signal, incumbent-only — `26ea2f6`. Seeded fills-replay watcher +
`make_watch_seed.py`. Divergence checker with per-direction rates +
dedupe. Zero-PnL close race (hydration skip) — shipped 2026-07-20.

---

## B. Day-3 runbook — Wednesday 2026-07-22, ready before 15:30 local

Day 3 is the **first full day on post-fix semantics**. Arm 1 is closed;
the week's remaining purpose is Phase-0 sign-off (unattended stability)
plus live-defect discovery. No experiment is waiting on trade counts.

### Pre-open (any time before ~15:25 local)
1. **Machine on, sleep disabled** (§10.10 — closed consoles and sleep
   killed three runs the week of 07-14).
2. **Rebuild the watcher seed:**
   `python scripts/make_watch_seed.py`
   (one command; prints the carried book — expect 4 longs: AMZN, MSTR,
   PLTR, TSLA ≈ $8.0k gross).
3. **Launch:** double-click `scripts/launch_paper_week.bat`.
   Preflight runs its 11 GO/NO-GO checks, then the trader starts in a
   visible titled console. **That window stays open.**
4. **Expected startup lines (all NORMAL):**
   - `platt_calibrator_stale_semantics ... checkpoints\platt_calibration.json`
     — the guard refusing the pre-fix artifact; correct behavior.
   - `memory_runtime_ready enabled=True` — memory on, calibration path
     active but running uncalibrated passthrough.
   - Reconciliation re-adopting the 4 carried longs.

### What day 3 should look like (post-fix signature, from 468-decision day-2 sample)
- Confidence spread ~0.47–0.95; LONG median ~0.86, SHORT ~0.51.
- EV-pass: long ~98%, short ~80% pre-constraint.
- Shorts executable ONLY on NFLX/MSTR (incumbent sizing ~$131 median);
  whole-share blocks on everything else are arithmetic, not bugs.
- Rejection mix dominated by `no_position_change` + deadband; churn-y
  close-and-reopen cycles on NFLX/META/MSFT/AMZN are the known baseline.

### During the day (assistant session, optional but recommended)
- Re-arm the seeded watcher + `check_postfix_signature.py` monitor
  (silent unless reality contradicts the backtest predictions;
  `logs/.postfix_signature_seen` persists dedupe — delete it to re-arm
  alerts from scratch).
- Anything NEW (not the known signature) gets pinged; routine re-opens do
  not.

### End of day
- **Stop ~21:30 local (19:30 UTC)** via Ctrl+C in the trader console —
  ahead of the final-30-minute window that built ~65% of day-1's
  overnight carry. (Alternative: let it run to the 22:00 close once this
  week to exercise close behavior for Phase-0 sign-off — but do that on a
  day someone is watching equity.)
- News poller keeps running regardless.

### The actual decision on the table (no deadline)
The §9 stopping rule is met; the pivot menu is live:
  (a) **30–60 min bars** — attacks the measured problem directly
      (§7.0.2: breakeven falls from 60–70% to ~54–58%); recommended.
  (b) **News features** — NEWSBOX N0 accruing since 07-19 (117k+
      articles); biggest build.
  (c) **Cost-viable universe** — spread/cost rationale only (the sizing
      argument was withdrawn 2026-07-21 evening — decoupled retrains
      size symmetrically).
Any pivot retrain runs decoupled heads → inherits the semantics fix, the
long-side confidence signal measurement, symmetric 5% sizing, and (after
A1/A2) clean MSTR data and a correctly-fitted calibrator.
