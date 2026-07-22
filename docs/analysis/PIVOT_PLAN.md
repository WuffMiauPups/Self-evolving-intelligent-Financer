# PIVOT PLAN — 30-minute bars, baseline-first, three learning loops

Written 2026-07-22. Authored from a 125-fact repo sweep (workflow
wf_b774632c-d88); every non-obvious claim below traces to a file:line in
this repo. This is the execution plan for the strategic pivot discussed
2026-07-22: the §9 stopping rule is MET (all four 5-min/OHLCV arms
negative), so the thesis changes — coarser bars where the cost math works,
a dumb-model-first discipline so complexity has to earn its place, and the
continuous-feedback loops that make "self-evolving" more than a name.

Nothing here is executed yet. This is the map.

---

## 0. The decisions this plan makes (and why) — read first

Three forks the research closed. If you disagree with any, the plan branches.

### 0.1 Timeframe = **30Min**, not 60Min (at least first)
`1Hour` is a worse first step on four measured counts:
- **Data starvation:** at 60Min, session-aware labels leave H=10 and H=15
  with **zero** training samples; even H=5 keeps only ~37k (vs ~131k at
  30Min, vs ~1.03M today). 30Min is ~8× less data than today; 60Min is
  ~28× less.
- **Regime machinery breaks:** `regime_snapshots._MIN_SESSION_BARS=12`
  (hardcoded, comment "a full session has ~78" — 5-min-denominated). A
  60-min session is ~6.5 bars → every session dropped → regime
  backward-transfer degrades to fallback. 30Min = 13 bars/session, passes
  by one.
- **Bucket alignment:** live buckets floor the UNIX epoch, so 60-min
  buckets align to clock hours and the 13:30-UTC open lands mid-bucket;
  30-min buckets align cleanly to the open.
- **Spelling trap:** the installed alpaca-py rejects `TimeFrame(60, Minute)`
  ("amounts between 1-59"); 60-min must be spelled `1Hour`. 30-min is
  `30Min`. (evidence: alpaca timeframe.py:86; alpaca_client.py:28-33)

**60Min stays on the menu as a follow-up** if 30Min shows signal — the
cost math is even better there (H=5 = 300min = the audit's H=60 row,
breakeven 53.9%). But prove the thesis where the data exists first.

### 0.2 Horizon = **short (primary ≈ 5 bars), intraday-only**
The horizon-cost audit's whole point maps mechanically: a 30-min-bar
**H=5 = 150 min** window is the audit's **H=30 row — median clearing move
53.2 bps vs 6 bps round-trip cost, breakeven 55.6%**, a ~2.1× better
move/cost ratio than the 5-min H=6 the current model trades. That is the
entire reason to pivot. But `session_aware_labels` (default on, drops any
label window spanning > H×bar_period + 1800 s) means:
- H=5 (150 min): keeps 131,671 of 206,270 samples (36% drop) — **usable**
- H=10 (300 min): keeps 58,695 (71% drop) — thin
- H=15 (450 min): keeps 4,705 (98% drop) — **empty**

So the pivot's horizon stack must be **short**: `forward_horizons` around
`[1, 3, 5]`, `primary_direction_horizon: 5`, auxiliary at most `[8]`. This
is a deliberate restatement, not an inheritance — today's `[1,5,15]` +
`auxiliary [30,60]` would train on an empty dataset at 30Min.
(evidence: data_module.py:86-109, 322-336; the sample-count replay in the
research)

### 0.3 **Intraday-only** (EOD flatten), NOT overnight carry — I'm reversing my earlier lean
On 2026-07-22 I said I "lean B, managed overnight carry." **The research
overturns that and I'm correcting it.** The reason is not risk appetite,
it's coherence:
- `session_aware_labels` trains the model **only on windows that do not
  cross the overnight gap** — the model is intraday *by construction*.
  Trading it overnight runs it on a regime its labels never contained.
- To train on overnight-spanning moves you would DISABLE the session mask,
  which reintroduces the overnight gap as label noise — the exact thing
  the mask exists to remove (MSTR's two >20% weekend gaps are handled by
  it).
- Overnight carry is the **measured dominant uncontrolled P&L driver**
  (equity gap −$220 / +$84 across runtime-down periods, ~100× the churn
  cost; MASTER_PLAN 2026-07-21).

So the coherent design is **intraday-only with an end-of-day flatten** —
which does **not exist** today (grep for flatten/eod/market_close finds
only the P&L position guard and shutdown order-cancel; shutdown explicitly
does NOT close positions). That's new machinery, scoped in Phase 4.
(evidence: trade_engine.py:28-119; the EOD-absent grep; MASTER_PLAN:378)

**If you want overnight carry anyway**, that's a different thesis (swing
trading on daily-ish bars) and a different plan — say so and I'll rewrite,
but it is not "30-min intraday with the current label machinery."

---

## 1. Success criteria & the new stopping rule (define before building)

This thesis gets the same discipline that killed the last one — no
open-ended tuning.

**Edge milestone (unchanged form, §9 step 5):** an OOS scorecard row with
**positive net expectancy AND excess-vs-buy-and-hold > 0**, under the §2.0
three-seed paired protocol, on 30-min data.

**Noise floor first:** re-read the floor at 30Min (seeds 7/13/29) BEFORE
any candidate claim — the §2.0 rule is "on the same data path," and the
0.92pt/2.00pt floor was measured on 5-min data. No claim is interpretable
until the 30-min floor exists.

**Baseline-first gate (new, the heart of the "revamp"):** the 15M brain
must **beat a logistic-regression / tiny-MLP baseline by more than the
30-min noise floor, 3/3 seeds paired**, on val balanced accuracy AND on
OOS expectancy. If it does not, the brain's complexity is unpaid — ship
the baseline and shrink the model (this is the measured-revamp, §3).

**Stopping rule (new):** if the 30-min retrain does not reach the edge
milestone within the planned seed sweeps AND the baseline is not beaten,
30-min OHLCV is declared exhausted too → the next fork is 60Min (cheap,
data permitting) or features-beyond-OHLCV (NEWSBOX, already accruing) —
NOT more knob-turning at 30Min.

---

## 2. Phase plan (each phase: steps → verify → kill criteria)

Phases are ordered so the cheapest disqualifying test runs first. **Do not
start Phase N+1 until Phase N's verify passes.**

### PHASE 0 — Pre-work that needs no retrain (do this week, trading stays live)
Goal: clear the ground so the pivot retrain is one clean action.

0.1 **Correct the MSTR "fix" (A1) — it was wrong.** My FIXES_AND_DAY3_PLAN
A1 said "suspected unadjusted split, refetch adjusted." **Research
refutes it:** `bar_adjustment` already defaults to `"all"` everywhere and
is threaded through every fetch; the cache IS split-adjusted (MSTR's
2024-08 10:1 split shows no price step). The two >20% moves are **real
Friday→Monday crash gaps** (2022-06-13 −27.5%, 2024-08-05 −23.9%), already
handled by the session mask. → **verify:** update FIXES_AND_DAY3_PLAN A1
to "no action; discontinuities are genuine gaps, masked." No code.

0.2 **Deletion-audit safe batches** (independent of the pivot): execute
A1 (whole dead modules ~1,020 LOC), A2 (dead functions ~330), A3 (unused
imports), B1 (delete-with-tests, porting the feature-contract assertion
first), D-delete (`compare_runs.py`, `run_backtest_parallel.py`),
D-archive (arm-3/4 launchers + `path_label_oracle_test.py` + the two
closed-arm configs `horizon_30.yaml`/`risk_target_norm.yaml`), E-safe
(stale root logs, `.pytest_cache`, egg-info, 2 memory backups). Run the
plan's verification recipe per batch (pytest + re-grep). → **verify:**
suite green after each batch; `git status` clean; DELETION_AUDIT_PLAN
batch checkboxes ticked. **Defer A4/F** (config-key deletions) — they
couple to `paper_trade.yaml` under `extra='forbid'` while paper trading is
live; they ride the pivot retrain commit (Phase 3).

0.3 **Commit or gitignore** `scripts/launch_paper_week_detached.vbs` and
`prompts/audit_unwanted_changes.md` (both untracked, no git safety net).

0.4 **Fast loop L0 wiring** (the self-monitor, weekend project already
scaffolded): finish S5, write `scripts/selfmon_replay.py`, wire
`record_decision`/`record_fill`/report-emit into `tick_processor` at the
exact named sites AND the backtest engine (shared path). Config under
`AppConfig` as `monitoring.self_monitor`, `enabled: false`. → **verify:**
`tests/test_self_monitor.py` green incl. un-xfailed S5; replay flags S3 on
2026-07-20; backtest decision_log byte-identical with monitor disabled;
full suite green. **This is independent of the timeframe and de-risks the
biggest new subsystem before the pivot.**

Kill criteria: none — this is all reversible prep.

### PHASE 1 — Data acquisition (the pivot's first irreversible cost)
Goal: 4.5y of real 30-min bars for 13 symbols, cached.

1.1 Create `configs/experiments/timeframe_30min.yaml` overlay (pattern:
`horizon_30.yaml`) setting **only** `data.timeframe: "30Min"` for the
training/fetch side. → **verify:** `load_config` with the overlay does not
raise (it won't, because runtime.timeframe stays 5Min so the stale_data
validator is satisfied for the fetch step).

1.2 Fetch with live Alpaca credentials:
`python scripts/fetch_historical.py --real-data --config configs/default.yaml --config configs/experiments/timeframe_30min.yaml`
(it reads `config.data.timeframe` at line 117; there is **no** offline
5→30 resampler, so this is a real broker fetch). → **verify:**
`artifacts/historical/{SYM}_30Min.pt` exists for all 13; row counts land
near the predicted ~14.7k regular-session buckets/symbol (206k total);
spot-check MSTR consolidates to ~13 bars/day (the 5-min thin-print
largely repairs at 30-min).

1.3 **Re-measure what the docs cannot carry over:** run
`scripts/audit_horizon_costs.py` on the real 30-min fetch for per-symbol
P(clear)/breakeven at 30-min entry grids, and record true per-bar move
from aggregated OHLC (not 5-min close deltas). → **verify:**
`runs/horizon_cost_audit_30min.json` written; per-symbol breakeven table
in the plan. (The 5-min per-symbol table only exists at one horizon.)

Kill criteria: if the real 30-min per-symbol breakevens are materially
WORSE than the mechanical H=30-row prediction (e.g. cheap symbols still
> 60% breakeven at H=5), the cost thesis is weaker than believed — pause
and reconsider 60Min before spending training compute.

### PHASE 2 — The atomic config switch + noise floor
Goal: a loadable, self-consistent 30-min config, and the floor to judge
against.

2.1 **One atomic edit** (nothing cross-validates most of these — the
research found exactly ONE load-time validator on timeframe). All in the
30-min config stack:
| key | file(s) | 5Min → 30Min | failure mode if forgotten |
|-----|---------|--------------|---------------------------|
| `data.timeframe` | default.yaml:47 | `5Min`→`30Min` | wrong bars |
| `runtime.timeframe` | default.yaml:88 | `5Min`→`30Min` | wrong live cadence |
| `backtest.seconds_per_bar` | default.yaml:265 | `300.0`→`1800.0` | **session mask drops all samples** |
| `backtest.bars_per_year` | default.yaml:264 | `19656`→`3276` | wrong Sharpe |
| `risk.stale_data_sec` | default.yaml:211 + paper_trade.yaml | `420`→`1920` | **config load raises (loud)** |
| `runtime.max_data_staleness_sec` | default.yaml:103 + paper_trade.yaml | `1200`→`2100` | **silent: every decision STALE_DATA-blocked** |
| `runtime.state_reset_gap_sec` | default.yaml:98 | `300`→`1800` | **silent: every bar DATA_GAP-blocked** |
| horizons | training.yaml:7-9 | `[1,5,15]`/prim `15`/aux `[30,60]` → `[1,3,5]`/prim `5`/aux `[8]` | empty dataset |
| `training.sequence_length` | default.yaml:296 | decide (48 bars = 24h at 30Min) | window meaning shifts |
| `runtime.warmup_bars` | default.yaml:89 | decide (48 bars = 3.7 days) | eligibility/warmup shift |

**Keep `features.windows` unchanged** (5/20/60) so the feature contract is
identical — this is what lets the baseline-vs-brain comparison and any
checkpoint tooling stay coherent; the windows just span 6× longer
wall-clock, which is fine (a retrain is happening regardless).

2.2 **Fingerprint fix (code, small):** add `backtest.seconds_per_bar` to
the sample-cache `_training_config_fingerprint` (sample_cache.py:107-148)
OR mandate a cache purge on the pivot. Reason: the session mask uses
`seconds_per_bar` but it's absent from the fingerprint, so a wrong value
serves stale/empty caches silently. Prefer the code fix (one line + a
test). → **verify:** a test that changing `seconds_per_bar` changes the
fingerprint.

2.3 **Preflight check-7 fix (code, small):** derive the staleness
threshold from `timeframe_delta(config.runtime.timeframe)` instead of the
hardcoded `300.0` (preflight_monday.py:50-52), or the check goes
decorative at 30-min. → **verify:** preflight prints the derived bar
period; passes at 30-min config, fails a deliberately-too-low value.

2.4 **A2 fix (Platt horizon, code):** make the train-time Platt fit consume
`return_h{primary}` (return_h5 post-pivot) instead of
`regression_targets['primary_return']` (= max horizon) at
supervised.py:581-585. The `return_h5` target already exists in caches
(`enable_future_return_targets: true`), so no rebuild for the fix itself.
→ **verify:** a test that the fit's outcome array equals the primary-
horizon return, not the max-horizon return.

2.5 **Batch A4/F deletions here** (the pivot retrain regenerates every
fingerprint anyway, absorbing the invalidation): the dead config keys +
`symbols.yaml` + the `decouple_confidence_from_soma` chain. → **verify:**
config loads under `extra='forbid'`; preflight green; suite green.

2.6 **Noise floor at 30Min:** point `launch_noise_floor_sweep.bat` at the
30-min config stack (it currently passes NO `--config` = default 5-min
stack — must add `--config`), run seeds 7/13/29. → **verify:**
`runs/sweep_<id>_run00{1,2,3}.json` written; record val/test balanced
accuracy → **this is the 30-min floor**, the denominator for every later
claim. Expect a cold-start cache rebuild (no collision with 5-min caches;
different bar fingerprint).

Kill criteria: if the 30-min floor is catastrophically high (val BA spread
across seeds ≫ the 0.92pt seen at 5-min, i.e. the model can't even
stabilise on the coarser data), the sample count (131k) may be too thin —
reconsider horizon or timeframe before spending the full protocol.

### PHASE 3 — Baseline-first model bake-off (the "does complexity earn its keep" test)
Goal: settle whether the 15M brain beats a dumb model on 30-min data,
under §2.0. This is the phase that answers "is the whole structure too
complex."

**Blockers to clear first (all found in the sweep):**

3.0a **Baseline can't be backtested today.** `BrainPolicy.__post_init__`
raises when `ev_gate_requires_signals=true` and the model has no
`risk_head`; `BaselineMLP` has none. **Decision required:** either (i) add
a minimal `mfe/mae` risk head to `BaselineMLP` so it feeds the same EV
gate, OR (ii) run both models through a **shared vol-prior/cost-floor EV
gate** (the §7.0.4 finding already says the brain's EV legs ARE just a
vol prior + cost floor, so this is arguably more honest and more
comparable). **Recommend (ii)** — it removes the risk-head confound
entirely and matches what the EV gate demonstrably is. Scoped as a small
gate variant. → **verify:** a baseline checkpoint backtests without
raising and both models pass through identical gate logic.

3.0b **Sweep can't 3-seed a baseline.** `sweep_train.py` has no
`use_baseline` plumbing and hardcodes the `brain.pt→brain_runNNN.pt`
rename. Add a `--baseline` pass-through + checkpoint-filename awareness.
→ **verify:** a 3-seed baseline sweep produces `baseline_run00{1,2,3}.pt`
+ run summaries.

3.0c **No baseline promotion path** — fine, baselines are never promoted;
they're a comparison bar. Just score them; do not touch `registry.json`.

3.0d **Sizing comparability:** the baseline's `target_fraction` is a
LEARNED head (legacy entangled), while the brain's is zeroed (risk layer
owns sizing). For a clean comparison, force the baseline's
`target_fraction` inert (use the same `default_target_fraction × action`
path). → **verify:** both models size identically; only direction/
confidence differ.

3.1 **Train the baseline, 3 seeds**, 30-min config, primary H=5. →
**verify:** `baseline_run*.json` with val/test BA.

3.2 **Train the brain, 3 seeds**, same config, same seeds, A2 fix live,
decoupled heads (default). → **verify:** `brain_run*.json`; confirm the
brain's confidence is P(resolved class) end-to-end (the 2026-07-21 fix is
model-agnostic — recomputed in `_build_proposal` — so it holds here).

3.3 **Paired scoring** on the SAME OOS window and the SAME (shared) EV
gate: brain − baseline per seed, on val BA AND OOS net expectancy. →
**verify:** `runs/baseline_vs_brain_30min.json` with the 3 paired deltas
and the 30-min floor printed alongside.

**Decision gate (the fork this whole plan is built around):**
- **Brain wins 3/3 by > floor on both metrics** → complexity earns its
  keep; proceed to Phase 4 with the brain.
- **Brain does NOT clearly beat baseline** → **the measured revamp**: ship
  the baseline as the model, and open a deletion pass on the 15M-brain
  roadmap items the audit reserved under C1/C2 (arbiter, memory damping,
  seed_ensemble, rl_scaffold, self_supervised, curriculum) — now with
  *evidence* that they never paid off, not a guess. This is the
  "restart the concept" you asked about, done surgically instead of by
  arson.
- **Neither clears the edge milestone (positive OOS expectancy + >B&H)**
  → the 30-min stopping rule fires → fork to 60Min or NEWSBOX features.

### PHASE 4 — Live enablement of the winning model (intraday-only)
Goal: the chosen 30-min checkpoint trades live, intraday-only, with the
new EOD flatten.

4.1 **EOD-flatten machinery (new, ~small):** a time-of-day trigger that
emits flatten proposals for all open positions before session close,
riding the existing position-guard dispatch path (halt-exempt, EV-gate-
free, risk-reducing). **Critical constraint from the sweep:** the
submission SessionGuard blocks ALL dispatches (including risk-reducing)
outside `regular`, so the flatten must fire **before** 16:00 ET, not at/
after it — e.g. at 15:45 ET. → **verify:** a backtest/replay test that
positions are flat at session end; a unit test that the trigger fires
inside `regular` and the flatten proposals clear the submission guard.

4.2 **Rebuild the memory bank** at the new checkpoint + 30-min bars
(`build_memory.py --real-data`) — preflight check 10 (bank fingerprint ==
checkpoint fingerprint) will FAIL until this is done, and the bank is
built from runtime-timeframe bars so a 5-min bank is doubly stale. →
**verify:** preflight check 10 green.

4.3 **Rotate `runs/observations.jsonl`** at the pivot boundary (move the
5-min-era file aside) so the retrain drift trigger's midpoint split
doesn't straddle two timeframes. → **verify:** `--check-retrain-trigger`
reads only post-pivot observations.

4.4 **Promote** the winning brain checkpoint through the existing gate
(`run_canonical_training_with_gate`; the Platt sidecar rides along
atomically). **Note the gate has no timeframe awareness** — it would
benchmark the 5-min incumbent on 30-min bars. **Decision:** treat the
pivot checkpoint as a NEW lineage (fresh `registry.json` entry, no
cross-timeframe incumbent comparison) rather than forcing an apples-to-
oranges gate. → **verify:** `registry.json` shows the 30-min checkpoint
active; `checkpoints/brain.pt` is the new one; `.bak` preserved.

4.5 **Re-budget every min-sample threshold** for ~6× fewer decisions/day
(13/symbol vs 78): self-monitor S1/S2/S3 windows, Platt refit 300, retrain
trigger 200. → **verify:** documented new thresholds in config; a note
that arm-1-style "300 resolved trades" now takes ~6× the wall-clock.

4.6 **Paper-run the winner** intraday-only for a validation week, self-
monitor L0 emitting `runs/self_report.jsonl`. → **verify:** daily
`paper_status` + `self_report` review; the monitor's own S1/S2 flags vs
the backtest expectation.

Kill criteria: if live intraday-only diverges hard from the backtest
(self-monitor S1/S2 fire persistently), stop and reconcile before enabling
any of the response ladder.

### PHASE 5 — Medium loop (weekly calibration refit) — only after Phase 4 is stable
Goal: the bot updates *how much it trusts itself* from realized outcomes,
at a sample size that supports 2 parameters.

5.1 **Per-side Platt machinery (new):** `refit_platt_live.py` carries no
side dimension and `PlattCalibrator` stores one (a,b) curve; the arm-1
design note requires per-side. Build two calibrators (or side-conditional
storage) + a side-aware apply site in `MemoryAwarePolicy`. (For the
incumbent both sides were null; for a decoupled 30-min retrain the long
side is where signal lives.) → **verify:** a test that a per-side refit
produces two curves and the apply site selects by proposal action.

5.2 **Promotion path for a live-refit calibrator (new):** today the ONLY
writer of the canonical sidecar is training promotion; there is no path
from `runs/platt_live_refit.calibrator.json` to the canonical sidecar.
Design a weekly-refit promotion rule (gated: refit only promoted if it
improves ECE/Brier on a held-out slice). → **verify:** the rule refuses a
worse calibrator and the sidecar move is atomic (fold in A3: make
`PlattCalibrator.save()` tmp+rename).

5.3 **Schedule it** (Windows Task Scheduler, weekend, trading offline) —
no scheduler exists today; refit is manual CLI only. → **verify:** a dry-
run scheduled task writes the isolated artifact.

### PHASE 6 — Slow loop hardening (retrain cadence) — standing, low priority
The retrain-gate-promote loop already works. Post-pivot the only changes:
rotate observations at boundaries (done 4.3), and decide the cross-
timeframe incumbent-comparison policy (done 4.4). The drift trigger
(`--check-retrain-trigger`, sharpe_drop 0.5 / win_rate_drop 0.05, min 200
obs) is timeframe-blind but works once the stream is rotated. No new work
beyond documenting the cadence.

---

## 3. The three learning loops, summarised (your "it should learn as it trades")
| loop | evidence it learns from | what it adjusts | cadence | status |
|------|------------------------|-----------------|---------|--------|
| **fast** self-monitor | ~13 decisions/day | size / cost-floor / stand-down (shrink-only) | intraday | scaffold built; wire L0 (Phase 0.4) |
| **medium** Platt refit | ~weekly resolved trades | how much it trusts its own confidence (2 params) | weekly | script exists; per-side + promotion new (Phase 5) |
| **slow** retrain+gate | ~1M bars of banked outcomes | the 15M weights | on drift trigger | works; rotate stream (Phase 4.3) |

The contract across all three: **learn at the rate the evidence supports;
never rewire the brain intraday.** Fast loop may only make the bot
*smaller*; it can never grow, flip, or override a veto.

---

## 4. Complete change inventory (nothing forgotten — the checklist)

**Config edits (Phase 2.1, one atomic commit):** data.timeframe,
runtime.timeframe, backtest.seconds_per_bar, backtest.bars_per_year,
risk.stale_data_sec ×2 files, runtime.max_data_staleness_sec ×2 files,
runtime.state_reset_gap_sec, training horizons, sequence_length decision,
warmup_bars decision.

**Code changes:** seconds_per_bar → sample-cache fingerprint (2.2);
preflight check-7 derive-from-timeframe (2.3); A2 Platt-horizon fix (2.4);
baseline EV-gate path — shared vol-prior gate (3.0a); sweep_train.py
--baseline plumbing (3.0b); baseline target_fraction inert (3.0d);
EOD-flatten trigger (4.1); per-side Platt + apply site (5.1); calibrator
promotion rule + atomic save/A3 (5.2); self-monitor L0 wiring +
`monitoring.self_monitor` config + `scripts/selfmon_replay.py` (0.4);
`regime_snapshots._MIN_SESSION_BARS` — 30Min passes by one, **only touch
if going to 60Min**.

**Data/artifacts:** fetch 13× 30-min histories (1.2); rebuild memory bank
(4.2); rotate observations.jsonl (4.3); after promotion, reclaim ~27 GB of
5-min sample/feature caches (deletion audit, was do-not-touch).

**Deletion audit:** safe batches now (0.2); A4/F + soma chain at pivot
commit (2.5); legacy PolicyHead not-decoupled branch deletable ONLY after
the incumbent 5-min brain.pt is retired (4.4); C1/C2 roadmap items become
delete candidates ONLY if Phase 3 ships the baseline (3 decision gate).

**Corrections to prior docs:** MSTR A1 was wrong — no refetch (0.1);
my 2026-07-22 "lean managed-carry" reversed to intraday-only (0.3).

**Must NOT touch:** runs/ + logs/ (the truth store the comparisons depend
on); memory/schemas loaders + market_memory.pt; the not-decoupled
PolicyHead branch until incumbent retired; artifacts/historical 5-min bars
if you later choose to aggregate rather than refetch.

---

## 5. What still needs a human decision (not assumptions I can make)
1. **30Min now, 60Min later** — or straight to 60Min accepting ~28× less
   data and the regime-machinery fix? (Plan assumes 30Min-first.)
2. **Intraday-only vs overnight carry** — plan assumes intraday-only +
   EOD flatten; overnight carry is a different thesis needing a rewrite.
3. **Baseline EV-gate approach** — shared vol-prior gate (recommended) vs
   adding risk heads to the baseline.
4. **If the brain does not beat the baseline** — ship the baseline and
   delete the roadmap, or keep the brain for a features pivot (NEWSBOX)?
5. **Sequence_length / warmup_bars at 30Min** — keep 48 bars (24h /
   3.7 days) or shrink to keep the wall-clock window comparable to 5-min?

---

## 6. Suggested order of execution (dependency-sorted)
```
Phase 0 (this week, trading live)   → prep + safe deletions + self-monitor L0
  ↓
Phase 1 (needs Alpaca creds)        → fetch 30-min data + re-measure costs   [KILL GATE: cost thesis]
  ↓
Phase 2 (offline)                   → atomic config + fingerprint/preflight/A2 fixes + 30-min noise floor
  ↓
Phase 3 (offline, GPU, ~6 trainings)→ baseline vs brain, 3-seed paired       [DECISION GATE: complexity / revamp / stop]
  ↓
Phase 4 (paper-trading)             → EOD flatten + bank rebuild + promote + validation week
  ↓
Phase 5 (weekend)                   → per-side Platt refit + promotion + schedule
  ↓
Phase 6 (standing)                  → retrain cadence, documented
```
Every arrow is a hard dependency. Phases 1 and 3 have kill/decision gates
that can end the pivot early — by design, same as the last thesis.
