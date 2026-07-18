# SETAI Master Implementation Plan — v2

**Revised:** 2026-07-08 (v1: 2026-07-07). **Sources:** `CODEBASE_REVIEW.md`, `RESEARCH_ROADMAP.md`, and the execution experience of Phase 0 + Phase 1 wave 1.
**Goal, restated for v2:** every remaining task must serve one end — **a measurably more accurate bot**: better labels in, honest measurement out, and a closed loop that compounds improvements instead of noise. Tasks that never touch that chain were cut (see "Cut from the plan").

**What changed in v2:**
- Completed work is compressed into a status ledger; detail lives in git history (task IDs appear in commit messages).
- Every remaining task now carries an explicit **→ verify:** criterion — a check that can pass or fail, not a vibe.
- The plan is re-spined around the **Accuracy Program** (§2): a fixed measurement protocol plus the ordered list of accuracy levers.
- Follow-ups discovered during wave-1 execution are folded in (§4.6).
- Phases 3–4 no longer duplicate `RESEARCH_ROADMAP.md`; they reference it and keep only entry criteria.
- 14 items were cut or parked with reasons (§8).

---

## 1. Status ledger

**Test suite:** 702 passed / 1 skipped (PySide6 not installed) as of commit `dd1b8d8`. Baseline before this effort: 374.

**PHASE 1 CODE-COMPLETE (2026-07-10).** All of §4.1–§4.7 has landed and the full suite is green. The only open items are operational, not code: the ⟲1 retrain gate (§3 runbook — needs Alpaca credentials + a training run) and the unattended paper-trading week from Phase 0. Phase 2 entry stays gated on the ⟲1 SCORECARD.md incumbent row.
*Correction (2026-07-14 audit): this banner overclaimed — parts of §4.6 (4/5), §4.7 (trainer hygiene), and several §4.1/§4.4/§4.5 verify criteria had not actually landed. All were fixed 2026-07-14; the itemized truth is in §4.9.*

| ID | Scope | Status | Commit |
|---|---|---|---|
| 0.0 | Commit the Great Simplification, tag `pre-master-plan` | ✅ done | `1a8d322` |
| 0.1–0.3 | C1 bar dedup/timeframe/backfill; C4 sync resilience; C3 broker-truth accounting + fill capture | ✅ done | `7f49de6` |
| 0.4 | C5 build_memory.py + scripts import-smoke test | ✅ done | `8cca120` |
| 0.5 | C9/C10 zoneinfo sessions, fail-closed, validators | ✅ done | `b418c18` |
| 0.6 | GUI stderr drain, unrealized_pnl formula, fusion_enabled removal | ✅ done | `bd5c2b4` |
| 1E | Risk correctness: daily reset, risk-reducing exemption, resize path, post-trade caps, EV spread, kill-switch isolation | ✅ done | `40f091d` |
| 1A | Adjusted bars (Adjustment.ALL), cache v2.0 invalidation, stylized-facts detector, fetch/cache fixes | ✅ done | `6540607` |
| 1B core | **C2 normalization skew fixed** (stats persisted + re-applied on every inference path), parity tests, effective fast-mode config, atomic checkpoint saves, label-affecting cache fingerprints, split-boundary embargo | ✅ done | `0499055`, `8defad5` |
| 1F | extra='forbid' + RiskConfig validators + 42-YAML hygiene test (70 tests), script footguns, pyproject fixes | ✅ done | `84a6ce4` |
| 1B rest | §4.1 labels/imbalance/selection metric, §4.2 direction-aware Platt + binding, §4.3 parity insurance | ✅ done | `02c3151`, `9efd6bc`, `66b56ab` |
| 1C | Evaluation honesty: OOS enforcement, next-open±half-spread fills, CVaR95, truthful --compare, benchmark protocol + SCORECARD.md | ✅ done | `b3ff151`, `02c3151` (§4.4) |
| 1D | Self-evolution loop: outcome ledger + decision events (`1639159`, `cf45cfa`), retrain trigger, transactional scorecard-gated promotion, KD/L2 anchors, backward-transfer floor | ✅ done | `e764010` |
| 1G | Runtime/docs hygiene + §4.6 follow-ups: graceful shutdown, session-id nonce, bar_adjustment threading, webhook off hot path, caching-engine parity, README/cheatsheet truth pass | ✅ done | `dd1b8d8` |
| ⟲ #1 | First honest retrain | ✅ **run 2026-07-13** — adjusted refetch clean (1.13M bars, no split artifacts); candidate `d748f28c3695` trained (best epoch 4, val BA 0.4435), normalization + split ranges + Platt sidecar verified in meta; OOS benchmark row in SCORECARD.md (BA 0.3642, excess vs buy&hold −3.96%); **promotion refused by the gate** (honest: no alpha yet), candidate staged at `checkpoints/20260713_185353/`. Gate defect found & filed: incumbent benchmarked on mismatched/in-sample window when meta lacks split ranges. **Then FORCE-promoted** (user-approved ⟲1 bootstrap — refusal reasons were "no alpha vs buy-and-hold" plus an invalid delta against a pre-C2 incumbent with no split ranges): `d748f28c` **is the current canonical brain** (registry status `active`). Bookkeeping caveats from the 2026-07-14 audit: the force-promote was re-run over an already-promoted brain, so `brain.pt.bak` holds `d748f28c` itself (the pre-⟲1 pre-C2 incumbent was NOT preserved) and the registry entry's `parent_fingerprint` is self-referential. |
| 🔍 audit | Phases 0–3.1 verified vs this plan | ✅ **2026-07-14** — five-agent audit read every "done" claim against code + test bodies. Safety-critical claims all held (risk invariants, C2 parity, embargo, scorecard, Phase 2 gates/artifacts). Overclaims found and **fixed same day** (commits `e3b0c98`..`59dfb77`, suite 879→921): see §4.9. |

**Operational validation still owed from Phase 0:** one unattended week of paper trading (buffers hold one bar per period, equity tracks broker, forced outage degrades and recovers, daily-loss trips on a simulated loss). Nothing blocks starting it now.

---

## 2. The Accuracy Program

Everything remaining is ordered by one question: *does it make the bot's decisions more accurate, or make accuracy measurable?* The chain:

```
labels (what the model learns)  →  training (how it learns)  →  calibration (what its confidence means)
      →  measurement (whether improvement is real)  →  the loop (whether improvement compounds)
```

### 2.0 The noise floor — measured 2026-07-15, and it dwarfs every claim so far

**Three fast-mode trainings, identical but for `training.seed`, scored val balanced accuracy 0.3865 / 0.4444 / 0.4132 — a 5.8pt spread. Test split: 8.1pt. Long recall swung 0.18 → 0.48.** Reproduce: `python scripts/sweep_train.py --real-data --sweep-config configs/sweep_seed_variance.yaml --config configs/default.yaml --config configs/training.yaml --config configs/experiments/fast_seed_variance.yaml --rank-by val_report.balanced_accuracy`.

Every accuracy claim this plan has recorded is smaller than that spread:

| Claim | Delta | vs floor |
|---|---|---|
| ⟲2 "classification improved, non-degradation verified" (§6.1) | +0.44pt | ~13× below |
| ⟲3 "market context is NEGATIVE" (§3) | −0.31pt | ~19× below |

Both conclusions are **withdrawn as unsupported** — not reversed (we cannot claim market context helps either), just not evidence. The measurement could not even be taken until 2026-07-14, because `seed_everything(7)` was hardcoded: every "different" run trained identically (§4.9 item 1).

Caveats, stated so this number is not over-claimed in turn: the sweep is fast mode (6 months, 50k samples, 5 symbols) and full-data runs should be quieter; n=3; val-BA is not the OOS scorecard surface.

**Full-data GPU floor — measured 2026-07-17 (this is now the authoritative floor).** Three full-data GPU trainings, identical but for `training.seed` ∈ {7, 13, 29}, scored val balanced accuracy **0.4384 / 0.4348 / 0.4440 — a 0.92pt spread** (mean 0.4391, pop-std 0.38pt); **test split 0.4395 / 0.4206 / 0.4195 — a 2.00pt spread.** Runs: `runs/sweep_20260717_142927_run00{1,2,3}.json` (seeds 7/13/29 in order), ~25–32 min each on the RTX 4070 Ti (cu128 + real AMP). Reproduce with `scripts/launch_noise_floor_sweep.bat` (hidden-window launcher; a visible console gets closed — §10.10). As predicted, full data is ~6× quieter than fast mode's 5.8pt — **but 0.92pt val / 2.00pt test is NOT small.** Both historical single-run claims sit UNDER it: ⟲2 +0.44pt and ⟲3 −0.31pt are each below the 0.92pt val floor, so both stay withdrawn on the full-data path too. *Erratum (same day):* an earlier note here claimed val long-recall 0.0 on all three seeds — that was a metrics-extraction bug (a nonexistent `val_report.long_recall` field silently read as zero), not a model property. The real per-class picture (`val_report.per_class_metrics`, all seeds): **long recall 0.22–0.24, short 0.18–0.21, flat ≈0.90**, prediction ratios matching label base rates (≈6.5/87/6.5) — no flat-hedging, directional precision ≈3× base rate. No collapse; nothing to fix.

**Protocol decision (resolves the §9.1 fork).** The floor is intermediate — between the "< ~0.5pt → single runs interpretable" and "≳ 1pt → 3-seed mandatory" branches — but decisively on the **mandatory-3-seed** side: the two claims it must adjudicate (⟲2, ⟲3) are both smaller than it. So the standing rule is the 3-seed paired protocol below, unconditionally; the "single full-path run per experiment" shortcut in §9 step 1 is **retired** (do not use it even for prioritization-to-promotion, only for pure ordering). n=3 and val-BA-≠-OOS caveats still hold; a promotion still ultimately needs the OOS scorecard, not just val BA.

**Standing rule (amends §2.1):** an improvement claim requires a delta that clears the noise floor **on the same data path**, demonstrated across seeds — not one row beating one row. A single-seed scorecard row is a sample, not a result.

**Operational recipe (so no judgment is needed at execution time):** to claim candidate > baseline, run BOTH with the same 3 seeds (7/13/29), pair the runs per seed, and require (a) all 3 paired deltas positive on the claimed metric AND (b) the mean delta reported side-by-side with the measured floor spread. 3/3 positive signs is p=0.125 — call the result "suggestive" in writing unless the mean delta also exceeds the floor spread. Anything cheaper (one seed, unpaired seeds, fast-vs-full comparison) may be used to PRIORITIZE experiments but never to claim or promote.

### 2.1 The scorecard — accuracy is a number, not a feeling

**Task SC — build the benchmark protocol** *(part of 1C, task 4.4.6; do it first within 1C)*. A single command (`scripts/run_benchmark.py` or a `run_backtest.py --benchmark` mode) that:

1. Loads a checkpoint and reads its `training_split_ranges` from meta (already persisted).
2. Evaluates **only on bars strictly after the checkpoint's train/val end** — hard-fails if none exist.
3. Uses honest fills (next-bar-open, spread-aware slippage, real `transaction_cost_bps`).
4. Runs the same bars through buy-and-hold and the momentum baseline (both tradeable, §4.4.4).
5. Reports: 3-class balanced directional accuracy, net-of-cost expectancy per trade, Sharpe, max drawdown, CVaR95, benchmark-relative return, trade count.
6. Appends one row to `docs/analysis/SCORECARD.md` (date, checkpoint id, data range, all metrics).

**→ verify:** running it twice on the same checkpoint+range appends two identical rows; running it on a checkpoint whose training range covers the requested window hard-fails with a clear message.

**Standing rule:** from the first scorecard row onward, "accuracy improved" means *a new row beats the incumbent row on balanced accuracy AND net expectancy, out of sample*. No claim of improvement is accepted into this plan's ledger without a row.

### 2.2 The accuracy levers, in order of expected effect

| # | Lever | Where | Why it moves accuracy |
|---|---|---|---|
| 1 | **Retrain gate #1** (§3) | training | Every deployed checkpoint is pre-C2 + trained on split-contaminated raw prices. The first retrain on adjusted data with normalization parity is the single largest accuracy delta available — it makes live behavior match validation for the first time. |
| 2 | **Label quality** (§4.1) | targets.py, losses.py | Cost-aware, horizon-scaled flat bands replace the symptom-patch (`force_directional_flip`). The model currently learns a flat band that is ~√H too narrow and cost-blind — mislabeled samples put a ceiling on any architecture's accuracy. |
| 3 | **One imbalance correction, accuracy-aligned selection** (§4.1.3) | class_balance, trainer | Triple-stacked class rebalancing pushes predicted distributions far from base rates; best-epoch selection on raw multi-task loss rewards flat-hedging. Selecting on balanced accuracy directly optimizes the scorecard metric. |
| 4 | **Calibration that means P(win)** (§4.2) | platt_calibration | Confidence feeds the EV gate — the actual trade/no-trade decision. Direction-agnostic Platt is anti-calibrated for shorts: correct short calls get deflated exactly when right. |
| 5 | **Honest measurement** (§4.4) | backtest/eval | In-sample evaluation makes memorization look like accuracy; same-bar fills inflate reactive strategies. Without this, levers 1–4 can't be confirmed. |
| 6 | **The closed loop** (§4.5) | training lifecycle | Anchored, gated retraining turns one-off accuracy into compounding accuracy under drift — and prevents a bad retrain from silently replacing a good checkpoint. |
| 7 | Parity insurance (§4.3) | features | Scalar-vs-batch equivalence and feature-cache salting don't raise accuracy; they prevent silent future regressions of it. |

Phase 2+ levers (memory prior, market-context features, OOD gating) build on this foundation and are gated on scorecard evidence (§6–7).

---

## 3. ⟲ Retrain gate #1 — the runbook

*Run after §4.1 and §4.2 land (they change training artifacts; retraining before them wastes the run). §4.3–4.7 are not blockers.*

1. **Refetch** — the v2.0 cache invalidated all raw-price caches by design: `python scripts/fetch_historical.py` for the full universe/date range. → verify: fetch log shows `adjustment=all`; stylized-facts advisory reports no >20% single-bar returns for the universe (NVDA specifically).
2. **Train** — `python scripts/run_train.py` (full mode, canonical path). → verify: checkpoint meta contains `normalization` (non-null), `training_split_ranges`, effective model config; log shows embargo purge counts per symbol.
3. **Calibrate + memory** — produced by the training run; Platt must be direction-aware (§4.2) and the bank rebuilt with the new encoder + stats. → verify: calibrator artifact records the checkpoint fingerprint; bank items carry the new encoder fingerprint.
4. **Score** — run the §2.1 benchmark. This row becomes the **incumbent baseline** all future work must beat. → verify: SCORECARD.md row exists; evaluation window starts after train/val end.
5. **Paper-validate** — start the unattended paper week (Phase 0 sign-off + first live look at the post-C2 stack). → verify: `model_decision` logs show varying confidence (not pinned); reason-code distribution is not dominated by `negative_ev` or `stale_data`.

| Retrain | After | Status |
|---|---|---|
| ⟲ 1 | §4.1 + §4.2 | ✅ run 2026-07-13 (see §1 ledger: gate-refused, then user force-promoted `d748f28c` as bootstrap incumbent) |
| ⟲ 2 | model-head disentanglement (§6, Phase 2) | ✅ run 2026-07-14 (§6.1: classification +0.44pt, trading worse → refused; `4575582c` staged) |
| ⟲ 3 | FeatureContract bump for market-context features (§7, Phase 3) | ✅ **run 2026-07-14 — INCONCLUSIVE (not "negative")** — isolated run (`checkpoints/retrain3_market_context/`, contract 44→52, honest 6bps labels, val BA 0.4418). Same-window OOS vs incumbent (cutoff 2026-03-18): BA 0.3617 vs 0.3648 (−0.31pt), expectancy −0.17 vs −0.10, excess vs buy&hold −6.2% vs −5.5% → NOT promoted (correct: it did not clear the bar), `features.market_context_enabled` stays **false**. **But the −0.31pt is ~19× below the 5.8pt seed noise floor measured 2026-07-15 (§2.0), so "market context hurts" is withdrawn — this run is one sample of an unknown distribution.** Deciding market context needs a multi-seed comparison, not another single run. Fingerprint `9b3ff04c6a7c` (re-stamped — this run exposed the encoder-fingerprint collision, fixed same day). |

---

## 4. Remaining Phase 1 work — detailed

### 4.1 Labels & training accuracy (1B-labels) — **do first**

1. **[M] Cost-aware, horizon-scaled flat band.** In `src/training/targets.py`: the direction-label epsilon must (a) include a realistic round-trip cost floor (`transaction_cost_bps` — set a non-zero default in `configs/training.yaml`, shared with backtest costs), and (b) scale with √(effective_primary_horizon) — a 1-bar std threshold applied to an H-bar return is ~√H too narrow, mislabeling noise as signal. Adaptive per-symbol epsilon keeps working on top.
   **→ verify:** unit test: with cost=10bps and H=6, a +8bps 6-bar move labels FLAT and a move > ε·√6 labels directional; label distribution diagnostics on synthetic bars shift accordingly.
2. **[S] Per-trade-normalized loss weighting** in `src/training/losses.py` so large-magnitude outcomes don't dominate the direction loss (JaxMARL-HFT degeneracy diagnosis).
   **→ verify:** loss unit test with two samples of equal correctness but 10× return magnitude contributes equal direction-loss.
3. **[M] One imbalance correction + aligned model selection.** Today balanced class weights + inverse-frequency sampler + adaptive focal gamma stack multiplicatively, and best-epoch selection uses raw multi-task loss. Pick ONE primary rebalancing mechanism (default: class weights; sampler and focal become opt-in, off by default) and make model selection configurable with default `val_balanced_accuracy`.
   **→ verify:** config hygiene test passes with new defaults; a training smoke run logs which mechanism is active; best-epoch restore keys on the configured metric (test with a crafted metric history).
4. **[M] Retire `force_directional_flip` as default.** With honest labels (1–3), the no-trade attractor should be fixed at the cause. Make the flag a single config field consumed identically by `policy_factory.py` and `runtime_bootstrap.py`, default **off** in both, kept only as an explicit experiment flag.
   **→ verify:** backtest-vs-runtime proposal parity test: same checkpoint + same snapshot ⇒ identical proposal through both construction paths; grep shows exactly one config source for the flag.

### 4.2 Calibration correctness (1B-calibration)

1. **[S] Direction-aware Platt labels:** fit on predicted-sign × realized primary return (flat excluded, net of costs), so calibrated confidence approximates P(chosen direction wins), not P(price up).
   **→ verify:** unit test: a synthetic set where shorts are 80% right yields calibrated confidence ≈0.8 for those shorts (was ≈0.2 under the old labels).
2. **[S] Artifact isolation + binding:** per-run calibrator path (experiment runs must not clobber the canonical one the runtime loads); calibrator records the checkpoint fingerprint; loaders (backtest + runtime, currently two hardcoded `runs/` literals) take the path from config and refuse a fingerprint mismatch.
   **→ verify:** test: isolated run writes its own artifact and the canonical file is byte-identical after; loading a calibrator against a different checkpoint raises.

### 4.3 Parity insurance (1B-parity)

1. **[S] Scalar-vs-batch feature equivalence test** — the load-bearing invariant behind train/live parity has no test. Default config AND a small-window config (exposes the known RSI `[:_RSI_PERIOD]` / momentum boundary off-by-ones — fix both while here); derive `first_valid_index` from ALL feature-family warmups.
   **→ verify:** the new test fails on current code for the small-window config (proving it bites), passes after the off-by-one fixes.
2. **[S] Feature-cache salt + signature:** add a code/schema version constant to the feature-cache fingerprint; strengthen the bars signature beyond endpoints (cheap content hash).
   **→ verify:** test: bumping the constant invalidates a warm cache; editing an interior bar changes the signature.
3. **[S] Live-path input sanitation:** `nan_to_num`/isfinite-reject in `build_from_tensor`; epsilon on `return_1` denominators; unify the epsilon constant (code default 1e-6 vs YAML 5e-4 — pick the YAML value, one source).
   **→ verify:** test: a zero-close bar produces finite features on the live path; grep shows one epsilon source.
4. **[S] Model input hardening:** remove/fix the `full_features=None` branch-order fallback (silently permuted inputs); fix `_extract_ev_signals_batch` zeroing both EV legs when one is missing (single-path handles them independently).
   **→ verify:** test: calling the model without `full_features` raises; batch EV extraction with mfe-only matches single-path behavior.
5. **[M] Batch-composition invariance** (masked mean-pool or length-bucketing instead of convolving left-zero-padding): `propose_batch([a,b]) == [propose(a), propose(b)]` for heterogeneous temporal lengths.
   **→ verify:** exactly that equality as a test, with different-length snapshots in one batch.

### 4.4 Evaluation honesty (1C)

*Simpler than v1 planned: `training_split_ranges` (per-split, per-symbol timestamps) already ships in checkpoint meta — no new persistence needed.*

1. **[M] Out-of-sample by default:** `run_backtest.py` reads the checkpoint's train/val end from meta and restricts evaluation to later bars; `--allow-in-sample` overrides but every metric line and report is tagged `IN-SAMPLE`.
   **→ verify:** test with a stub checkpoint meta: default run drops overlapping bars; override run tags output.
2. **[M] Honest fills and costs:** `backtest.fill_timing: same_close | next_open` (default `next_open`); slippage = base + k·spread where spread exists; `cvar_95` added to metrics; backtest costs and label costs read the same config values.
   **→ verify:** golden test: a策 buy decided on bar t fills at bar t+1's open; CVaR95 matches a hand-computed fixture; cost value asserted identical in targets and fills.
3. **[L] Real walk-forward:** per-fold retraining honoring `reset_model_per_fold` (fast_mode for tractability) — or, minimum: hard-refuse folds whose test window overlaps the checkpoint's training range. Paired candidate-vs-incumbent stats on identical folds (sign test); degenerate folds excluded from aggregation, not averaged as zeros.
   **→ verify:** walk-forward on a checkpoint trained through "now" refuses every fold (proving the guard); with per-fold retrain enabled, fold N's model trains only on fold N's train slice (assert via injected marker).
4. **[S] Tradeable baselines:** momentum/mean-reversion/buy-and-hold pass the EV gate (honest heuristic EV inputs or an explicit `ev_gate_requires_signals=false` for policies without risk heads); fail fast when the EV gate has no inputs (silent zero-trade sessions become impossible).
   **→ verify:** regression test: BuyAndHold produces >0 fills on synthetic bars; a risk-head-less checkpoint without the bypass flag raises at construction.
5. **[M] Metric math + engine floors:** golden hand-computed fixtures for Sharpe/Sortino (fix the negative-count divisor)/Calmar/round-trips (sign flips close trips)/profit-factor (all-winner case); bar loop starts at `max(warmup_bars, first_valid_index)`; check `ExecutionResult.accepted` before `register_fill`; timestamp-aligned multi-symbol iteration (or per-bar assertion).
   **→ verify:** the golden fixtures; a backtest with warmup < first_valid_index starts at the later index (assert first decision bar).
6. **[M] The scorecard (§2.1).**
   **→ verify:** as in §2.1.
7. **[S] Repair or delete `--compare`:** truthful scenario labels, no hardwired fusion zeros, no silent flip-flag divergence between scenarios.
   **→ verify:** `--compare` output contains no "fusion"/"veto" rows; both scenarios build via the same factory options minus the single toggled flag.

### 4.5 Close the loop (1D) — after ⟲1

*Ordering rationale: the loop should first exist around the honest retrain workflow (§3), then automate it.*

1. **[M] Durable outcome dataset:** persist the Ledger (JSONL, append-only); emit an execution-outcome event stream (decision id, proposal, risk decision, executed action, fill, realized PnL net of fees — charge fees into `realized_pnl` while here).
   **→ verify:** restart test: fills recorded before a restart are present after; each closed trade appears exactly once with net PnL.
2. **[S] Credit assignment:** record executed (risk-clamped) action + clamp/veto flag on every decision — `MemoryItem.action_annotation` (currently None), threaded through tick_processor Phase C and `BacktestEngine.run`; `intervention_rate` in runtime status.
   **→ verify:** paper-loop test: a resized order's memory item carries the executed size and `resized_to_cap`.
3. **[M] Retrain trigger + replay wiring:** un-reserve `retrain.py`/`replay_buffer.py`; per-decision observations flow into the replay buffer; `check_retrain_trigger` gets per-metric thresholds (0.15 on Sharpe is noise; 0.15 on win-rate is catastrophe) and a scheduler/CLI entry point.
   **→ verify:** unit tests per threshold; an end-to-end dry run: N synthetic observations → trigger fires → retrain entry point invoked with lineage metadata recorded.
4. **[M] Authoritative promotion:** gate wired into `run_train.py` (canonical `brain.pt` can never be overwritten ungated; `--force-promote` escape hatch); runtime resolves via the registry with config-path fallback; transactional promote (atomic write, lock, corrupt-JSON tolerance); the flat-vs-dotted metric-key mismatch fixed with one adapter + key-alignment test.
   **→ verify:** test: an ungated training run leaves `brain.pt` untouched and writes to its run dir; a gated promote updates file+registry atomically (kill-injection test leaves both consistent).
5. **[L] Promotion judged on the scorecard:** candidate must beat the incumbent row and buy-and-hold on identical OOS bars (§2.1 protocol) — not just classification metrics.
   **→ verify:** promotion of a deliberately-lobotomized candidate (shuffled labels) is refused.
6. **[M] Forgetting-safe retraining:** λ·KL self-distillation against the frozen incumbent + L2 proximal anchor (config-exposed); regime-stratified rehearsal quotas from the historical cache; per-module weight-drift (cosine) logged per retrain.
   **→ verify:** anchored fine-tune on a narrow recent slice degrades a frozen early-regime holdout ≤ configured floor, unanchored control degrades more (test with synthetic regimes).
7. **[S] Backward-transfer floor in the gate:** 3–6 frozen historical regime slices; promotion refused if any degrades beyond threshold.
   **→ verify:** gate test with a crafted candidate that wins recent + loses old slices → refused.

### 4.6 Wave-1 follow-ups (discovered during execution)

1. **[S] Thread `config.data.bar_adjustment` into client construction** (`runtime_bootstrap._build_client`, gui services, scripts) — the field exists and validates; the client currently hardcodes ALL (correct default, config is decorative). **→ verify:** grep shows constructors read the config; a config set to `raw` reaches the client in a unit test.
2. **[S] Sync training↔memory `forward_horizons`** — still `[1,3,6]` vs `[1,3,6,12]` in default.yaml; remove the explicit memory override so the intended auto-sync runs, and assert equality in a test. **→ verify:** hygiene test asserts the loaded config has identical lists.
3. **[S] Checkpoint-meta forward-compat note:** `ModelConfig.model_validate(meta["model_config"])` now runs under extra='forbid'; a future field *removal* would break old-checkpoint loads. Parse with a filtered dict (drop unknown keys, log them) at this one boundary. **→ verify:** loading meta containing a bogus key logs + loads.
4. **[S] `evaluate_checkpoint` uses stored stats:** it rebuilds a bundle and re-derives normalization from current data — metrics not comparable across runs. Inject the checkpoint's `NormalizationStats` into the rebuild (skip re-deriving) or refuse when stats differ materially. **→ verify:** evaluating the same checkpoint against two different data ranges applies identical stats (assert via injected spy).
5. **[M] Defaults alignment:** Python-side config defaults still diverge from default.yaml (risk limits, epochs, sequence_length, AMP/Platt/early-stopping, epsilon). Declare YAML canonical: align the Python defaults, and add a test that `AppConfig()` == `load_config("configs/default.yaml")` for the aligned fields (or an explicit documented diff-list). **→ verify:** that test.

### 4.7 Runtime & docs hygiene (1G) — batched, low priority vs the above

- **[S] Trainer hygiene:** `torch.no_grad()` for val/test epochs; `model.eval()` in `encode_batch`; BatchNorm size-1-batch guard (`drop_last` or skip); config-driven seed (hardcoded `seed_everything(7)` zeroes sweep variance); surface collate `nan_to_num` counts. → verify: existing training tests + one new test per item.
- **[S] `resolve_device('cuda')` without CUDA** → clean ConfigurationError. → verify: unit test.
- **[M] Graceful shutdown:** cancel tracked open orders, persist memory/pending, final status snapshot on KeyboardInterrupt. → verify: mock-run integration test asserting cancel calls + snapshot file.
- **[S] Ops nits:** webhook alerts off the rejection hot path (thread/queue); client_order_id restart-collision nonce. → verify: unit tests.
- **[M] Docs truth pass:** rewrite README (delete dendritic/soma/Hopfield/ESOM/veto/GRU claims; "approves, resizes, or rejects" is now true — say how; document the C2 stats, scorecard, and adjusted-bar cache); delete or rewrite `LIVE_LOG_CHEATSHEET.md`; fix factory/engine docstrings claiming fusion/cutoff behavior; GUI: remove dead toggles, fix `MemoryStatus` reading nonexistent bank keys. → verify: `grep -ri "hopfield\|dendritic\|soma\|esom" README.md docs/ src/` returns only historical notes explicitly marked as removed.

### 4.8 Post-⟲1 follow-up — ✅ **done 2026-07-14** (found by running the ⟲1 gate for real)

**[M] Promotion-gate benchmark windows must match.** `scripts/run_train.py:_benchmark_candidate_vs_incumbent` benchmarks candidate and incumbent each on their *own* OOS window (`src/evaluation/oos.py` reads each checkpoint's `training_split_ranges`). Two defects observed in the 2026-07-13 ⟲1 run:
1. A pre-C2 incumbent **without** `training_split_ranges` was benchmarked over the FULL history including its own training data (in-sample contamination: its row had 29,693 buy-hold fills vs the candidate's 9,703).
2. Even with ranges present, the two rows can cover different windows, making the gate's expectancy/balanced-accuracy **deltas apples-to-oranges**.

Fix: benchmark BOTH checkpoints on the intersection window — bars after `max(candidate_training_end, incumbent_training_end)`; if the incumbent has no `training_split_ranges`, either use the candidate's window for both or mark the incumbent row untrusted and skip the delta gates (absolute gates like `excess_vs_buy_hold` stay). Related code: `src/training/checkpoint_promotion.py:evaluate_promotion`, `src/evaluation/benchmark.py:run_benchmark`.
**→ verify:** regression test in `tests/test_close_the_loop.py` with a missing-split-ranges incumbent: both benchmark rows share the same window (equal buy-hold fill counts) or delta gates are skipped with a loud log; the 2026-07-13 scenario reproduced under the fix yields a fair comparison.

### 4.9 Audit gap-fix wave — ✅ **done 2026-07-14** (commits `e3b0c98`, `a6d86d5`, `8dcb56e`, `59dfb77`)

The 2026-07-14 audit found the "Phase 1 code-complete" banner overclaimed several items. All fixed the same day, each with its regression test:

1. **§4.7 trainer hygiene** (was largely absent): val/test epochs now run under `torch.set_grad_enabled(training)`; `encode_batch` forces eval-mode and restores caller state; training DataLoader drops size-1 remainder batches (`train_drop_last`, smoke datasets exempt); `seed_everything(7)` replaced by config-driven `training.seed`; collate `nan_to_num` counts surfaced (`count_nonfinite_columns`). → `tests/test_trainer_hygiene.py`.
2. **§4.6.5 defaults alignment** (was never done — 38 divergent fields): Python defaults aligned to `default.yaml`; 8 deliberate divergences documented with reasons and pinned by `tests/test_defaults_alignment.py` (notably `model.decoupled_heads=False` stays load-bearing for pre-Phase-2 checkpoint reconstruction).
3. **§4.6.4** (was detect-and-warn): `evaluate_checkpoint` now INJECTS the checkpoint's stored normalization stats into the bundle build (`normalization_override` threaded through); pre-C2 checkpoints keep the warning fallback. → `tests/test_stats_injection.py`.
4. **§4.4.2 cost parity** (verify was missing; fields were only comment-linked): `training.transaction_cost_bps` is now BY CONTRACT `2·(backtest.fee_bps + backtest.slippage_bps)` = 6.0 (was ≈5.0); EV-gate risk costs asserted equal to backtest costs. Label-affecting: sample caches re-fingerprint on the next training run.
5. **§4.4.4 EV-gate fail-fast** (silent zero-trade sessions were still possible): `risk.ev_gate_requires_signals` (default true) makes BrainPolicy REFUSE at construction to wrap a risk-head-less model; explicit bypass for experiments. → `tests/test_proposal_parity.py`.
6. **§4.1.4 proposal parity test** (promised, never written): backtest-vs-runtime construction paths produce identical proposals for the same checkpoint + snapshot (`build_runtime_brain_policy` extracted). Real parity hole found and fixed while writing it: the factory never passed `default_target_fraction`.
7. **§4.4.5 metric fixtures** (missing): hand-computed Sharpe (`√18900/30`) and Calmar (1.2) goldens; engine-floor regression (bar loop starts at `first_valid_index`).
8. **§4.4.7 --compare**: dead fusion/veto print block deleted; no-fusion-rows + factory-only-scenarios pinned by tests.
9. **§4.4.3 walk-forward** (weakest audit finding): in-sample folds now hard-refused at the RUNNER level (`_refuse_in_sample_folds`); `reset_model_per_fold` DELETED (defined-but-dead; per-fold retraining stays [L] future work); `compare_walk_forward()` = the paired candidate-vs-incumbent sign test on identical folds.
10. **§4.5.3 observation writer** (trigger had no data source): every executed fill (paper + broker) appends a `DecisionObservation` to `execution.observation_stream_path` with net PnL + submit-time clamp flag. Backtests deliberately do not write.
11. **§4.5.7 slice producer** (floor gate ran empty): `benchmark_backward_transfer_slices()` cuts the shared OOS window into `evaluation.promotion.backward_transfer_slices` chronological regime slices and feeds `evaluate_promotion(slice_metrics=…)` in the canonical gate. Truly pre-training "frozen historical" slices stay future work until incremental fine-tuning exists (full-history retraining makes them unhonest today).
12. **§4.5.6 weight drift** (never logged): `compute_weight_drift()` per-module cosine vs the anchor, logged as `anchor_weight_drift` on every anchored retrain. **Rehearsal quotas deliberately NOT built:** the canonical retrain path trains on the full configured history, so every regime is structurally rehearsed; quota machinery enters with incremental fine-tuning, not before.

Known limitations documented (not fixed, by choice): the paper/live runtime resolves checkpoints via `config.model.checkpoint_path` only (registry resolution is GUI-only), and the fills ledger JSONL is not rehydrated at boot (durability is file-level).

---

## 5. Standing rules (unchanged from v1, still binding)

1. Paper-first; the live-mode double interlock is never weakened.
2. Learned proposes, deterministic risk disposes — now test-enforced (`test_risk_correctness.py` safety invariants).
3. Every self-evolution mechanism must beat plain pretrain+fine-tune AND equal-budget random search before adoption.
4. Learn from executed actions, not intent.
5. Counterfactual credit, not correlational.
6. Anchor every self-update to the incumbent.
7. Protect rare-regime data from every eviction/rebalance rule.
8. Promotion is a statistical decision on the scorecard.
9. Every fix ships with its regression test in the same commit.
10. Verify-as-you-touch: opening a file obligates resolving its `CODEBASE_REVIEW.md` §7 findings in the same PR.

---

## 6. Phase 2 — after ⟲1 baseline exists (unchanged scope, gated entry)

*Entry criterion: SCORECARD.md has the ⟲1 incumbent row. Full task detail: v1 of this plan (git history) + RESEARCH_ROADMAP.md Phase 2. Summary, accuracy-first order:*

1. **Model-head disentanglement** — ✅ **code done + ⟲2 run 2026-07-14** (commit 897b1fe; candidate `4575582c9cd9`). Same-window OOS vs incumbent: balanced accuracy 0.3677 vs 0.3633 (+0.44pt — ~13× BELOW the 5.8pt seed noise floor measured 2026-07-15 (§2.0); the "non-degradation verified" claim is **withdrawn as unsupported**), sharpe −1.25 vs −3.49, val BA best 0.4466 vs 0.4435 — but excess vs buy&hold −5.2% and expectancy delta −0.09 → **gate refused, incumbent stays**. Candidate staged at `checkpoints/20260714_062943/`; scorecard row appended; bank rebuilt back to incumbent fingerprint. Takeaway: classification improved, trading didn't — sizing is now a fixed default fraction by design, so the P&L gap belongs to items 2 (memory damping) and 4 (measured risk / sizing).
2. **Memory correctness then bounded influence** — ✅ **done 2026-07-14** (commits 7811872, e830092, 41daf93, a701cf4). Correctness: temporal cutoff enforced in retrieval (+ self-exclusion), O(N) extend, build provenance stamped, saves debounced off the hot path, pending queues bounded, unstamped-bank fallback removed. Influence: damping-only size multiplier ∈ [0.5, 1.0], off by default, every rail test-pinned. Credit gate: `scripts/run_memory_credit.py` (paired same-seed OOS folds). **First real credit run: NOT EARNED** (1/4 folds positive, mean ΔTotalReturn −1.3bps, p=0.94 — `runs/memory_credit.json`) → influence stays off, re-run after items 1/4 improve the brain/bank. Pruning: MemoryConfig 123→21 fields (allow-list pinned in hygiene tests), fusion/topology stubs + 14 zombie presets deleted, GUI dead toggles removed.
3. **Strategy arbitration layer** — ✅ **done 2026-07-14** (commit dd4d198). StrategyArbiter: brain + momentum + mean-reversion registry; `strategy_id` threaded proposal → OrderIntent → FillRecord (broker path attributed via client_order_id map); per-strategy `budget_fractions` soft caps; PnL-EMA outcome-weighted allocation with a weight floor (throttled, never starved); memory delegation preserved. **OFF by default** (`arbitration.enabled`) — enable only after `run_backtest.py --use-arbitration` produces a scorecard row beating the brain-only incumbent.
4. **Measured risk** — ✅ **done 2026-07-14** (commits 6ea2493, 3480fa9, 1464656). Perturbed-R² harness (`scripts/run_perturbation.py`), cap-sweep harness with knee selection (`scripts/run_cap_sweep.py`, plus a found bug: RiskConfig accepted ≤0 caps), outcome-adaptive drawdown budget (deterministic multiplier ∈ [budget_min_multiplier, 1.0], risk-reducing exempt, add-to-trim clamp, logged per decision, off by default). **First real measurements** (incumbent d748f28c, OOS window): perturbation curve HEALTHY — sharpe degrades monotonically −3.57 → −11.67 as signal → noise, so returns are signal-driven, no leakage, not structure-dominated (the model genuinely loses less than noise; the edge is real but below costs). Cap sweep FLAT 3k–48k — max_position_notional never binds; sizing is dominated by the model's small learned fractions, so cap tuning is not currently a lever (artifacts: `runs/perturbation_curve.json`, `runs/cap_sweep_max_position_notional.json`).
5. **Anchored negatives + curriculum** — ✅ **done 2026-07-14** (commit 11a78a2). Permanent deduplicated store of worst executed decisions (loss > round-trip cost + noise buffer) appended on every triggered retrain; economic-margin pair-validity filter for the later contrastive phase; difficulty-based epoch-progressive CurriculumSampler (easy→hard by flat-band ambiguity) wired at `_finalize_training_bundle`. Both config-gated off (`training.curriculum_*`, `training.anchored_*`).

**PHASE 2 CODE-COMPLETE 2026-07-14.** All five items landed (suite 864 passed). Influence features ship measured-and-gated: memory damping off (credit gate negative), arbitration off (needs a winning scorecard row), curriculum/anchored/adaptive-budget off (experiment knobs for ⟲3+). Phase 3 entry still requires scorecard-VERIFIED improvement rows from items 1–2 — the machinery exists; the wins don't yet.

Each item: **→ verify** = a scorecard row (or paired-run delta) demonstrating non-degradation; memory influence additionally requires counterfactual-credit > 0 on OOS folds. *(2026-07-15 amendment: any such delta must also clear the §2.0 noise floor under the floor-appropriate protocol.)*

**Re-entry criteria for the gated-off influence features (2026-07-15):**
- **Memory damping** (off; credit −1.3 bps, p=0.94): re-run `scripts/run_memory_credit.py` only after the next §2.0-verified brain improvement or a bank rebuilt from a materially better encoder — re-measuring against an unchanged brain re-measures the same thing.
- **Arbitration — RESOLVED 2026-07-17: primitives leave the roadmap (do-NOT-list ethos applied early).** The measured answer (−12.52% vs brain-only −0.88% OOS) plus standalone-negative primitives (momentum −25%) means the capped-budget re-test would be new warm-up code in service of a measured loser. The `StrategyArbiter` plumbing and `strategy_id` attribution STAY (zero cost, useful for any future strategy with standalone merit); the momentum/mean-reversion primitives are removed from the experiment queue (§8). Re-entry bar: a primitive earns arbitration only with its OWN standalone OOS scorecard row ≥ buy-and-hold first.
- **Adaptive budget / curriculum / anchored negatives** (off): experiment knobs, enter through the §9.4 queue under the floor protocol; anchored negatives additionally need the store populated by ≥1 triggered retrain (producer shipped 2026-07-14, store currently empty).

---

## 7. Phases 3–4 — research adoption, detailed (refreshed 2026-07-15; supersedes the thin menu)

Task detail lives HERE now; RESEARCH_ROADMAP.md §4 keeps the research rationale and citations. The do-NOT-do list (ROADMAP §5) is binding on everything below. Gating: **§7.0 and §7.1 are UNGATED** (measurement and operations, not model research — they produce the evidence and safety the gated work needs). §7.2 items stay gated as marked. All accuracy claims are subject to the §2.0 noise-floor rule.

### 7.0 Measurement first (ungated — §9 steps 1 & 3 point here)

1. **[✅ done 2026-07-17] Full-data noise floor = 0.92pt val / 2.00pt test BA** (seeds 7/13/29 → val 0.4384/0.4348/0.4440, mean 0.4391; `runs/sweep_20260717_142927_run00{1,2,3}.json`). Written into §2.0 with the protocol decision: floor is intermediate but the 3-seed paired protocol is mandatory (both ⟲2 +0.44pt and ⟲3 −0.31pt sit under it); the single-run shortcut is retired. History, so no future session mis-reads the earlier artifacts:
   - **CPU sweep** `sweep_20260715_154016` (launched 15:40 UTC): seed 7 → val BA **0.4405** (78.2 min), seed 13 → **0.4460** (78.9 min); seed 29 never ran — the sweep was superseded mid-flight by the GPU migration. The pair is **reference only** (different arithmetic path than deployment).
   - **GPU migration, same evening:** torch 2.11.0+cu128 on the RTX 4070 Ti; `device.use_amp: true` became real AMP; full-data training ~2× faster (78 → 39 min/run); smoke run in `runs/gpu_smoke.json`. **Decision: the authoritative floor is the GPU floor** — the floor must be measured on the arithmetic path that produces every future candidate, and that path is now GPU.
   - **GPU sweep** `sweep_20260715_184357` (launched 18:43 UTC): seed 7 → val BA **0.4381** (39.4 min) — then the machine was shut down at 19:25 UTC, two minutes after the summary landed; seeds 13/29 never started (§10 note 10). That run is a 4th reference sample.
   - **Relaunch 2026-07-17 14:11 UTC:** fresh full 3-seed GPU sweep (not a 13/29-only resume — after a 2-day gap the per-run bar top-up would conflate seed variance with window drift; ~40 extra GPU-minutes buys same-day windows). Launched via Task Scheduler (`SETAI_noise_floor_sweep`, disabled after start) → `scripts/launch_noise_floor_sweep.bat` → `logs/sweep_gpu_20260717.log`.

   Caveat that held: each sweep run tops bars up to "now", so seeds saw windows differing by ~40–80 min of bars out of 4.5y — negligible against the 0.92pt spread. Reference samples from the interrupted sweeps agree closely (CPU seed 7 0.4405, GPU seed 7 0.4381 vs the clean GPU seed 7 0.4384), which is why seed identity, not window drift, is what the floor measures.
   **→ verify:** ✅ §2.0 updated with the spread + protocol decision; all three summaries committed under runs/.
2. **[S] Horizon-vs-cost audit — ✅ done 2026-07-17** (`scripts/audit_horizon_costs.py`, verdict in `docs/analysis/HORIZON_COST_AUDIT.md`, artifact `runs/horizon_cost_audit.json`; 1.13M pairs/horizon). Result in one line: **the timeframe is NOT a hard ceiling (moves clear 6 bps at 62–95% rates), but the configured horizons [1,3,6] demand 60–70% directional accuracy to break even (66–79% on SPY/QQQ/VOO), while H=15–30 bars need only ~54–58%** — so horizon extension is the cheapest structural cost lever, promoted into the §9.4 queue. Precondition discovered while writing it: labels had NO session-boundary handling — H-bar windows near the close included the overnight gap (~8% of samples at H=6, 22–42% at H=15–30). **Precondition ✅ shipped 2026-07-17** (`training.session_aware_labels`, on by default, in the cache fingerprint; commit `2338f26`) — the horizon-extension retrains are unblocked.
   **→ verify:** met — deterministic script + table + go/no-go committed; §7.2/§9.4 experiments must state targets against the audit's breakeven rates.
3. **[M] Platt refit from realized outcomes.** Live confidence pins at 0.488–0.515 — the EV gate can barely discriminate, so trade selectivity (the main no-retrain cost lever) is idle. Data path exists as of 2026-07-15: join `runs/decisions.jsonl` (confidence, `client_order_id`) with `runs/observations.jsonl` (`decision_id` = client_order_id, realized net PnL). **Pitfall — opening fills:** `realized_pnl` is nonzero only on closing/reducing legs; an opening fill's 0.0 is "unresolved", NOT a loss. Use only observations with |realized_pnl| > 0 (or aggregate per round trip); needs ≥ ~300 such resolved trades — the paper week supplies them. Reuse the fit in `src/training/platt_calibration.py` (win = realized_pnl > 0 replaces the return-sign label; the direction-aware outcome construction pattern is `direction_aware_outcomes`). Write a per-checkpoint sidecar through the existing fingerprint binding; canonical sidecar only via the normal gate.
   **→ verify:** reliability curve before/after; live confidence range measurably widens; `negative_ev` share of rejections changes (today it is near-constant because confidence is).
   *Tooling shipped 2026-07-17:* `scripts/refit_platt_live.py` (join + opening-fill exclusion + min-samples refusal + reliability/ECE/Brier before-after + isolated fingerprint-stamped artifact `runs/platt_live_refit.json` — the canonical sidecar still moves only through the gate; `tests/test_refit_platt_live.py`). First real run: 133 resolved trades of the required 300 (146 opening fills correctly excluded, 0 unmatched) → refused, as designed; **149/300 by Friday close**. Re-run after the paper week.
4. **[S] EV-input quality audit — ✅ done 2026-07-17** (`scripts/audit_ev_inputs.py`, verdict in `docs/analysis/EV_INPUT_AUDIT.md`, artifact `runs/ev_input_audit.json`; DecisionEvent + backtest decision log now carry the EV legs, additive + tolerant, `tests/test_ev_input_audit.py`). Result: over 77,631 OOS decisions, Spearman predicted-vs-realized is +0.15/+0.26 POOLED but **≈0 per symbol** (median +0.02) — the risk head learned cross-sectional vol ordering, not timing — and magnitudes are ~5× over-predicted (gain 137 vs realized 25 bps). **Recorded decision: the EV gate is a symbol-level vol prior + cost floor, not a per-decision signal, until the risk head improves; §4.4.4 re-scoped accordingly; §7.0.3 (Platt refit) is now the only live selectivity lever.** Improvement claims must move PER-SYMBOL correlation (pooled inflates on cross-sectional vol alone).
   **→ verify:** met — correlations + decision recorded in the audit doc; live DecisionEvents accumulate the paper-side dataset from the next launch.
   *Fix shipped 2026-07-17 (gated off):* `training.risk_target_per_symbol_norm` — mfe/mae targets divided by the symbol's train-split median |excursion| (post-cache trainer transform, raw sample caches untouched), scales persisted in checkpoint meta (`risk_target_scales`), `BrainPolicy._denormalize_ev` restores raw fractions per symbol on both inference paths, old checkpoints unaffected (`src/training/risk_target_norm.py`, `tests/test_risk_target_norm.py`). In normalized units every symbol's typical excursion is ~1.0, so the vol-level answer no longer minimizes the loss — the head must learn timing. Flipping it is a §9.4 experiment; success metric = PER-SYMBOL Spearman moving off ≈0 in this audit, not pooled.
5. **[S] Calibrated-selectivity threshold sweep (added 2026-07-17; after §7.0.3 lands).** Once confidence is calibrated on realized outcomes, sweep a minimum calibrated P(win) on the EV gate over OOS backtests (e.g. 0.50–0.60 in 0.01 steps); report the expectancy-vs-fill-count curve and pick the knee. This is the follow-through the §7.0.3 verify clause implies — calibration without a selectivity consumer changes nothing.
   **→ verify:** curve in the audit doc; the chosen threshold's OOS net expectancy beats no-threshold with fill count above a stated floor; threshold enters config as an explicit knob, default off until the curve exists.

### 7.1 Operational hardening (ungated — from paper-run evidence, 2 incidents in 2 days)

1. **[S] Failed-submit backoff — ✅ done 2026-07-17**, exactly per the pre-made decisions: `submit_failed_until` map on `RuntimeContext`, armed in tick_processor's not-accepted branch on terminal `submit_failed`, checked before order construction; `execution.submit_failure_cooldown_sec: 300.0` in both `ExecutionConfig` and `default.yaml`; `submit_failed_cooldown` log + `submit_failed_cooldown_skips` counter. Regression tests: `tests/test_submit_failure_cooldown.py` (same-side skip, other-side free, time-based expiry, fills clear nothing, zero disables, call-site source contracts).
   **→ verify:** met (unit + source-contract tests); live confirmation rides the next paper session.
2. **[S] VOO decision — ✅ executed 2026-07-17:** VOO removed from `runtime.symbols` in `configs/default.yaml` (with an in-file comment so nobody "fixes" the runtime/data asymmetry back); `data.symbols` untouched, cache fingerprints and checkpoint contract unaffected (suite green, no rebuild).
   **→ verify:** staleness ≈ 0 confirmation rides the next paper session (the config takes effect at next launch).
3. **[S] Unattended-week logistics.** The runtime must be launched OUTSIDE the assistant session — session-restart survival proven three times by 2026-07-17 (four separate sessions on 2026-07-15 alone; today's `20260717p2488` rehydrated carried positions correctly at 13:56 UTC). The binding constraint is now the MACHINE, not the software: see §10 note 10. Daily checklist and end-of-week drills: §9.2.
4. **[S] PLTR churn watch (observation, decision deferred).** Even with the `min_trade_notional` deadband, fills concentrate hard: 75 of 265 July fills are PLTR (28%), and PLTR+NVDA+TSLA = 72% — the same high-vol names the horizon audit says carry the tradeable volatility, so concentration per se is not wrong. Candidate follow-ups if the full week shows dust-churn rather than signal there: raise `risk.min_trade_notional`, or make the deadband %-of-position. **Explicitly a week-of-data decision — do not tune it early.**
5. **[S] Per-symbol cost model in the EV gate (added 2026-07-17).** The gate's cost side is a global 6 bps, but real per-symbol economics differ wildly: MSTR's quoted spreads alone tripped `spread_too_wide` 418 times in one morning, while SPY's breakeven is unreachable at any configured horizon (§7.0.2). Make cost per symbol: `cost_bps(symbol) = fee + slippage + measured median half-spread`, with the median spread measured from the paper week's live quotes. Cost-hopeless trades then price themselves out honestly instead of via ad-hoc symbol lists. Data source: the week (add spread stats to the §9.2 daily checklist).
   **→ verify:** risk gate consumes per-symbol costs (config pair + test); an OOS backtest shows ETF fill share dropping with expectancy not degrading.
6. **[S] Position guard — ✅ done 2026-07-17** (user-requested; `src/risk/position_guard.py`, `risk.position_stop_loss_pct` / `position_take_profit_pct`, `tests/test_position_guard.py`, commit `4c2c744`). Deterministic unrealized-P&L exits for held positions, checked every synced bar in BOTH engines (§10.4): a breach originates a FLATTEN riding the normal risk-reducing path (halt-exempt, deadband-exempt, outside the EV gate, one attempt per bar, exit-cooldown blocks instant re-entry; works under kill-switch). End-to-end pinned: a crashing long is liquidated and beats riding it. **OFF by default (both 0.0)** — pick thresholds from a backtest sweep before enabling live; candidate week-2 knob after the clean week baseline exists.
7. **[M] Horizon-aligned holding period (added 2026-07-17; experiment knob, backtest-first) — PROMOTED by arm-3 evidence 2026-07-18.** The model is trained to predict H-bar outcomes (H=15 incumbent) but the runtime may flip the position at the next bar — execution and objective are misaligned, and every early flip pays the round trip without collecting the predicted move. **Arm 3 (§9.4) made this concrete and measured:** training at H=30 *without* this knob was worse 3/3 on OOS net expectancy — the horizon-agnostic per-bar EV gate simply traded ~2× more on the larger expected moves, never holding to collect the 30-bar move, so the audit's lower breakeven (55.6%) was never realized. **Horizon-extension training and this knob are therefore coupled: neither pays off alone.** The natural re-test is H=30 *with* holding-to-30 (this item) as a single combined arm. Knob: a directional position opened on an H-bar-label model is re-decidable only after H bars (halts/kill/daily-loss exempt — risk always overrides). Cuts turnover ~H× if the model's churn is noise; hurts if mid-horizon reversals carry signal — that is exactly what the backtest measures. OFF by default; shared semantics between backtest and runtime or it's a parity bug.
   **→ verify:** backtest turnover drops ≈H× with the knob on; OOS expectancy delta reported under §2.0; parity test pins identical behavior in engine and tick processor.

### 7.2 Phase 3 items (gated as marked; roadmap numbering kept)

1. **[M] Familiarity/OOD gate** — *gate: a §2.0-verified win first; then build v0 before v1.* v0 (cheap, no new training): familiarity = kNN distance percentile against the existing memory bank, consumed through the SAME shrink-only damping rails as §6.2 (multiplier ∈ [floor, 1.0], never grow/flip/veto). Implementation shape: extend `memory.influence_mode` to a third value `"familiarity"` in `MemoryConfig` (remember its field allow-list in `tests/test_config_hygiene.py` if fields are added), apply at the same call site as damping (grep `influence_mode` for it), and copy the rail-pinning test structure of `tests/test_memory_damping.py` wholesale — bounds, never-grow, action/confidence untouched, temporal-cutoff shield, default-off. v1 (roadmap's masked-reconstruction readout + MAE pretraining) only if v0 shows credit — do not build the readout before the cheap variant earns it.
   **→ verify:** paired same-seed OOS folds (the `run_memory_credit.py` harness pattern), credit > 0 required before default-on.
2. **[M] Regime snapshots — offline half ✅ done 2026-07-17** (`src/evaluation/regime_snapshots.py`, `scripts/build_regime_snapshots.py`, artifact `runs/regime_snapshots.json`, `tests/test_regime_snapshots.py`). Per-session K-means (seeded local kmeans++, no sklearn) over the regime feature family (cross-symbol session-close mean of trend strength / vol ratio / volume ratio / return alignment); Mahalanobis distance to the assigned cluster with a p95 novelty threshold; K = `evaluation.regime_snapshot_k` (=6, its own knob per the decision). **First consumer wired:** the §4.5.7 backward-transfer gate slices by regime label by default (`evaluation.promotion.backward_transfer_slicing: regime`, chronological fallback when clustering is impossible; slices align by name because one shared clustering labels both logs). Real-data verify: 1137 sessions → 2022-bear dominant cluster (r1: 76→11→2 sessions across 2022→2024→2026) vs 2024-bull dominant (r3: 25→130) **SEPARATED**; held-out novelty (fit < 2026-03-10) flags 4/90 sessions at p95 — calibrated. Known caveat: partial sessions (≥12 bars) are admitted and skew volume features — today's half-session was the top novelty hit; harmless for slicing, mind it for the online half. **Online half (novelty event → shrink-only risk-cap tightening) stays future work** — it touches the live risk path and enters with the same rails as damping. Never hard-swap models on a regime signal (do-NOT #9).
   **→ verify:** met — synthetic bear/bull separate (test), real 2022-vs-2024 separate (script output above), novelty fires on a shocked held-out session (test), slice-gate wired to regime labels with the fallback pinned by source-contract test.
3. **Market context — code DONE (P3.1), verdict OPEN.** Re-test is an experiment under the §9.4 queue with the floor-appropriate protocol; the ⟲3 candidate already exists as one sample. No new plumbing.
4. **[M] Episodic embedding training** — *gate: memory counterfactual credit > 0 (currently NEGATIVE, −1.3 bps, p=0.94).* Parked until the credit gate flips; building retrieval-training before retrieval earns credit is inverted.
5. **[M] Sweep upgrade (Optuna TPE + ASHA)** — *gate: multi-seed machinery first (§2.0), which is now partially built (seed axis, 2026-07-15).* Fidelity audit BEFORE any pruned search — and do it honestly: the 3 shared seeds from the two noise-floor sweeps only permit a qualitative check (does the best/worst seed agree between fast and full? n=3 supports no correlation coefficient — do NOT compute Spearman on 3 points and report it as evidence). A real audit needs ≥8 paired configs: run the first 8 Optuna trials at BOTH fidelities, then correlate; only enable ASHA pruning if rank agreement is strong. Anti-snooping: objective = train/val walk-forward only; OOS scorecard touched exactly once per promoted candidate.
   **→ verify:** the ≥8-pair fidelity correlation reported before any pruned search; every promoted config confirmed by a full multi-seed run clearing §2.0.
6. **[M] Bootstrap-only synthetic stress — ✅ done 2026-07-17** (`src/evaluation/bootstrap_stress.py`, `scripts/run_bootstrap_stress.py`, `backtest.adverse_fills`/`adverse_fill_range_fraction`, `tests/test_bootstrap_stress.py`, artifact `runs/bootstrap_stress.json`). Session-block bootstrap conditioned on §7.2.2 regime labels (whole real sessions, same sampled sequence for every symbol, rebased splices on a synthetic 2030 timeline); stylized-facts gate allows flags inherited from real crash bars and REFUSES splice artifacts. Adverse-fill mode pushes next-open fills against the trade by a bounded range fraction. `evaluate_promotion(stress_metrics=…)` records the deltas verbatim on the promotion record; a test pins that they can never change eligibility. First run (incumbent, clusters {1,2} = the bear/stress regimes, 60 sessions): **baseline +0.12% / Sharpe +0.97 flips to −0.65% / Sharpe −5.12 under adverse fills** (expectancy +0.05 → −0.24, 2,648 fills) — on rare regimes the whole edge sits inside the fill assumption, consistent with "signal real, below costs".
   **→ verify:** met — inherited-vs-introduced flag gate test-pinned; promotion record carries baseline-vs-adverse deltas (informational only, test-pinned).
7. **[L] Memory plasticity schedules** — *gate: same as item 4 (credit > 0).* Parked.
8. **[L] Regime-tagged specialist archive** — *gate: item 2 shipped + a §2.0-verified win.* Specialists as distillation teachers only, never live swaps. Parked.
9. **[S] Governance leftovers** — most of this shipped in Phase 1/2 (promotion registry + notes, learned-proposes tests). Remaining: per-retrain error-analysis artifact. Low priority; do with the next retrain, not before.
10. **News program (added 2026-07-17)** — full plan in **`NEWSBOX_PLAN.md`**: point-in-time Alpaca/Benzinga archive (N0, ungated — API access VERIFIED on the existing account, historical to 2022), offline event study (N1), shrink-only event-risk gate (N2), news features through the contract (N3), local-LLM extractor last and only if boring extractors show signal (N4). Constitutional rules: news becomes FEATURES never orders; point-in-time or it didn't happen. N2+ gated behind the current §9.4 arms / the §9 step-5 stopping rule.

### 7.3 Phase 4 — menu COLLAPSED 2026-07-17: research breadth before an edge is a liability

With the floor measured and no OOS edge yet, the wide menu was inventory, not options. What remains:
- **Caruana/seed-ensemble — PULLED FORWARD (the pull-forward rule, exercised):** the three floor checkpoints already exist, so this is §9.4 arm 2 at zero training cost.
- **PBT-lite — parked** behind the §9 step-5 edge milestone (population methods amplify an edge; they do not create one).
- **Cut to §8:** SOM regime map (the §7.2.2 K-means snapshots cover the useful half), differentiable Hopfield retrieval, meta-gradients, capacity growth, generative strategy discovery, cross-play pool — each may be revived individually, with a stated reason, only after the edge milestone. The do-NOT list stays binding regardless: no GAN generators, no per-bar test-time training, no wholesale ESOM/Hopfield restoration.

---

## 8. Cut from the plan (v2)

Removed or parked — each either served no link in the accuracy chain or duplicated other documents. Revive only with a stated reason.

| Item (v1 location) | Disposition | Why |
|---|---|---|
| True ring buffer for BarBuffer | parked | Perf-only; hot path already O(1) amortized; zero accuracy effect |
| Multi-symbol historical requests (~13× fewer calls) | parked | Perf-only; fetch is a rare offline operation |
| Backtest preset zoo consolidation + GUI preset descriptions | parked | Presets now truthfully labeled (1F); cosmetic beyond that |
| Config provenance logging / resolved-config sha256 | parked | Observability nicety; hygiene test already fails loudly on drift |
| Unify config error contract (ConfigurationError vs ValidationError) | parked | Cosmetic; both fail loudly at startup |
| Episode-parallel backtest restructuring (Phase 4) | parked | Perf-only; revisit only if PBT/cross-play actually lands |
| Governance: schema-validated model-change statement | folded | One line in the promotion record (§4.5.4) suffices at this scale |
| Self-evolution "autonomy ladder" config | cut | Speculative config surface; the human promotion gate IS the policy |
| Calendar 7-day-cycle / exchange-tz feature rework | folded into §4.3.3 sanitation pass | Marginal; weekend bars barely occur for equities |
| `seed_everything` PYTHONHASHSEED note | folded into §4.7 trainer hygiene | Trivial |
| Per-session regime "snapshot" logging pre-Phase-3 | cut | Duplicates Phase 3 regime work; premature |
| Webhook alert full redesign | reduced | Only the hot-path fix stays (§4.7) |
| `--use-memory` runtime_update_mode plumbing repair | cut | Dead by design until Phase 2 memory work; deleting beats repairing |
| Doc rewrite of `LIVE_LOG_CHEATSHEET.md` as new doc | reduced | Delete or minimal rewrite in §4.7; new operator docs come with the scorecard |
| Arbitration momentum/mean-reversion primitives | deleted 2026-07-17 | Measured −12.52% vs brain-only −0.88%; momentum −25% standalone. Plumbing + attribution retained; re-entry needs a standalone OOS row ≥ buy&hold (§6 re-entry note) |
| Phase-4: SOM regime map | cut 2026-07-17 | §7.2.2 K-means snapshots + Mahalanobis novelty cover the useful half |
| Phase-4: Hopfield retrieval, meta-gradients, capacity growth, generative discovery, cross-play pool | cut 2026-07-17 | Research breadth before an OOS edge is a liability; revive individually after the §9 step-5 edge milestone |

---

## 9. Immediate next actions (strict order) — refreshed 2026-07-15

*(The previous v2 list — labels → calibration → scorecard → ⟲1 → loop → hygiene — completed 2026-07-10 → 2026-07-14; history in §1 and §4.9. Ordering principle unchanged: measurement before optimization, operations before experiments.)*

1. **✅ Full-data noise floor read 2026-07-17 = 0.92pt val / 2.00pt test BA** (seeds 7/13/29, §2.0/§7.0.1). The fork resolved to the **large** branch: every accuracy claim needs the 3-seed paired protocol (single full-path runs are NOT interpretable — both ⟲2 and ⟲3 sit under this floor). Batch hypotheses to amortize the 3× cost. **Seed-averaging/ensembling is NOT auto-promoted to the front** — at 0.92pt the floor is real but not catastrophic (fast mode was 6× worse), so ordinary candidate effects can still clear it with the 3-seed protocol; ensembling stays an available §7.3 pull-forward, not a forced first move. (An initially-reported "long-recall 0.0" flag was an extraction error — per-class recalls are healthy across seeds; §2.0 erratum.)

2. **The unattended paper week** (Phase 0 sign-off, still owed). Day one (2026-07-15) already paid for itself: four defects found and fixed same day (fractional shorts, staleness < bar period, churn deadband, submit-failure map leak — commits `2b6a44a`, `96e64d7`). For the real week: **double-click `scripts/launch_paper_week.bat`** (added 2026-07-17) — it runs `scripts/preflight_monday.py` (11 GO/NO-GO checks: interlock, VOO drop, backoff, bar gate, credentials, bank fingerprint, disk) and only then starts `run_paper.py` in a visible, titled console that must stay open all week. Launching OUTSIDE the assistant session remains the rule. Daily review checklist: fills vs deadband-reject ratio, reason-code distribution (fail if `negative_ev`/`stale_data` dominate — with VOO out of `runtime.symbols` (§7.1.2), ANY staleness block is now anomalous), `runs/observations.jsonl` growth, equity tracks broker, **resolved-trade count vs the 300 Platt threshold (149 banked as of Fri), median holding time + turnover (evidence for §7.1.6), per-symbol spread stats (input for §7.1.5)**. End of week: forced-outage drill + simulated daily-loss trip (the Phase 0 checklist), then **Phase 0 CLOSED**.
   *Status 2026-07-17: the clean week has NOT started.* Wed = day-of-fixes (valuable, not unattended); **Thu = machine off all day, zero trading** (order journal has no 2026-07-16 entries); Fri = one 2.5-minute session (13:55–13:57 UTC) ended by a closed console. Four long positions (AAPL, AMZN, META, NFLX, ≈$4.3k notional) are parked on the paper account with no runtime attached — benign on paper, and reconciliation re-adopts them at next launch. **Restart the week Monday 2026-07-20** with the §10-note-10 machine-on commitment; the VOO drop and failed-submit backoff are live from the next launch, so the week also serves as their live verification.

3. **Cost-bound work before more training.** The perturbation harness already established: the signal is real but below costs. Two no-retrain tasks, in order:
   1. **Horizon-vs-cost audit — ✅ done 2026-07-17** (§7.0.2, `docs/analysis/HORIZON_COST_AUDIT.md`). Verdict: timeframe is not a hard ceiling; the configured horizons are the expensive part of the curve (breakeven 60–70% there vs ~54–58% at H=15–30) → horizon extension enters the §9.4 queue with the session-aware-labeling precondition.
   2. **Calibration refit from realized outcomes** (pulled forward from the §7 Phase 3 menu — it is observation-driven, not training-driven). Once the paper week's observation stream has a few hundred records, refit Platt on realized trade outcomes and verify live confidence stops pinning (today's live range: 0.488–0.515 — the EV gate can barely discriminate). Confidence → EV gate → trade selectivity is the main no-retrain cost lever.

4. **Edge experiments — money-per-effort order (re-engineered 2026-07-17), protocol per (1), claims per the §2.0 rule.** One change per arm; each arm is judged ONLY on its named metric. Cheap no longer means first — expected effect on NET-OF-COST OOS expectancy orders the queue:
   1. **Platt refit + selectivity threshold** *(no retrain; unblocks mid-week at 300 resolved trades — 149 banked)*: run `scripts/refit_platt_live.py`, then §7.0.5's calibrated-confidence-floor sweep. Named metric: OOS net expectancy at a bounded fill-count drop.
   2. **Seed-ensemble scorecard — ✅ MEASURED 2026-07-17: NO BENEFIT, crossed off.** Majority-vote ensemble of the three floor checkpoints on the shared OOS window (from 2026-03-20; `runs/seed_ensemble.json`, `scripts/run_seed_ensemble.py`): ensemble BA **0.3609** vs seeds 0.3652–0.3659 and incumbent 0.3661; expectancy −1.23/trade vs incumbent −0.14. Voting reduced nothing — variance reduction cannot fix a negative mean. Suggestive tier, but decisively the wrong direction; do not retune. Side-finding worth keeping: seeds nearly tied on val BA diverge enormously on the TRADING surface (seed 29: −0.015/trade on 1,533 fills vs seed 7: −1.26 on 2,575) — the noise floor on expectancy is far larger than the 0.92pt BA floor, which strengthens the §2.0 rule for any expectancy claim.
   3. **Horizon-extension program — ✅ MEASURED 2026-07-18: NO BENEFIT, worse 3/3, crossed off.** 3-seed retrains (7/13/29) at H=15 (baseline, `configs/training.yaml` primary=15) vs H=30 (candidate, `configs/experiments/horizon_30.yaml`, the ONE change `primary_direction_horizon:30`), both with `session_aware_labels` on, one batch → shared window; scored on one OOS cutoff (2026-05-05→07-17), each at its own native horizon (`scripts/score_horizon_arm3.py`, `runs/horizon_arm3.json`; checkpoints `checkpoints/sweep_20260718_065658` / `_072039`). **Paired OOS net expectancy (H=30−H=15): −1.64 / −0.32 / −0.64 — signs `---`, mean −0.87/trade, 0/3 positive** (val-BA also −0.006/−0.007/−0.000, under floor). Suggestive-tier by §2.0, decisively the wrong direction — same disposition as arm 2. **Mechanism (from the numbers):** the backtest EV gate is horizon-agnostic (per-bar, `build_brain_policy`); horizon feeds only the direction *target* + the BA metric, NOT the holding period. The H=30 model was trained on larger 30-bar moves → bigger expected-gain signals → the EV gate fires ~2× more (689–1065 trades vs 317–425) with **no** better directional accuracy → more cost bleed. So the audit's lower breakeven (57.7%→55.6%) is never realized: **it assumed you HOLD 30 bars to catch the 30-bar move, but the engine exits per-bar.** Consequence: horizon-extension training is COUPLED to §7.1.6 (horizon-aligned holding); extending the label horizon alone is measured-negative. Side-finding: session masking (max_horizon=60) dropped ~73% of samples and cost the H=15 arm ~3pt val BA vs the unmasked 2026-07-17 noise-floor runs (0.409 mean vs 0.439). Also: no arm reached positive OOS net expectancy AND >buy&hold (best row H=15 seed7 expectancy +0.019 but excess −0.0005) — the §9 step-5 edge milestone is still unmet.
   4. **Risk-target per-symbol norm — ✅ MEASURED 2026-07-18: NO BENEFIT, crossed off.** 3-seed (7/13/29) knob-ON (`configs/experiments/risk_target_norm.yaml`, the ONE change `risk_target_per_symbol_norm:true`, primary=15) vs knob-OFF baseline = arm-3's H=15 checkpoints (byte-identical to a fresh knob-off run — the knob is a post-cache transform, so the knob-ON runs reused the same sample cache; checkpoints `checkpoints/sweep_20260718_080530` carry `risk_target_scales`, the baseline `sweep_20260718_065658` does not). Audited both via `scripts/audit_ev_inputs.py` (§7.0.4), paired per seed (`scripts/score_risk_norm_arm4.py`, `runs/risk_norm_arm4.json`). **Named metric — per-symbol Spearman — did NOT move off ≈0; it moved TOWARD/below it:** loss/MAE per-symbol median Δ(on−off) = −0.099/−0.010/−0.128 (signs `---`, 0/3 up), gain/MFE = −0.055/−0.061/+0.019 (`--+`, 1/3 up); symbols with both corr >0.05 went 6/6/0 → 1/1/3. **Why it fails:** dividing the target by the per-symbol median |excursion| removes the level, but the head doesn't discover per-symbol *timing* to modulate around 1.0 — it predicts the normalized ≈constant, so after de-normalization the per-symbol *variation* (what the audit measures) collapses toward 0. Pooled stayed ~flat/slightly up (loss 0.28→0.35) = exactly the cross-sectional-vol ordering §7.0.4 discounts. **Conclusion: the risk head's per-decision blindness is not a target-scaling problem; §7.0.4 stands (EV gate = symbol-vol prior + cost floor).** So the EV legs remain uninformative and **arm 1 (Platt refit + calibrated-confidence threshold) is the last no-retrain lever that can make the EV inequality discriminate** — its weight goes up. Same disposition as arms 2 and 3.
   5. **Curriculum** *(demoted 2026-07-17: cheap, but it carries no cost/money thesis — it rides along whenever a retrain batch runs anyway)*. Named metric: val BA under §2.0.
   6. Tail: **anchored negatives** (still blocked — store empty until a first triggered retrain) and **market-context re-judge** (existing ⟲3 candidate only; suggestive-tier evidence at best).

5. **The edge milestone and the stopping rule (added 2026-07-17).** "An edge exists" means exactly one thing: **an OOS scorecard row with positive net expectancy AND excess vs buy-and-hold > 0, under the §2.0 three-seed protocol.** Every §7.2 gate that reads "a §2.0-verified win" points at this milestone. **Stopping rule:** if §9.4 arms 1–4 complete and no configuration reaches positive OOS net expectancy, the 5-minute/OHLCV thesis is declared EXHAUSTED — no further knob-tuning; convene the pivot decision instead. **Status 2026-07-18: arms 2, 3, 4 all MEASURED-NEGATIVE (seed-ensemble, horizon-extension, risk-target norm); no config has cleared the milestone. Only arm 1 (Platt refit + calibrated-confidence threshold) remains — and it is DATA-GATED, not runnable until the paper week banks ~300 resolved trades (149 as of 2026-07-17). So the pivot decision is one data-gated arm away; if arm 1 also fails to produce a positive-expectancy row, the thesis is exhausted and the pivot menu below is convened.** Pivot menu (decide then, not now): (a) rebuild on 30–60-minute bars, where the §7.0.2 breakeven is materially lower; (b) features beyond OHLCV — **now concretely specified as the news program in `NEWSBOX_PLAN.md`** (its N0 archive is ungated and may start immediately; its N2+ consumers wait for this stopping rule or for the current arms to resolve); (c) universe restriction to the cost-viable high-vol names. Phase 3's remaining gated items (familiarity gate, Optuna, episodic embeddings, plasticity, specialists) stay gated until the milestone; Optuna additionally waits for the ≥8-pair fidelity audit, and enters *with* the multi-seed machinery.

6. **Live-mode discussion happens only after** the paper week is green AND the §9 step-5 edge milestone is met. Standing rule 1 (paper-first, double interlock) unchanged.

*Traceability: v1 of this plan (git `00523e7`) holds the original full enumeration; CODEBASE_REVIEW.md §5/§7 items remain covered by the verify-as-you-touch rule; nothing was dropped silently — every removal is in §8.*

---

## 10. Execution notes for whoever picks up a task (added 2026-07-15 — read before coding)

Hard-won repo mechanics that are NOT obvious from the code and have each burned a session:

1. **Config changes come in pairs.** Any new/changed default goes in BOTH the pydantic model (`src/core/config/*.py`) and `configs/default.yaml`, or `tests/test_defaults_alignment.py` fails. Deliberate divergences need a reasoned entry in its `DOCUMENTED_DIVERGENCES`. New `MemoryConfig` fields additionally need the allow-list in `tests/test_config_hygiene.py`. Files under `configs/` that are NOT AppConfig overlays (sweep grids) must be listed in that test's `NOT_APP_CONFIG`.
2. **Every fix ships with its regression test in the same commit** (§5 rule 9), and the full suite (`.venv\Scripts\python.exe -m pytest -q`, ~20s, currently 962 passed) runs before every commit. If a safety test fails after your change, suspect your change first — `test_risk_correctness.py` has already caught one too-broad "fix" this week.
3. **Never touch canonical artifacts directly:** `checkpoints/brain.pt`, `checkpoints/registry.json`, `artifacts/market_memory.pt`, and the canonical Platt sidecar move ONLY through `promote_checkpoint`/the gate. Experiments run isolated via `run_train.py --run-id <name>` (auto-isolates) and write scorecard rows, not promotions.
4. **Sizing/eligibility logic goes in the SHARED risk path** (`src/risk/risk_manager.py`), never in the live adapter — otherwise the backtest books fills the broker rejects (this exact divergence produced three live incidents on 2026-07-15).
5. **Broker physics learned so far (Alpaca):** no fractional short sells; no fractional orders that flip position side; bars are timestamped at bucket START (a bar is one full period old at bucket close — that is not stale data); VOO barely prints on the IEX feed.
6. **Claims follow §2.0.** No "improved/regressed" statements from single runs — use the §2.0 recipe verbatim. Fast mode prioritizes; it never proves.
7. **Windows shell traps:** commit messages with quotes/parentheses break PowerShell 5.1 argument quoting — write the message to a file and `git commit -F <file>`. Long-running jobs (paper runtime, sweeps) survive assistant-session restarts only if launched OUTSIDE the session.
8. **When a background run seems wrong, read its log before touching anything** — both 2026-07-15 retry loops looked like broker flakiness and were actually deterministic order-shape bugs, diagnosable from a single `order_transition` line.
9. **Interrupted-sweep recovery.** Each sweep run writes its `runs/sweep_<id>_runNNN.json` on completion, so completed seeds survive any interruption. If the gap is HOURS: resume the missing seeds via a partial grid yaml (`sweep: {seed: [13, 29]}`) and pair with the completed summaries. If the gap is DAYS: re-run the full grid instead — every run tops bars up to "now", so a stale completed seed would conflate seed variance with window drift (this is why the 2026-07-17 relaunch re-ran seed 7 rather than resuming; one ~40-min GPU run bought same-day windows). `scripts/launch_noise_floor_sweep.bat` is the canonical relauncher — start it through `launch_noise_floor_sweep_hidden.vbs` (or the trigger-less Task Scheduler task `SETAI_noise_floor_sweep`) so no console window exists to close: the first 2026-07-17 relaunch died at minute 4 to a window-CLOSE event (`forrtl: error (200)` in the log), not a crash.
10. **The machine being OFF is the top operational killer — three for three this week.** The GPU sweep died because the box was shut down 2 min after seed 7's summary landed (Wed 19:25 UTC); Thursday the box was powered on for ~2 minutes total, so the "paper week" had a zero-trading day; Friday's trader ran 2.5 minutes before its console window was closed. None of these were software failures — restart survival and crash-free operation are proven; power/window management is the binding constraint. For any unattended run: the machine stays ON (check Windows sleep settings), the process launches via Task Scheduler or a dedicated terminal that nobody closes, and a shutdown is treated as a deliberate end-of-run decision. There is currently NO auto-relaunch task for the paper runtime — launching it stays a deliberate manual act (drills and interlocks depend on knowing when it runs).
