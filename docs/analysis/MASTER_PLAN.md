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
| ⟲ #1 | First honest retrain | ⬜ **open — operator step** (needs Alpaca creds; §3 runbook) |

**Operational validation still owed from Phase 0:** one unattended week of paper trading (buffers hold one bar per period, equity tracks broker, forced outage degrades and recovers, daily-loss trips on a simulated loss). Nothing blocks starting it now.

---

## 2. The Accuracy Program

Everything remaining is ordered by one question: *does it make the bot's decisions more accurate, or make accuracy measurable?* The chain:

```
labels (what the model learns)  →  training (how it learns)  →  calibration (what its confidence means)
      →  measurement (whether improvement is real)  →  the loop (whether improvement compounds)
```

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
| ⟲ 1 | §4.1 + §4.2 | **open — next milestone** |
| ⟲ 2 | model-head disentanglement (§6, Phase 2) | blocked on ⟲1 scorecard |
| ⟲ 3 | FeatureContract bump for market-context features (§7, Phase 3) | blocked on ⟲2 |

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

1. **Model-head disentanglement** (per-horizon return heads; confidence = probability of the chosen class; sizing separated) → ⟲2. *Directly raises accuracy: three conflicting targets currently share one scalar.*
2. **Memory correctness then bounded influence:** temporal cutoff made real, per-fold rebuild fixed, provenance stamped, O(N²) extend fixed, hot-path saves debounced; then kNN outcome stats as a **damping-only** prior (size multiplier ∈ [0.5, 1.0]) → then counterfactual utility credit (paired brain-only vs memory-fused runs as the memory promotion gate). *Also prune the ~100 dead MemoryConfig fields + inert GUI toggles here (moved from 1F).* 
3. **Strategy arbitration layer** (pillar 2): registry of brain + bounded primitives, `strategy_id` through orders/fills, per-strategy budgets, outcome-weighted allocation — paper-only until it beats the brain-only scorecard row.
4. **Measured risk:** perturbed-R² leakage diagnostic; cap sweeps → knees replace hand-picked caps; outcome-adaptive risk budgets (bounded, logged).
5. **Anchored negatives + curriculum** for retrain dataset construction.

Each item: **→ verify** = a scorecard row (or paired-run delta) demonstrating non-degradation; memory influence additionally requires counterfactual-credit > 0 on OOS folds.

---

## 7. Phases 3–4 — research adoption (see RESEARCH_ROADMAP.md §4, phases 3–4)

Kept as a gated menu; no detail duplicated here. Entry criteria:

- **Phase 3** (OOD/familiarity gate, regime snapshots, market-context features → ⟲3, Optuna/Hyperband with anti-snooping audits, bootstrap-only synthetic stress, calibration refit from realized outcomes): enter when Phase 2 items 1–2 have scorecard-verified rows.
- **Phase 4** (PBT-lite, small SOM regime map, differentiable Hopfield retrieval, meta-gradients, capacity growth, Caruana ensemble): each item enters only behind experiment isolation after beating (a) plain fine-tune and (b) equal-budget random search. The 20-item **do-NOT-do list** in RESEARCH_ROADMAP.md §5 is binding.

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

---

## 9. Immediate next actions (strict order)

1. §4.1 labels & training accuracy (1–4) — *the label fix is the accuracy foundation everything retrains on.*
2. §4.2 calibration (1–2).
3. §4.4.1/2/4 + §2.1 scorecard — *enough evaluation honesty to measure the retrain.*
4. **⟲ Retrain gate #1** (§3 runbook) → incumbent scorecard row.
5. Start the unattended paper week in parallel with:
6. §4.3 parity insurance, §4.4 remainder (walk-forward, metric math), §4.6 follow-ups.
7. §4.5 close the loop.
8. §4.7 hygiene/docs, then Phase 2 entry.

*Traceability: v1 of this plan (git `00523e7`) holds the original full enumeration; CODEBASE_REVIEW.md §5/§7 items remain covered by the verify-as-you-touch rule; nothing was dropped silently — every removal is in §8.*
