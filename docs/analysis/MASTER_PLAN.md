# SETAI Master Implementation Plan

**Date:** 2026-07-07
**Sources:** `CODEBASE_REVIEW.md` (confirmed defects C1–C10, per-subsystem improvements §5, unverified findings §7, test/config gaps §8, top-20 list §9) and `RESEARCH_ROADMAP.md` (paper digests, gap analyses, 4-phase adoption roadmap, do-not-do list).
**Goal:** Take SETAI from "sound foundation with broken rails and no learning loop" to a genuinely self-evolving trading system serving all three vision pillars: (1) self-developed strategies, (2) known strategies under soft caps, (3) outcome-driven weight adaptation.

Every improvement item from the codebase review and every research-roadmap step is mapped into a phase below. Task IDs are stable — reference them in commits (`[0.3]`, `[1B.4]`).

---

## How to read this plan

- **Phases are ordered by dependency, not preference.** Phase N's "definition of done" gates Phase N+1. Within a phase, workstreams (letters) are largely parallelizable; numbered tasks within a workstream are ordered.
- **Effort:** S = hours, M = 1–3 days, L = 1–2 weeks (solo developer).
- **⟲ RETRAIN gates:** some tasks invalidate all existing checkpoints. They are grouped so you retrain a minimum number of times (three planned retrains total: end of 1B, end of 2F, Phase 3 contract bump).
- **Verification discipline:** every fix ships with its regression test in the same commit. The §7 unverified-medium findings are folded into the workstream that touches their file — *verify-as-you-touch*: when you open a file for a planned task, resolve that file's §7 items in the same PR (confirm + fix, or refute + note).

## Standing rules (apply to every phase)

1. **Paper-first, always.** The live-mode double interlock is never weakened by any task in this plan.
2. **Learned proposes, deterministic risk disposes** — this invariant gets a test in Phase 1 and no task may violate it.
3. **Every self-evolution mechanism must beat two dumb baselines** before adoption: a plain pretrain+fine-tune, and an equal-budget random-search control.
4. **Learn from executed actions, not intent** — every outcome-driven update keys off the post-risk-manager action.
5. **Counterfactual, not correlational credit** — memory/strategies earn utility only for decisions they actually changed.
6. **Anchor every self-update** — KD/proximal anchors to the promoted checkpoint whenever weights move.
7. **Protect the rare** — crisis-regime episodes are explicitly exempt from every eviction/rebalancing rule.
8. **Promotion is a statistical decision** — paired folds, significance tests, backward-transfer floors, benchmark-relative requirements.
9. **The do-NOT-do list (bottom of this file) is binding** until a phase explicitly revisits an item behind an experiment gate.

---

# Phase 0 — Stop the bleeding (critical hotfixes)

*Target: ~1–2 weeks. Until Phase 0 lands, no paper session and no backtest number means anything. No model changes; no retrain needed.*

### 0.0 Commit the in-flight refactor — S
The "Great Simplification" (~94 files, −7,928/+5,299 lines) is **uncommitted**. Commit it as-is first (tag `pre-master-plan`), so every subsequent fix is a reviewable diff and nothing can be lost. Then branch per workstream.

### 0.1 Fix live bar ingestion (C1) — M
`src/data/alpaca_client.py:289,327`, `src/data/stream_manager.py`, `src/execution/tick_processor.py:221`, `src/data/buffers.py`.
- Dedup by timestamp before append: skip (or overwrite-in-place for the forming bar) when `bar.timestamp <= buffer.last_bar().timestamp`.
- Honor the configured timeframe: aggregate polled 1-min bars into the runtime timeframe and commit only on bucket close, **or** fetch the latest *completed* bar via `get_historical_bars_range`. Remove the `del timeframe` lines.
- Move the append after (or gate it on) the session-guard/staleness check so stale bars can't poison features while trading is blocked.
- Backfill missed bars via `get_historical_bars_range` on reconnect/gap.
- **Tests:** polling the same mock bar twice grows the buffer by 1; a gap triggers backfill; a 1-min bar never lands in a 5-min buffer.

### 0.2 Survive transient broker errors (C4) — S/M
`src/execution/tick_processor.py:207`, `src/execution/trade_engine.py:63-75`, `src/data/alpaca_client.py:141-171`.
- Wrap `_sync_live_state` in try/except: log, skip reconciliation this tick, count consecutive failures, block trading (not crash) after N failures — mirroring the stream layer's circuit-breaker pattern.
- Bounded retry/backoff + explicit request timeouts on Alpaca read calls.
- Transient-vs-fatal split in `trade_engine` with an error budget.
- Also (§7.7): make stream circuit-breaker exhaustion exit nonzero so the GUI shows ERROR, not STOPPED; wrap the Phase-A `save_memory_bank`/`save_pending` calls so a disk error can't kill the session.

### 0.3 Broker-truth accounting + outcome capture (C3) — M/L
`src/execution/tick_processor.py:57-66`, `reconciliation.py:158-181`, `order_router.py:84-100,185`, `paper_executor.py:49-53`, `src/risk/risk_manager.py:56-61`.
- Fetch `client.get_account()` each tick (or short cadence); overwrite `portfolio.cash`/equity from broker truth.
- Derive `daily_pnl` from broker equity vs a **trading-day-anchored** snapshot (resurrects the daily-loss circuit breaker).
- Detect fill transitions: when an order leaves open orders, fetch it by id (Alpaca's dedicated endpoint — also fixes `get_order_by_client_order_id` missing closed orders) → apply PaperExecutor's close-out math to PortfolioState, record a `FillRecord` in the Ledger, call `risk_manager.register_fill`, update the order journal with real fill prices.
- Preserve position `entry_time` across per-tick hydration (fixes always-zero peak/drawdown tracking).
- Make `MockBrokerClient` stateful (update positions on submit) so mock rehearsals stop hiding this class of bug.
- **Tests:** simulated fill transition updates cash/realized/daily PnL, ledger, and cooldowns; daily loss limit fires in broker mode.

### 0.4 Fix the broken entry point (C5) + import-smoke test — S
`scripts/build_memory.py:21`, `src/gui_services/memory_service.py:196`.
- Strip the deleted `bank_cache` import and `--no-bank-cache` branches (build via `build_memory_bank_from_bars` + `save_memory_bank`), fix the summary print (deleted attributes) and the `config.logging.json` → `json_logs` bug.
- Add the **scripts import-smoke test** (importlib over `scripts/*.py`) so this class of breakage can never be silent again. Include the bench scripts' missing sys.path bootstrap (§7.7).

### 0.5 Timezone-correct, fail-closed sessions (C9/C10) — S
`src/core/time_utils.py:18-30`, `src/execution/session_guards.py`, `src/core/config/runtime.py`, `risk.py`.
- Define sessions in `America/New_York` with `zoneinfo`; convert timestamps before comparison. Broker clock stays authoritative where available; static windows are the fallback.
- `within_session` fails **closed** (raise/False) on unknown session names; `field_validator`s on `allowed_sessions` in both configs; fix the session boundary overlap (13:30/20:00 double-claimed).
- **Tests:** winter + summer boundary tests (15:30 ET in January must be tradeable; 9:00 ET must not).

### 0.6 Operational hazards that can silently kill or deceive a session — S
- `src/gui_services/runtime_service.py:298`: add a stderr drain thread (TrainingService already has one) — the trading subprocess currently deadlocks once the OS pipe buffer fills.
- `src/execution/tick_processor.py:239`: fix the equity-log `unrealized_pnl` formula (currently reports gross notional).
- Remove the `fusion_enabled` runtime-status field (false assurance about a deleted safety layer).

**Phase 0 definition of done:** a paper session runs for a week unattended: buffers contain exactly one bar per period; equity/daily-PnL track the broker; a forced Alpaca outage degrades and recovers instead of crashing; the daily-loss limit demonstrably trips in a simulated loss; `build_memory.py` runs; January and July session boundaries are correct.

---

# Phase 1 — Trustworthy measurement + close the loop

*Target: ~4–8 weeks. This is the biggest phase because the analysis was unambiguous: every number the system shows you is currently untrustworthy, and self-evolution built on dishonest numbers evolves toward artifacts. Ends with the first honest retrain.*

## Workstream 1A — Data integrity

1. **[M] Split/dividend-adjusted bars.** `adjustment=Adjustment.ALL` in historical fetches (config field), bump the cache version to invalidate all raw caches, refetch. The current cache contains NVDA's 10:1 split as a fake −90% crash inside training data.
2. **[M] Stylized-facts data-quality gate.** New `src/evaluation/stylized_facts.py` (~200 lines): return ACF, |return| ACF, kurtosis, leverage correlation, cross-symbol correlation distance. Run over the refreshed cache in `data_loading` as a gate and as a promotion sanity check. (Its first payoff is proving 1A.1 worked.)
3. **[S each] Data-layer defect burn-down** (§7.1, verify-as-you-touch): atomic cache writes (tmp+`os.replace`); fix the stdlib-logger kwarg crash in the cache legacy fallback; serialize `CacheManifest` record+save globally (single lock; unique tmp names on Windows); zero-bar fetches return empty + gap report instead of crashing; make `--force-refresh` actually refetch; fix `_historical_lookback_delta` for multi-minute/hourly timeframes; timeframe-aware cached-range tolerance for offline loads.
4. **[M] Resolve the spread train/live skew.** Historical bars hardcode `spread=0.0`, live carries real spread — on a contract column. Either estimate historical spread, zero it live too, or drop the column from the contract until historical quotes exist. Decide once, document at the decision site.
5. **[M] Delete or re-wire dead prep code** (`parallel_prep.py`, `BarWindowDataset`, unused `PipelineConfig` fields) so the training path reads as it runs.

## Workstream 1B — Train/serve integrity ⟲ RETRAIN at the end

1. **[M] Fix normalization skew (C2) — the big one.** `normalize_columns` returns per-tensor mean/std; persist in checkpoint meta (`build_checkpoint_meta`) beside the FeatureContract; apply via **one shared helper** in `BrainPolicy.propose/propose_batch`, `ModelStateEncoder.encode_batch`, backtest snapshot building, and `memory_builder`. (Alternative: move normalization into the pipeline itself — decide before implementing.) Also fix `evaluate_checkpoint` re-deriving stats from current data (§7.7).
2. **[S] Train-vs-inference input parity test** — same bar through the training path and the inference path must produce identical model inputs. This test is the permanent guard for the whole C2 class.
3. **[S] Scalar-vs-batch feature equivalence test** — the load-bearing parity invariant has no test today. Default config AND a small-window config (exposes the RSI `[:_RSI_PERIOD]` and momentum boundary off-by-ones — fix both while here); derive `first_valid_index` from **all** feature-family warmups, not just `windows['long']`.
4. **[S] Feature-cache correctness:** salt the fingerprint with a schema/code version constant (cheapest correct fix in the repo relative to risk); strengthen the bars signature (content hash or interior samples); absolute-pathed, size-capped, disabled in test conftest.
5. **[S] Live-path feature sanitation:** `nan_to_num`/isfinite-reject in `build_from_tensor`; epsilon on `return_1` denominators; unify epsilon between scalar and rolling helpers (code default 1e-6 vs YAML 5e-4 — pick one).
6. **[S] Checkpoint integrity:** save the **effective** (fast-mode-override-applied) model config in meta + load round-trip test; atomic checkpoint saves (tmp+replace) with a previous-version backup of `brain.pt`.
7. **[S] Label/cache integrity:** add label-affecting fields (adaptive-epsilon knobs, `transaction_cost_bps`) to sample-cache fingerprints + version bump; purge `max_horizon` samples at split boundaries (real embargo) and strengthen the split test to assert the gap; use `effective_primary_horizon` everywhere (targets, diagnostics, Platt outcomes).
8. **[M] Degeneracy fixed at the cause** (replaces the `force_directional_flip` patch): cost-widened flat band using a realistic `transaction_cost_bps` **shared between labels and evaluation**; epsilon scaled by √(primary_horizon); per-trade-normalized loss weighting. Unify `force_directional_flip` into a single config consumed identically by `policy_factory.py` and `runtime_bootstrap.py` — or delete it once the cause-fix works. Add the backtest-vs-runtime proposal parity test.
9. **[S] Review the triple class-imbalance stack** (balanced weights + inverse-frequency sampler + adaptive focal, all on by default, and val loss shares the weights): pick at most one primary mechanism; make model selection configurable (balanced accuracy / macro-F1 / cost-adjusted expectancy proxy instead of raw multi-task loss).
10. **[S] Platt correctness:** direction-aware labels (predicted-sign × primary_return, flat excluded, minus costs); isolate the artifact per run (experiment runs must not clobber the canonical calibrator); config-driven path (kill both hardcoded `runs/` literals); bind the calibrator to its checkpoint fingerprint and refuse mismatches.
11. **[S] Model input hardening:** remove/fix the `full_features=None` branch-order fallback (silently permuted inputs); make batched inference invariant to batch composition (masked pooling or length-bucketing; `propose_batch([a,b]) == [propose(a), propose(b)]` test); fix `_extract_ev_signals_batch` zeroing both EV legs when one is missing.
12. **⟲ RETRAIN + refetch + rebuild.** With 1A.1 (clean data), 1B.1 (normalization), 1B.7 (labels), 1B.8 (cost-aware targets) landed: full retrain, new Platt fit, new memory bank. Everything downstream evaluates *this* checkpoint.

## Workstream 1C — Evaluation honesty

1. **[S] Persist per-split train/val date ranges in checkpoint meta** at save time (prerequisite for 2 and 3).
2. **[M] Out-of-sample by default (C7):** `run_backtest.py` restricts to bars after the checkpoint's train/val end, or hard-warns and tags every metric **IN-SAMPLE** when overlapping.
3. **[L] Real walk-forward (C6):** per-fold retraining (fast_mode) honoring `reset_model_per_fold` — or, minimum viable: hard-refuse folds whose test window overlaps the checkpoint's training range. Paired candidate-vs-incumbent evaluation on identical folds with a paired t-test/sign test; drop degenerate folds from aggregation (don't average hard zeros); fix the module docstring.
4. **[M] Honest fills and costs:** `backtest.fill_timing: same_close | next_open` with **next_open default for promotion decisions**; slippage = base_bps + k·spread_bps where spread exists; `cvar_95` added to `compute_backtest_metrics`.
5. **[S] Backtest validity floor:** start the bar loop at `max(warmup_bars, first_valid_index)`; check `ExecutionResult.accepted` before `register_fill` (phantom-cooldown bug); validate `warmup_bars >= max(windows)` at config load; cross-validate `sequence_length` vs `warmup_bars`.
6. **[S] Tradeable benchmarks:** baselines get honest heuristic EV estimates (or an explicit `ev_gate_requires_signals` bypass for policies without risk heads) + a regression test that BuyAndHold produces >0 fills. Fail fast when the EV gate has no inputs (no risk head → ConfigurationError) — silent zero-trade sessions become impossible.
7. **[S] Repair or delete `--compare`:** post-simplification scenario labels, remove hardwired zero veto/fusion rows, stop `_make_memory_aware_policy` silently flipping the flip flag.
8. **[M] Metric math correctness + tests:** golden hand-computed fixtures for `performance.py` (Sharpe, Sortino — fix the negative-count divisor; Calmar; round-trip detection — sign flips must close trips; profit_factor=0-for-all-winners bug; fill_count vs trade_count); engine tests (fees/slippage arithmetic, flip on/off, warmup floor, timestamp alignment).
9. **[M] Structural cleanups:** timestamp-aligned multi-symbol iteration (aligned grid or per-bar assertion); de-duplicate `_CachingBacktestEngine` into `BacktestEngine` (precomputed-features hook; the GUI copy already diverged — unsorted symbols, skipped cutoff); timestamped equity curve + per-symbol attribution in `BacktestReport`; guard empty-fold aggregation; benchmark-relative and deflated-Sharpe statistics in reporting.
10. **[S] Annualization coupling:** tie `bars_per_year`/`seconds_per_bar` to the configured timeframe (validator or warning) — a 1Min backtest currently mis-annualizes Sharpe by ~√5.

## Workstream 1D — Close the self-evolution loop (C8, pillar 3 skeleton)

*Depends on 0.3 (outcomes exist) and 1C (honest promotion evidence).*

1. **[M] Durable outcome dataset:** persist the Ledger (JSONL); emit an execution-outcome event stream (decision id, features fingerprint, proposal, risk decision, executed action, entry/exit, realized PnL net of fees, slippage vs reference, reason codes). Charge fees into `realized_pnl` (currently gross). This is the raw material for everything below.
2. **[S] Credit assignment (shield principle):** record the **executed (risk-clamped) action**, clamp/veto flag, and rejection reason on every decision — extend `MemoryItem` (`src/memory/schemas.py`), fix `record_snapshot(action_annotation=None)`, thread `RiskDecision` through `tick_processor` Phase C and `BacktestEngine.run`; expose `intervention_rate` in runtime status.
3. **[M] Replay buffer wired:** per-decision observations (state, executed action, realized outcome) flow into `ReplayBuffer`; un-reserve it; age/size caps; populate `paper_observation_count`.
4. **[M] Retrain trigger wired:** `check_retrain_trigger` called from a scheduler/CLI/GUI; fix the shared 0.15 threshold (per-metric thresholds — 0.15 on Sharpe is noise, 0.15 on win-rate is catastrophe); `RetainMetadata` lineage recorded on each retrain.
5. **[M] Promotion becomes authoritative:** gate wired into `run_train.py` (brain.pt can never be overwritten ungated; `--force-promote` escape hatch); runtime resolves via `get_active_checkpoint` from `checkpoints/registry.json` (config-path fallback); transactional promote (atomic registry write, corrupt-JSON tolerance, lock file); shared metric-flattening adapter between the flat and dotted key schemas + key-alignment test (the current mismatch would silently zero every gate).
6. **[L] Promotion gates on trading metrics,** not just classification: walk-forward Sharpe / max drawdown / CVaR95 / benchmark-relative floors via the existing engine; candidate must beat buy-and-hold **and** the incumbent on identical bars.
7. **[M] Forgetting-safe retraining:** λ·KL self-distillation against the frozen previous promoted checkpoint + L2 proximal anchor, both in `configs/training.yaml` (λ loosened on detected regime breaks); regime-stratified rehearsal quotas from the historical cache in every retrain batch mix.
8. **[M] Measure forgetting:** 3–6 frozen held-out slices from distinct historical regime windows; AP/AF (backward-transfer) matrix in `walk_forward.py`; backward-transfer floor in the promotion gate.
9. **[S] Weight-space telemetry:** per-module cosine(θ_prev, θ_new) + norm trajectories logged on each retrain into runtime_status/GUI; alert on monotonic shrink or frozen modules; contrast-gap monitor on training pairs.

## Workstream 1E — Risk-layer correctness (pillar 2 groundwork)

1. **[S] `daily_pnl` resets at trading-day boundaries** (US/Eastern from the broker clock; in backtests too). Today it's dead in broker mode and a lifetime cumulative cap in sim.
2. **[M] Risk-reducing orders are exempt from halt reasons** (kill switch / daily loss / drawdown / caps pass for pure exposure reductions; optional flatten-only mode). A tripped limit must de-risk the book, not freeze it.
3. **[S] Post-trade cap evaluation:** aggregate-notional/leverage checks include the proposed order (currently overshootable by up to `max_position_notional`); boundary tests.
4. **[M] Resize path (soft caps, as the README already claims):** shrink `target_qty` to the remaining budget instead of rejecting; emit a `resized` reason code. This is the first concrete piece of "soft caps."
5. **[S] EV-gate cost model includes spread** (half-spread per leg; resolve the per-leg vs round-trip ambiguity against the simulator — they currently disagree by ~2x).
6. **[S] Sizer/portfolio hardening:** clamp equity ≤ 0 (negative equity currently flips order direction); fix drawdown-window sampling (one pnl_history entry per bar; currently 2×/bar in backtests + per fill); enforce or delete `max_position_per_symbol`; move `_recent_orders` duplicate-window marking to post-route confirmation; stop `RiskManager` mutating the shared config (instance copies; kill_switch as manager state); harden same-day-restart `client_order_id` collisions (process nonce or seed from closed orders).
7. **[M] Risk test expansion:** ~3 of ~14 gates are tested. Add cooldown, duplicate windows + seeding, daily-loss/drawdown/leverage/concurrency boundaries, sizer rounding, flatten-bypass, resize semantics, risk-reducing exemption.
8. **[S] Safety-invariant test:** no learned module can mutate `RiskConfig`/`PortfolioState` — the learned-proposes/risk-disposes contract becomes enforced, not conventional.

## Workstream 1F — Config hygiene (makes drift impossible, not just fixed)

1. **[M] `extra='forbid'` across the config package** (keep `ignore` only on the two documented inert stubs); fix all fallout keys in `configs/*.yaml`.
2. **[S] `RiskConfig` validators** — the most safety-critical model currently has zero: positivity/sign/range on every limit, `max_aggregate_notional >= max_position_notional`, pct bounds.
3. **[M] Config-files hygiene test:** load every `configs/**/*.yaml` through `load_config`, recursively diff keys against model fields (allowlist for intentional legacy). This single test would have caught every dead key in the review.
4. **[M] Purge/rewrite no-op presets:** both confidence sweeps, both fusion-veto sweeps, `backtest_strict_fusion.yaml`, memory-updates/topology sweeps, ~14 of 25 experiment ablations; rewrite `_experiment_base.yaml` for the EV-gate era so ablations generate fills; sync `run_experiments.ALL_EXPERIMENTS` (13 YAMLs missing).
5. **[M] Prune the dead-knob graveyard:** ~105 MemoryConfig fields + default.yaml blocks + inert GUI toggles (training_tab/memory_tab); dead ModelConfig ablation flags and branch/regime dims; `runtime.decision_interval_sec`; `data.bars_per_trading_day`+lookbacks; `pipeline.validation_strictness`/`max_prep_workers`; `broker.base_url` (decorative); `training.pretrain_*`/`rl_*`/`temporal_jitter_prob`/`contrastive_*`; dead `evaluation.walk_forward.reset_*` once 1C.3 wires them. Wire `memory.top_k` and `min_similarity` (the two knobs users would actually tune — `MemoryAwarePolicy` hardcodes k=5).
6. **[M] Align programmatic defaults with default.yaml** (risk limits, epochs, sequence_length, AMP/Platt/early-stopping, epsilon, memory sizes) — or declare YAML canonical with a test that all entry points load it first.
7. **[S] Horizon-sync integrity:** remove the explicit `memory.forward_horizons` override so the training→memory sync actually runs (shipped stacks currently train [1,5,15] and label memory [1,3,6,12]); assert equality in a test.
8. **[S] Script/CLI footguns:** `--config` append-over-default bug in all six scripts (default=None + post-parse fallback); `--seed 0` truthiness; scope `sweep_train.find_best_run` to the sweep's own runs; relative-date parsing for `backtest_quick`.
9. **[S] pyproject hygiene:** declare numpy; PySide6 as a `gui` extra; fix the `authors = "OpenAI Codex"` attribution; verify the `seif-brain` entry point; dev extras.
10. **[S] Config provenance:** log which file supplied each overridden key at startup; persist resolved-config sha256 per run.

## Workstream 1G — Core/runtime hygiene

1. **[S] `resolve_device('cuda')` without CUDA:** clean ConfigurationError (or warn+CPU) instead of a later opaque torch crash; validate `DeviceConfig` fields.
2. **[S] Seeding:** config-driven seed (the hardcoded `seed_everything(7)` silently zeroes sweep variance); seed numpy; drop the ineffective runtime `PYTHONHASHSEED` write.
3. **[S] Logging hardening:** swap handlers before closing old handles; guard None streams (`pythonw`); rotate the runtime log.
4. **[S] Trainer hygiene (§7.7):** `torch.no_grad()` for val/test epochs; `model.eval()` in `encode_batch`; `drop_last` or size-1-batch guard with BatchNorm; restore-best also restores meta/optimizer-state consistency; register safe globals so `weights_only=True` loads succeed (kill the permanent unsafe-fallback path + warning noise); surface collate `nan_to_num` counts (silent corruption becomes visible); fix `_compute_class_weights` absent-class blowup when the cap is nulled.
5. **[M] Graceful shutdown:** cancel tracked open orders, persist memory/pending, final status snapshot on KeyboardInterrupt/fatal error.
6. **[S] Unify the config error contract** (ConfigurationError subclassing ValueError; wrap YAML errors with file context).
7. **[M] Documentation truth pass:** rewrite README (delete dendritic/soma/Hopfield/ESOM/veto/GRU claims, fix "approves, resizes, or rejects" once 1E.4 makes it true, fix the paper-fills framing); delete or rewrite `LIVE_LOG_CHEATSHEET.md`; fix factory/engine/script docstrings claiming fusion/cutoff behavior; GUI: remove dead toggles, fix `MemoryStatus` reading nonexistent bank summary keys, fix the Promote-button tooltip; derive `metrics.CLASS_NAMES` from `constants.Action`.

**Phase 1 definition of done:** an honest number exists. A retrained checkpoint (1B.12) is evaluated strictly out-of-sample with next-bar fills and real costs against tradeable benchmarks; promotion is gated on trading metrics + backward transfer and physically protects `brain.pt`; realized paper outcomes accumulate in a durable dataset and can trigger a gated, anchored retrain; every YAML key is either read or rejected; the money-path test suite (performance math, engine, risk gates, parity tests, script imports, config hygiene) is green.

---

# Phase 2 — Causal memory, measured risk, strategy arbitration

*Target: ~4–6 weeks. Makes memory earn its keep, turns risk caps from guesses into measured operating points, and delivers pillar 2 (known strategies under soft caps).*

## Workstream 2A — Memory correctness (before memory influence)

1. **[S] Make the backtest temporal cutoff real** (filter `query_top_k_simple` by `_temporal_cutoff`; exclude the query's own timestamp/symbol) — or delete `set_temporal_cutoff` and the engine's false comment. The half-implemented state is the most dangerous option.
2. **[S] Repair the walk-forward per-fold memory rebuild** (guards reference deleted attributes; every fold currently reuses the full-history bank whose outcomes postdate the fold) + regression test: fold-bank max timestamp ≤ train_end − embargo.
3. **[S] Provenance at build time:** stamp `builder_embargo_end_ts`, warmup/embargo bars, and **always** the encoder fingerprint (including `memory_builder` — currently None, silently bypassing stale-bank detection); make `policy_factory`'s embargo check read what builders actually write; validate `feature_contract_id` per item; enforce or delete `require_encoder_match`.
4. **[M] Performance fixes:** vectorize `MemoryBank.extend` (O(N²) today — ~200 GB cumulative memcpy at 20k items); move `save_memory_bank`/`save_pending` off the per-tick hot path (debounced/background); age-purge for pending items that can never finalize; isoformat timestamps so `weights_only=True` works.
5. **[S] Reconcile `similarity_metric`** (euclidean/attention silently run as cosine); payload version check on load; item-id dedup on append.

## Workstream 2B — Memory hygiene (what gets stored and retrieved)

1. **[M] Novelty-gated writes:** merge within ε of an existing key (update its outcome stats/count) instead of inserting near-duplicates.
2. **[S] Retrieval hygiene:** temporal-adjacency exclusion window (top-k is currently dominated by neighboring bars including the query itself), same/cross-symbol filters, enforce the now-wired `min_similarity` floor.
3. **[M] Salience write-gate + uncertainty read-gate:** store only when feature surprise / regime flip / outcome magnitude clears a bar; retrieve+blend only when the policy is uncertain (entropy / low calibrated confidence).
4. **[M] Protected set + priority eviction:** reservoir sample + tagged tail episodes (drawdowns, vol spikes, rare path labels) exempt from eviction; replace FIFO with utility/priority eviction; quantile-threshold retrieval.

## Workstream 2C — Memory → decisions, with counterfactual credit (pillar 3 at the memory layer)

1. **[M] Stage 1 — bounded prior:** blend top-k neighbor `OutcomeSummary` stats (direction agreement, MAE-p10) into the proposal's `expected_gain/loss_bps` and/or a size multiplier in [0.5, 1.0] **before** the EV gate. Damping-only at first (never creates trades); hard-capped band (e.g. ±30%); every adjustment logged.
2. **[S] Reuse the brain's forward for the memory latent** (expose `fused_representation`; halves brain+memory inference cost — memory currently doubles inference for diagnostics).
3. **[L] Stage 2 — counterfactual utility:** paired brain-only vs memory-fused walk-forward runs as the **memory-bank promotion gate**; per-item utility EMA credited only when fusion changed the action; retrieval ranked by α·sim + (1−α)·(utility + UCB bonus); utility-based pruning with a protection window; retire internalized memories whose utility decays post-retrain.
4. **[M] Encoder/memory compatibility guard:** re-encode the bank on checkpoint promotion (or an encoder-stability penalty during retraining); encoder fingerprint as a hard staleness alarm triggering rebuild.

## Workstream 2D — Strategy arbitration layer (pillar 2 delivered)

1. **[L] `StrategiesConfig` + registry:** per-strategy `enabled`, capital-fraction soft cap, drawdown attenuation; `AdaptationConfig` (retrain triggers, promotion criteria, rollback policy). The vision finally gets a declarative home in config.
2. **[L] The arbiter:** new `src/strategies/` component wrapping brain + bounded strategy primitives (momentum entry, mean-revert, passive/aggressive exit, flat — each with hard parameter bounds); starts equal-weight or PnL-EMA-weighted allocation. Paper-only behind a flag until it clears promotion.
3. **[M] Strategy identity end-to-end:** `strategy_id` on TradeProposal → OrderIntent → client-order-id → journal → FillRecord; per-strategy notional budgets enforced in `RiskManager`; per-strategy PnL attribution persisted.
4. **[M] Honest strategy primitives:** true EMA/Wilder MACD/RSI alongside (or honestly renamed) SMA variants; VWAP/Bollinger %B under existing branch prefixes — the "known real-world strategies" should compute what practitioners mean by them.

## Workstream 2E — Measured risk + degeneracy diagnostics

1. **[M] Perturbed-R² harness** (`run_backtest.py --perturb-signal`): mix policy logits with calibrated noise (R² = 1.0/0.5/0.1/0.01), plot Sharpe vs R². Detects leakage (profit only at implausible R²) and structure-not-signal returns (flat curve). ~1 day, high information.
2. **[M] Cap-sweep harness** (`run_backtest_parallel.py`): sweep `max_position_notional`, aggregate notional, EV padding across ~5 values on identical bars; report mean return/vol/CVaR95; pick the knee. Soft caps become measured operating points instead of guesses.
3. **[M] Guardrail-toggle diagnostic:** backtest mode with the risk gate in log-only mode recording would-be-violation frequency/magnitude per retrain generation — a degeneracy early-warning curve.
4. **[L] Outcome-adaptive risk budget:** rolling-expectancy multiplier on target_fraction / max notional per symbol/strategy, fed by the persisted ledger; hard bounds; changes logged and reversible. (Pillar 3 at the risk layer — deterministic, inspectable adaptation.)
5. **[M] Volatility/EV-aware sizing:** capped fractional-Kelly from the risk head's own MFE/MAE predictions (currently used only as a binary gate).

## Workstream 2F — Model-head disentanglement ⟲ RETRAIN at the end

1. **[M] Split position sizing from return prediction:** per-horizon return heads (or drop the multi-horizon loss); dedicated sizing head or deterministic sizing from expected return + risk head. `target_fraction` is currently regressed toward primary return AND every horizon, then consumed as position size.
2. **[M] Give confidence one meaning:** proper scoring rule on direction correctness (or the softmax probability of the chosen class) — today it's trained toward "stability"/MFE and consumed as P(win) by the EV gate and Platt.
3. **[S] Prune dead `ModelOutput` fields** (nine legacy placeholders), dead config flags, dead GRU bookkeeping in `brain_policy`; make no-op ablations fail loudly; align FLAT-flip confidence semantics if the flip still exists.
4. **[M] Anchored negatives + pair-validity filter** for retrain dataset construction: frozen worst decisions retained permanently as regime-conditioned negatives; contrasts only when outcome gap > round-trip cost + noise buffer.
5. **[M] Adaptive curriculum sampler:** replay-buffer wrapper scheduling regime buckets easy→hard by measured per-bucket accuracy.
6. **⟲ RETRAIN** with the new heads, via the (now honest) promotion pipeline. This is also the first live test of the forgetting-safe machinery from 1D.7–1D.8.

**Phase 2 definition of done:** memory demonstrably changes decisions and is promoted/pruned on counterfactual PnL credit; a strategy ensemble runs in paper with per-strategy soft caps and attribution; risk caps sit at measured knees; the perturbed-R² curve says the edge is signal, not structure or leakage; model heads each mean one thing; the second retrain passed the anchored, backward-transfer-gated promotion.

---

# Phase 3 — Regime awareness, richer features, smarter search

*Target: ~2–3 months, item-by-item as capacity allows. Everything here assumes Phases 0–2 are the operating baseline.*

1. **[M] Familiarity/OOD gate:** small masked-reconstruction readout on the feature window; normalized residual feeds position sizing and the EV bar — the brain currently trades at full confidence in never-seen regimes. Wire `self_supervised.py` (MAE-style pretraining) as its training path.
2. **[L] Regime-snapshot recurrence layer:** per-session K-means centroids over regime features; two-stage MMD/Mahalanobis matching (NOT alpha-shapes); on match, activate per-regime Platt params / retrieval weights / sizing caps; on no-match, emit a novelty event that tightens risk caps; prequential error over resolved trades as the live rollback/retrain trigger.
3. **[M] Cross-sectional market context:** new `src/features/market_context.py` — SPY/QQQ return + realized vol, symbol-vs-SPY relative strength, universe breadth; computed once per tick, broadcast to all symbols; FeatureContract version bump (fingerprints invalidate old checkpoints cleanly). ⟲ implies the Phase-3 retrain.
4. **[M] Episodic embedding training + the killer baseline:** prototypical auxiliary loss so retrieval similarity predicts outcomes; standing walk-forward comparison vs plain pretrain+fine-tune with the identical encoder — if memory can't beat it, that decides the architecture question.
5. **[L] Sweep upgrade:** Optuna TPE + Hyperband/ASHA in `sweep_train.py` with walk-forward Sharpe objective (not macro-F1); **fidelity-correlation (Spearman) audit before enabling aggressive pruning** (a short window is a different regime, not a cheap proxy); Kendall-Tau audit of promotion proxies vs realized Sharpe; fANOVA importance to shrink the space; config-portfolio memory warm-starting future sweeps; multi-seed promotion t-tests.
6. **[L] Synthetic scenarios — bootstrap only, no GAN:** regime-conditioned block-bootstrap + rolling-factor-beta generator in `src/training/augmentation.py` (replacing the token Gaussian noise; wire or delete `temporal_jitter`) for rare-regime oversampling; correlated multi-symbol tail stress for the risk manager; parametric shock overlay (exp-OU vol + tanh shock) strictly for stress-testing session guards/limits/flattening; adverse-fill mode in the simulator (probabilistic worse-price fills, partial fills above a volume-participation fraction) with baseline-vs-adverse reporting in promotion. All synthetic data gated by stylized-facts; never used for final promotion decisions.
7. **[M] Memory plasticity schedules:** per-slot maturity/saturation (decay-unless-reinforced, leaky-Hebbian `w ← (1−λ)w + c·retrieval·outcome`); LRUA-style write/eviction; prototype splitting on regime-bimodal outcome stats; BAM-style reverse "danger set" retrieval (state signatures preceding large adverse excursions) feeding the risk gate.
8. **[M] Regime-tagged specialist archive:** regime signatures in checkpoint meta; archived specialists as **distillation teachers** during rehearsal (not live swaps); noise-robustness promotion gate (perturb weights, require stable behavior); recency-bias affine logit correction before Platt.
9. **[M] Calibration refit from realized outcomes:** rolling refit of Platt (per-regime once 3.2 exists) from the persisted outcome stream; calibrator versioned with the checkpoint it calibrates.
10. **[M] Feature/scale hygiene:** scale-invariant `regime_trend_strength` (returns-based); richer calendar features (minutes-to-open/close, 7-day cycle, explicit exchange tz, dtype passthrough); defensive copies or enforced immutability for `snapshot_at` aliasing; branch-prefix coverage validation.
11. **[M] Domain-shift fold design:** walk-forward folds deliberately placed across detected vol/regime breaks, weighted into promotion — selects for exactly the live-vs-backtest robustness gap.
12. **[M] Governance:** schema-validated model-change statement at promotion (training window, cost assumptions, WF-vs-incumbent, weak regimes) beside `brain.pt`; self-evolution ladder in configs (per-axis autonomy policy: what may change automatically vs behind the human gate); automated per-retrain error-analysis artifact surfaced in the GUI.

**Phase 3 definition of done:** the bot knows when it doesn't know (OOD gate + regime novelty tightening caps); features see the market, not just the symbol; hyperparameter search optimizes out-of-sample trading metrics with anti-snooping audits; stress and adverse-fill results are part of every promotion decision; calibration tracks reality.

---

# Phase 4 — Experiment-gated advanced mechanisms

*No timeline — each item enters only behind experiment isolation, and only after beating (a) plain pretrain+fine-tune and (b) equal-budget random search. These are the research payoffs, not the foundation.*

1. **PBT-lite population:** 3–5 concurrent checkpoints with perturbed hyperparameters, exploit/explore between retrain cycles; the promotion gate picks the canonical member; optional cross-distillation (fitness-weighted soup).
2. **Small SOM regime map:** 8×8 toroidal probabilistic SOM auxiliary head; per-cell outcome stats as a memory index; assignment entropy → sizing; Moran's-index acceptance test; GUI regime-map tab. (The lean, evidence-gated return of the deleted ESOM idea.)
3. **Differentiable Hopfield/attention retrieval:** energy-attention readout over the bank (learned temperature/projections) trained end-to-end on outcomes — only after Phase 2 counterfactual credit proves memory adds value at all.
4. **Meta-level learning:** online meta-gradient tuning of LR/replay-mix/class-balance; meta-learned instance weighting; Reptile/FOMAML regime-episodic meta-init for nightly retrains; evolved surrogate trading loss (CMA-ES, walk-forward Sharpe fitness). All require the purged/embargoed splits and deflated metrics from Phases 1–3.
5. **Capacity growth:** Net2Wider/Net2Deeper morphism or DEN-style threshold-triggered head expansion with group lasso, on sustained plateau + reliable regime detection; rehearsal is the baseline it must beat.
6. **Generative strategy discovery (pillar 1 beyond retraining):** future-guided positive selection via experiment checkpoints (adopt a candidate's decisions as targets only where they out-score the incumbent); reflection-on-failure trade autopsies writing corrective memories; latent-manifold sampling over 10+ profitable checkpoints; bounded TTRL-style episodic head fine-tune before low-confidence decisions (frozen encoder, discarded after).
7. **Caruana mixed ensemble:** greedy weighted ensemble over top-k checkpoints + non-neural baselines (LightGBM/CatBoost on FeatureContract features, rule strategies) — the mature form of soft-cap weighting, re-learned from outcomes each retrain; budgeted against intraday inference latency.
8. **Cross-play promotion pool + episode-parallel backtesting:** candidate vs a frozen pool of past promoted checkpoints and baselines across regime slices (require non-degradation); tensor-contiguous episode-parallel rollouts for cheap frequent re-evaluation.
9. **Deferred engineering nice-to-haves (P3s from the review):** outcome-driven feature relevance loop (permutation importance over realized trades → contract-versioned pruning); true multi-timeframe aggregation (30-min/daily families); true ring buffer for BarBuffer; multi-symbol historical requests (~13× fewer calls); utility-aware eviction generalization; backtest preset zoo consolidation; GUI preset descriptions from metadata.

---

# Continuous workstreams (run alongside every phase)

- **Testing:** each fix ships with its test. Standing order of new suites: performance math → engine behavior → parity tests (feature scalar/batch, train/serve, backtest/runtime) → risk-gate breadth → memory policy/updater lifecycle → tick_processor/router state machines → config hygiene → script imports.
- **§7 triage (verify-as-you-touch):** ~77 unverified medium + the low list are attached to their files; opening a file for a planned task obligates resolving its §7 items in the same PR. Any §7 item still unresolved when its subsystem's phase completes gets an explicit accept/defer note.
- **Docs truth:** README/cheatsheet/docstrings updated in the same PR as the behavior they describe; the Phase-1G.7 truth pass clears the backlog.
- **Ops journal:** after each phase, run a week of paper trading and write a short observed-vs-expected note (equity tracking, reason-code distribution, intervention rate, retrieval stats). These notes are the input to re-prioritizing the next phase.

# Retrain schedule (minimize churn)

| Retrain | After | Why |
|---|---|---|
| ⟲ 1 (end of 1B) | clean data + normalization fix + label/cost fixes | current checkpoints were validated on inputs deployment never sees |
| ⟲ 2 (end of 2F) | model-head disentanglement | new head semantics; first live test of anchored promotion |
| ⟲ 3 (Phase 3.3) | FeatureContract bump (market context) | contract change invalidates checkpoints by design |

# Binding do-NOT-do list (from the research roadmap; revisit only via Phase 4 experiment gates)

No NLS/soliton/quantum pricing models. No WGAN-GP market generator (bootstrap first, stylized-facts-gated). No per-bar test-time RL micro-training. No wholesale restoration of the deleted ESOM/Hopfield/dendritic stack. No Energy-Transformer pattern completion as a forecaster. No alpha-shape regime matching (MMD/Mahalanobis instead); never hard-swap models on raw regime signals (hysteresis + teachers first). No full NAS / entire-structure search. No second-order MAML (Reptile/FOMAML only). No zero-knowledge audit layer. No literal LLM self-rewarding machinery (only the contrast-collapse diagnostic transfers). No unconstrained HPO against a single backtest window. No autoencoder weight compression. No SSM/Mamba adoption for long context (take the gating/eviction patterns only). No causality-breaking augmentation (mixup/time-warp). No keeping `force_directional_flip` as the degeneracy answer. No autonomous self-generated objectives — objective changes, architecture growth, and strategy-code changes stay behind the human promotion gate. No weight-space fixpoints as goals. No treating per-cell/per-regime expectancy as alpha (priors and gates only).

---

*Traceability: Phase 0 ⊇ review §9 items 1–6 + §5.9/§7.6 operational hazards; Phase 1 ⊇ §9 items 7–14 + §5.1–5.12 P0/P1 items + roadmap Phase 1; Phase 2 ⊇ §5 P1/P2 memory/risk/strategy items + roadmap Phase 2; Phase 3 ⊇ remaining P2 items + roadmap Phase 3; Phase 4 ⊇ P3 items + roadmap Phase 4. Every §5 improvement item and §8 remedy appears above; §7 items are covered by the verify-as-you-touch rule attached to their files.*
