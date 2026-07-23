# DELETION AUDIT PLAN

**Date:** 2026-07-20 · **Status (updated 2026-07-22): EXECUTED — the pivot-independent safe batches are done.**

**Executed 2026-07-22** (Phase 0.2 of PIVOT_PLAN.md; each batch re-verified for 0 refs AFTER the week's confidence/self-monitor work, suite green throughout, per-batch commits): **A1** (5 dead modules + the memory/schemas 14-class cluster, 842→229 LOC), **A2** (all dead functions/methods/fields), **B1** (all 5 delete-with-tests — live properties preserved by porting the BarWindowDataset contract check to `pipeline.build().contract` and the ObservationBuffer JSONL roundtrip to `append_observation`), **D-delete** (compare_runs.py, run_backtest_parallel.py) + **D-archive** (→ `scripts/archive/`) + **E-safe** (regenerable junk), and the grep-verified subset of **A3** unused imports. Verification caught 3 stale A3 line-refs (checkpoints_tab `Path`, risk_tab `Qt`, widgets `COLOR_BG` are actually USED — kept).

**NOT executed (deliberate):** **A4 + F** (config-key deletions — batch with the pivot retrain commit per rule 2), **C** (gated experiments / reserved scaffolding — user/roadmap decision), **E user-decision** (oldsrc/, experiments/, irreversible untracked), the **`__init__` re-export blocks** (intentional public API via `__all__`, not clearly dead), and the multi-line-block remainder of A3 (pure F401 lint — sweep with `ruff --fix`).

**Method:** 18 scoped finder agents (one per `src/` package + config-fields + `scripts/` + repo hygiene) produced 151 candidates; every candidate was then re-verified by an independent adversarial agent instructed to *refute* the "unused" claim (bare-name grep across `src/`, `scripts/` incl. `.bat`/`.vbs`, `tests/`, `configs/*.yaml`, `docs/`, `README`; quoted-string / `getattr` / `importlib` / registry dynamic-dispatch hunts; `__init__` re-export checks; git-log recency). A final completeness critic swept for cross-scope misses. Verdicts: **115 DEAD · 19 TEST_ONLY · 10 GATED_EXPERIMENT · 4 USED (false alarms, dropped) · 3 UNCERTAIN.**

**Two ground rules before deleting anything:**

1. **Git-tracked deletions are reversible** (recoverable from history). **Untracked/gitignored deletions are NOT** — `oldsrc/`, `experiments/`, `artifacts/*`, `.claude/` have no git safety net. Everything irreversible is marked *user-decision* below.
2. ⚠ **The paper week is live as of Monday 2026-07-20.** `configs/paper_trade.yaml` sets three config keys this plan deletes (lines 15/18/19). Because all config models are `extra="forbid"`, deleting a pydantic field without removing its YAML keys **in the same commit** makes config loading crash — which would kill the paper run. **Recommendation: execute batches A4/F after the paper week, or at minimum run `scripts/preflight_monday.py` after any config-touching batch.**

**Verification recipe (per batch):** delete the batch → `pytest` full suite green (~940 tests; the suite includes source-contract tests that read files textually — all flagged items were checked against those) → re-grep the deleted names (expect 0 hits) → for batches touching runtime/config, `python scripts/preflight_monday.py`.

Suggested batch order (safe → policy-heavy): **A1 → A2 → A3 → B1 → D-delete → E-regenerable → A4+F (post-paper-week) → C/E-user-decision (your call)**.

---

## A. Verified DEAD in `src/` — safe to delete (~1,530 LOC)

### A1. Whole modules (zero importers anywhere) — ~1,020 LOC

| Module | LOC | Evidence |
|---|---|---|
| `src/data/parallel_prep.py` | 326 | All 4 public symbols: 0 refs repo-wide; superseded by `training/data_loading.py` + `sample_cache.py`; its config knobs are also dead (→ A4). CODEBASE_REVIEW already flags it. |
| `src/monitoring/events.py` | 159 | All 14 event classes: 0 refs incl. quoted-string; no `import_module`/`__import__` exists anywhere in the repo, so no dynamic escape hatch; "kept so old logs stay parseable" docstring has no reader code behind it. |
| `src/models/encoders.py` | 49 | `FeedForwardBlock`/`MultiTimeframeEncoder`: 0 refs since first commit; CODEBASE_REVIEW: "encoders.py is dead code". |
| `src/utils/serialization.py` | 20 | `to_jsonable`: referenced only inside its own file. |
| `src/utils/math_utils.py` | 14 | `clamp`/`mean`: string `math_utils` occurs in **zero** files; stdlib covers both. |
| `src/backtest/fills.py` | 11 | Only importer is `simulator.py:5`, called only inside dead `preview_cost`; live fill math is duplicated inline in `paper_executor.py:20-23`. |
| `src/memory/schemas.py` dead cluster | ~586 of 842 | 14 classes orphaned by the "Great Simplification" (`f8ecffe`): `MemoryUnit`(124), `FusionHealth`(122), `TopologyHealth`(59), `PathStatistics`(49), `PrototypeCluster`(44), `RetrievalCalibrationBucket`(35), `RiskStatistics`(32), `RetrievalCalibrationProfile`(26), `MemoryHealthReport`(26), `MemoryDecisionSupport`(20), `EncoderQualityReport`(16), `UnitStatus`(13), `RetrievedStateMatch`(11), `RetrievedClusterMatch`(9). They reference only each other; verified saved memory banks are payload **dicts** (never pickled instances) and `MemoryBank.from_payload` ignores their legacy keys — old banks keep loading. Delete as one block. **Exception:** `RetrievalHealth`(78) → C3 (documented P1 revival plan). Live classes `MemoryItem`/`OutcomeSummary`/`RiskSummary`/`PendingMemoryState` stay. |

### A2. Dead functions / classes / methods — ~330 LOC

| Symbol | LOC | Evidence |
|---|---|---|
| `BacktestSimulator.preview_cost` (`backtest/simulator.py:16`) | 9 | 0 callers; engine calls only `.execute()`. Remove the now-orphan `fee_bps`/`slippage_bps` fields + their pass-in at `engine.py:53-54`. |
| `RiskViolation` (`core/exceptions.py:5`) | 3 | Never raised/caught/imported; all importers use named imports of the other exceptions. |
| `is_same_trading_day` (`core/time_utils.py:17`) | 2 | Definition-only hit repo-wide. |
| `DEFAULT_FEATURE_DTYPE`, `ACCOUNTING_DTYPE` (`core/constants.py:18-19`) | 2 | Superseded: dtype comes from the feature contract / `device.py:47` hardcodes float64. |
| `HistoricalSequenceDataset` (`data/dataset.py:92`) | 12 | Sole ref = unused import `data_module.py:13` — **trim that import line in the same edit** (critic finding). |
| `BarBuffer.clear` (`data/buffers.py:95`) | 5 | Every `.clear(` hit repo-wide is dict/set/Qt, not this method. |
| `HistoricalDataService.cache_path_for` (`data/historical_data.py:262`) | 2 | Definition-only. |
| `HistoricalBarCache.exists` (`data/cache.py:241`) | 2 | All `.exists(` calls are `Path.exists()`. |
| `CacheManifest.total_rows_for_symbol` (`data/fetch_planner.py:251`) | 2 | Definition-only; `summary()` computes totals independently. |
| `aggregate_fold_metrics` (`evaluation/reporting.py:16`) | 42 | Superseded by live `walk_forward._aggregate_metrics`. |
| `PerturbationReport.metric_curve` (`evaluation/perturbation.py:120`) | 3 | Definition-only; class stays. |
| `OrderJournal.read_all` (`execution/order_journal.py:43`) | 11 | Definition-only (distinct from `DecisionLogWriter.read_all`, which stays → B2). |
| `OrderRouter.async_route` (`execution/order_router.py:200`) | 3 | Never awaited; runtime is synchronous. Remove `import asyncio` (line 3) with it. |
| `SessionGuard.reset_symbol` (`execution/session_guards.py:96`) | 2 | 0 callers since first commit (other `reset_symbol_state` hits are policy classes). |
| `COLD_START_MAX_WORKERS` (`execution/runtime_bootstrap.py:48`) | 1 | Both executors use `config.runtime.max_parallel_workers`; the "fallback" comment describes a call site that doesn't exist. |
| `RunningStandardizer` (`features/normalization.py:173`) | 39 | The C2 fix it awaited shipped via `NormalizationStats` instead; CODEBASE_REVIEW calls it dead. |
| `rolling_zscore` (`features/rolling.py:244`) | 16 | Live zscore path is `normalization.safe_zscore`. |
| `FeaturePipeline.infer_branch_dimensions` (`features/feature_pipeline.py:844`) | 5 | Definition-only; duplicates the live contract path. |
| `_reload_history` (`gui/tabs/checkpoints_tab.py:240`) | 2 | No call, no `.connect()`; live path is the worker-refresh chain. |
| `get_backtest_equity_series` (`gui_services/equity_service.py:92`) | 16 | Self-described placeholder returning empty data; 0 callers. |
| `RuntimeService.get_config_paths` (`gui_services/runtime_service.py:82`) | 3 | Definition-only. |
| `CheckpointFile.display_name` (`gui_services/state.py:315`) | 5 | GUI builds rows from raw fields; 0 attribute reads. |
| `list_available_experiments` (`gui_services/experiment_service.py:55`) | 12 | Critic finding: its only ref is the unused import at `experiments_tab.py:50` — delete function **and** import together. |
| `BrainTradingModel.initial_memory_state` (`models/brain_model.py:100`) | 9 | Self-described GRU leftover; protocol requires only `forward()`. |
| `PolicyHead.forward` `soma_activation` param (`policy_head.py:50`) | 6 | Accepted then `del`'d; sole caller `brain_model.py:165` allocates a tensor per forward just to feed it. `ModelOutput.soma_activation` **field stays** (live). |
| `TemporalEncoderOutput.activations` (`models/cnn_encoder.py`) | 4 | Write-only; sole consumer reads `.embedding` only. Matching edit at `brain_model.py:157`. |
| `HealthState.restore_trading` (`monitoring/health.py:29`) | 5 | Nothing ever un-blocks trading (by design); 0 refs. |
| `PortfolioState.leverage_ratio` (`risk/portfolio_state.py:39`) | 3 | The real check needs *prospective* leverage, computed inline at `risk_manager.py:279`. |
| `PositionState.notional` (`risk/portfolio_state.py`) | 3 | All `.notional` hits are the *monitoring* dataclass or inline recomputation. |
| `load_checkpoint` (`training/checkpoints.py:46`) | 12 | Everyone uses `load_checkpoint_payload`. |
| `columnar_from_shards` (`training/sample_columns.py:465`) | 30 | Role implemented in-module by `data_module.py`; never called. |
| `ColumnarDataset` (`training/sample_columns.py:417`) | 27 | Sole ref = unused import `data_module.py:36` — trim it too (critic). |
| `StartupReport.clock`, `ReconciliationResult.synchronized_at` | 2 | Write-only dataclass fields; `summary()` omits them, no reader anywhere. |

### A3. Unused imports & dead re-exports — ~30 lines, zero-risk

- `backtest/engine.py:5,10` — `torch`, `bars_to_tensor` (remaining "torch" hits are comments).
- `gui/tabs/training_tab.py:14-15` — `Qt`, `QColor`, `QTextCharFormat`, `QTextCursor` (keep `QTimer`).
- `gui/tabs/checkpoints_tab.py:11,13,22` — `Path`, `Qt`, `QLabel` (keep `QThreadPool`, `QUrl`).
- `gui/header.py` — top-level `ServiceError` + `get_live_equity_series` (line 135 is bound by the *local* import at 129).
- `gui/tabs/positions_tab.py:24-25` — `QHBoxLayout`, `QHeaderView` (**`Qt` is used here — keep it**).
- `gui/tabs/logs_tab.py:4`, `gui/tabs/risk_tab.py:4` — `Qt` only.
- `gui/tabs/memory_tab.py:6` — `QCheckBox`. · `gui/tabs/experiments_tab.py:16` — `Any`.
- `gui/widgets.py:27` — `COLOR_BG` constant (hex is hard-coded where used).
- Critic: dead re-export blocks in `src/gui_services/__init__.py:31-47` and `src/memory/__init__.py` — zero package-level imports exist; all callers import submodules directly.

### A4. Dead config keys — ⚠ every row couples pydantic field + YAML key(s) in ONE commit (`extra="forbid"`)

| Field | Defined | YAML keys to remove with it | Note |
|---|---|---|---|
| `runtime.decision_interval_sec` | `runtime.py:15` | `default.yaml:90`, **`paper_trade.yaml:15`** | Pacing is `broker.polling_interval_sec`. |
| `runtime.regime_hysteresis_enabled` | `runtime.py:40` | `default.yaml:122`, **`paper_trade.yaml:18`**, `experiments/all_new_features.yaml:9`, `experiments/no_regime_hysteresis.yaml:7` | Tombstone of a *removed* feature; "kept for GUI" comment is refuted (GUI excludes it). Retire the now-no-op preset yaml too (→ F). |
| `runtime.market_priors_enabled` | `runtime.py:41` | `default.yaml:123`, **`paper_trade.yaml:19`**, `all_new_features.yaml:10`, `no_market_priors.yaml:8` | Same tombstone pattern; not the (gated, live) `market_context`. |
| `model.feature_dim_hint` | `model.py:9` | `default.yaml:159` | Dims come from the feature contract. Checkpoint-safe: loader drops unknown meta keys (`factory.py:143-165`); incumbent `d748f28c` unaffected; only *future* fingerprints change. |
| `model.branch_depth` / `regime_hidden_dim` / `regime_embedding_dim` / `regime_count` / `regime_prototype_init_scale` | `model.py:12-37` | `default.yaml:162,164,165,166,185` | Knobs of the removed 1.x architecture; same checkpoint-safety as above. |
| `model.decouple_confidence_from_soma` | `model.py:36` | `default.yaml:172` | Whole write-only chain dies together: `PolicyHead.decouple_confidence` attr, `brain_model.py:88` literal, GUI checkbox `training_tab.py:238-244` + `training_service.py:157-158` ("Legacy toggle" per its own tooltip). |
| `training.scheduler_warmup_epochs` | `training.py:87` | `default.yaml:281` | No warmup scheduler exists (`supervised.py:237-251`). |
| `data.validation_strictness` / `data.max_prep_workers` | `data.py:17-18` | `default.yaml:65-66` | Existed for `parallel_prep.py` (A1), which never read them either. |
| `data.bars_per_trading_day` + `min_lookback_days_minute/_hourly/_multiplier_daily/_days_daily` | `data.py:54-58` | `default.yaml:55-59` | Sole call site passes positionally (`alpaca_client.py:286`) so the YAML can't reach the helper. **Keep the same-named function parameters** — live, and the known lookback bug (CODEBASE_REVIEW:473) is a separate fix. |

Note: `model.regime_input_dim` looked dead but is **USED** (`brain_model.py:45`, overridden from contract at `factory.py:84`) — keep.

---

## B. TEST_ONLY surface

### B1. Delete-with-tests (~165 src LOC + their tests)

| Symbol | LOC | Coupled edits |
|---|---|---|
| `BarWindowDataset` (`data/dataset.py:35`) | 57 | **Port** the feature-contract-consistency assertion in `tests/test_features.py:22-42` to the live columnar pipeline first — it tests a live property through a dead vehicle. |
| `collate_training_samples` (`data/dataset.py`) | 50 | Trim import `data_module.py:13`; tests in `test_state_path_training.py:382-410` test the dead function itself. |
| `ObservationBuffer` (`training/replay_buffer.py`) | 41 | Live loop uses `append_observation`/`load_observations` directly (those stay); update prose refs `replay_buffer.py:3-8`, `curriculum.py:100`; tests `test_close_the_loop.py:39,172,197`. |
| `RuntimeService.set_config_paths` (`runtime_service.py:73`) | 8 | GUI overrides flow via `runs/gui_runtime_override.yaml`; tests `test_gui_services.py:120,127`. |
| `build_brain_memory_policy(conservative_thresholds, verbose_fusion)` inert params | 8 | Self-documented "inert legacy parameters"; edits: `policy_factory.py:188-189,203-205,318-319`, shim `run_backtest.py:57-58,74-75`, existence-tests `test_backtest_policy_factory.py:43-48`. |

### B2. TEST_ONLY but KEEP (verified justified)

`serialize_bar` (fixture counterpart of the live legacy-load path) · `compare_walk_forward` (deliberate §4.4.3 audit tooling) · `RegimeSnapshot.from_dict` (deserializer of a persisted format; "offline half" implies an online consumer) · `Ledger.load` (documented durability contract, MASTER_PLAN §1D) · `RiskManager.emergency_stop` (sole setter of the live kill-switch read at `risk_manager.py:173,196` — line 196 added 2026-07-19) · `WebhookAlerter.flush` (test infra for the async queue) · `DecisionLogWriter.read_all` (documented tolerant-reader for the arm-1 outcome dataset, active file) · `app_name` (anchors the config smoke test).

---

## C. Gated experiments & reserved scaffolding — **your call, not cleanup**

### C1. Measured-negative arms & gated features — KEEP per project policy (truth artifacts)
`strategies/arbiter.py` (fully wired, gated at `arbitration.enabled=false`) · `MemoryAwarePolicy._apply_damping` (runs in the live path, gate inside) · `training/curriculum.py` incl. scalar helpers `pair_is_valid`/`sample_difficulty` · `training/risk_target_norm.py` · `evaluation/seed_ensemble.py` · `evaluation/memory_credit.py` (standing re-run gate, MASTER_PLAN:273) · `features/market_context.py` incl. the tests-only `reset_market_context_warnings` hook · `training/platt_calibration.py` (ACTIVE arm 1) · `data/news_archive.py` (NEWSBOX live) · `run_triggered_retrain` + `_file_fingerprint` (`retrain.py` — the gated curriculum arm's only integration path).

One redundancy *inside* a kept arm: **`build_arbitration_policy`** (`policy_factory.py:323`, 23 LOC) has zero callers even when arbitration is enabled (live path calls `build_arbiter` directly at `run_backtest.py:347` and `runtime_bootstrap.py:134`). Deleting the wrapper does not reduce the arm's capability — *user-decision*.

### C2. Reserved "self-evolution" scaffolding — never wired, but the roadmap names it (delete = roadmap decision; git history keeps it recoverable)

| Cluster | LOC | Status |
|---|---|---|
| `training/self_supervised.py` + `pretrain_*` config keys ×3 (`training.py:209-211`, `default.yaml:394-396`) | ~95 | Zero code refs; RESEARCH_ROADMAP:280 reserves it for planned P2 SSL-pretraining/familiarity-gate work. |
| `training/rl_scaffold.py` + `Experience`/`ReplayBuffer` classes (`replay_buffer.py`) + `rl_*` config keys ×6 (`training.py:202-207`, `default.yaml:387-392`) | ~92 | Zero refs; CODEBASE_REVIEW C8 pillar-3 material. Dies together or not at all. Rest of `replay_buffer.py` is **live**. |
| `retrain.py` lineage cluster: `RetainMetadata`+5 helpers (135, test-only, backbone of active close-the-loop program — lean keep) · `get_checkpoint_lineage` (25, zero refs even in tests) · `validate_retrain_prerequisites` (24) · `list_retrain_history` (12) | ~195 | File actively worked (2026-07-14 wave). Decide the cluster as one unit. |
| `RetrievalHealth` (`memory/schemas.py:521`) | 78 | Unreachable, but CODEBASE_REVIEW:246 has a P1 plan to revive its adaptive min-similarity. |
| `TrainingAugmenter.temporal_jitter` + `temporal_jitter_prob` plumbing | ~25 | Verified no-op end-to-end; config comment says "reserved until batched jitter lands"; review says wire-or-delete. |
| Contrastive knobs: `contrastive_enabled` (**lying knob** — read at `supervised.py:254` and logged, but `trainer.py` stores and never reads it; flipping changes nothing) + `contrastive_margin` | ~10 + plumbing | Multi-file cleanup (supervised/trainer/run_summary/tests/yaml) already queued as P2 in CODEBASE_REVIEW:278; MASTER_PLAN:266 still mentions a later contrastive phase. |
| `risk.max_position_per_symbol` (`risk.py:17` + 3 yaml) | ~5 | Unenforced no-op; review says **enforce or delete**. |
| `promote_if_eligible` + `evaluate_for_promotion` (`checkpoint_service.py`) | 63 | Test-only; the disabled v1 GUI "Promote" button reserves this flow. |

---

## D. `scripts/` (entry points — "dead" = superseded/completed, not unimported)

**Delete (superseded, nothing else dies):**
- `compare_runs.py` (96) — fully superseded by `run_train.py --list-runs/--best-run/--compare` (lines 668-705).
- `run_backtest_parallel.py` (172) — its documented purpose shipped as `run_cap_sweep.py`; update the two stale RESEARCH_ROADMAP lines (223, 271) in the same change.

**Archive (e.g. `scripts/archive/` — only committed reproduction recipe of measured arms whose `runs/*.json` artifacts are gitignored):**
- `launch_horizon_arm3.bat` + `_hidden.vbs` (horizon program closed).
- `launch_risk_norm_arm4.bat` + `_hidden.vbs` (arm 4 crossed off).
- `path_label_oracle_test.py` (192) — one-off analysis, no recorded verdict; project keeps analysis artifacts.

**User-decision (working manual tools grep can't prove unused):**
- `profile_backtest.py` (83) / `profile_backtest_small.py` (102) — complementary profilers (real-data vs synthetic no-network), used in two completed perf passes; if dropped, simplify the `_isolated_globals` fixture in `test_scripts_smoke.py:30`.
- `bench_cold_start.py` (105) / `bench_memory_build.py` (90) — reference-dead but *deliberately repaired 2026-07-08* (commit `03e4ee3`) — maintainer signal to keep.
- `run_experiments.py` (342) — **open P0** in CODEBASE_REVIEW:428: "purge or rewrite"; anchors `_experiment_base.yaml` + two ablation configs. Resolve the P0, don't side-delete.

**Keep (verified live/required):** `score_horizon_arm3.py`, `score_risk_norm_arm4.py`, `audit_horizon_costs.py` (cited by MASTER_PLAN as the reproducibility recipes behind committed verdicts), `run_seed_ensemble.py`, `run_memory_credit.py` (gated-arm harnesses), `launch_noise_floor_sweep.*` (canonical 3-seed relauncher), `audit_ev_inputs.py`, `smoke_test.py`, and the active set (collect_news + launchers, refit_platt_live, preflight_monday, run_paper, launch_paper_week, run_gui, run_train, sweep_train, run_backtest, run_benchmark, build_memory, fetch_historical, trading_dashboard).

---

## E. Local files & folders

**Safe to delete now (regenerable or worthless; gitignored):**
`pytest_arb.log`, `pytest_mc.log`, `pytest_prune.log`, `pytest_curr.log`, `benchmark_out.log` (stale console captures) · `.pytest_cache/` (regenerated every run) · `seif_trading_brain.egg-info/` (build byproduct; the venv uses PEP 660 dist-info, verified no egg-link) · `artifacts/market_memory.pt.bak_pre40symbols_*` ×2 (16 MB; restore-incompatible pre-40-symbol snapshots, no code globs `market_memory.pt.*`).

**User-decision (untracked → deletion is IRREVERSIBLE):**

| Item | Size | Notes |
|---|---|---|
| `oldsrc/` | 64 files, ~16.7k LOC | Confirmed pre-rewrite copy (contains `dashboard/`, `ml/`… absent from current src); zero refs. Zip it somewhere first if in doubt. |
| `artifacts/memory_bank_cache/` | 182 MB | Producer `src/memory/bank_cache.py` was deleted; contents non-regenerable but also unusable. |
| `experiments/` | 360 files | Write-only results from Apr 15-16 (pre-honest-eval numbers); dir auto-recreated by code; `runs/` is the real truth store. |
| `.claude/agents/` | 2 files | Unity/game agents from the Harvest-Havoc era, byte-identical copies exist at `~/.claude/agents/` — zero loss. |

**Do NOT touch:** `artifacts/features_cache` (17.1 GB), `artifacts/training_cache` (9.9 GB), `artifacts/historical`, `market_memory.pt`, `pending_observations.pt`, `checkpoints/`, `runs/` (145 truth artifacts, 28 git-tracked), `logs/`.

**Housekeeping note:** `prompts/audit_unwanted_changes.md` is untracked and *not* gitignored (permanent `??` limbo) — either commit it or add `prompts/` to `.gitignore`.

---

## F. Config YAMLs

- `configs/symbols.yaml` — **delete** (stale 4-symbol overlay superseded by `default.yaml:75-87`'s 13-symbol universe; only ref is the wrong `README.md:204` line — fix it in the same commit; as a GUI preset it's a silent universe-shrinking footgun).
- `configs/backtest_sweep_confidence_050.yaml` / `_070.yaml` — *user-decision*: bodies are literally `{}` (no-op); an in-file comment records a deliberate keep decision, but its "referenced by name" claim is refuted (0 refs).
- `configs/experiments/no_regime_hysteresis.yaml` / `no_market_priors.yaml` — retire together with their dead flags (A4); running them measures nothing vs baseline (critic finding).
- All other `configs/**` are reachable (name-keyed dynamic dispatch via experiment runner + GUI globs) and stay.

---

## G. False alarms caught by verification (kept, for the record)

`scripts/score_horizon_arm3.py` and `audit_horizon_costs.py` (finder said stale; MASTER_PLAN cites both as reproducibility recipes) · `model.regime_input_dim` (structurally required, only the YAML value is inert) · `contrastive_enabled` (read+logged, hence USED — but see C2: behaviorally a lying knob).

The 3 UNCERTAIN items (`run_experiments.py`, `bench_cold_start.py`, `bench_memory_build.py`) are filed under D user-decision with the ambiguity stated.

---

### Totals if you take every "delete"/"delete-with-tests" (Categories A+B1+D-delete+E-safe+F-delete)

≈ **1,530 LOC** of verified-dead `src/` code, ≈ **165 LOC** test-only surface (+ its tests), ≈ **270 LOC** of superseded scripts, 5 stale root logs, 2 regenerable cache/build dirs, 2 stale memory backups, 1 stale config. Category C adds ~500 LOC more *if* you decide to drop the reserved scaffolding, and E adds ~17k LOC (`oldsrc/`) + ~180 MB local clutter — both irreversible-side decisions that are yours.
