# SETAI Codebase Review

**Date:** 2026-07-07
**Scope:** Full working-tree review (uncommitted "Great Simplification" refactor included) of `src/`, `scripts/`, `configs/`, `tests/`. Legacy copies (`.venv/`, `oldsrc/`) excluded. All findings verified against the current working tree, not git history.
**Audience:** Project owner. Findings are judged against the stated vision: (1) the bot develops strategies on its own, (2) it also uses known real-world strategies under soft caps, (3) it changes its weights based on realized trading outcomes.

---

## 1. Executive Summary

The codebase is in a much better structural state than its documentation suggests. The uncommitted "Great Simplification" refactor deleted the speculative dendritic/soma/Hopfield/ESOM/fusion stack (~5,000+ LOC) and left a clean, layered pipeline: data → features (contract-enforced) → CNN+MLP brain → passive memory → deterministic risk → execution → backtest/walk-forward → supervised training. Several load-bearing mechanisms verified **clean**: feature causality and lookahead hygiene, FeatureContract enforcement end-to-end, order-safety chokepoints (risk-approval required at three layers, live-mode double interlock, idempotent submission), frozen-brain enforcement, AMP/class-ordering/loss numerics, and atomic observability writers.

However, the review confirmed **four critical defect clusters that make live/paper results untrustworthy and disable the primary safety rails**, plus a cluster of high-severity evaluation-validity defects:

1. **Live data corruption (critical).** The polling stream fetches Alpaca's latest **1-minute** bar (the `timeframe` argument is explicitly `del`eted) every 2 seconds and appends it to buffers seeded with **5-minute** bars, with **no timestamp dedup anywhere**. Within minutes, feature buffers are dominated by duplicated wrong-timeframe bars; volatility/momentum features collapse to values the model never saw in training.
2. **Train/serve normalization skew (critical).** Training z-scores every model input with train-set statistics that are never persisted; backtest and live inference feed **raw** pipeline outputs to the model. Every deployed logit, confidence, and EV-gate input comes from out-of-distribution inputs; val/test metrics cannot detect it. This plausibly explains the observed "brain degeneracy" that recent commits patched downstream.
3. **Dead safety rails in broker mode (critical).** In the only mode the bot actually runs (broker-routed Alpaca paper), fills never touch the local portfolio: cash/equity/realized/daily PnL are frozen at startup values, the Ledger stays empty, and the daily-loss limits compare against a permanently-0.0 `daily_pnl` — the loss circuit-breaker can never fire. Position sizing compounds on phantom equity.
4. **Fragile runtime (critical).** Per-tick broker reconciliation is unwrapped; one transient API error kills the whole trading session.

At the high tier: the offline memory-build script imports a deleted module (hard crash), walk-forward never retrains per fold and default backtests are ~90% in-sample (so promotion decisions are steered by memorization), session windows are hardcoded to EDT (wrong for ~4 months/year), and — most important against the vision — **the self-evolution loop does not exist**: retrain/replay/RL modules are explicitly RESERVED, realized outcomes are not even captured on the broker path, and model weights change only when a human reruns offline training.

The report below gives the deduplicated confirmed-defect table, refuted claims, per-subsystem reviews with all prioritized improvements, verified-clean areas, ~100 unverified medium/low items, test/config hygiene gaps, and a top-20 action list.

---

## 2. Architecture Assessment

### 2.1 The pipeline as it exists in the working tree

```
Alpaca (paper) ──► data lane
  AlpacaClient (BrokerDataClient protocol; MockBrokerClient fallback)
  PollingStreamManager (2s batched latest-bar polls, circuit breaker)
  HistoricalBarCache (.pt columnar tensor_v1) + chunked parallel backfill (manifest/resume)
  BarBuffer (deque + preallocated tensor view — O(1) snapshot on hot path)
        │
        ▼
features: FeaturePipeline — torch-native, dual path
  scalar build/build_from_tensor (live hot path) ⇄ vectorized build_batch (train/backtest/memory)
  FeatureContract (frozen, checkpoint-persisted, assert_matches at every consumer) ← the parity spine
        │
        ▼
models: BrainTradingModel = MarketTemporalCNN (dilated residual Conv1d) ⊕ flat features → MLP backbone
  heads: PolicyHead (3-class logits, confidence, target_fraction), ValueHead, optional RiskHead (MFE/MAE → EV gate)
  factory: self-describing checkpoints (meta v2.0: configs + contract + encoder fingerprint)
        │
        ▼
strategies: BrainPolicy (propose/propose_batch, one packed GPU→CPU sync)
  MemoryAwarePolicy — PASSIVE wrapper: second forward → latent → kNN diagnostics + Platt calibration.
  Memory does NOT gate/veto/size anything (fusion layer deleted).
        │
        ▼
risk: deterministic RiskManager (collect-all-reasons; EV gate replaces magic confidence threshold;
  caps, cooldowns, dedup, sessions) + linear PositionSizer + PortfolioState + in-memory Ledger
        │
        ▼
execution: trade_engine (thin loop) / runtime_bootstrap (all wiring) / tick_processor (Phase A eligibility →
  Phase B batched forward → Phase C sequential risk+route) / OrderRouter (quarantine state machine,
  idempotent client ids, retry+recovery) / PaperExecutor (backtest-only in practice)
        │
        ▼
backtest/eval: BacktestEngine (vectorized precompute, batched propose, same RiskManager/PaperExecutor),
  WalkForwardRunner (chronological folds), performance metrics, experiment reporting
        │
        ▼
training: supervised loop (forward-path targets, focal/CE + regression multitask, class balancing,
  columnar sample store, two-tier content-addressed cache, checkpoint isolation, Platt fit,
  post-training memory build). retrain/replay/RL/self-supervised: RESERVED stubs.
```

### 2.2 What the in-flight refactor changed

The uncommitted working tree deletes: `src/models/dendritic.py`, `soma.py`, `regime_map.py`, `memory.py` (GRU); `src/memory/hopfield_retrieval.py`, `esom_map.py`, `clustering.py`, `retrieval.py`, `dynamic_units.py`, `calibration.py`, `maintenance.py`, `bank_cache.py`, `path_stats.py`, `similarity.py`, `metrics.py`, `encoder_eval.py`; and the monolithic `src/core/config.py` (replaced by the `src/core/config/` package with domain-split pydantic models re-exported for import compatibility). The brain is now an honest CNN+MLP classifier; the memory bank is a flat append-only cosine-kNN store; the fusion/VETO layer is gone and `DecisionFusionConfig`/`TopologyGrowthConfig` are inert `enabled`-only stubs.

Consequences of the refactor that matter:

- **The simplification is defensible** (the module docstrings quantify the evidence: the complex model was not better), and the protocol layer (`TradingPolicy` → `BatchTradingPolicy` → `MemoryCapablePolicy`) survived cleanly.
- **The config/docs/GUI surface did not follow.** README, `docs/LIVE_LOG_CHEATSHEET.md`, ~85% of `configs/default.yaml`'s memory block (~100 dead knobs), ~14 of 25 experiment YAMLs, the fusion "sweep" presets, GUI checkboxes, backtest `--compare` scenario labels, and even the recent commit message ("gate fusion VETO") all describe machinery that no longer exists. Because every config model uses pydantic `extra='ignore'`, this drift is structurally invisible — dead YAML validates silently.
- **One entry point broke outright**: `scripts/build_memory.py` still imports the deleted `bank_cache` module.
- **Half-deleted safety plumbing remains**: `set_temporal_cutoff` is written but never read; the walk-forward per-fold memory rebuild guard checks an attribute that no longer exists; `reset_model_per_fold` is defined but read nowhere.

### 2.3 Fit against the vision

| Vision pillar | Status |
|---|---|
| (1) Develops strategies on its own | Not present. One supervised 3-class direction classifier; RL scaffold ~47 unwired lines; no strategy pool, search, or discovery mechanism. "Self-developing" = offline retrain of the same architecture. |
| (2) Known real-world strategies with soft caps | Not present. Momentum/mean-reversion/buy-and-hold exist only as backtest baselines (and cannot pass the EV gate); no runtime ensemble, no strategy identity on orders, no per-strategy budgets; all risk caps are global and hard (binary reject — no resize path despite README claiming one). Indicators are nonstandard SMA variants of MACD/RSI. |
| (3) Weights change with realized outcomes | 0% implemented for model weights. retrain/replay_buffer/rl_scaffold are RESERVED with zero callers; on the broker path realized outcomes are not even *captured* (empty Ledger, unfilled journal). The only outcome-adjacent adaptation is the memory bank finalizing bar-path statistics (not the bot's own trade P&L) — and memory is decision-inert. Platt calibration is fit once, offline. |

The architecture is a sound *foundation* for the vision — the outcome taxonomy (`OutcomeSummary`), reason-code observability, checkpoint promotion gates, and protocol layer are exactly the right raw material — but no loop closes.

---

## 3. Confirmed Defects (deduplicated, ordered by severity)

Severities reflect post-verification adjustments (several claims were verified true but downgraded/upgraded during evidence review; noted inline).

| ID | Sev | Location | Defect | Failure scenario | Suggested fix |
|----|-----|----------|--------|------------------|---------------|
| **C1** | **Critical** | `src/data/alpaca_client.py:289,327`; `src/data/stream_manager.py:33-101`; `src/execution/tick_processor.py:221-222`; `src/data/buffers.py:50-66` | Live polling feeds **duplicated, wrong-timeframe bars** into feature buffers. `get_latest_bar(s)` explicitly `del timeframe` and return Alpaca's latest **1-minute** (possibly still-forming) bar; the stream yields it every 2s with no novelty check; `process_tick` appends unconditionally **before** any guard; `BarBuffer.append` and `SessionGuard` (equal timestamps pass) have no dedup. Buffers are cold-started with true 5Min bars, then polluted ~30x/minute. | Within minutes of a paper run the 120-slot buffer is mostly duplicates of the last 2–4 real bars; a "48-bar" window spans minutes, not 4 hours; returns/volatility collapse toward 0; warmup gate passes on duplicates; the model trades on inputs unlike anything in training/backtest. MockBrokerClient generates a fresh bar per poll, so tests never expose it. Overnight the buffer becomes one repeated bar; missed bars after outages are never backfilled. | Dedup by timestamp before append (skip or overwrite-in-place when `bar.timestamp <= buffer.last_bar().timestamp`); honor the timeframe (aggregate 1-min bars into the configured bucket, commit only on bucket close, or fetch the latest completed bar via `get_historical_bars_range`); backfill gaps on reconnect; add a regression test that polling the same mock bar twice grows the buffer by 1. |
| **C2** | **Critical** | `src/training/data_module.py:496`; `src/training/sample_columns.py:494-544`; `src/models/factory.py:59-81`; `src/strategies/brain_policy.py:172-192`; `src/models/state_encoder.py:61-106`; `src/training/memory_builder.py:61-74` | **Train/serve normalization skew.** `normalize_columns` z-scores features, regime_input, temporal_tensor (per-channel), and every branch input in-place with train-split mean/std (clamp ±10); the stats are never persisted (checkpoint meta carries none; `RunningStandardizer` has zero callers). Every inference path — `BrainPolicy.propose/propose_batch`, backtest, `ModelStateEncoder`, post-training memory build — feeds **raw** pipeline outputs. Per-sample LayerNorm cannot reproduce per-feature dataset z-scoring; raw channel scales span ~1e-4 to ~1e2, so the transform is large and channel-dependent. | Checkpoint is trained/validated on z-scored inputs, deployed on raw inputs: logits, confidence, target_fraction, and the MFE/MAE risk-head outputs feeding the EV gate are all produced out-of-distribution; the Platt calibrator (fit on normalized-input confidences) is applied to raw-input confidences; memory-bank latents are encoded from a distribution the encoder never saw. Val/test metrics use normalized loaders, so reported quality has no relationship to deployed behavior — consistent with the observed degeneracy the "force directional trades" commit patched. | Have `normalize_columns` return per-tensor stats; persist them in checkpoint meta (`build_checkpoint_meta` extra) and beside the FeatureContract; apply them via a single shared helper in `propose/propose_batch`, `encode_batch`, backtest snapshot building, and `memory_builder` (or bake them into the model as buffers). Alternatively drop dataset-level normalization and move normalization into the pipeline shared by all consumers, then retrain. Add a train-vs-inference input parity test. |
| **C3** | **Critical** | `src/execution/tick_processor.py:57-66,207,470,529`; `src/execution/order_router.py:185-193`; `src/execution/paper_executor.py:49-53`; `src/execution/reconciliation.py:158-181`; `src/risk/risk_manager.py:56-61`; `src/execution/runtime_bootstrap.py:117-149,352-363` | **Broker-mode accounting is frozen and the daily-loss rails are dead.** `_build_client` always returns a client, so `intent.paper` is always False and every order routes to `_submit_live_order` — `PaperExecutor` (the *only* writer of `cash`, `realized_pnl`, `daily_pnl`, and the only caller of `Ledger.record` and the only trigger of `register_fill`) is unreachable. `_sync_live_state` hydrates positions and open orders only; no `get_account()` call exists in the loop; `sync_open_orders` treats a filled order vanishing from open orders as a silent removal (no transition, no journal fill price). | Buy $10k AAPL on Alpaca paper: broker cash drops to $90k but local cash stays $100k while the position hydrates in → `mark_to_market` reports ~$110k equity; realized losses vanish the same way. `daily_pnl` stays 0.0 forever, so `daily_loss_limit`/`daily_loss_pct_limit` can **never fire** in the only mode that touches a broker; exit cooldowns never register; sizing compounds on phantom equity; `equity_live.jsonl` reports fills=0 and realized_pnl=0 permanently — and no realized-outcome record exists for the self-evolution loop. | In `_sync_live_state`, fetch `client.get_account()` each tick (or cadence) and overwrite cash/equity from broker truth; derive `daily_pnl` from broker equity vs a session-start/day-anchored snapshot; detect fill transitions (closed-order polling or by-id fetch when an order leaves open orders), apply them to PortfolioState with the PaperExecutor close-out math, record `FillRecord`s, and call `risk_manager.register_fill`. |
| **C4** | **Critical** | `src/execution/tick_processor.py:57-66,207-208`; `src/execution/trade_engine.py:63-75`; `src/data/alpaca_client.py:141-171,211-242` | **One transient broker error kills the runtime.** `_sync_live_state` (`get_positions` + `get_open_orders`, every tick) has no exception handling; any 5xx/429/network blip propagates to the trade loop, which marks health blocked and **re-raises**, terminating `asyncio.run`. The stream layer and `get_clock` were hardened against exactly this class of failure; the per-tick reconciliation was not, and the Alpaca client has no request timeout or retry beneath it. | A 30-second Alpaca hiccup at any tick ends the paper-trading session; GUI shows ERROR; no trading resumes until manual restart — while positions remain open with no protective orders (market DAY orders only). | Wrap the sync in try/except: log, skip reconciliation for the tick, count consecutive failures, block trading (not crash) after N failures — mirroring the stream layer's pattern. Add bounded retry/backoff and explicit timeouts around the client's read calls. Distinguish transient vs fatal in `trade_engine`. |
| **C5** | **High** (downgraded from critical: runtime/training bank builds still work) | `scripts/build_memory.py:21-25,183-210,216-219`; `src/gui_services/memory_service.py:196` | Imports `src.memory.bank_cache` (deleted in the refactor) at module level → `ModuleNotFoundError` before argparse. The CLI **and** the GUI "Build Memory" subprocess are both dead; the final summary print also references removed `bank.prototypes/memory_units/esom_map` attributes, and `config.logging.json` resolves to the deprecated pydantic method instead of `json_logs`. Not blocking: `runtime_bootstrap._build_or_load_memory_bank` and the post-training build still produce banks. | Any invocation of `python scripts/build_memory.py` or the GUI build action crashes at import time. | Strip the bank_cache import and `--no-bank-cache` branches (build via `build_memory_bank_from_bars` + `save_memory_bank` unconditionally) or restore a minimal `bank_cache` module; fix the summary print to `bank.summary()` fields and `config.logging.json_logs`. |
| **C6** | **High** (downgraded: evaluation-validity, not money-movement) | `scripts/run_backtest.py:193-211`; `src/evaluation/walk_forward.py:184-245`; `src/core/config/evaluation.py:21` | **Walk-forward never retrains per fold.** `_wf_policy_builder` resolves the same canonical checkpoint (`checkpoints/brain.pt`) for every fold; fold train bars are used only for a contract-validation snapshot; per-fold val windows are never used; `reset_model_per_fold` is defined but read nowhere; the module docstring's "resetting model weights between folds" is false. Since training uses the same `config.data` range, every "test" fold lies inside the checkpoint's own train/val data. | `run_backtest.py --walk-forward` reports positive fold Sharpe and the operator concludes the model generalizes — a genuinely overfit model passes this "walk-forward" with flying colors, steering promotion toward memorization. | Train (or fine-tune) per fold on the fold's train slice, honoring `reset_model_per_fold`; or cheaper: persist the training data range in checkpoint meta and refuse/warn when a fold's test window overlaps it; fix the docstring; wire or delete `reset_model_per_fold`. |
| **C7** | **High** (downgraded, same rationale as C6) | `scripts/run_backtest.py:128-139`; `src/training/data_loading.py:51-84`; `src/training/supervised.py:265-266,480-498` | **Default backtests are ~90% in-sample.** `_load_bars_by_symbol` loads the identical `config.data` range training used (2022-01-01→now, split 70/20/10): ~70% of the backtest window is training data, ~20% is the val set that drove early stopping, best-epoch restore, sweep ranking, promotion gating, and Platt fitting. Nothing checks backtest/train disjointness (only a comment in `backtest_quick.yaml`). | All headline numbers from `run_backtest.py`, `--compare`, and every `backtest_*.yaml` scenario are dominated by in-sample performance; the "self-evolving from outcomes" loop is steered by memorization, not edge. | Persist per-split train/val date ranges into checkpoint meta at save time; have `run_backtest.py` default to bars after the checkpoint's train/val end, or hard-warn and tag metrics IN-SAMPLE when overlapping. |
| **C8** | **High** (verified true; reclassified from "critical defect" to "unimplemented core feature") | `src/training/retrain.py:6`; `replay_buffer.py:3`; `rl_scaffold.py:3`; `src/core/config/runtime.py:31`; `src/execution/runtime_bootstrap.py:462-472` | **The self-evolution loop never closes anywhere.** All three learning-from-experience modules are stamped "RESERVED — not wired into any active code path"; `check_retrain_trigger`/`validate_retrain_prerequisites` have zero non-test callers; training consumes only historical exchange bars — no code reads the ledger, journal, or equity log into any training signal; runtime weights are frozen by design. One partial channel exists: the memory bank finalizes bar-path outcomes each tick — but it learns market paths, not the bot's own realized P&L, and memory is decision-inert. | Six months of paper trading, including sustained losses in a specific regime, produce a next retrain identical in expectation to one trained without ever having traded. The user's pillar 3 is 0% implemented for weights. | Persist per-decision observations (features, action, realized P&L) into `ReplayBuffer`; call `check_retrain_trigger` from a scheduler/CLI; add a training mode blending paper observations with historical bars; populate `paper_observation_count` and RetainMetadata lineage; gate via existing checkpoint promotion. (Depends on C3: outcomes must be captured first.) |
| **C9** | **High** | `src/core/time_utils.py:24-30`; `src/execution/session_guards.py:40-48`; `src/risk/risk_manager.py:75-78` | **Session bounds are hardcoded EDT-only UTC times** (regular = 13:30–20:00 UTC). No zoneinfo/DST handling exists anywhere in `src/`. During EST (Nov–Mar): 8:30–9:30 ET premarket falls *inside* "regular" (and the broker-clock backstop doesn't block because `within_session('regular')` is True, and is skipped entirely when `get_clock` fails), while 15:00–16:00 ET falls *outside* it. | For ~4 months/year the bot **blocks all trading in the final regular hour** and **permits an hour of unintended premarket trading**. Premarket/afterhours bounds are equally EDT-only. | Define sessions in `America/New_York` local time and convert timestamps with `zoneinfo` before comparison; prefer the broker clock as authoritative with static windows as fallback; add winter+summer boundary tests. |
| **C10** | **Medium** (downgraded: requires a config typo plus partial mitigations exist) | `src/core/time_utils.py:18`; `src/core/config/runtime.py`, `risk.py` (no validators) | `within_session` silently falls back to the always-open `'crypto'` window for any unrecognized session name (`sessions.get(name, sessions['crypto'])`), and nothing validates `allowed_sessions` entries. A typo like `'regulars'` permanently disarms the session gates — **fail-open on bad input in a safety gate**. `require_market_open` + staleness checks bound the damage in the default config. | Typo in YAML → session gate passes ~24/7; genuine 24/7 exposure additionally needs broker-clock unavailability or `allow_prepost`. | Fail closed (raise or return False) on unknown session names; add `field_validator`s on `RuntimeConfig.allowed_sessions`/`RiskConfig.allowed_sessions` asserting membership in `session_bounds()`. |

> Note on deduplication: the original findings list contained multiple reports converging on C1 (stream/client/tick-processor facets), C2 (three reports), C3 (four reports: cash sync, daily-loss, fill capture, PaperExecutor unreachability — one root cause), and C5 (two reports). They are merged above with all evidence retained.

---

## 4. Refuted / Unproven Claims

Recorded so they are not re-reported.

- **`RiskConfig` negative `max_daily_loss` flows into nonsensical comparisons — REFUTED.** `RiskManager.__init__` explicitly normalizes the sign (`max_daily_loss = abs(...)`, risk_manager.py:17-20), so `max_daily_loss: -350` behaves identically to `350`. What remains is only a low-severity hardening gap: `RiskConfig` has no validators at all (unlike every sibling config), which is tracked as an improvement item (core P0) rather than a defect.

---

## 5. Per-Subsystem Review

Each subsystem: architecture note, strengths/weaknesses (condensed), **all improvement items with priorities**, and stale-docs highlights (full config-hygiene detail in §8).

### 5.1 Core (`src/core/`: config package, constants, device, exceptions, logging, seeds, time_utils)

**Architecture.** Shared-services layer. Config is a pydantic-v2 package split by domain; `load_config` deep-merges YAML overlays left-to-right and validates once; an after-validator back-fills data defaults and *intends* to sync training→memory horizons (defeated in practice — see below). `DecisionFusionConfig`/`TopologyGrowthConfig` are honest inert stubs. Support modules: Action enum, device context (float32 model / float64 accounting), seeding, structlog-optional logging with tee-to-file, and the UTC-hardcoded session windows consumed by the risk/session gates.

**Strengths.** Import-compatible package split (zero caller churn); predictable deep-merge overlay semantics; genuinely defense-in-depth broker gating (paper/live validator + config `confirm_live` + CLI `--confirm-live`); frozen-brain policy is explicit config pinned by tests; thoughtful TrainingConfig validation with a backward-compat loss-weights migration; clean model/accounting dtype separation; logging degrades gracefully without structlog; inert stubs are documented as inert.

**Weaknesses.** Session/timezone model too weak for an equities bot (no DST/calendar/holidays — C9/C10); pydantic `extra='ignore'` everywhere makes config drift invisible; `RiskConfig` — the most safety-critical model — has **zero validators**; no config surface for any vision pillar (no strategy registry, no soft caps, no adaptation policy — self-evolution is fenced off by `allow_online_model_updates=False` with no configured alternative); MemoryConfig is a graveyard (~100 dead fields, validated, GUI-exposed); the training/memory horizon sync is self-defeating (default.yaml sets `memory.forward_horizons` explicitly, so shipped configs train on [1,5,15] and label memory on [1,3,6,12]); no config provenance/versioning; inconsistent error contract (ConfigurationError vs ValidationError vs YAMLError); `tests/test_config.py` covers 2 tests.

**Improvements.**
- **[P0, small]** Timezone-correct, fail-closed sessions: `zoneinfo('America/New_York')` bounds; `within_session` raises/returns False on unknown names; validators on `allowed_sessions`; winter+summer tests. *(Fixes C9/C10.)*
- **[P0, medium]** `extra='forbid'` across the config package (keep `ignore` only on the two intentional stubs); fix the fallout keys in `configs/*.yaml`. Converts silent fail-open typos into startup errors.
- **[P0, small]** Add validators to `RiskConfig` (positivity/sign/range on every limit, `max_aggregate_notional >= max_position_notional`, precision bounds, pct in (0,1]).
- **[P1, small]** Fix `resolve_device` cuda fallthrough (raise ConfigurationError or warn+CPU); validate `DeviceConfig.preferred`/`matmul_precision`.
- **[P1, medium]** Prune dead MemoryConfig fields (+ validator clauses, default.yaml blocks, and the inert GUI toggles in training_tab/memory_tab).
- **[P1, small]** Enforce training/memory horizon consistency: remove the explicit default.yaml memory override so the sync runs, or raise loudly on divergence.
- **[P1, large]** Introduce `StrategiesConfig`: per-strategy registry (enabled, capital_fraction soft cap, drawdown attenuation) + `AdaptationConfig` (retrain triggers, promotion criteria, rollback). Gives the vision a declarative home.
- **[P2, small]** Unify the config error contract (ConfigurationError subclassing ValueError; wrap YAML/model_validate with file-path context).
- **[P2, small]** Record resolved-config provenance (source paths + sha256 fingerprint; persist model_dump per run).
- **[P2, small]** Harden logging reconfiguration (swap handlers before closing the old handle; guard None streams; rotate the main runtime log).
- **[P2, small]** Seed numpy; drop/annotate the ineffective runtime `PYTHONHASHSEED` write.
- **[P2, small]** Couple `bars_per_year`/`seconds_per_bar` to the timeframe (map + validator) or warn on mismatch.
- **[P2, medium]** Expand `test_config.py`: broker validator rejections, deep-merge semantics, horizon sync, loss-weights migration, session validation, RiskConfig validators, "default.yaml contains no unknown keys".
- **[P3, small]** Derive `metrics.CLASS_NAMES` from `constants.Action` (single-source the 0/1/2 ordering).

**Stale docs.** README documents deleted subsystems as current (dendritic/soma/Hopfield/ESOM throughout); `LIVE_LOG_CHEATSHEET.md` greps for log events that no longer exist; default.yaml sets dozens of knobs for deleted features as active; keys with no model field silently dropped (`training.sequential_memory_training`, runtime regime/market-prior sub-keys); paper_trade.yaml "enables" inert flags; GUI exposes dead checkboxes; the app.py "must match" comment about horizons is guaranteed false by the shipped stack.

### 5.2 Data (`src/data/`)

**Architecture.** Three lanes: live (AlpacaClient behind a protocol + MockBrokerClient; PollingStreamManager batched 2s polls with circuit breaker; BarBuffer deque+tensor); historical (per-symbol columnar `.pt` cache with leading/trailing extension merges; chunked parallel backfill with manifest/resume and gap validation); training (data_loading → training/data_module; `parallel_prep.py` and `BarWindowDataset` are orphaned dead code).

**Strengths.** Clean protocol boundary with a behaviorally faithful mock; batched latest-bar polling with careful partial-failure semantics; production-grade stream failure policy (empty-dict yields, rate-limited warnings, circuit breaker); excellent BarBuffer hot-path design; columnar cache format with legacy fallback and consistent UTC normalization; deterministic, resume-safe, atomic chunk planner/manifest with real gap analysis; validation reports instead of silent "fixes"; thorough planner/manifest tests.

**Weaknesses.** No bar-newness or aggregation contract at the live boundary (root of C1); historical `spread=0.0` vs live real spread — systematic feature skew on the contract-shared column; **no corporate-action adjustment** (raw prices for a universe containing NVDA's 10:1 split → ~-90% one-bar discontinuities poisoning labels and memory outcomes); OHLCV+spread is the only data type (no room for known-strategy inputs); two historical systems share files but not metadata (coverage truth can drift); dead prep code presented as the training path; cache reads always rehydrate full Python object lists; gap detection exists but `validation_strictness` is never consumed.

**Improvements.**
- **[P0, medium]** Dedup + timeframe-aggregate live bars before BarBuffer (newness contract in the stream; honor timeframe in `get_latest_bar(s)` or fetch latest completed bar; guard in `BarBuffer.append`; regression test). *(Fixes C1.)*
- **[P0, small]** Fix the stdlib-logger kwarg crash in cache legacy fallback (`cache.py:242` — `%s` formatting; test with a weights_only-incompatible payload).
- **[P0, small]** Serialize `CacheManifest` record+save globally in `ParallelFetchExecutor` (single manifest lock or main-thread batching; unique tmp filename on Windows).
- **[P0, medium]** Timeframe-aware cached-range tolerance so offline loads don't demand sliver refetches; serve cached bars when client is None and only a sliver is missing.
- **[P1, small]** Fetch split/dividend-adjusted bars (`adjustment=Adjustment.ALL`, config field, bump cache version to invalidate raw caches).
- **[P1, small]** Return empty results (or typed `DataUnavailableError`) instead of crashing on zero-bar fetches.
- **[P1, small]** Make `--force-refresh` actually refetch (thread `skip_covered` into `_fetch_single_chunk`).
- **[P1, medium]** Fix lookback math for multi-minute/hourly timeframes; wire the decorative `DataConfig` lookback knobs or page backwards until `limit` bars return.
- **[P1, small]** Atomic cache writes (tmp + `os.replace`).
- **[P1, medium]** Delete or re-wire dead prep code (`parallel_prep.py`, `BarWindowDataset`, unused PipelineConfig fields).
- **[P1, medium]** Resolve the spread train/live skew (estimate historical spread, zero live too, or drop the column from the contract until historical quotes exist).
- **[P2, medium]** Keep the columnar format end-to-end (`load_columnar` + tensor-mask range filtering).
- **[P2, small]** Cross-check manifest coverage against cache files (drop stale entries).
- **[P2, small]** Namespace batch-level stream errors (stop counting `'_quote_batch'` as a symbol).
- **[P3, medium]** Multi-symbol historical requests (batch symbols per chunk window; ~13x fewer requests).
- **[P3, small]** True ring buffer for BarBuffer (drop the O(maxlen) shift-with-clone on hot-path appends).

**Stale docs.** `data.bars_per_trading_day`/lookback knobs never reach the helper; `pipeline.validation_strictness`/`max_prep_workers` consumed by nothing; `broker.base_url` never used to construct the endpoint (paper flag decides); `parallel_prep` docstring claims contract preservation it doesn't do; README's "replace polling with websockets" framing understates the current duplicate-bar defect.

### 5.3 Features (`src/features/`)

**Architecture.** (T,6) OHLCV+spread tensor → FeatureSnapshot (sorted flat vector, prefix-matched branch tensors, regime vector, (9,T) temporal tensor). Dual paths with a documented equivalence invariant: scalar `build/build_from_tensor` (live hot path) vs vectorized `build_batch` (training/backtest/memory) with post-zeroing mirroring scalar guards. Batch results disk-cached keyed on an endpoint-only bars signature + config. FeatureContract pins ordering/branches/dtype and is enforced everywhere. Deliberate SMA (not EMA) indicators to keep values window-local so both paths agree. `RunningStandardizer` is dead code.

**Strengths.** FeatureContract as a checkpoint-persisted, assert-enforced artifact is genuinely good engineering; the dual implementation is done with unusual care and documented boundary semantics; the SMA tradeoff is recorded at the decision site; torch-native, allocation-light, with profiled micro-caches; a real future-leakage regression test for the batch path; atomic cache writes with kill-switch and config-fingerprint invalidation; the volume_latest removal note preserves institutional memory.

**Weaknesses.** The feature set is fixed and hand-crafted — no outcome-driven feature relevance loop (vision pillar 1 has no representation here); "known strategy" fidelity is weak (SMA-MACD/SMA-RSI carry the names but not the semantics; no VWAP/Bollinger/ADX); **no normalization layer exists** (RunningStandardizer dead; scaling delegated to model LayerNorm; no winsorization of unbounded ratios) — the structural root of C2; `multi_timeframe.py` is single-timeframe window variation despite its name; `regime_trend_strength` is price-scale-dependent (a $500 stock reads as "weaker trend" than a $20 stock); `first_valid_index = windows['long']` doesn't cover MACD/RSI/volatility warmups; **no scalar-vs-batch equivalence test exists** — the invariant the whole train/live parity story rests on is unverified; calendar features assume 5-day weeks and are timezone-naive; `snapshot_at` returns aliased views into shared batch tables.

**Improvements.**
- **[P0, small]** Salt the feature-cache fingerprint with a schema/code version constant — otherwise every future formula edit silently serves stale features to training/backtest while live uses new code. Cheapest correct fix in the repo relative to its risk.
- **[P0, small]** Add the scalar-vs-batch equivalence test (default config AND a small-window config that would expose the RSI/momentum off-by-ones).
- **[P0, small]** Sanitize features on the live path (`nan_to_num` or isfinite-reject in `build_from_tensor`/tick site; epsilon on `return_1` denominators) — training data is sanitized, live is not.
- **[P1, small]** Strengthen the bars signature (cheap content hash or add sum(close)/sum(volume)/midpoint samples) to kill interior-revision collisions.
- **[P1, small]** Fix the RSI (`[:_RSI_PERIOD]`) and momentum boundary off-by-ones; derive `first_valid_index` from **all** family warmups.
- **[P1, small]** Unify epsilon between scalar and rolling return helpers (note default.yaml's epsilon is 5e-4, 500x the code default); update "bit-equivalent" docstrings to the actual tolerance.
- **[P1, medium]** Delete or actually wire `RunningStandardizer`: fit per-feature stats on train, store in contract/checkpoint, apply identically everywhere; winsorize unbounded ratios. *(Structural fix for the C2 class.)*
- **[P1, small]** Make the feature cache absolute-pathed, size-capped, and disabled in the test conftest; prefer plain-dict payloads loadable with `weights_only=True`.
- **[P2, medium]** Offer true EMA/Wilder indicators alongside (or rename the SMA variants honestly); add VWAP/Bollinger %B under existing branch prefixes.
- **[P2, small]** Defensive copies or an enforced immutability contract for `snapshot_at`; stop sharing mutable name lists across snapshots.
- **[P2, small]** Validate branch prefix coverage (warn on unmatched/multiply-matched features; remove the redundant `volume_` prefix).
- **[P2, medium]** Scale-invariant `regime_trend_strength` (returns-based vol); richer calendar (minutes-to-open/close, 7-day cycle, explicit exchange tz, pipeline dtype passthrough).
- **[P3, large]** Outcome-driven feature relevance loop (permutation importance over replay/realized trades feeding contract-versioned pruning/promotion).
- **[P3, large]** True multi-timeframe aggregation (30-min/daily families) + cross-sectional context (vs-SPY, breadth).

**Stale docs.** README's dendritic/multi-timescale claims; `model.regime_input_dim: 4` is dead (actual regime input is 10-dim, overridden from the contract); equivalence docstrings overstate ("bit-equivalent"); epsilon divergence code-vs-yaml; feature_cache docstring omits its two biggest staleness holes; RunningStandardizer docstring describes an active mechanism that nothing uses.

### 5.4 Models (`src/models/`)

**Architecture.** Two `TradingModelProtocol` implementations: BaselineMLP and BrainTradingModel (optional dilated-conv temporal CNN ⊕ flat features → MLP backbone; policy/value/optional risk heads). `ModelOutput` retains nine legacy fields populated with placeholders. `factory.py` builds from the contract, embeds full reconstruction meta + encoder fingerprint (v2.0), auto-disables the CNN when temporal weights are absent, blocks untrained checkpoints, and optionally torch.compiles. `ModelStateEncoder` batches snapshots into L2-normalized latents for the memory bank. `encoders.py` is dead code. Tests are shape-only.

**Strengths.** Contract-driven construction with checkpoint-vs-runtime assertion; self-describing checkpoints; graceful CNN auto-disable; explicit untrained guard; the simplification itself is honest, evidence-quantified engineering; careful single-sync GPU→CPU discipline; defensive shape validation throughout; opt-in, documented compile wrapping.

**Weaknesses.** No mechanism for self-developed strategies (single 3-class classifier; value head is a drawdown regressor, not a critic — the RL scaffold has nothing to build on); no strategy-prior structure to soft-cap (pillar 2); no outcome-conditioned input path (memory context deleted; regime/branch inputs explicitly ignored) — realized feedback can only reach weights via full offline retrain; `target_fraction` semantically conflated (regressed to primary return AND every horizon, then consumed as position size — sizing is an accidental byproduct); confidence is not a probability (trained against "stability"/MFE depending on config, then treated as P(win) by the EV gate and Platt); nine dead ModelOutput fields drag through every caller with misleading diagnostics; one shared head trained against conflicting horizons; ablation config flags silently ignored (GUI/experiment YAMLs toggle nothing); shape-only tests hide all of the above.

**Improvements.**
- **[P0, small]** Save the **effective** (override-applied) model config in checkpoint meta (fast-mode small-profile checkpoints are currently unloadable); round-trip test.
- **[P0, small]** Remove or fix the branch-order `full_features` fallback (silently permuted reconstruction) — raise "full_features required" or scatter by the contract's global order.
- **[P0, medium]** Make batched inference invariant to batch composition (bucket by temporal length, or masked mean pooling instead of AdaptiveAvgPool over zero padding); regression test `propose_batch([a,b]) == [propose(a), propose(b)]`.
- **[P1, medium]** Split position sizing from return prediction (per-horizon return heads; dedicated sizing head or deterministic Kelly-style sizing from expected return + risk head).
- **[P1, medium]** Give confidence a fixed, calibrated meaning (proper scoring rule on direction correctness, or softmax probability of the chosen class).
- **[P1, large]** Add an outcome-conditioned input path (retrieved memory statistics as a contract-defined context vector) — the minimal step toward pillar 3 between retrains.
- **[P1, medium]** Prune dead ModelOutput fields, dead config flags, and the dead GRU stacking in brain_policy; make no-op ablations fail loudly; update GUI accordingly.
- **[P2, large]** Strategy-prior head structure with soft caps (K interpretable strategy heads mixed by cap-clamped learned weights, per-strategy attribution) — pillar 2 at the model layer.
- **[P2, small]** Delete dead `encoders.py` (or document why kept; fix FeedForwardBlock depth semantics).
- **[P2, small]** Pass `head_hidden_dim` to RiskHead; validate CNN kernel sizes at construction (even kernels crash at runtime).
- **[P2, small]** Harden checkpoint loading (narrow the weights_only fallback; trusted-source flag) and the compile wrapper (`dynamic=True`, scoped `suppress_errors`, recompile-count logging).
- **[P3, medium]** Expand tests beyond shapes: batch-vs-single equivalence, fast-mode round-trip, eval determinism, CNN gradient flow, golden ModelOutput semantics.

**Stale docs.** README/cheatsheet document deleted architecture; default.yaml model section carries ~10 dead fields; the three "ablation" experiment YAMLs train baseline-identical models; the propose_batch docstring describes dead GRU state handling; the brain_model comment about full_features concatenation is wrong.

### 5.5 Memory (`src/memory/`, `memory_policy.py`, `build_memory.py`, `memory_builder.py`)

**Architecture.** Flat append-only associative store: MemoryBank (items + latent matrix, single cosine-kNN op), MemoryItem (latent, OutcomeSummary with per-horizon returns/MFE/MAE/vol/path labels, contract+encoder fingerprints), tolerant torch.save persistence. Writers: offline `build_memory_bank_from_bars` (batch-encoded, md5-seeded per-symbol stratified downsampling) invoked by the (broken) CLI, post-training builder, runtime cold start, and walk-forward; online `AssociativeMemoryUpdater` (pending → finalize when future bars exist, persistable pending store). Consumer: MemoryAwarePolicy — **diagnostics only**; backtests force `updater=None`. ~600 lines of schemas and ~80 config fields are dead fusion-era remnants.

**Strengths.** Minimal legible core with explicit invariants and honest docstrings; conceptually right provenance guards (contract + encoder fingerprints, fail-fast loads); a solid reusable outcome taxonomy — exactly the raw material outcome-driven adaptation needs; deterministic stratified sampling; chunked batch encoding; leakage awareness (read-only banks in backtests, CPU pinning with rationale); restart-safe pending persistence; tolerant forward-compatible serialization.

**Weaknesses.** Memory is **write-only with respect to decisions** — pillar 3 is not served here at all today; the outcome-adaptive machinery exists only as orphaned dead schemas (RetrievalHealth/FusionHealth adaptive thresholds); no path-label→strategy-prior mapping (pillar 2 hook unused); no pattern discovery beyond raw kNN (pillar 1); retrieval is naive for financial data (no temporal-adjacency exclusion — top-k dominated by neighboring bars including the query itself; no recency weighting, no min-similarity floor despite the config field); pure FIFO eviction; the two build paths produce non-interchangeable banks (different warmup, missing fingerprint) with invisible differences; the config surface actively lies about capabilities.

**Improvements.**
- **[P0, small]** Fix `scripts/build_memory.py` (import, cache branch, summary print) — CLI and GUI builds are both dead. *(C5.)*
- **[P0, medium]** Make the backtest temporal cutoff real: filter `query_top_k_simple` by `_temporal_cutoff` and exclude the query's own timestamp/symbol — or delete `set_temporal_cutoff` and the engine comment so the code stops claiming a guarantee it doesn't provide.
- **[P0, small]** Repair the walk-forward per-fold memory rebuild (drop the dead `hasattr(policy,'retriever')` and `updater is not None` guards; regression test that fold-bank max timestamp ≤ train_end − embargo).
- **[P1, small]** Stamp embargo + provenance metadata at build time (`builder_embargo_end_ts`, warmup/embargo bars, always the encoder fingerprint — including from `memory_builder`); remove/downgrade the bootstrap fingerprint-stamping fallback.
- **[P1, medium]** Vectorize `MemoryBank.extend` (bulk stack+cat or preallocation) — the current per-append full-matrix `torch.cat` makes builds O(N²) (~200 GB cumulative memcpy at 20k items); fix the docstring claiming amortization.
- **[P1, medium]** Fix the online update lifecycle: default mode vs `observe()` semantics; age/size caps on `pending_by_symbol`; move `save_pending`/`save_memory_bank` off the per-symbol hot path (debounced, background); isoformat timestamps so `weights_only=True` works.
- **[P1, large]** Reconnect memory to decisions with **bounded, outcome-driven influence**: direction agreement + MAE-p10 from top-k neighbors → a size multiplier in [0.5, 1.0] and advisory confidence adjustment (damping only, never creating trades); revive RetrievalHealth's adaptive min-similarity from realized fills; per-path-label exposure caps as the "known strategy" soft-cap hook. *(Vision alignment.)*
- **[P2, small]** Validate `feature_contract_id` per item; enforce (or delete) `require_encoder_match`.
- **[P2, small]** Reconcile `similarity_metric` config vs implementation (euclidean/attention silently execute as cosine; 'dot' unreachable).
- **[P2, medium]** Purge ~600 lines of dead schemas and ~80 inert config fields (+ default.yaml blocks, GUI reads).
- **[P2, medium]** Retrieval hygiene: temporal-adjacency exclusion window, same/cross-symbol filters, min_similarity floor (config field exists, unused).
- **[P3, medium]** Utility-aware eviction instead of pure FIFO.
- **[P3, small]** Payload version check on load; item_id dedup on append (restart replay safety).

**Stale docs.** README's entire memory story (Hopfield/ESOM/dynamic units/veto); `require_encoder_match: true` promises a hard-raise that doesn't exist; ~14 experiment YAMLs toggle deleted memory features; the engine/factory comments claim cutoff/fusion behavior that is gone; `memory_service.MemoryStatus` reads bank.summary() keys that no longer exist (GUI shows 0s); updater comments reference removed functions.

### 5.6 Training loop (`src/training/`: trainer, supervised, losses, targets, splits, data_module, class_balance, augmentation, metrics)

**Architecture.** `run_supervised_training` orchestrates: bar resolution (Alpaca/cache/synthetic, fast-mode overrides) → vectorized per-symbol sample build (batch features + forward-path outcomes → direction/path/regression targets with per-symbol adaptive epsilon) → two-tier content-addressed caches → columnar store → contiguous per-symbol chronological splits → in-place train-stat z-scoring → weighted loaders. Trainer: AMP-aware focal/CE + three regression losses reusing the model's heads, adaptive focal gamma from val minority recall, early stopping + best-state restore. Post-fit: detailed reports, isolated checkpoint save with provenance meta, run summary, optional Platt fit, post-training memory build. retrain/rl/replay/self-supervised: RESERVED.

**Strengths.** Real leakage awareness in the data path (train-only epsilon and normalization stats, causality tests); excellent, measured memory engineering (columnar 22KB→3KB/sample, shard absorption, VRAM-gated GPU preload); production-grade caching; rich observability (target diagnostics, collapse warnings, logit diagnostics, epsilon sensitivity sweeps, structured summaries); a serious class-imbalance toolkit; checkpoint isolation protecting the canonical brain.pt; robust trainer details (NaN diagnosis naming the offending loss component, skip-step on bad grads).

**Weaknesses.** The self-evolving core is absent (RESERVED modules; outcomes never reach the optimizer); no known-strategy priors/imitation (pillar 2); the objective is not trading-aware (fixed-horizon direction CE, no PnL weighting, cost defaults 0, best-epoch selection on raw multi-task loss rewards flat-hedging); output heads structurally overloaded (target_fraction ← primary + every horizon; confidence ← stability/MFE/vol; value ← drawdown+MAE); single chronological split with no purge/embargo; val is triple-used (adaptive gamma, early stopping, Platt); adaptive epsilon is horizon-blind (1-bar std thresholding a 15-bar return — ~√15 too narrow); augmentation is token (noise on flat features only; temporal_jitter never invoked).

**Improvements.**
- **[P0, medium]** Persist + apply normalization stats at inference (or drop dataset normalization for pipeline-level causal normalization); train-vs-inference input parity test. *(Fixes C2.)*
- **[P0, small]** Save the effective model config in fast-mode checkpoints (+ load test).
- **[P0, small]** Add label-affecting fields (adaptive-epsilon knobs, `transaction_cost_bps`) to cache fingerprints; bump cache versions.
- **[P0, small]** Purge `max_horizon` samples at split boundaries (embargo); strengthen the split test to assert the gap.
- **[P0, small]** Make `primary_return` use `effective_primary_horizon` everywhere (targets, diagnostics, Platt outcomes).
- **[P1, medium]** Real per-horizon return heads (or drop the loss); stop overloading confidence/value; risk targets only when the risk head exists.
- **[P1, large]** Wire outcome-driven retraining: retrain triggers ← paper monitor, realized outcomes → replay buffer, warm-start fine-tune entry point, promotion-gated. *(C8; pillar 3.)*
- **[P1, small]** Direction-adjust Platt calibration and isolate its artifact per run.
- **[P1, small]** Configurable model-selection metric (balanced accuracy / macro-F1 / cost-adjusted expectancy proxy instead of raw multi-task loss).
- **[P1, small]** Horizon-aware adaptive epsilon (scale by √horizon); feed the actually-used per-symbol epsilon into diagnostics.
- **[P2, small]** Remove residual per-batch GPU syncs (tensor loss components, on-device accumulation, epoch-end grad-norm sync).
- **[P2, small]** Vectorize the confusion matrix (bincount instead of a per-sample Python loop).
- **[P2, small]** Consistent augmentation across all model inputs (or drop feature-only noise); wire or delete `temporal_jitter`.
- **[P2, small]** Config/seed hygiene: config-driven seed instead of hardcoded 7; delete dead contrastive/latent plumbing and stale fast-mode overrides for removed modules.
- **[P3, medium]** Calendar-aligned global splits; walk-forward retrain-evaluate as a first-class training mode.
- **[P3, small]** Surface NaN sanitization counts from the collate (silent corruption becomes visible).

**Stale docs.** README's training-objective claims (latent-consistency/contrastive) describe removed losses; `loss_weights` validation *requires* a 'latent' key nothing reads; `temporal_jitter_prob` is a no-op; fast-mode overrides shrink dimensions of deleted modules; the sample-cache "MUST be in the fingerprint" invariant is violated; the leakage-test header claims GRU/ablation coverage whose sections are empty; RESERVED modules ship config knobs (`rl_*`, `pretrain_*`) nothing reads.

### 5.7 Training lifecycle (checkpoints, promotion, retrain, replay, RL, Platt, sample cache, run summaries)

**Architecture.** Three tiers: ACTIVE (supervised orchestration, two-tier cache, self-describing checkpoints, isolation, run summaries + compare CLI, optional Platt to `runs/platt_calibration.json` loaded by backtest and runtime); SEMI-WIRED (PromotionGate + registry consumed only by GUI services with the Promote button disabled; run_train never gates; runtime ignores the registry); RESERVED (retrain lineage/triggers, replay, RL, self-supervised).

**Strengths.** Incident-driven, well-tested checkpoint isolation; production-grade sample cache (content-addressed, fail-open, atomic, sharded, kill-switch, size-capped, cross-shard contract enforcement); self-describing checkpoints; fail-safe promotion gating with readable decisions and retire-previous semantics; honestly labeled dead scaffolding; a real experiment-tracking workflow; operator-protective checkpoint manager (soft delete, registry-active guard); clean passthrough-when-unfit calibrator.

**Weaknesses.** No end-to-end evolution loop (C8); **promotion is decorative** — run_train overwrites the canonical brain.pt with no gate, and the registry is not authoritative (runtime resolves from config path); gates judge only classification metrics — a checkpoint can pass every gate and lose money; calibration is one-shot and historical (never refit from realized outcomes); no strategy identity anywhere in the lifecycle; the RL scaffold is ~the whole feature away from usable; checkpoint fingerprints are nominal (never verified against file contents); two parallel metric-key schemas (flat vs dotted) with no adapter — future wiring will silently read 0.0 and block everything.

**Improvements.**
- **[P0, small]** Label-affecting fields into the sample-cache fingerprint (+ version bump, tests). *(Shared with 5.6.)*
- **[P0, small]** Direction-aware Platt fit (predicted-sign × primary_return, flat excluded, minus costs).
- **[P0, small]** Isolate `platt_calibration.json` like the checkpoint (fast/experiment runs must not clobber the canonical calibrator the runtime loads); replace the hardcoded `runs/` in both loaders with `config.training.runs_dir`.
- **[P0, small]** Atomic checkpoint save (tmp + replace) + previous-version backup for the canonical brain.pt.
- **[P1, medium]** Wire promotion into run_train and make the registry authoritative (gate before overwriting brain.pt; runtime resolves via `get_active_checkpoint` with config-path fallback; `--force-promote` escape hatch).
- **[P1, large]** Implement the outcome-driven retrain loop (un-reserve retrain.py; per-metric drift thresholds — fix the shared 0.15 applied to both Sharpe and win-rate; scheduled/GUI trigger; rolling-window retrain; gated promote; lineage records). *(C8.)*
- **[P1, small]** Shared metric-flattening adapter (checkpoint meta / run summary → gate keys) with a key-alignment test.
- **[P1, large]** Gate promotion on trading metrics (walk-forward Sharpe/drawdown/profit-factor floors via the existing engine), not just classification metrics.
- **[P2, medium]** Refit calibration from realized paper outcomes; version the calibrator with the checkpoint it calibrates.
- **[P2, medium]** Harden the registry (tolerate corrupt JSON, tmp+replace writes, sha256-verified fingerprints, lock file for concurrent promotion).
- **[P2, small]** Un-nest the val-loss regression gate from `require_improvement_over_parent`.
- **[P2, small]** Register safe globals so `weights_only=True` actually succeeds; stop logging "legacy format" on FileNotFoundError.
- **[P3, small]** Touch shard mtime on cache hit (true LRU eviction).
- **[P3, small]** Namespace run-summary files (stop globbing platt/retrain JSON into `--list-runs`); fix the RetainMetadata typo and columnar type hints/log counts.

**Stale docs.** The Promote-button tooltip claims run_train promotes (it doesn't); the sample-cache fingerprint invariant and `enforce_cache_limit` docstrings are wrong; README presents replay/self-supervised/RL as part of the stack without the RESERVED caveat; dead `pretrain_*`/`rl_*` config; the Platt docstring undersells the one-shot limitation.

### 5.8 Backtest & evaluation (`src/backtest/`, `src/evaluation/`, `scripts/run_backtest.py`)

**Architecture.** Bar-driven event-loop simulator: vectorized per-symbol feature precompute → per-bar mark-to-market → one batched policy forward → sequential risk evaluation → PaperExecutor fills (same-bar close ± flat bps) → Ledger → ~22 metrics. WalkForwardRunner generates chronological folds, calls a policy_builder (no retraining in practice — C6), runs the test slice, aggregates. Reporting persists configs/folds/metrics/comparison tables. `experiment_service` ships a diverged copy-paste `_CachingBacktestEngine`.

**Strengths.** Structural leakage discipline (fold tests, per-fold seeding, full state resets, embargo warnings, fingerprint validation on bank load); shared policy factory across CLI/GUI/parallel; vectorized precompute with documented equivalence; deterministic idempotent order identity matching the live scheme; a rich fill-derived metric set; good experiment hygiene in reporting; defensive zero-bar symbol handling in the GUI service; CPU-pinned banks with rationale.

**Weaknesses.** Walk-forward never retrains (C6) and the evolve-from-outcomes loop is structurally unevaluable (updater hard-disabled, memory passive, no before/after-retrain counterfactual harness); no soft-cap/strategy-blending evaluation (baselines exist only as separate experiments — and can't pass the EV gate); the fill model is optimistic and fixed (same-bar close + constant bps; no next-bar-open, volume caps, partial fills, spread-dependent slippage, borrow costs); the `--compare` harness is vestigial (scenarios 2–4 identical post-fusion-removal; hardcoded zero veto counts); ~120 lines of engine loop copy-pasted and already diverged; index-based (not timestamp) symbol alignment; naive model-selection statistics (raw mean Sharpe, no deflation, zero-metric folds averaged in); duplicated aggregation logic with divergent behavior; equity curve has no timestamps or attribution; test coverage is one smoke test.

**Improvements.**
- **[P0, small]** Check `ExecutionResult.accepted` before `register_fill` (phantom-exit cooldown lockout after rejected fills); log rejections into metrics.
- **[P0, small]** Start the bar loop at `max(warmup_bars, first_valid_index)` — with defaults 48 < 60, every backtest feeds ~12 bars of zero-filled long-window features to the policy; or validate `warmup_bars >= max(windows)` at config time.
- **[P0, large]** Make walk-forward actually out-of-sample (per-fold retrain/fine-tune honoring `reset_model_per_fold`, or hard-fail on checkpoint/test-window overlap via persisted training ranges). *(C6/C7.)*
- **[P0, medium]** Remove or repair the dead per-fold memory rebuild (unreachable guards; rebuild from train bars unconditionally when requested, or error loudly).
- **[P1, medium]** De-duplicate `_CachingBacktestEngine` into BacktestEngine (precomputed-features parameter/hook; delete the diverged copy).
- **[P1, medium]** Add a `fill_timing: same_bar_close | next_bar_open` config and implement the one-bar delay — the cheapest realism upgrade before trusting self-developed strategies.
- **[P1, small]** Temporal filtering in memory kNN or delete the cutoff plumbing (the half-implemented state is the most dangerous option).
- **[P1, small]** Config-driven, provenance-checked Platt calibrator path (warn/refuse when the fit range overlaps the backtest window).
- **[P1, medium]** Timestamp-aligned multi-symbol iteration (aligned grid or per-bar timestamp assertion).
- **[P1, medium]** Unit tests for performance metrics and reporting (golden hand-computed fixtures) — currently zero direct tests on the money math.
- **[P2, small]** Fix Sortino (divide by N, not negative-count) and round-trip detection (sign flips close trips) — matters most under `force_directional_flip`.
- **[P2, small]** Guard folds against empty equity curves (validate test_bars vs warmup; mark failed instead of aggregating hard zeros); delete the duplicate aggregator.
- **[P2, medium]** Benchmark-relative and stability statistics (per-fold buy-and-hold excess, profitable-fold pct, deflated Sharpe/t-stat before self-selection).
- **[P2, medium]** Timestamped equity curve + per-symbol attribution in BacktestReport.
- **[P3, small]** Prune vestigial fusion surface (hardcoded zero counts, dead params, unused imports, `preview_cost`); distinguish fill_count vs trade_count; fix profit_factor=0 for all-winning strategies.

**Stale docs.** README's `--use-memory` metric list and retrieval narrative; the engine's `_temporal_cutoff` comment; factory "brain+memory+fusion" docstrings; `--compare` help text and scenario labels; `--use-memory` mutating a mode the factory ignores; `reset_model_per_fold`/`reset_memory_per_fold` defined-but-dead; the walk-forward docstring's weight-reset claim; README's realism caveat omitting the same-bar fill assumption.

### 5.9 Execution (`src/execution/`)

**Architecture.** trade_engine (thin loop) / runtime_bootstrap (client build, startup checks, symbol validation, reconciliation, warmup loads, frozen checkpoint, guards, writers) / tick_processor (sync → append → mark → Phase A eligibility → Phase B batched forward → Phase C sequential risk+route) / OrderRouter (risk-approved chokepoint, duplicate-id blocking, symbol quarantine state machine, retries with by-client-id recovery) / PaperExecutor (unreachable in the runtime — backtest-only in practice). Observability writers are exception-isolated by design. Any loop exception blocks health and re-raises.

**Strengths.** Clean single-purpose decomposition; defense-in-depth order safety (three risk-approval checks, strict id schema, quarantine); genuinely rigorous startup gating (double live interlock, mock refused for live, reconciliation can hard-block); idempotency-aware submission with non-retryable classification; consistent ReasonCode taxonomy (a prerequisite for outcome-driven adaptation); exception-isolated telemetry with atomic snapshots; the Phase A/B/C design preserves guard-before-batch and sequential-portfolio invariants; frozen-brain enforcement; broad session-guard hazard coverage.

**Weaknesses.** The realized-outcome feedback loop is not wired on the path that actually runs (C3); no strategy identity on orders (pillar 2 unenforceable at this layer); single-policy runtime (no arbitration hook for multi-strategy); PaperExecutor is unreachable dead weight yet the backtest source of truth — guaranteed sim/live accounting divergence; brittle failure model (C4; no graceful shutdown, no open-order cancellation on exit); market DAY orders only (all protective exits depend on the Python loop staying alive); symbol validation silently requires shortable AND fractionable (long-only strategies on non-shortable names impossible for no stated reason); reconciliation is positions+open-orders only; 7 tests total for this risk surface.

**Improvements.**
- **[P0, small]** Re-sync account cash/equity from the broker every tick; derive daily_pnl from a day-start anchor. *(C3 — the single highest-value fix in the subsystem.)*
- **[P0, medium]** Capture real fills into Ledger + risk manager (closed-order polling or by-id fetch on open-order disappearance → FillRecord + journal fill price + `register_fill`). *(C3; restores the raw material for pillar 3.)*
- **[P0, small]** Fix `get_order_by_client_order_id` to include closed orders (use Alpaca's dedicated endpoint) so submit-timeout recovery finds instantly-filled market orders.
- **[P0, small]** Dedup polled bars before appending (timestamp check or replace-in-place; regression test with a repeated-bar stream). *(C1.)*
- **[P1, medium]** Make the loop survive transient broker errors (bounded retry/skip on sync and memory persistence; transient-vs-fatal split in trade_engine with an error budget). *(C4.)*
- **[P1, small]** Preserve position `entry_time` across per-tick hydration (fixes always-zero peak/drawdown tracking and position-age semantics).
- **[P1, medium]** Strategy identity + per-strategy soft caps on orders (strategy_id on OrderIntent/TradeProposal → journal/client-id → RiskManager per-strategy budgets). *(Pillar 2 prerequisite.)*
- **[P1, medium]** Emit an execution-outcome event stream (decision id, entry/exit, realized PnL, slippage vs reference, reason codes → JSONL for training/memory).
- **[P1, small]** Make MockBrokerClient stateful (update positions on submit) so mock rehearsals exercise the real lifecycle instead of hiding the broker-mode accounting bugs.
- **[P2, small]** Harden client_order_id uniqueness across restarts (process nonce in session id, or seed sequences from recent closed orders).
- **[P2, medium]** Graceful shutdown (cancel tracked open orders, persist memory/pending, final status snapshot on KeyboardInterrupt/fatal).
- **[P2, small]** Fix the equity-log `unrealized_pnl` formula (Σ(market−avg)×qty; current formula reports position notional).
- **[P2, small]** DST-correct sessions (shared with core P0; broker clock primary).
- **[P2, small]** Move `_recent_orders` duplicate-window marking to post-route confirmation (approval alone currently throttles legitimate retries after router blocks).
- **[P2, medium]** Test the per-tick pipeline and router state machine (process_tick phases, sync transitions, quarantine release, sequence seeding, recovery path).
- **[P3, small]** Delete or repurpose the unreachable PaperExecutor plumbing in the runtime (or add an explicit offline-simulation mode).
- **[P3, small]** Per-symbol stream error metrics (symbol + error class; stop counting `'_quote_batch'`).

**Stale docs.** README's "paper fills" framing predates broker routing; README's runtime-retrieval narrative describes deleted memory modules; `runtime.decision_interval_sec` is a dead knob (pacing is `polling_interval_sec`); paper_trade.yaml's hopfield/strong-match flags are dead; `broker.base_url` checks validate a decorative string; the reconciliation doc implies restart-safe dedupe that terminal orders don't get; `AccountSnapshot.min_trade_confidence` is deprecated but still published.

### 5.10 Risk (`src/risk/`)

**Architecture.** Deterministic gate between policies and execution. Frozen TradeProposals (action, calibrated confidence, target_fraction, spread, request_id, MFE/MAE→EV bps) → `RiskManager.evaluate` collects **all** failing reasons (kill switch, dedup, staleness, spread, daily-loss abs/pct, EV gate, sessions, cooldowns, duplicate windows incl. side-flip guard, concurrency, per-order/aggregate notional, leverage, rolling drawdown). Linear PositionSizer (fraction×equity, capped, floor-rounded); delta vs current position determines the order; action=0 flattens and bypasses the EV gate. PortfolioState tracks cash/equity/positions/pnl history. Ledger is in-memory only. Crucially, all portfolio mutation happens only through PaperExecutor (see C3).

**Strengths.** Clean propose/dispose separation with frozen dataclasses and collect-all-reasons observability; the EV gate is a principled replacement for magic thresholds (prices trades against cost physics, rejects unknown signals, well-tested); serious idempotency/restart handling (bounded request-id dedup, duplicate windows, startup seeding); defensive normalizations (daily-loss sign, at-limit spread rejection, floored leverage denominator, injectable clock); critical rejections escalate to the alerter with never-raise guards; the GUI risk service is explicitly read-only.

**Weaknesses.** No outcome-driven adaptation anywhere (pillar 3: every cap is a static YAML constant; nothing shrinks/expands risk from realized expectancy); no strategy attribution (pillar 2 impossible at this layer; metadata exists but is never read); everything is binary reject — **no downsizing path** despite README claiming "approves, resizes, or rejects" (soft caps, the user's stated preference, do not exist); PositionSizer ignores volatility/confidence/EV magnitude (MFE/MAE used only as a binary gate, never for sizing); risk-off states block risk-**reducing** orders (a tripped limit freezes the book instead of de-risking it); no margin/buying-power model (naked shorting unbounded in the simulator); no time/price-based position management despite per-trade MFE/MAE predictions; tests cover ~3 of ~14 gates.

**Improvements.**
- **[P0, medium]** Wire broker fill transitions into portfolio PnL + `register_fill`. *(C3, shared with execution.)*
- **[P0, small]** Reset `daily_pnl` at trading-day boundaries (US/Eastern day from the broker clock; same in backtests) — currently a cumulative lifetime stop where it works at all.
- **[P0, medium]** Exempt risk-reducing orders from halt reasons (kill switch / daily loss / drawdown / caps pass for pure exposure reductions; optional flatten-only mode).
- **[P1, small]** Make exposure caps post-trade (prospective aggregate/leverage including the proposed order) with boundary tests.
- **[P1, medium]** Add a resize path (shrink target_qty to the remaining budget instead of rejecting; emit a 'resized' reason) — implements the vision's soft caps and matches the README's existing claim.
- **[P1, large]** Outcome-adaptive risk budget (rolling-expectancy multiplier on target_fraction/max notional per symbol/strategy, fed by a persisted ledger) — the missing link for pillar 3 at the risk layer.
- **[P1, medium]** Strategy attribution on TradeProposal/FillRecord (prerequisite for per-strategy caps and outcome attribution).
- **[P1, small]** Enforce or delete `max_position_per_symbol`; `extra='forbid'` on RiskConfig so stale preset keys (min_trade_confidence) fail loudly.
- **[P1, small]** Include spread in the EV gate cost (half-spread per leg; document/fix the per-leg vs round-trip ambiguity — the comment and the simulator disagree by ~2x).
- **[P1, medium]** Expand `test_risk.py` to the untested gates (cooldown, duplicate windows + seeding, daily-loss/drawdown/leverage/concurrency boundaries, sizer rounding, flatten-bypass semantics).
- **[P2, small]** Fix drawdown-window sampling (one pnl_history entry per bar/tick; currently 2×/bar in backtests + per-fill, so "window 100" ≈ 50 bars and differs live).
- **[P2, small]** Charge fees into realized_pnl; persist the Ledger to disk (JSONL) so fills survive restarts and can feed the adaptive budget.
- **[P2, medium]** Volatility/EV-aware sizing (capped fractional-Kelly from the risk head's own predictions).
- **[P3, small]** Stop mutating shared config in RiskManager (instance copies; kill_switch as manager state).

**Stale docs.** README's daily-loss/cooldown claims are inert in broker mode; the "resizes" claim is false; `max_daily_loss_pct` "overrides" comment is wrong (both apply); many presets set the removed `min_trade_confidence`; `max_position_per_symbol` is a dead knob; the EV "round-trip" comment contradicts the simulator; `seed_existing_positions` docs promise both-sides marking and cooldown restore that don't happen.

### 5.11 Strategies (`src/strategies/`, `src/main.py`, policy factory)

**Architecture.** Two-layer policy stack behind runtime-checkable protocols. BrainPolicy: contract validation, one no_grad forward, single packed sync, `_build_proposal` (argmax, optional FLAT→directional flip, sign-normalized fraction, MFE/MAE→EV bps); vestigial GRU bookkeeping (model always returns None). MemoryAwarePolicy: passive wrapper — delegates, runs a **second** full forward for the latent, attaches kNN diagnostics (no gating), applies Platt, feeds the updater. The only remaining "gate" on learned trades is the RiskManager EV gate. Factory shared by CLI/GUI backtests; runtime bootstrap builds the frozen live stack.

**Strengths.** Formal protocols replacing duck typing; propose_batch preserves the sequential path's per-symbol safety invariants with documentation; disciplined transfer management and bounded self-disabling diagnostics; construction-layer leakage hygiene (updater=None in backtests, fingerprint fail-fast, untrained refusal, weight freezing); the EV gate's principled inputs; clean passthrough Platt design; L2-normalized latents with the rationale recorded; one shared factory ending CLI/GUI drift.

**Weaknesses.** All three vision pillars unserved at this layer (one learned strategy, no discovery; baselines never available at runtime and untradeable through the EV gate even in backtests; the deleted fusion layer's outcome-adaptive scaffolding is dead code; Platt is the sole outcome→decision path and is offline). Memory is pure overhead (doubled inference for diagnostics; the brain's own `fused_representation` is thrown away). Backtest≠live parity broken by construction: `force_directional_flip` defaults ON in factory-built backtests, OFF in the live runtime — promotion evidence describes a different policy than the one that trades. `record_snapshot` passes `action_annotation=None`, so memory can never condition on decisions. Flipped proposals carry the FLAT-belief confidence into the EV gate. The config surface is dominated by dead fusion/veto knobs actively misleading the operator.

**Improvements.**
- **[P0, small]** Make baselines pass the EV gate (honest heuristic EV estimates, or an `ev_gate_requires_signals` bypass for policies without risk heads) + a regression test that BuyAndHold produces >0 fills — without this every ablation/benchmark number is fiction.
- **[P0, small]** Unify `force_directional_flip` across backtest and runtime (shared config; pass it and `default_target_fraction` at every construction site; backtest-vs-runtime proposal parity test).
- **[P0, small]** Fail fast when the EV gate has no inputs (no risk head → ConfigurationError or prominent warning) — cheap insurance against silent zero-trading sessions.
- **[P1, small]** Delete or implement `set_temporal_cutoff` (and fix the engine comment). *(Shared with memory/backtest.)*
- **[P1, medium]** Reuse the brain forward for memory encoding (expose `fused_representation`; halve brain+memory inference cost).
- **[P1, large]** Wire memory outcomes into decisions as a soft, capped prior (kNN outcome stats adjust EV/confidence within a hard-capped band; adjustments logged). *(Pillar 3, minimal safe version.)*
- **[P1, small]** Record the executed action (and RiskDecision) in `record_snapshot` — the prerequisite for conditioning on "what did we do and how did it go".
- **[P1, small]** Bind Platt calibration to the checkpoint (fingerprint stored/verified; path from config, not two hardcoded literals).
- **[P2, large]** Runtime strategy ensemble with per-strategy soft caps (brain + selected baselines, outcome-EMA-weighted combiner as just another TradingPolicy; paper-only behind a flag). *(Pillar 2.)*
- **[P2, medium]** Align FLAT-flip confidence semantics (directional confidence from the long/short logit margin, or tag flipped proposals; separate Platt fits if the flip stays).
- **[P2, small]** Mask or trim padded temporal rows in propose_batch (length-bucketed sub-batches or repeat-padding; heterogeneous-length equivalence test).
- **[P2, medium]** Purge dead fusion/veto/hopfield config, `fusion_enabled` status surfacing, run_backtest fusion reporting, and dead health-tracking schemas; give BaselineMomentumPolicy reset methods.
- **[P3, small]** Wire `config.memory.top_k` and `runtime.default_target_fraction` through the factory (both currently hardcoded/ignored in backtests).

**Stale docs.** README's veto/dampen claims; default.yaml fusion/priors blocks; factory docstrings claiming fusion forcing; the engine cutoff comment; `--compare` fusion labels and zeroed counts; GUI fusion/hopfield surfaces; the "byte-identical" `_make_memory_aware_policy` docstring (it silently flips the flip flag).

### 5.12 Configs, scripts, tests (cross-cutting)

**Architecture.** Overlay YAML → single pydantic validation; default.yaml (~570 lines) + small overlays + 10 backtest presets + 25 experiment overlays + sweep configs. Scripts bootstrap sys.path, append `--config` over a non-empty default stack, mutate the config from flags. 33 test files, ~416 tests, no conftest; slow tests marker-gated.

**Strengths.** Clean overlay design with genuinely useful validators; consistent safety-first defaults; fast-mode is a model of explicit opt-in design with 80 dedicated tests; presets single-sourced with structural tests; leakage-awareness tested; scripts share the policy factory; parallel backtests keep the one-portfolio-per-process invariant; sensible pytest config; unusually good YAML rationale comments.

**Weaknesses.** The config surface describes the old architecture (~120 dead memory knobs presented as active self-tuning); no config home for pillar 2 or outcome-driven strategy weighting; drift is structurally invisible (`extra='ignore'` + stubs + zero preset-loading tests — the whole ablation suite silently degraded to no-ops); run_experiments omits 13 newer YAMLs and ~14 of 25 toggle removed features; Python-side defaults diverge widely from default.yaml (risk limits, epochs, sequence_length, AMP/Platt/early-stopping, epsilon, memory sizes) so programmatic construction tests a different system; test coverage is inverted relative to risk (training/GUI heavy; performance math, engine, memory learning loop, runtime loop, and script imports untested — which is why build_memory.py is broken right now); sweeps rank on classifier metrics, decoupled from PnL.

**Improvements.**
- **[P0, small]** Fix `scripts/build_memory.py`. *(C5; also listed under memory.)*
- **[P0, medium]** Config-hygiene test: load every `configs/**/*.yaml` through `load_config` and recursively diff keys against model fields (allowlist for intentional legacy keys) — would have flagged every dead key in this review.
- **[P0, medium]** Purge or rewrite the no-op experiment/sweep presets (confidence/fusion sweeps, ~14 dead ablations) against surviving flags; rewrite `_experiment_base.yaml` for the EV-gate era so ablations generate fills; sync `run_experiments.ALL_EXPERIMENTS`.
- **[P1, small]** Reconcile training vs memory `forward_horizons` (remove the default.yaml override or match it; assert equality in a test).
- **[P1, small]** Scripts import-smoke test (importlib over `scripts/*.py`) — cheap insurance during large refactors.
- **[P1, large]** Cover the untested money paths, in order: performance.py golden-value tests; BacktestEngine fee/slippage/flip/equity tests; MemoryAwarePolicy (Platt application, diagnostics, batch parity); AssociativeMemoryUpdater lifecycle + pending roundtrip; risk-gate breadth.
- **[P1, small]** Make `--compare` honest post-simplification (rename scenarios, delete zeroed fusion counts, drop reserved params).
- **[P1, medium]** Align programmatic defaults with default.yaml, or declare YAML canonical with a test that all entry points load default.yaml first.
- **[P1, large]** Config surface + presets for the strategy layer (per-strategy enabled/soft-cap/weight; baselines runnable in paper; persisted per-strategy PnL stats consumed by a weight-update rule). *(Pillars 1–3.)*
- **[P2, small]** Fix the `--config` append footgun (default=None + post-parse fallback in all six scripts); `--seed 0` truthiness in run_experiments; scope sweep `find_best_run` to the sweep's own runs; sys.path bootstrap in the bench scripts.
- **[P2, small]** pyproject hygiene (declare numpy; PySide6 as a gui extra; fix the "OpenAI Codex" authors attribution; verify the `seif-brain` entry point; dev extras).
- **[P2, small]** Relative-date support for backtest_quick (`now-42d`-style parsing; the hardcoded start has drifted past its own freshness rule).
- **[P2, medium]** Prune dead memory knobs from MemoryConfig/default.yaml (~105 unused fields); wire `memory.top_k` since it's the one retrieval knob users would tune.
- **[P3, medium]** Consolidate the backtest preset zoo post-purge; GUI dropdown reads descriptions from metadata.
- **[P3, small]** Log config provenance at startup (which file supplied each overridden key).

**Stale docs.** Extensive — see §8.2 for the consolidated inventory (README, cheatsheet, ~200 lines of dead default.yaml memory keys, dead runtime/training keys, no-op experiment headers, backtest_quick comments about removed knobs, training.yaml duplication, sweep_state_only comment).

---

## 6. Verified-Clean Mechanisms

These areas were adversarially checked and hold up; future reviews can deprioritize them.

1. **Lookahead/leakage in the data→feature→target→memory→backtest chain.** Outcomes use strictly forward bars with bounds checks; all rolling features are trailing windows (batch mirrors scalar, covered by a causality test); multi-timeframe/calendar features are causal; normalization stats come from train only (the defect is serving-side, not split leakage); adaptive epsilon uses train-period bars only; sample windows use past+current bars, labels strictly forward; online memory finalization is causal; backtest banks are read-only; the policy at bar t sees only bars ≤ t; walk-forward folds are chronological and non-overlapping.
2. **Train/serve parity where the FeatureContract governs.** Ordering/dtype enforced end-to-end (dataset build, checkpoint load, per-snapshot policy validation, encoder, bank load); action-decode/fraction mapping is one shared code path; batch/scalar feature parity within train/backtest; memory encoding consistent between offline build and runtime; identical left-padding conventions across all batching sites; Platt applied symmetrically in backtest and runtime.
3. **Risk-control bypass & order safety.** `risk_approved` required at the router, the validator, and the executor; all three OrderIntent construction sites set it only after approval; no other order-creating path exists; approved quantity/side flow unmutated; the live-mode interlock is sound (config validator + confirm_live + CLI flag; GUI cannot enable live); idempotent submission with recovery-by-id and quarantine; startup fails closed; frozen-brain and untrained-checkpoint guards; division-by-zero guards; GUI services read-only.
4. **Evaluation plumbing that is correct as far as it goes.** Annualization from config (matches the 5Min default); correct max-drawdown; correct fold construction and per-fold seeding/state reset; directionally correct slippage/fee arithmetic; backtest-time bank mutation disabled; post-training memory build train-split-only by default; bank/checkpoint compatibility enforced; chronological splits; class-weight arithmetic correct.
5. **ML/numerical correctness.** Consistent 0=short/1=flat/2=long everywhere; no softmax double-application; memory pathway gradient-isolated; no recurrent-state leakage (GRU fully removed); correct train/eval mode handling; no off-by-one in window/horizon boundaries; sound AMP mechanics; NaN guards on the training input path; solid checkpoint metadata round-trip.
6. **Concurrency & operational robustness.** Atomic positions-snapshot handoff; torn-line-tolerant JSONL writers/readers; atomic versioned caches that fail through to rebuild; RLock-protected GUI services with a stdout drain thread for training; idempotency-hardened router; per-symbol-locked cache merges; buffer mutation confined to the loop thread; guarded `get_clock`; deduped, corruption-tolerant pending restore; fail-closed startup; stream-level transient failures no longer kill the runtime.
7. **Self-evolution scaffolding that exists is safe.** Frozen-brain enforcement is real; encoder-generation hygiene on banks blocks retrieval-against-stale-embeddings; memory outcome labeling has no lookahead; PaperExecutor PnL math is correct where it runs; no own-trade contamination of training data; experiment checkpoint isolation works; the promotion registry is internally consistent.

---

## 7. Unverified Medium/Low Findings

**⚠ Everything in this section is UNVERIFIED** — plausible per code reading but not independently confirmed through the full evidence chain. Treat as a triage queue, not established fact. Deduplicated against §3 and each other.

### 7.1 Medium — core & data

| Location | Issue |
|---|---|
| `src/core/device.py:19-27` | `resolve_device('cuda')` without CUDA falls through to the catch-all and returns a cuda device anyway; crashes later with an opaque torch error instead of a clean startup failure. |
| `configs/default.yaml` + `src/core/config/app.py:64-68` | Explicit `memory.forward_horizons=[1,3,6,12]` disables the training→memory sync; with training.yaml the brain trains on [1,5,15] while memory labels on [1,3,6,12] — mismatched semantics in every shipped stack (build_memory.py's "auto-synced" comment is wrong). |
| `src/core/config/common.py` | `bars_per_year=19656`/`seconds_per_bar=300` hardcode 5-minute assumptions; a 1Min backtest silently mis-annualizes Sharpe by ~√5. |
| `src/data/parallel_fetch.py` | `--force-refresh` is a no-op for covered chunks (`_fetch_single_chunk` short-circuits on manifest coverage regardless of `skip_covered=False`) — corrupted caches "repair" to success with zero network calls. |
| `src/data/historical_data.py:242` | Zero-bar range fetches crash (`_build_record` ValueError) instead of returning an empty result + gap report (weekend/holiday ranges, no-IEX-data symbols). |
| `src/data/alpaca_client.py:33-52` | `_historical_lookback_delta` assumes 1-minute bars (ignores the N-minute multiplier; hourly branch undershoots) — cold-start buffers under-fill across weekends; memory builds silently get fewer bars. |
| `src/data/alpaca_client.py:283` | Historical bars hardcode `spread=0.0` while live bars carry real quote spread — systematic train/live shift on a contract column and on the `max_spread_bps` gate (the spread_too_wide rejection can only fire live). |
| `src/data/cache.py:263-267` | `HistoricalBarCache.save` writes in place with bare `torch.save` (no tmp+rename); a crash mid-write leaves a truncated file the manifest still claims is covered. |
| `src/data/cache.py:242` | Legacy-format fallback calls stdlib `logger.warning(..., path=...)` with a structlog kwarg → TypeError instead of falling back, making legacy cache files permanently unreadable. |
| `src/data/parallel_fetch.py:117-123` | Manifest record+save is serialized only per-symbol while the manifest is shared across symbols — concurrent mutation during save iteration / same-tmp-path replaces on Windows can crash or drop chunks. |

### 7.2 Medium — features & models

| Location | Issue |
|---|---|
| `src/features/feature_pipeline.py:429-441` | Bars signature keys the cache on endpoints only (count, first/last ts+close); interior data revisions collide → stale cached features served for different data. |
| `src/features/feature_pipeline.py:224-237` | `snapshot_at` returns live views into shared batch tables with shared mutable name lists; one downstream in-place op corrupts every other snapshot. Unguarded aliasing hazard on the hottest structure. |
| `src/features/price_features.py:21-23` | `return_1` has no epsilon/isfinite guard; a zero-close bar yields inf into the model on the live path (training sanitizes, live doesn't). |
| `src/models/brain_model.py:119-134` | The `full_features=None` fallback concatenates branches in branch order, not the pipeline's global sorted order — silently permuted inputs; only the width check passes. Loaded trap (no current caller omits full_features). |
| `src/strategies/brain_policy.py:379-392` (+ `state_encoder.py:91-98`) | propose_batch left-zero-pads unequal temporal tensors; the CNN convolves the padding and pools unmasked → a symbol's proposal depends on which other symbols share its batch (cold-start symbols most affected); violates the docstring's batch==sequential promise. |

### 7.3 Medium — memory

| Location | Issue |
|---|---|
| `src/memory/memory_bank.py:75-94` | `extend` is a per-item append loop, each doing a full-matrix `torch.cat` → O(N²) builds (~200 GB cumulative copy at 20k items); the docstring's claimed amortization doesn't exist. |
| `src/execution/tick_processor.py:304-319` | Under `append_when_ready`, every finalization triggers a synchronous full-bank `save_memory_bank` on the decision hot path — multi-second stalls at 20k items. |
| `src/training/memory_builder.py:61` | The post-training encoder is built without `encoder_fingerprint` → training-built banks carry None; downstream fail-fast checks require both non-None, so stale-bank detection is silently bypassed and bootstrap stamps the *current* fingerprint over unknown provenance. |
| `src/backtest/policy_factory.py:224-230` | The embargo check reads `meta['builder_embargo_end_ts']` which no builder ever writes — the warning fires on every load and enforces nothing; `embargo_bars` is unverifiable from a saved bank. |
| `src/memory/updater.py` (finalize_ready) | Pending items whose timestamps roll out of the bounded live buffer before max_horizon bars arrive are retained and rescanned forever (no age purge). |

### 7.4 Medium — training & lifecycle

| Location | Issue |
|---|---|
| `src/training/supervised.py:43-56` + `factory.py:94-106` | Fast-mode 'small' checkpoints save `config.model` (full dims) instead of the effective override config → strict `load_state_dict` size-mismatch crash; the artifact is unloadable. |
| `src/training/splits.py` + `data_module.py` | No purge/embargo between contiguous splits: the last ~max_horizon train samples have labels computed from val-window bars (and val→test); inflates val metrics that gate promotion and Platt. The guard test checks only decision-timestamp ordering. |
| `src/training/supervised.py:480-514` + `platt_calibration.py:42` | Platt labels are direction-agnostic (`return > 0` regardless of predicted side) → calibrated "confidence" ≈ P(price up), anti-calibrated for shorts (correct shorts get deflated exactly when right); and the artifact is written to the shared runs dir even for isolated experiment runs, clobbering the canonical calibrator the runtime loads. |
| `src/training/checkpoint_promotion.py:172-203` | `promote_checkpoint` is non-transactional: copy succeeds, then registry update can raise (corrupt JSON unguarded, non-atomic write, no locking) → deployed file/registry divergence. |
| `src/training/sample_cache.py:101` | `_bars_fingerprint` uses count+endpoints; interior bar corrections preserve the fingerprint → stale cached samples silently served. |
| `src/gui_services/checkpoint_service.py` vs `checkpoint_promotion.py:73-89` | Flat vs dotted metric-key schemas with no adapter — the moment the Promote button is enabled, every gate reads its 0.0 default and blocks with confusing reasons. |
| `configs/default.yaml` / `config/training.py` | Triple class-imbalance correction stacks by default (balanced weights + inverse-frequency sampler + adaptive focal) — the prior is corrected two-to-three times over; output distribution/confidence pushed far from base rates (consistent with the force-directional workaround), and val loss applies the same weights so selection optimizes the skew. |
| `src/training/losses.py:84-127` | Confidence-head multi-role conflict: trained toward the 'stability' path statistic while consumed as trade confidence and Platt input; without separate risk heads it would also regress toward MFE and volatility — three incompatible targets on one scalar. |

### 7.5 Medium — backtest & evaluation

| Location | Issue |
|---|---|
| `src/backtest/engine.py:81` | Bar loop starts at `warmup_bars` (48) but batched snapshots are valid from `first_valid_index` (60) — every backtest/fold feeds ~12 bars of partially zero-filled long-window features to the policy. |
| `src/evaluation/walk_forward.py:216-240` | The per-fold memory rebuild is unreachable dead code (requires `updater is not None` — always None in backtests — AND a `retriever` attribute that no longer exists); every fold silently reuses the full-history bank whose outcomes postdate the fold's test window. |
| `src/strategies/memory_policy.py:40-43` + `memory_bank.py:96-119` | `set_temporal_cutoff` is written and never read; kNN has no temporal filter — backtest memory diagnostics include future-outcome neighbors; the engine's leakage-guard comment is false. Diagnostics-only today; any future consumer inherits the leak. |
| `src/backtest/policy_factory.py:254` | Platt loaded from a hardcoded path with no provenance check; a calibrator fit on data overlapping the backtest window injects in-sample information into the EV gate's approvals. |
| `src/backtest/engine.py:81-84` | Symbols aligned by list index, not timestamp; halts/gaps desynchronize symbols — mark-to-market mixes prices from different wall-clock times under one timestamp; min-length truncation silently drops tails. |
| `src/backtest/engine.py` + `fills.py` | Same-bar close fills: the decision uses bar t's close and fills at that close ±2bps with no next-bar delay, no volume caps, no spread model — a systematic optimistic bias that inflates reactive strategies vs paper. |
| `scripts/run_backtest.py:73-107,255-333` | `--compare` misrepresents itself: `_make_memory_aware_policy` silently applies `force_directional_flip=True` via the factory while inline scenarios build without it (differences attributed to "memory" are partly the flip); fusion scenario labels and veto/dampen rows are hardwired zeros for a removed layer. |

### 7.6 Medium — execution & risk

| Location | Issue |
|---|---|
| `src/execution/reconciliation.py:158-181` | Per-tick hydration clears positions and re-creates them with `entry_time=utc_now()` (Alpaca positions carry no created_at) → PositionPeakTracker re-seeds every tick; GUI per-position peak/drawdown is always ~0; any age-based logic is meaningless. |
| `src/execution/order_router.py:84-100` | An order vanishing from open orders (the normal fill path between polls) is a silent removal: tracking popped, quarantine released, **no transition emitted** — fills never journaled with prices, never converted to ledger/risk events; fill detection is only implicit via position hydration. |
| `src/data/alpaca_client.py:529-553` | MockBrokerClient never updates `self.positions` on submit; hydration wipes the local book every tick → mock sessions are permanently flat, re-order on every duplicate-window expiry, and validate none of the risk controls they appear to exercise. |
| `src/execution/order_router.py:55-76,209-218` + `runtime_bootstrap.py:562` | Same-day restart client_order_id collision: session id is just the date and sequences seed only from OPEN orders; a same-minute restart after a fill regenerates the id, Alpaca rejects it, and recovery misattributes the OLD order's state to the new intent (fail-safe direction, but journal/metrics record a fill that didn't happen). |
| `src/risk/risk_manager.py:116-119` | Aggregate-notional/leverage caps are evaluated pre-trade excluding the proposed order — every cap overshootable by up to `max_position_notional`. |
| `src/risk/risk_manager.py:66-74` | The EV gate's cost model excludes spread (3bps modeled costs vs `max_spread_bps=100`) → systematically approving negative-real-EV trades in wide-spread names; per-leg vs round-trip ambiguity vs the simulator. |
| `src/risk/portfolio_state.py` | `rolling_drawdown(window=100)` counts pnl_history *entries*; mark_to_market runs 2×/bar in backtests + per fill → "window 100" ≈ 50 bars in backtests and a fill-dependent span live. |
| `src/risk/risk_manager.py:46-127` | Halt reasons (daily loss, drawdown, caps, kill switch) block risk-REDUCING exits too — a tripped limit locks the bot into holding losers until manual intervention. |
| `src/risk/portfolio_state.py` | `daily_pnl` never resets at any day boundary — where it works (in-process sim) `max_daily_loss` is a cumulative lifetime cap; in broker mode it's always 0 (C3). Either way "daily" is not implemented. |
| `src/risk/position_sizer.py` + `risk_manager.py:93` | Negative equity flips order direction: no equity≤0 clamp → negative target notional → `action × negative` inverts side; a "buy" executes as an additional short. Low reachability in broker mode. |
| `src/gui_services/runtime_service.py:298` | The runtime subprocess's stderr pipe is only read after exit; nothing drains it live — once the ~64KB OS buffer fills (warnings/tracebacks), the trading subprocess blocks forever on a stderr write while the GUI shows RUNNING. TrainingService has a drain thread; RuntimeService doesn't. |
| `src/execution/tick_processor.py:239-240` | Equity-log `unrealized_pnl = equity − cash − realized_pnl` equals gross position **notional**, not unrealized PnL — wrong by orders of magnitude whenever positions are open (GUI equity chart included). |
| `src/execution/tick_processor.py:554` + `memory_service.py:135` | Runtime status reports `fusion_enabled` from an inert config stub; no fusion layer exists — operator-facing false assurance that a memory VETO safety layer is active. |
| `src/strategies/brain_policy.py` | If the checkpoint lacks a risk head, `expected_gain_bps=0` → the EV gate rejects **every** directional proposal; the system silently produces zero trades with only per-proposal `negative_ev` breadcrumbs. No construction-time guard. |
| `src/strategies/memory_policy.py` + configs | The memory VETO/fusion soft-cap layer claimed by configs (`backtest_strict_fusion.yaml`, sweep presets, paper_trade thresholds), the runtime status field, and the recent commit message does not exist in the working tree — the only gates on learned trades are the EV gate and deterministic caps. |
| `src/core/config/risk.py:9` + `config/memory.py:51` | `max_position_per_symbol` and `require_strong_match_for_trade` are dead risk/gating knobs — defined, set in configs, read by nothing. |
| `src/risk/ledger.py` + `tick_processor.py:500-515` | Even in-process fills produce no durable outcome dataset: the Ledger is in-memory only; the journal persists submit-time snapshots never updated with async fill prices. After months of operation, no queryable (state, action, realized PnL) record exists on disk — the raw material for retraining/RL/Platt-refit is not being banked. |
| `src/training/rl_scaffold.py` | RL is ~0%: ~45 lines (batch builder + TD(0) target formula); Experience/ReplayBuffer never instantiated anywhere; no reward definition, collection, learner, or episode semantics; the frozen-brain policy would forbid online updates as designed. Building RL here is a from-scratch project. |
| `configs/backtest_sweep_*.yaml` | All sweep presets are exact no-ops: confidence sweeps set the removed `min_trade_confidence`; fusion-veto sweeps set thresholds the stub swallows; memory-updates/topology presets target mechanisms already inert in backtests. A "3-point sweep" produces three identical runs. |
| `scripts/run_train.py:284` (+5 other scripts) | `--config` uses `action='append'` with a non-empty default — user files are appended AFTER the base stack, which can never be replaced or reordered. |
| `configs/default.yaml` memory keys | `memory.top_k`, `min_similarity`, `attention_temperature`, etc. accepted but unused (~15 of ~100+ fields read anywhere); `MemoryAwarePolicy` hardcodes `knn_top_k=5`. |
| `tests/test_leakage.py` | Coverage gaps: Tests 5/8 are headers with no bodies; no split-boundary horizon-purge test; no memory temporal-cutoff test (why the dead setter went unnoticed); no fill-timing test; no train/serve normalization parity test (would have caught C2). |

### 7.7 Low (condensed; all unverified)

- **Core:** `seed_everything` sets PYTHONHASHSEED too late (child-process-only) and never seeds numpy; logging reconfiguration closes the old file handle before swapping handlers (racy `ValueError` from background threads; `pythonw` None-stdout AttributeError); session windows overlap at boundaries (13:30/20:00 claimed by two sessions); `metrics.CLASS_NAMES` duplicated from `Action`.
- **Data:** `'_quote_batch'` pseudo-symbol pollutes per-symbol stream metrics; `parallel_prep.SymbolPrepResult.error` is dead code (module itself uncalled); bench scripts (`bench_cold_start.py`, `bench_memory_build.py`) lack the sys.path bootstrap and fail on documented usage.
- **Features:** RSI zeroing off-by-one (`[:_RSI_PERIOD+1]` vs scalar guard) and momentum_{W} boundary mismatch — masked by `first_valid_index=60`, real skew for small-window configs; rolling return epsilon asymmetry (batch adds `+epsilon`, scalar doesn't; default.yaml epsilon=5e-4 makes it systematic); calendar features hardcode float32 (float64 pipeline crashes) and encode day-of-week on a 5-day cycle (weekend bars alias weekdays); feature cache dir is CWD-relative with no size cap and `weights_only=False` loads.
- **Models:** even CNN kernels crash at runtime (no construction validation); `torch.compile(dynamic=False)` + global `suppress_errors=True` → recompile storms then silently-masked errors; untrained models all share one constant encoder fingerprint (defeats stale-bank detection in dev paths); RiskHead ignores `head_hidden_dim` (sweeps misattribute).
- **Memory:** `save_pending` stores raw datetimes → `weights_only=True` always fails → misleading "legacy format" warning + unsafe pickle path every restart; `_validate_item` never checks `feature_contract_id`; `require_encoder_match` is dead; `similarity_metric` euclidean/attention silently execute as cosine; `build_memory_bank_from_bars` validates bar sufficiency for the first symbol only.
- **Training:** validation/test epochs run without `torch.no_grad()` (wasted compute, VRAM spike; combined with 50%-free-VRAM preload can OOM); `restore_best_on_finish` leaves last-epoch optimizer state and last-epoch metrics in meta describing restored-best weights; BatchNorm1d + no `drop_last` → size-1 final batch crashes an epoch (~1/384 of size combinations); `encode_batch` doesn't force `model.eval()` (dropout noise into bank latents; BN stats update under no_grad); `seed_everything(7)` hardcoded (sweep variance silently zeroed); sample-cache/checkpoint `weights_only=True` can never succeed for custom classes → warning noise + double parse + permanent unsafe fallback (any exception, incl. FileNotFoundError, triggers it in checkpoints.py); the nested val-loss-regression gate is disabled by turning off improvement-over-parent; `--list-runs` globs platt/retrain JSON into junk rows; `check_retrain_trigger` applies one 0.15 threshold to both Sharpe (noise) and win rate (catastrophic) — latent while RESERVED; `evaluate_checkpoint` re-derives normalization stats from current data (metrics non-comparable across runs — same root cause as C2); `_compute_class_weights` gives absent classes huge weight when the cap is nulled; `make_columnar_collate`'s silent `nan_to_num` masks corruption (incl. regression targets → trains toward 0) and diverges from the legacy collate; `memory_build_split='train'` extends max_horizon into val (and `'all'` builds over val+test with no warning).
- **Backtest/eval:** Sortino divides by negative-return count instead of N; round trips only close at net-zero (direction flips never close → undercounts, meaningless holding times — exactly the flip behavior the recent commit forces); `trade_count` = raw fill count; `profit_factor=0` for all-winning strategies ranks them worst; empty-fold zero metrics averaged as real results; `_CachingBacktestEngine` iterates unsorted symbols and skips `set_temporal_cutoff` (order-dependent risk caps → GUI experiments not comparable to CLI); `--use-memory`'s `runtime_update_mode` mutation is dead (updater=None).
- **Execution:** stream circuit-breaker exhaustion ends the generator → process exits rc=0 → GUI shows STOPPED, not ERROR, after a broker outage; `save_memory_bank`/`save_pending` unwrapped in Phase A (disk error kills the session); PaperExecutor accepts all sells/shorts with no margin/borrow check (backtests can hold impossible positions); `seed_existing_positions` marks one side only, never `(symbol,'_any')`, never cooldowns (docstring wrong); RiskManager mutates the shared pydantic config in place; FillRecord.realized_pnl excludes fees (`del fee`) → gross, optimistic per-trade stats; Ledger has no persistence; webhook alerts are synchronous urlopen in the rejection path (a 15-symbol critical rejection tick can stall >1 minute); `max_position_notional` caps the target while the submitted DELTA can be ~2x on a flip; `sequence_length` (48) vs `warmup_bars` (48) are independent knobs with no cross-validation (silent train/serve window divergence if either changes).
- **Strategies:** `_extract_ev_signals_batch` zeroes both gain and loss if either is missing (single-path handles them independently) — latent batched-vs-sequential EV divergence; BaselineMomentumPolicy lacks reset methods (protocol split; silent state leak if it ever grows state); `build_memory.py`'s `config.logging.json` resolves to the deprecated method (truthy → JSON logs always on).
- **Scripts/configs:** `run_experiments --seed 0` is falsy and silently replaced; `sweep_train.find_best_run` scans ALL of runs/ (can crown a checkpoint from an unrelated session); `training.sequential_memory_training`/`memory_truncation_steps` and runtime regime/priors keys don't exist on any model (silently dropped); paper_trade.yaml "enables" inert flags.

---

## 8. Test-Coverage Gaps & Config Hygiene

### 8.1 Test coverage (inverted relative to risk)

**Well covered:** training internals (fast mode: 80 tests; caches; presets: 253 lines), GUI services, config preset structure, data planner/manifest/merge, checkpoint isolation, leakage tests 1/4 (feature causality, normalization causality), risk EV gate/spread/staleness.

**Missing where the money is:**

| Area | Gap |
|---|---|
| `src/backtest/performance.py` | Zero direct tests on Sharpe/Sortino/Calmar/round-trip/expectancy math. |
| `BacktestEngine` | One smoke test (`fill_count >= 0` — masks the baselines-can't-trade defect); no fee/slippage, flip on/off, warmup-vs-first_valid_index, timestamp-alignment tests. |
| Scalar-vs-batch feature equivalence | **The load-bearing train/live parity invariant has no test.** |
| Train/serve input parity | No test that train-time and inference-time model inputs for the same bar are identical (would have caught C2). |
| `MemoryAwarePolicy` / `AssociativeMemoryUpdater` | Zero direct tests (Platt application, propose_batch parity, pending lifecycle/persistence roundtrip). |
| `tick_processor` / `trade_engine` / router state machine | Untested (phases, sync transitions, quarantine release, sequence seeding, recovery, duplicate-bar streams). |
| Risk gates | ~3 of ~14 gates tested; no cooldown/duplicate-window/daily-loss/drawdown/leverage/concurrency/sizer tests. |
| Scripts | No import-smoke test — which is exactly why `build_memory.py` is broken right now. |
| Config files | No test loads the 10 backtest presets or 25 experiment overlays; the entire ablation suite degraded to no-ops without a failure. |
| `tests/test_leakage.py` | Sections 5 and 8 are empty headers; no split-boundary purge, memory-cutoff, fill-timing, or normalization-parity tests. |
| Model tests | Shape-only; no batch-vs-single equivalence, fast-mode checkpoint round-trip, or golden-output semantics. |

### 8.2 Config hygiene (consolidated)

Root cause: pydantic `extra='ignore'` on every model + two inert stubs + zero YAML-loading tests. Consequences, by class:

1. **Dead keys silently dropped (no model field):** `training.sequential_memory_training`, `training.memory_truncation_steps`; runtime `regime_consecutive_threshold`, `regime_high_confidence_threshold`, `regime_cooldown_bars`, `regime_shift_*`, `market_priors_*_filter`; all removed `decision_fusion`/`topology_growth` sub-keys; `risk.min_trade_confidence` across ~8 preset/experiment files.
2. **Accepted-but-unread fields:** ~100 MemoryConfig fields (hopfield/ESOM/dynamic-unit/calibration/fusion/topology/saturation, `top_k`, `min_similarity`, `require_encoder_match`, `require_strong_match_for_trade`); ModelConfig ablation flags + branch/regime dims; `risk.max_position_per_symbol`; `runtime.decision_interval_sec`; `data.bars_per_trading_day` + lookback knobs; `pipeline.validation_strictness`/`max_prep_workers`; `broker.base_url` (endpoint chosen by the paper flag); `training.pretrain_*`/`rl_*`/`temporal_jitter_prob`/`contrastive_*`; `evaluation.walk_forward.reset_model_per_fold` (+ effectively `reset_memory_per_fold`).
3. **No-op preset/experiment files:** both confidence sweeps, both fusion-veto sweeps, `backtest_strict_fusion.yaml`, memory-updates/topology-off sweeps, ~14 of 25 experiment ablations, the three model-ablation experiments (train baseline-identical models).
4. **Python defaults vs default.yaml divergence:** risk limits (500/5000/15 vs 350/12000/100), training (epochs 3 vs 5, sequence_length 64 vs 48, AMP/Platt/early-stopping off vs on), features epsilon (1e-6 vs 5e-4), memory sizes — programmatic construction tests a different system.
5. **Defeated invariants:** the training→memory horizon sync (explicit default.yaml override → shipped stacks train [1,5,15] / label [1,3,6,12]); `backtest_quick.yaml`'s hardcoded start date drifted past its own freshness rule; training.yaml duplicates default values verbatim against its own header.
6. **Stale documentation:** README (dendritic/soma/Hopfield/ESOM/fusion/veto/GRU throughout; `--use-memory` metrics; "approves, resizes, or rejects"; daily-loss/cooldown claims dead in broker mode); `docs/LIVE_LOG_CHEATSHEET.md` (greps for nonexistent log events); factory/engine/script docstrings claiming fusion/cutoff behavior; GUI tabs exposing dead toggles and reading nonexistent bank summary keys; pyproject `authors = "OpenAI Codex"`.

**Remedies (tracked in §5 improvements):** `extra='forbid'` + the config-files hygiene test (P0), RiskConfig validators (P0), preset purge/rewrite (P0), defaults alignment (P1), dead-knob pruning (P1–P2), provenance logging (P2–P3).

---

## 9. Top-20 Prioritized Action List

**P0 — correctness of the live path and of every number you look at**

1. **Fix live bar ingestion (C1):** timestamp dedup before buffer append + honor the configured timeframe in latest-bar fetches (aggregate or fetch last completed bar); backfill gaps on reconnect; regression test with a repeated mock bar. Until this lands, every paper session trades on corrupted features.
2. **Fix train/serve normalization skew (C2):** persist `normalize_columns` stats in checkpoint meta and apply them in every inference path via one shared helper (or move normalization into the pipeline and retrain); add a train-vs-inference input parity test. Then **retrain** — current checkpoints were validated on inputs the deployment never sees.
3. **Sync broker truth into the portfolio (C3):** `get_account()` each tick for cash/equity; day-anchored `daily_pnl`; detect fill transitions → PortfolioState + Ledger FillRecords + `register_fill`. This resurrects the daily-loss circuit breaker, correct sizing, and outcome capture in one change.
4. **Survive transient broker errors (C4):** guard `_sync_live_state` (skip-tick + failure budget), retry/timeouts on client reads, transient-vs-fatal split in `trade_engine`.
5. **Fix `scripts/build_memory.py` (C5)** (import, cache branch, summary print, `json_logs`) and add the scripts import-smoke test so this class of breakage can't recur silently.
6. **Timezone-correct, fail-closed sessions (C9/C10):** `America/New_York` zoneinfo bounds, raise on unknown session names, `allowed_sessions` validators, winter+summer tests.
7. **Make evaluation out-of-sample (C6/C7):** persist training data ranges in checkpoint meta; `run_backtest` defaults to post-training bars (or hard-warns IN-SAMPLE); per-fold retrain or overlap refusal in walk-forward; wire or delete `reset_model_per_fold`.
8. **Config hygiene enforcement:** `extra='forbid'` across configs, RiskConfig validators, and the load-every-YAML key-diff test; purge/rewrite the no-op sweep and experiment presets.
9. **Backtest validity floor:** start the bar loop at `max(warmup_bars, first_valid_index)`; check `ExecutionResult.accepted` before `register_fill`; make baselines pass the EV gate (or bypass) so benchmarks mean something; unify `force_directional_flip` across backtest and runtime.
10. **Fast-mode checkpoint integrity:** save the effective model config in meta (+ load round-trip test); atomic checkpoint saves with a brain.pt backup.
11. **Risk-halt semantics:** exempt risk-reducing/flattening orders from halt reasons; reset `daily_pnl` at trading-day boundaries (backtest too).
12. **Feature parity guarantees:** scalar-vs-batch equivalence test (default + small-window configs); salt the feature cache with a schema version; sanitize features on the live path.
13. **Calibration correctness:** direction-aware Platt fit; isolate the artifact per run (stop experiment runs clobbering the canonical calibrator); config-driven path + checkpoint-fingerprint binding + overlap provenance check.
14. **Label/cache integrity:** add label-affecting fields to sample-cache fingerprints (version bump); purge max_horizon samples at split boundaries; use `effective_primary_horizon` everywhere.

**P1 — close the vision gaps and the trust gaps**

15. **Wire the outcome-driven retrain loop (C8, pillar 3):** persist per-decision observations (needs #3), replay buffer, `check_retrain_trigger` with per-metric thresholds, warm-start fine-tune entry point, promotion-gated (and make the registry authoritative — run_train currently clobbers brain.pt ungated); gate promotion on trading metrics, not just classification.
16. **Complete broker-side outcome capture:** closed-order polling / by-id fetch (Alpaca's dedicated endpoint) with journaled fill prices; preserve `entry_time` across hydration; persist the Ledger; emit the execution-outcome JSONL stream; make MockBrokerClient stateful.
17. **Make memory matter, safely (pillar 3):** real temporal cutoff (or delete it), adjacency-excluded retrieval with a min-similarity floor, action annotation on stored items, then a bounded (damping-only) kNN-outcome influence on EV/confidence; fix the O(N²) extend and hot-path bank saves; stamp embargo/fingerprint provenance at build time (including `memory_builder`).
18. **Strategy identity and soft caps (pillar 2):** strategy_id on TradeProposal/OrderIntent/FillRecord; per-strategy budgets in RiskManager; a resize path instead of binary rejects; `StrategiesConfig` + a runtime ensemble (brain + capped baselines) behind a paper-only flag.
19. **Test the money paths:** golden-value tests for `performance.py`; engine fee/slippage/alignment/flip tests; MemoryAwarePolicy + updater lifecycle tests; risk-gate breadth; tick_processor/router state-machine tests; de-duplicate `_CachingBacktestEngine` into the engine.
20. **Backtest realism + honest reporting:** next-bar-open fill option; spread in the EV cost model; post-trade cap evaluation; timestamp-aligned symbols; honest `--compare` scenarios; fix Sortino/round-trip metrics; benchmark-relative fold statistics before letting the system self-select configurations.

---

*Sections 5 and 7 retain every improvement item and unverified finding from the underlying subsystem analyses; §3 deduplicates fourteen confirmed reports into ten root-cause defects with all evidence preserved.*
