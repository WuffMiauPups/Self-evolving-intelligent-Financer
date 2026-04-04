# SEIF Trading Brain Foundation

SEIF is a research-first, brain-inspired trading framework for Alpaca that keeps PyTorch at the center of feature processing, model inference, and training. It is designed to be safe by default, paper trading first, and modular enough to support future supervised, self-supervised, and reinforcement-learning experimentation without letting learned components bypass deterministic risk controls.

This repository is intentionally not a claim of profitability. It is a production-style foundation for research, observability, and controlled iteration.

## Highlights

- PyTorch-first numerics with CUDA auto-detection and CPU fallback
- Brain-inspired `BrainTradingModel` with dendritic branches, regime conditioning, soma-style gating, lightweight explicit memory, and an optional temporal CNN encoder
- Associative market memory with latent-state encoding, Hopfield-style retrieval, ESOM-like topology-aware units, and forward path statistics
- Simpler `BaselineMLP` plus a deterministic `BaselineMomentumPolicy` for comparison
- Centralized Alpaca adapter layer using the official [`alpaca-py`](https://alpaca.markets/sdks/python/) SDK
- Real historical bar fetch + local cache pipeline for multi-symbol training and memory prebuild
- Deterministic risk manager separating AI proposal generation from order approval
- Explicit feature-contract validation so training, backtest, and paper runtime share the same feature ordering, branch assignment, regime inputs, and dtype contract
- Checkpoint metadata that stores model build settings plus the feature contract required to reload trained weights safely
- Startup broker/account validation, symbol eligibility checks, restart reconciliation, and session/data guards before trading begins
- Strict live-mode interlocks: paper remains default, and live requires both config confirmation and CLI confirmation
- Deterministic `client_order_id` generation for dedupe and restart-safe open-order synchronization
- File-backed memory bank persistence for historical analog retrieval and conservative no-trade handling on unknown states
- Paper execution path, minimal backtest engine, and supervised/self-supervised/RL scaffolds
- Structured logging for model decisions, risk outcomes, execution events, and device/runtime state

## Repository Layout

```text
configs/        Runtime, broker, training, and symbol configuration
scripts/        Convenient entrypoints for training, backtesting, paper mode, and smoke tests
src/core/       Config, logging, devices, errors, time helpers, reproducibility
src/data/       Alpaca adapter, historical loading, polling stream manager, buffers, datasets
src/features/   Torch-native feature engineering and multi-timescale aggregation
src/memory/     Associative memory schemas, storage, clustering, retrieval, and updater scaffolds
src/models/     Brain-inspired and baseline PyTorch models
src/training/   Training loop, losses, replay buffer, self-supervised and RL scaffolds
src/risk/       Portfolio state, deterministic limits, ledger, and sizing
src/execution/  Order validation, routing, paper fills, and paper runtime loop
src/backtest/   Backtest engine, simulator, fills, and performance metrics
src/strategies/ Policy adapters used by backtest and runtime
src/monitoring/ Events, counters, and runtime health state
tests/          Unit and integration-style smoke coverage
```

## Architecture Overview

The runtime path is:

1. The runtime validates broker connectivity, account status, live-vs-paper mode consistency, and data-feed usability.
2. Symbols are checked for asset metadata, tradability, activity, asset-class compatibility, and data availability.
3. Startup reconciliation hydrates broker positions/open orders and sets `SAFE_TO_TRADE` before the loop begins.
4. Market data arrives through the Alpaca adapter or mock client.
5. Bars are stored in symbol buffers and converted into Torch tensors by the feature pipeline.
6. Every snapshot is validated against an explicit feature contract before inference when a checkpoint-backed policy is active.
7. Session and timestamp guards run before proposal generation and again before order submission.
8. The feature pipeline also produces a temporal tensor so CNN-style modules can learn local bar-pattern structure without breaking the feature contract.
9. If associative memory is enabled, the current latent state queries similar historical states, mature units, and ESOM-neighbor structure through Hopfield-style retrieval.
10. The policy model produces a proposal: direction, confidence, and target position fraction.
11. Associative memory can veto weak/unknown states or reduce size, but it cannot create an execution path on its own.
11. The deterministic risk manager approves, resizes, or rejects the proposal.
12. The order router only accepts intents that carry explicit risk approval and refuses duplicate or quarantined orders.
13. The ledger, portfolio state, and structured logs capture the outcome.

The learning path is:

1. Real Alpaca bars can be fetched into a local cache keyed by symbol and timeframe, or synthetic bars can still be used for deterministic smoke and fixture workflows.
2. Multi-symbol datasets are built with time-based train/val/test splits so there is no leakage across splits.
3. Every sample conforms to one shared feature contract and includes both snapshot features and temporal CNN input.
4. Future path targets are computed deterministically from forward bars, not from runtime risk-manager decisions.
5. The trainer runs a PyTorch training loop with optional AMP on CUDA.
6. Checkpoints are stored under `checkpoints/` together with the model kind, model config, feature config, feature contract, data config, training target config, and symbol universe required for reuse.

## Brain-Inspired Model Design

`BrainTradingModel` is a practical first-pass modular network, not a pseudo-neuroscience toy:

- Multi-timescale perception comes from feature windows and multi-timeframe summaries.
- A temporal CNN encoder can consume feature-contract-compatible bar sequences to learn local price, volatility, range, momentum, and volume motifs.
- Dendritic branches specialize on configurable feature subsets such as trend, volatility, volume, and regime.
- A regime encoder produces a market-state embedding through differentiable soft assignments over learned prototypes.
- A soma aggregator learns gate weights across branch activations and suppresses low-alignment states through an activation score.
- A lightweight GRU-based memory carries latent state across sequential decisions.
- Policy and value heads expose action logits, confidence, target position fraction, and value estimates.

## Hybrid Memory Brain

The current research stack is now a hybrid memory-learning architecture with distinct responsibilities:

- CNN / temporal conv blocks learn short- and mid-range local market motifs from sequential feature windows.
- The unified latent state encoder fuses temporal embeddings with the existing branch/regime pathway into one canonical latent state `z`.
- Hopfield-style associative retrieval performs attention-like lookup over stored mature memory units and returns a soft context vector, top-k analogs, similarity, entropy, and conflict information.
- An ESOM-like map builds topology-aware neighbor structure over dynamic memory units so nearby units can be inspected as a local memory neighborhood rather than a flat table.
- Outcome and path memory attach forward returns, excursions, realized volatility, drawdown, and path labels to both items and units.
- Deterministic risk gating remains unchanged and fully authoritative.

This is deliberately not a single monolithic network. It is a modular hybrid system where learned temporal structure, latent encoding, associative recall, and memory organization each have a clear job.

## Feature Contract

The repository now treats the feature specification as a first-class contract. A `FeatureContract` captures:

- feature ordering for the full feature tensor
- branch-to-feature assignments for dendritic inputs
- regime feature ordering
- tensor dtype for model-facing features

This contract is:

- validated while building the training dataset
- stored in checkpoints
- checked again in backtests and paper runtime before model inference

This keeps training, backtest, and paper execution on the same feature boundary instead of relying on implicit assumptions.

## Real Historical Training

The repository now supports a real-data training foundation built around Alpaca historical bars with local caching.

### Historical Cache

Historical bars can be fetched per symbol/timeframe and stored under `data.cache_dir`, which defaults to `artifacts/historical`.

Each cache record stores:

- symbol
- timeframe
- cached start/end coverage
- source
- timezone normalization
- row count
- cache generation/update timestamps

When you request a range:

- the loader reuses the cache when the requested window is already covered
- missing leading or trailing coverage is fetched incrementally
- merged bars are deduplicated by timestamp and saved back to disk

This keeps repeated training and memory-build workflows reproducible without re-fetching every bar set on each run.

### Dataset and Target Flow

Training datasets are built from real or cached bars using time-based splits per symbol:

- `train`
- `val`
- `test`

Each sample contains:

- the standard feature vector
- branch inputs
- regime input
- temporal CNN tensor
- symbol/timestamp metadata
- deterministic future path targets

Targets are derived from future bars only. They include:

- short / flat / long direction label
- future return over the primary horizon
- max favorable excursion
- max adverse excursion
- realized volatility
- drawdown
- path labels such as `breakout_up`, `breakout_down`, `mean_reversion_up`, `mean_reversion_down`, and `chop`

The training pipeline does **not** train on:

- stale-data runtime rejections
- session guard rejections
- spread guard decisions
- broker/account blocks
- deterministic risk-manager approvals or denials

This separation is intentional. The model learns market structure and future path tendencies, while runtime risk controls remain a separate authoritative layer.

### Multi-Task Training

The supervised trainer now uses a shared encoder path with:

- direction classification over `short / flat / long`
- auxiliary regression targets from real future outcomes and risk summaries
- a lightweight latent-consistency objective to make encoded states less arbitrary for later memory retrieval

The CNN/temporal path and unified latent state are therefore trained on real forward market behavior rather than synthetic-only placeholders.

### Class Balance and Diagnostics

Training now logs:

- class counts by split
- flat-rate by split
- flat-rate by symbol
- class ratios by split
- primary-return quantiles and near-boundary rates around `training.label_epsilon`
- regression target summaries
- warnings for minority-class collapse or pathological flat dominance

This keeps `flat` as a valid class without letting it silently dominate the whole dataset.

Overall accuracy is not enough. The training/evaluation flow now also reports per-class precision, recall, F1, macro F1, weighted F1, balanced accuracy, class-ratio mismatch, collapse warnings, confusion matrices, and logit/probability summaries so LONG/SHORT/FLAT behavior can be inspected directly.

## Memory and Sequence State

`BrainPolicy` manages model state explicitly on a per-symbol basis:

- state is tracked independently for each symbol
- state resets on runtime start and backtest episode start
- state resets if timestamps move backward or stop being monotonic
- state resets if the gap between observations for a symbol exceeds `runtime.state_reset_gap_sec`

This keeps latent memory aligned with actual sequence boundaries instead of letting hidden state leak across unrelated episodes.

## Associative Market Memory

The associative memory subsystem is now a hybrid research scaffold:

- A canonical latent state is produced from the main model path plus the optional temporal CNN encoder.
- Hopfield-style retrieval performs soft associative lookup over stored memory units and historical states.
- Dynamic self-expanding memory units can grow, mature, merge, split, calibrate retrieval quality, and prune stale or low-utility structure over time.
- An ESOM-like map organizes those units into a topology-aware neighborhood graph.

It is not a second execution engine and it does not bypass risk management.

The memory subsystem now does seven things:

1. encodes a feature snapshot into a latent market-state vector using the same `FeatureContract` as the brain model
2. stores historical latent states with timestamps, regime weights, outcome summaries, and risk summaries
3. learns local temporal motifs with a CNN encoder and fuses them into the canonical latent state used by memory
4. clusters similar states into simple prototype centroids and dynamic units with aggregated outcome-path statistics
5. retrieves top-k historical analogs and prototype matches for a new state, with explicit weak-match and unknown-state handling
6. performs Hopfield-style soft retrieval over stored units to produce a memory context vector plus confidence/conflict diagnostics
7. builds an ESOM-like neighborhood graph over units so local topology can be inspected and used for memory health analysis
8. evaluates latent-space quality offline so retrieval quality can be audited instead of assumed
9. calibrates raw similarity against historical outcome consistency and conflict
10. organizes dynamic memory units that can stay candidate, mature with evidence, merge when redundant, split when overly broad, and prune when stale

### What Memory Stores

Each memory item includes:

- symbol and timestamp
- latent vector
- feature-contract fingerprint
- regime soft weights when available
- branch confidence/norm diagnostics
- forward returns over configured horizons
- max favorable excursion, max adverse excursion, realized volatility, and drawdown
- a simple path label such as `breakout_up`, `breakout_down`, `mean_reversion_up`, `mean_reversion_down`, or `chop`

### What Retrieval Returns

For a live or backtest state, retrieval returns:

- top historical matches with similarity scores
- top prototype/cluster matches
- Hopfield-style soft-attention context over mature units
- match strength and average similarity
- predicted path-label distribution
- aggregated risk statistics from similar states
- explicit `weak_match` / `no_trade_suggested` flags
- calibrated match strength, retrieval confidence, and conflict score
- ESOM neighbor identifiers for the most relevant unit when available
- a conservative directional bias and size multiplier

Unknown or weak states are not coerced into confident analogs. By default, weak matches suggest no-trade.

### Dynamic Memory Units

Dynamic memory units are practical associative storage nodes, not literal online-mutating model neurons.

Each unit tracks:

- a prototype latent vector
- candidate vs mature status
- support count and assignment similarity distribution
- path and risk statistics
- regime, symbol, and session composition
- utility and maturity scores
- creation and update timestamps

The default lifecycle is conservative:

- genuinely novel states create candidate units
- repeated similar evidence can promote candidates into mature units
- strong matches update mature units instead of always growing memory
- mature units can merge when they are both similar and outcome-compatible
- broad, high-variance units can split when evidence suggests mixed structure
- stale or low-utility units can be pruned or archived

This keeps the memory bank open to novelty without forcing endless unit growth from noise.

### ESOM-Like Memory Map

The ESOM component is intentionally practical rather than a full classical fixed-grid SOM.

- Each mature or candidate unit becomes a node in a topology-aware map.
- Neighbor links are built from latent similarity rather than a hard-coded static lattice.
- Local density and neighbor similarity provide a lightweight notion of memory topology.
- This gives research tooling a way to inspect whether memory is growing as coherent neighborhoods or fragmenting into noise.

For vNext research, this is the bridge between prototype memory and a richer self-organizing memory surface.

### Latent-Space Versioning

Memory items are now tagged with an **encoder fingerprint** derived from the checkpoint metadata that produced them. This prevents silent mixing of latent vectors from incompatible model versions:

- The `MemoryBank` stores an `encoder_fingerprint` identifying which model checkpoint built it.
- Each `MemoryItem` carries its own `encoder_fingerprint`.
- On append, the bank checks fingerprint compatibility and logs a warning (or raises, if `require_encoder_match` is set).
- Checkpoint metadata now includes an `encoder_fingerprint` field (version 1.4+).

This is a foundation for safe memory reuse across model iterations.

### SatSOM-Inspired Saturation and Adaptive Plasticity

Inspired by SatSOM (saturation-aware self-organizing maps), memory units now carry explicit saturation and plasticity scores:

- **Saturation** (0→1) measures how well-established a unit is based on support count, outcome consistency, and cluster tightness.
- **Plasticity** (0→1) controls how much a unit's prototype shifts when new evidence arrives. Mature, saturated units resist being overwritten by new data.
- **Maturity-based attenuation**: prototype updates are scaled by a plasticity factor that decays with cumulative updates, configurable via `maturity_update_attenuation`.

This prevents catastrophic forgetting of well-established memory patterns while keeping newer units responsive to new data. All SatSOM features default to disabled and can be turned on via config.

### Adaptive GNG-Inspired Topology Foundations

Inspired by Growing Neural Gas (specifically AiGAS-dEVL-RC), the ESOM map now maintains explicit **graph edges** with:

- **Edge age**: tracks how recently each connection was activated (GNG competitive Hebbian learning concept)
- **Edge lifecycle**: edges can be created, refreshed on activation, aged, and pruned when stale
- **Local error accumulation**: each unit tracks quantization error, enabling future graph-aware insertion decisions
- **Insertion candidate scoring**: combines local error with saturation to identify where new units would be most useful

These are foundational abstractions. The full adaptive GNG insertion/removal logic is a future research layer.

### Outcome-Aware Unit Assignment

When `require_outcome_alignment_for_strong_update` is enabled, similar-looking states with contradictory outcomes are no longer merged into the same unit:

- If a unit has a clear directional bias (e.g., predominantly positive outcomes) and a new item contradicts it, the item creates a new competing candidate unit instead.
- This preserves the predictive coherence of established units and allows the system to distinguish market regimes that look similar but resolve differently.

### Conflict History Tracking

When `conflict_tracking_enabled` is enabled, each unit maintains a rolling conflict history:

- Every assignment records how much the new item's outcome direction conflicts with the unit's established direction.
- Units with persistently high conflict rates (>60% over recent history) can be quarantined.
- Quarantined units are excluded from strong match retrieval but preserved for analysis and potential future splitting.

### Retrieval Calibration and Unknown States

Raw similarity is treated as only a first signal.

The memory bank now builds retrieval-calibration buckets offline from historical nearest-neighbor behavior. These buckets estimate:

- how often similar states agreed on direction
- how dispersed later outcomes were
- how conflict-heavy the analog set was
- how much trust a given similarity level deserves

The retriever uses that calibration to expose:

- `calibrated_match_strength`
- `retrieval_confidence`
- `conflict_score`

If a match is weak, sparse, conflicting, or poorly calibrated, memory stays explicit about uncertainty and can suggest no-trade support instead of forcing false confidence.

### Important Safety Boundary

Associative memory is only a decision-support layer.

- It can reduce or veto a base model proposal.
- It cannot invent a trade when the base policy is flat.
- It cannot bypass deterministic risk approval.
- It cannot route orders directly.

This keeps retrieval transparent and safe even when similarity is weak or historical support is sparse.

## Operational Safety Flow

v1.2 hardens runtime operation around one central runtime gate: `SAFE_TO_TRADE`.

- If `SAFE_TO_TRADE=false`, the runtime does not generate proposals, build `OrderIntent`s, or call the router.
- Startup validation runs before the loop and checks broker connectivity, credentials or explicit mock mode, account usability, buying power visibility, endpoint mode consistency, market clock access, and a data-feed probe.
- Live mode is impossible to start accidentally. It requires:
  - `broker.paper=false`
  - `broker.live_enabled=true`
  - `broker.confirm_live=true`
  - CLI confirmation via `--confirm-live`
- The runtime never silently falls back from live to paper or paper to live.

### Symbol Eligibility and Quarantine

Each symbol is validated once at startup and cached. Checks include:

- asset lookup success
- `active` status
- `tradable`
- supported asset class (`us_equity` / stock for v1.2)
- `shortable`
- `fractionable`
- market-data availability

Ineligible symbols are quarantined with explicit metadata:

- `reason_code`: why the symbol was blocked
- `quarantined_at`: when it was blocked
- `release_mode`: how it can be released
- `release_condition`: human-readable release rule

Current release modes are:

- `until_restart`: symbol remains blocked for the full runtime session
- `until_order_terminal`: symbol is released only after the tracked broker order becomes terminal or disappears during sync

By default, ineligible symbols are skipped and the runtime continues with the remaining eligible set. Startup aborts if no eligible symbols remain.

### Session and Data Guards

The runtime uses both local time-window logic and broker clock information.

- Local configured sessions (`runtime.allowed_sessions`) are always enforced.
- Broker clock is used as an additional guard when `runtime.require_market_open=true`.
- Stale bars, backwards timestamps, and large data gaps are blocked with machine-readable reason codes.
- A large gap can force state reset and suppress trading for that tick when `runtime.block_on_data_gap=true`.

### Reconciliation Severity

Startup reconciliation distinguishes mismatch severity levels:

- `warn`: informative only, runtime may proceed
- `block`: trading remains disabled until restart or operator action
- `fatal`: startup aborts immediately

For v1.2:

- broker positions hydrated into local portfolio state are `warn`
- broker open orders present at startup are `block` by default
- broker sync failures are `fatal`

### Deterministic Client Order IDs

`client_order_id` is not random. It follows a reproducible schema:

```text
seif-{session_id}-{SYMBOL}-{b|s}-{YYYYMMDDHHMM}-{sequence}
```

This gives the router stable dedupe behavior and lets startup reconciliation seed local sequence state from broker open orders.

## Setup

### 1. Create an environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### 2. Configure environment variables

Copy `.env.example` into `.env` and set:

```env
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_PAPER=true
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_FEED=iex
```

If credentials are missing, the runtime can fall back to synthetic mock data when `broker.mock_data_on_missing_credentials` is enabled.

## Configuration

Configuration is centralized in YAML files:

- `configs/default.yaml` defines safe defaults, features, model dimensions, checkpoint path, state-reset behavior, risk limits, and backtest settings.
- `configs/paper_trade.yaml` keeps the runtime in paper mode with shorter loop settings and slightly more permissive demo thresholds so Alpaca paper-routing can be validated more easily.
- `configs/training.yaml` adjusts training-specific settings.
- `configs/symbols.yaml` expands the symbol universe without touching source code.

Load multiple files together; later files override earlier ones.

Useful config knobs for inference:

- `model.checkpoint_path`: optional checkpoint to load for backtests or paper runtime
- `runtime.state_reset_gap_sec`: per-symbol sequence gap threshold that forces memory reset
- `model.use_cnn_encoder`: enable temporal CNN pattern extraction inside the brain model
- `model.cnn_channels`: per-block channel sizes for the temporal encoder
- `model.cnn_kernel_sizes`: kernel sizes rotated across temporal blocks
- `model.temporal_blocks`: number of temporal residual blocks

Useful runtime safety knobs:

- `broker.live_enabled`: explicit live-routing enablement
- `broker.confirm_live`: config-level live confirmation
- `broker.require_account_match`: abort if configured mode and endpoint mode disagree
- `broker.base_url`: explicit broker endpoint used for mode checks
- `runtime.require_startup_reconciliation`: force broker state sync before trading
- `runtime.block_on_reconciliation_mismatch`: block when reconciliation finds blocking mismatches
- `runtime.allowed_sessions`: local allowed session windows
- `runtime.allow_prepost`: allow premarket/after-hours windows
- `runtime.max_data_staleness_sec`: stale-bar threshold
- `runtime.block_on_backwards_time`: reject backwards timestamps
- `runtime.block_on_data_gap`: reject large data gaps until state reset is handled
- `runtime.block_on_symbol_ineligible`: abort instead of skipping ineligible symbols
- `runtime.require_market_open`: require broker clock to indicate open market for regular-session trading
- `execution.max_submit_retries`: bounded live submit retries
- `execution.retry_backoff_sec`: retry backoff between live submit attempts

Useful logging knobs:

- `logging.file_enabled`: write runtime logs to disk in addition to stdout
- `logging.file_path`: file that receives the full runtime trace, default `logs/seif_runtime.log`

Useful memory knobs:

- `memory.enabled`: enable associative memory support
- `memory.build_from_real_checkpoint`: require checkpoint-backed encoder metadata for real-data memory builds
- `memory.latent_dim`: optional latent-dimension assertion for the state encoder
- `memory.top_k`: number of retrieved historical matches
- `memory.similarity_metric`: `cosine`, `euclidean`, or `attention`
- `memory.min_similarity`: minimum normalized similarity for a strong match
- `memory.allow_unknown_state`: explicit unknown-state / no-trade behavior
- `memory.cluster_method`: currently `incremental_prototypes`
- `memory.max_items`: maximum in-memory bank size
- `memory.rebuild_on_startup`: rebuild from recent history instead of loading persisted memory
- `memory.persist_path`: local file-backed bank path
- `memory.forward_horizons`: forward return / path-summary horizons
- `memory.path_label_rules`: deterministic path-label thresholds
- `memory.min_cluster_size`: minimum prototype size for confident cluster use
- `memory.runtime_update_mode`: `disabled`, `pending_only`, or `append_when_ready`
- `memory.require_strong_match_for_trade`: legacy compatibility flag kept for older configs/checkpoints
- `memory.use_hopfield_retrieval`: enable attention-like associative retrieval over memory units
- `memory.use_esom_map`: build an ESOM-like topology-aware map over dynamic units
- `memory.match_calibration_enabled`: calibrate raw similarity into a stronger confidence signal
- `memory.min_calibrated_match`: minimum calibrated match strength for strong support
- `memory.require_strong_match_for_memory_support`: veto memory reinforcement when retrieval is weak or unknown
- `memory.build_history_bars`: recent bar count used when building memory offline or at runtime startup
- `memory.min_effective_matches`: minimum strong analog count before retrieval is trusted
- `memory.dynamic_growth_enabled`: enable adaptive candidate/mature unit organization
- `memory.novelty_mode`: novelty detection strategy, currently conservative and config-driven
- `memory.novelty_quantile`: quantile used to derive adaptive novelty thresholds
- `memory.candidate_promotion_mode`: promotion rule for candidate units
- `memory.min_candidate_support`: minimum confirming evidence before a candidate can mature
- `memory.max_candidate_age`: stale-candidate age before prune/archive becomes eligible
- `memory.merge_enabled`: enable mature-unit merge maintenance
- `memory.merge_similarity_mode`: merge threshold mode
- `memory.merge_outcome_consistency_threshold`: require compatible outcome/risk structure before merge
- `memory.split_enabled`: enable split maintenance for broad heterogeneous units
- `memory.split_variance_trigger`: variance trigger used by split heuristics
- `memory.prune_enabled`: enable stale/low-utility unit pruning
- `memory.utility_decay`: decay factor used in utility scoring
- `memory.min_utility`: utility floor before prune/archive
- `memory.max_units`: maximum retained dynamic units before overflow pruning
- `memory.archive_pruned`: keep archived metadata for explainability
- `memory.calibration_enabled`: build similarity-to-confidence calibration tables
- `memory.match_conflict_penalty`: penalize conflicting analog sets during retrieval
- `memory.require_calibrated_strong_match_for_trade_support`: require calibrated support before memory can reinforce a trade
- `memory.topology_neighbor_k`: neighbor count used by the ESOM-like topology graph
- `memory.health_report_enabled`: emit richer memory-health summaries
- `memory.symbol_specific_profiles`: reserved switch for symbol-aware calibration profiles
- `memory.regime_specific_profiles`: reserved switch for regime-aware calibration profiles

Useful real-data knobs:

- `data.symbols`: symbol universe for cache fetch and training
- `data.timeframe`: historical bar timeframe
- `data.start_date`: cache/training start date
- `data.end_date`: cache/training end date, defaults to now when omitted
- `data.cache_dir`: local historical cache directory
- `data.force_refresh`: ignore existing cache coverage and refetch requested ranges
- `data.max_missing_gap_handling`: `ignore`, `warn`, or `error` when large gaps are detected in the requested range
- `data.timezone_normalization`: timezone label stored with cache metadata
- `training.use_real_historical_data`: build the training dataset from cached/fetched Alpaca bars
- `training.train_split`, `training.val_split`, `training.test_split`: time-based split ratios
- `training.forward_horizons`: horizons used for future return/path targets
- `training.class_weight_mode`: currently `balanced` or `none`
- `training.class_weight_cap`: cap extreme class weights
- `training.use_weighted_sampler`: optional weighted train sampler
- `training.weighted_sampler_mode`: `inverse_frequency` or `effective_number`
- `training.minority_class_floor_warning`: emit warnings when a class becomes too rare
- `training.multitask_enabled`: enable auxiliary regression and latent-consistency losses
- `training.contrastive_enabled`: explicitly enable the lightweight latent-consistency objective
- `training.classification_loss`: `cross_entropy` or `focal`
- `training.focal_gamma`: focal-loss gamma when focal loss is enabled
- `training.label_smoothing`: conservative label smoothing for classification
- `training.loss_weights`: weighting for classification / regression / latent losses
- `training.report_logit_stats`: include logit/probability diagnostics in evaluation
- `training.report_prediction_distribution`: include predicted-vs-target class ratios
- `training.report_per_class_metrics`: include per-class precision/recall/F1
- `training.collapse_warning_thresholds.long_prediction_floor`: warn when predicted LONG ratio collapses
- `training.collapse_warning_thresholds.minority_recall_floor`: warn when any class recall becomes too small
- `training.inspect_target_boundaries`: report how many samples live near the flat boundary
- `training.regression_targets`: real forward/risk targets used by the multitask path
- `training.save_memory_after_training`: build and persist a memory bank immediately after training
- `training.memory_build_split`: choose `train`, `val`, `test`, or `all` for post-training memory build
- `training.report_target_stats`: log class/target distributions during training

## Running

### Smoke test

```bash
python scripts/smoke_test.py
```

### Supervised training

```bash
python scripts/run_train.py
python scripts/run_train.py --baseline
python scripts/run_train.py --real-data
python scripts/run_train.py --real-data --inspect-only
python scripts/run_train.py --real-data --save-memory
python scripts/run_train.py --real-data --class-weight-mode effective_number --use-weighted-sampler --classification-loss focal
```

Training saves checkpoints with the full model reconstruction metadata under `checkpoints/`.

If you request `--real-data`, training uses the local historical cache and fetches missing coverage from Alpaca when credentials are available. If the requested range is already cached, the same path can be used offline without silently switching back to synthetic data.

The post-training logs and checkpoint metadata now also include:

- prediction counts by class
- target counts by class
- predicted-vs-target class ratios
- confusion matrix
- per-class precision / recall / F1
- macro F1, weighted F1, balanced accuracy, and MCC
- per-symbol accuracy
- long/short-only accuracy
- logit/probability summaries and collapse warnings

### Fetch real historical data

```bash
python scripts/fetch_historical.py
python scripts/fetch_historical.py --force-refresh
python scripts/fetch_historical.py --inspect-cache
```

This pulls the configured `data.symbols` / `data.timeframe` / `data.start_date` / `data.end_date` range into the local cache.

`--inspect-cache` only reports local cache coverage and does not fetch missing ranges.

### Backtest

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --checkpoint checkpoints/brain.pt
python scripts/run_backtest.py --use-memory
python scripts/run_backtest.py --checkpoint checkpoints/brain.pt --use-memory
python scripts/run_backtest.py --real-data --checkpoint checkpoints/brain.pt --use-memory
```

The default backtest compares the deterministic baseline policy with a neural policy. If you provide `--checkpoint` or set `model.checkpoint_path`, the checkpoint-backed model is loaded and its feature contract is validated against the current runtime config before evaluation.

With `--use-memory`, the backtest wraps the neural policy in a memory-aware adapter. Retrieval metrics such as coverage, average similarity, average confidence, weak-match rate, veto rate, dampened rate, and bank/unit growth are added to the report.

### Build Associative Memory

```bash
python scripts/build_memory.py
python scripts/build_memory.py --from-checkpoint checkpoints/brain.pt
python scripts/build_memory.py --evaluate
python scripts/build_memory.py --from-checkpoint checkpoints/brain.pt --real-data
```

This builds a local file-backed memory bank from historical or synthetic bars using the current feature pipeline and latent encoder, then saves it to `memory.persist_path`.

With `--real-data`, the builder reads cached/fetched historical bars and uses the trained checkpoint-compatible encoder path to create real memory items for later backtests and runtime retrieval.

With `--evaluate`, the script also prints offline encoder and retrieval-quality diagnostics such as nearest-neighbor consistency, outcome dispersion, retrieval coverage, weak-match rate, conflict rate, calibrated confidence, and dynamic-unit compactness/separation.

The same evaluation now also reports hybrid-memory health such as mature/candidate ratio and ESOM topology density.

### Paper trading loop

```bash
python scripts/run_paper.py
python scripts/run_paper.py --checkpoint checkpoints/brain.pt
python scripts/run_paper.py --use-memory
python scripts/run_paper.py --checkpoint checkpoints/brain.pt --use-memory
```

This is paper-first and safe by default. If a checkpoint is configured, the runtime reloads the trained weights and verifies that the current feature config matches the checkpoint metadata before inference. Live routing remains disabled unless explicitly enabled in config.

If memory is enabled, the runtime loads a persisted memory bank when available, or builds one from recent history when necessary. Retrieval results are logged, and weak or unknown states can suppress trades before the deterministic risk manager evaluates the proposal.

When the hybrid path is enabled, runtime retrieval also exposes Hopfield-style context quality, ESOM neighbor metadata, calibration scope, and whether memory was only informative, dampened a proposal, or vetoed support altogether, while keeping execution safety unchanged.

By default, the same runtime events are also appended to `logs/seif_runtime.log`, including startup checks, session guards, model decisions, risk decisions, order submit attempts, order transitions, and runtime completion.

### Live-mode confirmation

Live mode requires config changes and an explicit CLI confirmation:

```bash
python scripts/run_paper.py --config configs/default.yaml --confirm-live
```

If any live safety condition is missing or inconsistent, startup aborts before the runtime loop begins.

## Safety Notes

- Paper trading is the default mode.
- The model never places orders directly.
- `SAFE_TO_TRADE` is a hard runtime gate. When false, the loop stops generating proposals and cannot route orders.
- No order is routed unless a deterministic risk decision succeeded and the resulting `OrderIntent` is explicitly marked as risk-approved.
- Associative memory can only veto or shrink a proposal. It cannot originate trades or bypass risk approval.
- Runtime startup validates broker/account state, symbol eligibility, and reconciliation before trading.
- Risk controls enforce stale-data rejection, spread checks, daily loss limits, max notional, max concurrent positions, cooldowns, duplicate-order protection, idempotency, and a kill switch.
- Live submit retries are bounded and recovery-aware; the router checks broker state by `client_order_id` before retrying.
- The baseline runtime is for research and infrastructure validation, not for unattended live deployment.

## Limitations

- The default runtime uses a polling abstraction; websocket trade updates and streaming reconciliation are intentionally deferred.
- The backtest is intentionally simple and bar-based; it does not yet model partial fills, venue microstructure, or borrow constraints with live parity.
- The memory encoder uses snapshot-level latent states from the current model output plus the optional temporal CNN path; it does not perform uncontrolled online weight mutation.
- Real-data training currently relies on Alpaca historical bars plus local cache files; the repository does not yet include a broker-independent market data warehouse.
- Prototype clustering is incremental centroid grouping, not a full nonparametric clustering system.
- Runtime memory updates only append statistics when configured; model weights still remain offline-trained.
- Hopfield retrieval is implemented as a modern attention-like associative lookup over unit prototypes rather than a literal classical recurrent Hopfield network.
- The ESOM component is an ESOM-like topology-aware memory map over dynamic units, not a full fixed-grid classical ESOM implementation.
- Dynamic memory maintenance is intentionally heuristic in v2. Candidate promotion, merge, split, and prune rules are evidence-driven and configurable, but not yet learned end-to-end.
- The current multitask trainer uses the existing runtime model heads plus auxiliary real-target losses; richer dedicated path heads can be added later without changing the feature or checkpoint contract.
- RL support is scaffolded, not a full PPO/actor-critic implementation.
- Reconciliation currently hydrates positions and open orders only; it does not rebuild a full broker-authoritative historical ledger.
- Session rules rely on configured windows plus broker clock; venue-specific holiday edge cases remain intentionally simplified.
- No profitability claims are made, and untrained model outputs are not meaningful strategies.

## Extension Points

- Add richer market microstructure, quote, and order book features in `src/features/`.
- Replace polling with websocket streams inside `src/data/stream_manager.py`.
- Expand memory retrieval to richer regime-aware prototypes, topology-aware maintenance, and calibration diagnostics.
- Replace heuristic novelty, promotion, merge, and split policies with stronger learned or regime-aware calibration once the offline metrics justify it.
- Train a stronger dedicated state encoder or contrastive latent objective once the hybrid temporal/memory diagnostics justify the extra complexity.
- Add research tooling for memory coverage, cluster stability, and outcome-calibration analysis across checkpoints.
- Add richer checkpoint management, experiment tracking, and model versioning across research runs.
- Extend `src/training/rl_scaffold.py` into a full RL trainer after the supervised stack is stable.
- Add portfolio-level optimizers, multi-asset coordination, and more advanced risk overlays.
