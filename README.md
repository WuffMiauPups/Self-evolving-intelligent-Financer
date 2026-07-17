# SEIF Trading Brain Foundation

SEIF is a research-first trading framework for Alpaca that keeps PyTorch at the center of feature processing, model inference, and training. It is paper-trading-first, safe by default, and built so learned components can never bypass deterministic risk controls.

This repository is not a claim of profitability. It is a foundation for controlled research, honest evaluation, and observability.

## Highlights

- PyTorch-first numerics with CUDA auto-detection and CPU fallback
- `BrainTradingModel`: an optional temporal CNN over OHLCV history concatenated with a flat feature vector, feeding a 2-layer MLP backbone with policy, value, and optional risk heads
- Explicit `FeatureContract` validation so training, backtest, and paper runtime share the same feature ordering and dtype
- Persisted `NormalizationStats`: z-score statistics computed on the training split are stored in the checkpoint and re-applied at inference, eliminating train/serve normalization skew
- Passive associative memory: a file-backed bank of encoded latent states with forward outcomes, exposed as kNN diagnostics only — it never gates, vetoes, or resizes trades
- Deterministic `RiskManager` separating AI proposal generation from order approval, with an expected-value gate that charges fees, slippage, and the full quoted spread
- Out-of-sample backtests by default: the engine enforces checkpoint training ranges and refuses overlapping evaluation windows unless explicitly overridden
- Honest fill model: orders execute at the next bar's open plus/minus half the quoted spread
- Deterministic benchmark harness (`scripts/run_benchmark.py`) that appends comparable result rows to a scorecard
- Adjustment-aware historical bar cache (v2.0) with incremental fetch, deduplication, and data-sanity gating
- Centralized Alpaca adapter using the official [`alpaca-py`](https://alpaca.markets/sdks/python/) SDK
- Startup broker/account validation, symbol eligibility checks, restart reconciliation, and session/data guards before trading begins
- Strict live-mode interlocks: paper remains default; live requires config plus CLI confirmation
- Deterministic `client_order_id` generation for dedupe and restart-safe open-order synchronization
- Structured logging for model decisions, risk outcomes, execution events, and runtime state, plus JSONL outcome datasets under `runs/`

## Repository Layout

```text
configs/          Runtime, broker, training, backtest-preset, and symbol configuration
scripts/          Entrypoints for training, backtesting, benchmarking, paper mode, GUI
src/core/         Config, logging, devices, errors, time helpers, reproducibility
src/data/         Alpaca adapter, historical loading + cache, polling stream, buffers, datasets
src/features/     Torch-native feature engineering and the feature contract
src/memory/       Passive memory bank: schemas, storage, kNN retrieval, outcome updater
src/models/       Brain and baseline PyTorch models, state encoder, model factory
src/training/     Training loop, losses, targets, metrics, Platt calibration, retraining
src/evaluation/   Out-of-sample enforcement, walk-forward engine, benchmark harness
src/risk/         Portfolio state, deterministic limits, ledger, sizing
src/execution/    Order validation, routing, tick processing, runtime loop
src/backtest/     Backtest engine, simulator, fills, performance metrics
src/strategies/   Policy adapters used by backtest and runtime
src/monitoring/   Events, counters, decision log, runtime health state
src/gui/          PySide6 operator GUI (tabs over the service layer)
src/gui_services/ Framework-free service layer between GUI and backend
tests/            Unit and integration-style coverage
docs/analysis/    Codebase review, master plan, scorecard
```

## Architecture: Runtime Path

1. The runtime validates broker connectivity, account status, live-vs-paper mode consistency, and data-feed usability.
2. Symbols are checked for asset metadata, tradability, activity, asset-class compatibility, and data availability; ineligible symbols are quarantined with explicit metadata.
3. Startup reconciliation hydrates broker positions/open orders and sets `SAFE_TO_TRADE` before the loop begins.
4. Market data arrives through the Alpaca adapter (or mock client). Polled bars are aggregated into the configured timeframe (default `5Min`) buckets; gaps are backfilled from the historical API.
5. Bars are stored in per-symbol buffers and converted into Torch tensors by the feature pipeline. Every snapshot is validated against the checkpoint's feature contract, and the checkpoint's persisted normalization statistics are applied before inference.
6. Session and timestamp guards run before proposal generation and again before order submission.
7. Once per symbol per bar bucket (`runtime.decide_once_per_bar`, on by default), the policy model produces a proposal: direction, confidence, and target position fraction. Re-polls of the same forming bar are skipped so the model is never re-run on unchanged data, while fill capture, halts, and the data lane still run every poll. If a fitted Platt calibrator is present next to the checkpoint (and its fingerprint matches), the confidence is calibrated before risk evaluation.
8. If memory is enabled, the encoded latent state is stored and nearest-neighbor diagnostics are attached to the decision record. Memory is informational only.
9. The deterministic risk manager approves, resizes, or rejects the proposal. Oversized orders are resized to the notional cap and tagged `resized_to_cap` instead of being rejected.
10. The order router only accepts intents that carry explicit risk approval and refuses duplicate or quarantined orders. With Alpaca credentials, `broker.paper=true` routes to Alpaca's paper endpoint.
11. Broker sync captures fills, updates the portfolio, and anchors daily PnL to broker equity. The ledger, decision log, and structured logs capture every outcome.

## Model

`BrainTradingModel` is intentionally simple:

- An optional temporal CNN (`model.use_cnn_encoder`) encodes the OHLCV feature-history window.
- The CNN embedding is concatenated with the flat feature vector and passed through a LayerNorm + 2-layer MLP backbone.
- A policy head produces action logits, confidence, and target position fraction; a value head produces a value estimate; an optional risk head (`model.separate_risk_heads`) predicts expected gain/loss magnitudes used by the risk manager's EV gate.

Earlier experimental machinery — dendritic branches, soma gating, regime prototypes, GRU latent memory, Hopfield/ESOM retrieval, and the decision-fusion layer — was removed after evaluation showed no measurable benefit. Some config keys and log fields survive as inert placeholders for backward compatibility; they do not affect behavior.

## Training Path

1. Real Alpaca bars are fetched into the local adjustment-aware cache (v2.0) keyed by symbol and timeframe; synthetic bars remain available for deterministic smoke tests.
2. Multi-symbol datasets are built with time-based train/val/test splits and split-boundary embargo so labels never leak across splits.
3. Direction labels (short/flat/long) use a cost-aware flat band: the band is derived from `training.transaction_cost_bps` (6.0 in `configs/default.yaml`, held by contract at `2 × (backtest.fee_bps + backtest.slippage_bps)`) and scales with the square root of the label horizon, so "flat" means "not worth trading after costs". With `training.session_aware_labels` (on by default), samples whose outcome window crosses a session close are dropped, so overnight gaps are never labeled as tradeable moves.
4. Class imbalance is handled by one mechanism: balanced class weights. Focal loss and the weighted sampler exist but are opt-in and off by default.
5. The trainer selects the best epoch on `val_balanced_accuracy` and restores those weights at the end of training (`training.restore_best_model`).
6. Checkpoints persist the model config, feature contract, normalization statistics, per-symbol training date ranges, and an encoder fingerprint.
7. After training, a direction-aware Platt calibrator is fitted on validation outputs and stored next to the checkpoint, bound to the checkpoint fingerprint. At load time a fingerprint mismatch degrades to the uncalibrated passthrough.

The training pipeline does **not** train on runtime risk-manager decisions, session-guard rejections, or broker/account blocks. The model learns market structure; runtime risk controls remain a separate authoritative layer.

Training also logs class counts and ratios by split, flat rates, near-boundary rates, per-class precision/recall/F1, macro/weighted F1, balanced accuracy, MCC, confusion matrices, and collapse warnings.

### Checkpoint staleness — retrain required

Checkpoints trained before the normalization-persistence fix (C2) do not contain `NormalizationStats` and were evaluated under train/serve skew. **All pre-C2 checkpoints are stale: retrain before trusting any result derived from them.**

## Feature Contract

A `FeatureContract` captures feature ordering, regime-input ordering, and the tensor dtype for model-facing features. It is:

- validated while building the training dataset
- stored in checkpoints
- checked again in backtests and paper runtime before model inference

Persisted normalization statistics travel with the contract inside the checkpoint, so training, backtest, and paper execution see identically scaled inputs.

## Evaluation

### Out-of-sample by default

Backtests enforce the checkpoint's persisted training ranges: if the requested evaluation window overlaps the data the model was trained on, the run aborts with instructions. Passing `--allow-in-sample` permits the run but tags the results as in-sample.

### Fill model and metrics

- Fills execute at the **next bar's open**, plus half the quoted spread for buys and minus half for sells.
- Reports include total return, max drawdown, Sharpe/Sortino, profit factor, expectancy, turnover, exposure, win rates, and **CVaR95** (expected shortfall of the worst 5% of per-bar returns).

### Walk-forward

`src/evaluation/walk_forward.py` runs chronological train/val/test folds (rolling or expanding). Policy state is reset between folds, memory banks can be rebuilt per fold from training bars with an outcome embargo, and degenerate folds (zero fills) are excluded from aggregate statistics and reported separately.

### Benchmark scorecard

`scripts/run_benchmark.py` runs a deterministic, seeded benchmark for a checkpoint and appends one comparable row (checkpoint fingerprint, data range, balanced accuracy, expectancy, Sharpe, max drawdown, CVaR95, excess vs buy-and-hold, trade count, in/out-of-sample tag) to `docs/analysis/SCORECARD.md`. Model claims should reference scorecard rows, not ad-hoc runs.

## Outcome Dataset

Runtime execution persists machine-readable outcomes under `runs/`:

- `runs/fills.jsonl` — the fill ledger (one row per fill, including broker-captured fills with realized PnL)
- `runs/decisions.jsonl` — one row per model decision with the proposal, the risk decision, the **executed** action, the risk head's expected gain/loss (bps) exactly as the EV gate consumed them, and whether the order was resized (`resized_to_cap`)
- `runs/observations.jsonl` — one row per executed fill (realized net PnL and the submit-time clamp flag); the data source the retrain drift trigger reads

These files are the ground truth for outcome-driven analysis: they record what actually happened after risk clamping, not the model's raw intent.

The structured runtime status snapshot additionally exposes `intervention_rate` — the fraction of proposals the risk layer rejected or resized — so a run where the risk manager is constantly overriding the model is visible at a glance.

## Passive Memory

The memory subsystem is a deliberately minimal scaffold:

- `MemoryBank` stores encoded latent states with timestamps, forward returns, excursion/volatility/drawdown summaries, and path labels.
- Each runtime-recorded item carries an executed-action annotation (action, approval, side, quantity, resized flag) — memory credits what the risk-clamped execution actually did, not the model's raw intent.
- Items and banks carry an encoder fingerprint; loading a bank built by a different checkpoint fails fast.
- Retrieval is a simple top-k nearest-neighbor lookup exposed in decision diagnostics (`memory_knn`).
- The updater finalizes outcomes once enough forward bars exist and appends items to the bank; backtests never mutate the bank at runtime (offline builds only, with an embargo boundary recorded in bank metadata).

Memory does **not** gate, veto, dampen, or resize proposals, and it cannot originate trades.

## Risk Management

The deterministic `RiskManager` is the only component that can approve an order:

- **EV gate**: a trade must satisfy `confidence * gain_bps - (1 - confidence) * loss_bps - (fee_bps + slippage_bps + spread_bps) > 0`. The full quoted spread is charged because a round trip crosses half the spread twice. With `model.separate_risk_heads`, gain/loss come from the trained risk head.
- **Spread guard**: proposals with `spread_bps >= risk.max_spread_bps` are rejected (`spread_too_wide`).
- **Daily-loss limits** work in broker mode: daily PnL is equity-based and anchored to the broker's previous close (`last_equity`), so limits act on real account swings.
- **Risk-reducing exemption**: orders that reduce or close an existing position are exempt from kill-switch and loss-limit halts, so the system can always de-risk.
- **Resize instead of reject**: oversized proposals are resized down to the notional cap and approved with reason `resized_to_cap`.
- Plus: stale-data rejection, max aggregate notional, leverage cap, rolling drawdown limit, max concurrent positions, cooldowns, duplicate-order protection, idempotent client order IDs, and a kill switch.

## Operational Safety Flow

`SAFE_TO_TRADE` is the central runtime gate. When false, the loop generates no proposals and cannot route orders.

- Startup validation checks broker connectivity, credentials (or explicit mock mode), account usability, buying power visibility, endpoint mode consistency, market clock access, and a data-feed probe.
- Live mode requires `broker.paper=false`, `broker.live_enabled=true`, `broker.confirm_live=true`, **and** CLI confirmation via `--confirm-live`. The runtime never silently switches modes.
- Symbol eligibility is validated at startup; ineligible symbols are quarantined with a reason code and release mode (`until_restart` or `until_order_terminal`).
- Session guards enforce configured windows plus the broker clock; stale bars, backwards timestamps, and large data gaps block trading with machine-readable reason codes.
- A failed broker sync skips trading for that tick; repeated failures block trading entirely.
- Startup reconciliation distinguishes `warn` / `block` / `fatal` mismatch severities.

`client_order_id` follows a reproducible schema:

```text
seif-{session_id}-{SYMBOL}-{b|s}-{YYYYMMDDHHMM}-{sequence}
```

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

Configuration is centralized in YAML files; later files override earlier ones. The config schema is strict (`extra='forbid'`): a YAML key without a model field is a hard load error, never a silent no-op.

- `configs/default.yaml` — safe defaults: features, model dimensions, risk limits, memory, backtest settings
- `configs/paper_trade.yaml` — paper-mode runtime with shorter loop settings
- `configs/training.yaml` — training-specific settings
- `configs/symbols.yaml` — symbol universe
- `configs/backtest_*.yaml` — backtest presets (the old fusion/topology-growth presets were deleted together with the config stubs they toggled)

Frequently used knobs:

- `model.checkpoint_path` — checkpoint for backtests or paper runtime
- `model.use_cnn_encoder`, `model.cnn_channels`, `model.cnn_kernel_sizes`, `model.temporal_blocks` — temporal CNN
- `model.separate_risk_heads` — enable the trained risk head feeding the EV gate
- `training.transaction_cost_bps` — cost assumption for the label flat band and calibration
- `training.platt_calibration_enabled` — fit/apply the confidence calibrator
- `training.session_aware_labels` — drop training samples whose outcome window crosses a session close
- `training.risk_target_per_symbol_norm` — per-symbol normalization of the risk head's MFE/MAE targets (off by default; experiment)
- `runtime.decide_once_per_bar` — generate at most one proposal per symbol per bar bucket
- `runtime.force_directional_flip` — retired experiment; default `false` and identical in backtest and live
- `runtime.state_reset_gap_sec`, `runtime.allowed_sessions`, `runtime.require_market_open`, `runtime.max_data_staleness_sec`, `runtime.block_on_data_gap`
- `execution.submit_failure_cooldown_sec` — cooldown before re-submitting a `(symbol, side)` the broker terminally refused
- `risk.fee_bps`, `risk.slippage_bps`, `risk.max_spread_bps`, `risk.max_position_notional`, `risk.max_daily_loss`, `risk.min_trade_notional`
- `memory.enabled`, `memory.persist_path`, `memory.max_items`, `memory.forward_horizons`, `memory.require_encoder_match`
- `data.symbols`, `data.timeframe`, `data.start_date`, `data.end_date`, `data.cache_dir`, `data.force_refresh`
- `broker.live_enabled`, `broker.confirm_live`, `broker.require_account_match`
- `logging.file_enabled`, `logging.file_path` (default `logs/seif_runtime.log`)

## Running

### Smoke test

```bash
python scripts/smoke_test.py
```

### Fetch historical data

```bash
python scripts/fetch_historical.py
python scripts/fetch_historical.py --force-refresh
python scripts/fetch_historical.py --inspect-cache
```

### Supervised training

```bash
python scripts/run_train.py --real-data
python scripts/run_train.py --real-data --inspect-only
python scripts/run_train.py --baseline
```

Checkpoints are stored under `checkpoints/` with full reconstruction metadata (model config, feature contract, normalization stats, training ranges, fingerprint).

### Backtest

```bash
python scripts/run_backtest.py --checkpoint checkpoints/brain.pt --real-data
python scripts/run_backtest.py --checkpoint checkpoints/brain.pt --real-data --allow-in-sample
```

Backtests are out-of-sample by default (see Evaluation). `--allow-in-sample` overrides the guard and tags results.

### Benchmark scorecard

```bash
python scripts/run_benchmark.py --checkpoint checkpoints/brain.pt
```

Appends a deterministic result row to `docs/analysis/SCORECARD.md`.

### Promotion gate

Promoting a retrained checkpoint is a statistical decision, not a file copy. A candidate must beat the incumbent scorecard row on balanced accuracy and expectancy and beat buy-and-hold out of sample, and no backward-transfer slice may degrade beyond a configured floor. Slices are regime-labeled by default (`evaluation.promotion.backward_transfer_slicing`): per-session K-means over regime features groups the out-of-sample window into regimes, with a chronological fallback. An optional adverse-fill stress mode reports baseline-vs-adverse deltas but is never itself a promotion criterion. The canonical `checkpoints/brain.pt`, registry, and memory bank move only through this gate.

### Build memory (offline)

```bash
python scripts/build_memory.py --from-checkpoint checkpoints/brain.pt --real-data
```

Builds the file-backed passive memory bank from cached bars with the checkpoint's encoder and saves it to `memory.persist_path`.

### Paper trading loop

```bash
python scripts/run_paper.py
python scripts/run_paper.py --checkpoint checkpoints/brain.pt
```

Paper-first and safe by default. Runtime events stream to stdout and `logs/seif_runtime.log`; see `docs/LIVE_LOG_CHEATSHEET.md` for the per-tick log lines worth watching.

### Unattended paper run (Windows)

```bash
scripts\launch_paper_week.bat
```

Runs `scripts/preflight_monday.py` — a GO/NO-GO gate over the paper interlock, hardening flags (VOO drop, submit-failure cooldown, once-per-bar gate, staleness floor), credentials, and checkpoint/memory-bank fingerprint integrity — and only then starts the paper loop in a persistent, titled console. Keep the window open for the duration of the run; a closed console kills the trader.

### Live dashboard

```bash
python scripts/trading_dashboard.py
# then open http://127.0.0.1:8787
```

A dependency-free local page that reads the `runs/` artifacts and shows equity, open positions, recent fills, the rejection-reason mix, and model-signal state, refreshing every few seconds. Read-only.

### GUI

```bash
python scripts/run_gui.py
```

Operator GUI over the `src/gui_services` layer: runtime control, backtests, training, checkpoints, memory bank inspection, equity, risk, and logs.

### Live-mode confirmation

```bash
python scripts/run_paper.py --config configs/default.yaml --confirm-live
```

If any live safety condition is missing or inconsistent, startup aborts before the runtime loop begins.

## Safety Notes

- Paper trading is the default mode; the model never places orders directly.
- No order is routed unless a deterministic risk decision succeeded and the resulting `OrderIntent` is explicitly risk-approved.
- Memory is diagnostics-only; it cannot originate, veto, or resize trades.
- Risk-reducing orders remain possible during halts so positions can always be closed.
- Live submit retries are bounded and recovery-aware; the router checks broker state by `client_order_id` before retrying, and an order shape the broker terminally refuses puts its `(symbol, side)` on a cooldown (`execution.submit_failure_cooldown_sec`) instead of resubmitting every window.
- This is research infrastructure, not an unattended live-deployment system.

## Limitations

- The runtime uses a polling abstraction; websocket trade updates and streaming reconciliation are deferred.
- The backtest is bar-based; it does not model partial fills, venue microstructure, or borrow constraints.
- Real-data training relies on Alpaca historical bars plus the local cache; there is no broker-independent data warehouse.
- The passive memory bank stores diagnostics; nothing consumes it for decisions yet.
- RL support is scaffolded, not a full implementation.
- Reconciliation hydrates positions and open orders; it does not rebuild a full broker-authoritative historical ledger.
- No profitability claims are made, and untrained model outputs are not meaningful strategies.
