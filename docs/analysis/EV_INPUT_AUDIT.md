# EV-Input Quality Audit (§7.0.4)

**Run:** 2026-07-17, `python scripts/audit_ev_inputs.py --real-data` (CPU inference — deliberately never contends with a GPU training sweep). Artifact: `runs/ev_input_audit.json`.
**Setup:** incumbent `d748f28c` OOS backtest (post-training-cutoff bars, next-open fills), 77,631 usable decisions of 77,636. Predicted legs = the proposal's `expected_gain_bps` / `expected_loss_bps` exactly as the EV gate consumed them (risk-head MFE ×10⁴, |MAE| ×10⁴). Realized legs = MFE / |MAE| of the relative close path over 1..6 forward bars — the same construction `targets.py` uses (`compute_outcome_summary`), verified equal by test.

## Result

| Metric | Value |
|---|---|
| Spearman predicted gain vs realized MFE, pooled | **+0.149** |
| Spearman predicted loss vs realized MAE, pooled | **+0.256** |
| Same, per symbol (13 symbols) | **−0.03 … +0.18, median ≈ +0.02** |
| Predicted gain bps (mean, p5–p95) | 136.9, [38, 262] |
| Realized MFE bps (mean) | 24.6 |
| Predicted loss bps (mean, p5–p95) | 117.7, [30, 237] |
| Realized MAE bps (mean) | 31.0 |

Per-symbol detail: AAPL +0.001/+0.022, AMZN +0.001/+0.021, GOOGL −0.005/+0.066, META −0.019/+0.004, MSFT +0.016/+0.057, MSTR +0.089/+0.001, NFLX +0.014/+0.000, NVDA +0.050/+0.058, PLTR −0.028/−0.021, QQQ +0.073/+0.152, SPY +0.119/+0.175, TSLA +0.006/−0.048, VOO +0.127/+0.131 (gain-vs-MFE / loss-vs-MAE).

## Interpretation

1. **The pooled-vs-per-symbol gap is the whole story.** Pooled +0.15/+0.26 collapses to ≈0 within symbols: the risk head has learned the *cross-sectional volatility ordering* (MSTR moves more than SPY — true, and trivially learnable from feature scale) and carries **almost no temporal information** about when a given symbol will move. What survives per-symbol clusters in the low-vol ETFs (SPY/QQQ/VOO ≈ +0.12–0.18), which the horizon audit says are the least cost-viable instruments anyway.
2. **Magnitudes are ~5× over-predicted** (gain 137 vs realized 25 bps; loss 118 vs 31). The EV gate therefore runs on inflated legs whose difference (≈ +19 bps mean spread of gain over loss) dwarfs the honest realized spread (≈ −6 bps) — approvals lean on a systematically optimistic prior.
3. **Formal decision (per the §7.0.4 verify clause): the EV gate is, today, a symbol-level volatility prior plus the cost floor — NOT a per-decision signal.** §4.4.4's "EV gate requires signals" is re-scoped accordingly: the construction-time refusal (risk-head-less models) still guards against zero-input fiction, but a present-and-weak risk head passes it while contributing ≈nothing per decision. Trade selectivity currently rests on confidence (pinned, §7.0.3 pending) and costs alone.

## Consequences

- The §7.0.3 Platt refit (realized-outcome calibration) carries MORE weight: with EV legs uninformative, calibrated confidence is the only live lever that can make the EV inequality discriminate.
- Any future risk-head improvement claim must show **per-symbol** rank correlation moving, not pooled (pooled can be inflated by cross-sectional vol alone). This audit is the baseline; the script is deterministic given cache + checkpoint.
- Live decisions now record the EV legs (`DecisionEvent.expected_gain_bps/expected_loss_bps`, additive fields) so the paper week accumulates the live-side dataset for the same comparison against fills.

---

## 2026-07-20 addendum — direction-split audit: the short side's EV legs are SWAPPED

**Trigger:** paper-week day 1 measured 92 live short intents with only 6 executed (longs: 73/206), at near-identical confidence (0.489 vs 0.509) but wildly asymmetric EV legs (shorts gain 97 / loss 198 bps; longs 450 / 221). A secondary blocker was `fractional_short_unsupported` (56/92 — Alpaca requires whole-share shorts, so shorts are only feasible on low-priced names; the session's executed shorts were exactly 1.000-share GOOGL/SPY).

**Run:** `python scripts/audit_ev_direction.py --real-data` (incumbent `d748f28c`, same OOS window/protocol as above; artifact `runs/ev_direction_audit.json`; 13,413 directional decisions).

| | LONG (n=6,988) | SHORT (n=6,425) |
|---|---|---|
| mean confidence | 0.4336 | 0.3681 |
| predicted MFE / \|MAE\| (bps) | 230.3 / 103.9 | 126.8 / 221.1 |
| realized MFE / \|MAE\| (bps) | 47.6 / 62.2 | 45.2 / 53.1 |
| EV **as consumed** (mean, share > 0) | +40.8, **76.9%** | **−92.6, 0.3%** |
| EV **direction-corrected** | +40.8, 76.9% (unchanged) | **+1.7, 48.4%** |
| Spearman predMFE~realMFE / pred\|MAE\|~real\|MAE\| | +0.131 / +0.226 | +0.136 / +0.163 |

**Mechanism (code-verified, not inferred):** MFE/MAE are *price-path* excursions — `max_favorable = relative_path.max()`, `max_adverse = relative_path.min()` (`src/memory/outcomes.py`), trained direction-agnostically (`targets.py`). At inference `_extract_ev_signals_*` (`src/strategies/brain_policy.py`) maps `gain = MFE`, `loss = |MAE|` **unconditionally**, and the EV gate (`risk_manager.py`) is direction-blind. For a SHORT proposal the legs are therefore inverted: the gate charges the short's win (the down-move) as its loss. The day-1 numbers decode exactly: as-consumed short EV 0.489×97 − 0.511×198 = −53.5 (measured −53.5); corrected ≈ +47.

**Findings:**
1. **The risk head is symmetrically direction-coherent.** Conditional on a long it predicts up/down ≈ 230/104; conditional on a short, 127/221 — a clean ~2.2:1 mirror. There IS directional signal in the head's levels; the original audit's "vol prior" verdict described the per-symbol *timing* dimension, not this conditional-on-intent dimension.
2. **The gate structurally vetoes half the model's directional opinion.** 6,425 OOS short proposals pass the (pre-cost) EV inequality at 0.3% as consumed vs 48.4% corrected. Every backtest and scorecard row ever produced shares this defect (same policy path), so historical comparisons remain internally consistent — but the short side has never actually been evaluated, anywhere.
3. **Not a training-data artifact.** Train targets are near-symmetric: MFE 145.5 vs |MAE| 137.3 bps (ratio 1.06, from the incumbent's own `target_diagnostics`). The "rising market taught a bullish prior" hypothesis is refuted.
4. **Realized excursions mildly vindicate the short side.** Conditional on short intents, realized down-moves exceed up-moves (53.1 vs 45.2 bps); conditional on long intents it is the reverse tilt (down 62.2 vs up 47.6) — the long book's negative expectancy is consistent with this. Corrected short EV (+1.7 mean) is still ~break-even pre-cost — the point is not that shorts are good, it is that they were never evaluated.
5. **Levels remain ~3–4× over-predicted** (230/221 predicted vs 48/53 realized) — the original audit's inflation finding stands unchanged.

**Consequences:**
- The leg mapping is an execution-layer correctness bug (same class as the 2026-07-20 zero-PnL race), NOT a training knob — fixing it is not barred by the §9 step-5 stopping rule, but it changes live behavior (the system would begin taking shorts), so it ships with tests + a backtest impact measurement and is deployed deliberately, not silently.

**FIXED 2026-07-21** (swap in `_build_proposal`, the shared single+batch funnel, keyed on the RESOLVED action; 5 tests in `tests/test_ev_leg_direction.py` fail pre-fix; `scripts/audit_ev_inputs.py` un-swaps shorts back to price-relative so its per-symbol Spearman stays comparable with this file's baseline). **Paired impact, same OOS window (`scripts/measure_ev_leg_fix_impact.py`, `runs/ev_leg_fix_{pre,post}.json`):** long side byte-identical (6,988 proposals, 76.87% pass, +40.8 mean EV); short EV-pass 0.28% → **48.42%**, mean EV −92.6 → **+1.8** (matches the corrected prediction above to the decimal); short participation 1 → 626 opening fills, 1 → 400 round trips. **Economics: the unlocked shorts LOSE — −$320.67 realized over the 400 short round trips (≈ −$0.80 each); overall expectancy −0.157 → −0.215/trade, total return −0.71% → −1.21%, sharpe −2.79 → −4.64.** Reading: the swapped legs were accidentally suppressing half of a below-breakeven model's trades (the same shape as arm 3's churn-as-damage-limiter); the fix removes a false veto, it does not create edge. Corrected short EV (+1.8 bps pre-cost mean) is ~breakeven before costs and negative after. **Scorecard comparability note: rows produced pre- vs post-fix are not apples-to-apples (the post-fix policy trades ~44% more round trips including shorts).** Deployment argument FOR next-launch pickup despite worse expected paper PnL: the paper week's currency is resolved trades and calibration coverage, not PnL — post-fix the system generates short-side outcomes at ~44% higher resolution rate, and the §9.4 arm-1 per-side Platt fit is impossible without short samples.
- **§9.4 arm 1 must be direction-aware end to end:** §4.2's direction-aware Platt already exists at the calibration layer; the refit on realized outcomes must fit and evaluate per side (short-side confidence is also systematically lower, 0.368 vs 0.434). Fixing the legs without per-side calibration — or vice versa — leaves the short side broken.
- The whole-share short constraint (§ broker physics) remains a real, separate throttle on short participation for high-priced names regardless of the above.
