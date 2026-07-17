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
