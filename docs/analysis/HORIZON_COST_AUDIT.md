# Horizon-vs-Cost Audit (§7.0.2)

**Run:** 2026-07-17, `python scripts/audit_horizon_costs.py` (deterministic given the bar cache; JSON artifact: `runs/horizon_cost_audit.json`).
**Data:** the exact bars the training path loads (`_resolve_bars_by_symbol`, default+training stack): 13 symbols, 5-min bars, 2022-01-03 → 2026-07-17, ~1.13M forward-return pairs per horizon.
**Cost:** `training.transaction_cost_bps = 6.0` (= 2·(fee 1 + slippage 2), the backtest round trip, by §4.4.2 contract).
**Session-boundary rule:** pairs spanning more than H×5min + 30min are excluded (the overnight gap is not a tradeable intraday move); the excluded fraction is reported — it rises mechanically with H because an H-bar window must fit inside the remaining session.

## Pooled result (all symbols)

| H bars | ≈ time | pairs | excluded | P(\|move\| > 6bps) | median \|move\| | median clearing \|move\| | breakeven hit rate (uncond. / clearing) |
|---|---|---|---|---|---|---|---|
| 1 | 5 min | 1,133,361 | 1.6% | 62.3% | 8.8 bps | 15.4 bps | 84.1% / 69.5% |
| 5 | 25 min | 1,133,309 | 7.8% | 81.5% | 19.7 bps | 25.6 bps | 65.2% / 61.7% |
| 15 | 75 min | 1,133,179 | 22.4% | 89.1% | 33.5 bps | 39.2 bps | 58.9% / 57.7% |
| 30 | 150 min | 1,132,984 | 41.9% | 92.3% | 47.8 bps | 53.2 bps | 56.3% / 55.6% |
| 60 | 300 min | 1,132,594 | 77.1% | 94.9% | 71.5 bps | 76.7 bps | 54.2% / 53.9% |

Breakeven hit rate = (m + c) / (2m) with c = 6 bps: the directional accuracy at which expected PnL is zero for typical-move-size trades. The two columns bracket reality — "uncond." assumes you trade the median bar, "clearing" assumes you only ever catch moves that clear cost.

## The symbol split is large

Per-symbol at H=5 (the neighborhood of the configured primary label horizon of 6 bars):

| Symbol | P(clear) | breakeven (clearing) | | Symbol | P(clear) | breakeven (clearing) |
|---|---|---|---|---|---|---|
| MSTR | 94.4% | 54.5% | | AAPL | 79.0% | 64.3% |
| PLTR | 91.1% | 57.3% | | MSFT | 78.3% | 64.8% |
| TSLA | 90.2% | 57.8% | | QQQ | 72.6% | 66.7% |
| NVDA | 88.3% | 59.0% | | VOO | 69.2% | 68.4% |
| NFLX | 85.6% | 60.7% | | SPY | 65.0% | 70.2% |

The cost-viable surface concentrates in the high-vol single names; the broad-market ETFs are structurally cost-dominated at short horizons (SPY at H=1 needs 79% directional accuracy to break even).

## Caveats

- **Spread:** historical bars carry `spread = 0`, so 6 bps understates the true round trip. Every breakeven number above is therefore a *lower bound* — the negative findings below only get stronger.
- **MSTR advisory:** the stylized-facts check flags two >20% single bars (2022-06-13, 2024-08-05 — both real crash mornings, two days before the 2024 split window). Two bars out of 64k do not move medians.
- H=60 mostly does not fit inside a session (77% excluded): the pairs that remain are morning-start windows only.

## Verdict (go/no-go on "the 5-min timeframe is the ceiling")

**The timeframe is NOT a hard ceiling — but the currently configured horizons sit in the worst part of the curve.**

1. **No-go for the strong claim.** Moves clear the round trip at high rates from H=5 up (81.5% pooled), so "no horizon clears costs" is false; 5-min bars remain a viable substrate.
2. **The configured label horizons (`forward_horizons` [1, 3, 6], primary = 6 ≈ 30 min) demand 60–70% directional accuracy** on the moves they see (interpolating H=5→15; worse on ETFs). Nothing measured on this model class is near that; the seed noise floor alone is ~6pt of val BA. This is the quantified version of the perturbation finding ("signal real, below costs").
3. **Horizon extension is the cheapest structural cost lever measured so far:** moving the primary horizon toward 15–30 bars (75–150 min) roughly halves the required edge (breakeven 57.7% → 55.6% conditional) while keeping 58–78% of pairs intra-session-valid. A horizon experiment (retrain with a longer primary horizon) belongs in the §9.4 experiment queue AHEAD of pure training-knob experiments — it moves the breakeven bar, not just the model.
4. **Symbol selection is a cost decision, not just a staleness decision:** SPY/QQQ (and VOO, already dropped from runtime for staleness — §7.1.2) need 66–70% hit rates at the configured horizons. If the EV gate's inputs ever become trustworthy (§7.0.4), per-symbol cost floors are the mechanism to encode this; until then the finding argues for concentrating evaluation attention on the high-vol names.

**Binding consequence for §7.2/§9.4 (per the §7.0.2 verify clause):** any accuracy experiment at the current horizons must state its target against the ~62% (pooled, clearing-conditional, H≈6) breakeven; experiments that extend the primary horizon to 15–30 bars are promoted into the queue as the structural alternative — with one precondition found while writing this audit: **`src/training/targets.py` has no session-boundary handling** (grep: no session/overnight logic), so H-bar labels near the close already include the overnight gap today (~8% of samples at H=6, matching the audit's exclusion rate) and would at H=15–30 corrupt 22–42% of labels. Session-aware label masking ships BEFORE any horizon-extension retrain, or the experiment measures gap noise.
