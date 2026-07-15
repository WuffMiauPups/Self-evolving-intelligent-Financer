# SETAI Accuracy Scorecard

One row per benchmark run (MASTER_PLAN §2.1). Protocol: bars strictly
after the checkpoint's train/val end, next-bar-open fills, real costs,
vs tradeable baselines. "Accuracy improved" = a new row beats the
incumbent on balanced accuracy AND net expectancy.

**⚠ Read every delta below against the NOISE FLOOR (measured 2026-07-15).**
Three fast-mode trainings identical except `training.seed` scored val
balanced accuracy **0.3865 / 0.4444 / 0.4132 — a 5.8pt spread from the RNG
alone** (test split: 8.1pt; long recall 0.18→0.48). The ⟲2 (+0.44pt) and
⟲3 (−0.31pt) row deltas are an order of magnitude SMALLER than that, so
neither is evidence of anything. Full-data runs should be quieter than
fast mode, but the floor is unmeasured there — until it is, a single-seed
row cannot support an "improved"/"regressed" claim. Compare multi-seed
distributions, not single rows.

**⚠ Rows dated ≤ 2026-07-15 bank unexecutable short fills.** They predate
the whole-share short constraint (Alpaca cannot short fractional shares);
sub-share shorts were simulated as filled but the live broker rejects them.
Absolute returns on these rows are optimistic for short-heavy windows.

Determinism (verified 2026-07-14 by byte-identical back-to-back runs):
a row is deterministic over the tuple (checkpoint weights+meta, bar
cache content, OOS cutoff TIMESTAMP, config, code version). Caveats:
- The "data start" column is date-only; the true window boundary is the
  cutoff timestamp (checkpoint train/val end, or the promotion gate's
  shared max-of-both-ends cutoff — logged as `oos_bars_dropped`).
  Two rows dated the same day can cover different windows.
- Fill counts are PATH-SENSITIVE to the window start: a 13-bar (65 min)
  cutoff shift moved buy-and-hold fills 9703 -> 10145 on identical data
  and code (entry prices shift every later cap-crossing trim). Compare
  fill counts only between rows sharing the exact cutoff.

| checkpoint | fingerprint | data start | data end | bal. acc | expectancy | sharpe | max DD | CVaR95 | vs buy&hold | trades | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brain.pt | d748f28c3695 | 2026-03-17 | 2026-07-13 | 0.3642 | -0.15 | -3.567 | -0.0115 | -0.00014 | -0.0396 | 940 | oos |
| brain.pt | 4575582c9cd9 | 2026-03-17 | 2026-07-13 | 0.3698 | -0.24 | -1.251 | -0.0610 | -0.00083 | -0.0517 | 872 | oos |
| brain.pt | 9b3ff04c6a7c | 2026-03-18 | 2026-07-14 | 0.3617 | -0.17 | -0.896 | -0.0578 | -0.00080 | -0.0617 | 1007 | oos |
| brain.pt | d748f28c3695 | 2026-03-18 | 2026-07-14 | 0.3648 | -0.10 | -2.305 | -0.0115 | -0.00014 | -0.0549 | 929 | oos |
