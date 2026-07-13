# SETAI Accuracy Scorecard

One row per benchmark run (MASTER_PLAN §2.1). Protocol: bars strictly
after the checkpoint's train/val end, next-bar-open fills, real costs,
vs tradeable baselines. "Accuracy improved" = a new row beats the
incumbent on balanced accuracy AND net expectancy. Rows are
deterministic for (checkpoint, data range).

| checkpoint | fingerprint | data start | data end | bal. acc | expectancy | sharpe | max DD | CVaR95 | vs buy&hold | trades | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| brain.pt | d748f28c3695 | 2026-03-17 | 2026-07-13 | 0.3642 | -0.15 | -3.567 | -0.0115 | -0.00014 | -0.0396 | 940 | oos |
