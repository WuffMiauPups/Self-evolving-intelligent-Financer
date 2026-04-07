# Live Log Cheat Sheet

Quick reference for reading `logs/*.log` or the GUI log tail during a paper
run, scoped to the fixes shipped on 2026-04-07. Tells you at a glance
whether the brain + memory + fusion stack is actually benefiting from the
changes or still degenerate.

## What each log line means

The paper loop emits three lines per tick per symbol:

```
session_guard     allowed=... reason_code=...
memory_retrieval  match_strength=... calibrated_match_strength=... ...
model_decision    action=... confidence=... soma_weights=... branch_confidences=...
risk_decision     approved=... reasons=[...]
```

## The six signals to watch

These are the fields that should demonstrably change after the
encoder-L2 / soma-gating / ESOM-aggregation / graded-conflict fixes.

### 1. `soma_weights` vs `branch_confidences` (from `model_decision`)
The soma now bakes `log(branch_confidence)` into its softmax exponent, so
the branch with the highest weight should be among the highest-confidence
branches — not the lowest one.

- **Bug signature (pre-fix):** `regime` had weight ~0.65 while its confidence
  was ~0.42 (lowest of four).
- **Expected after fix:** whichever branch has the max confidence should
  also have the max (or near-max) soma weight. If confidences are all
  similar, soma weights will be near-uniform (~0.25 each).
- **Red flag:** the lowest-confidence branch still has the highest weight.

### 2. `calibrated_match_strength` (from `memory_retrieval`)
Should now *vary* per query instead of pinning at ~0.08.

- **Bug signature (pre-fix):** locked at 0.082 for every tick on every
  symbol because the calibration bucket confidence had collapsed (binary
  conflict rule + saturated cosine).
- **Expected after fix:** values spread across the bucket edges — typical
  range 0.05–0.40 depending on match quality. Should correlate with
  `average_similarity`.
- **Red flag:** still pinned at one value across all ticks.

### 3. `hopfield_top_indices` + `esom_neighbor_ids` (from `memory_retrieval`)
Both should DIFFER across consecutive ticks and across symbols. ESOM
neighbor slate was previously identical everywhere.

- **Bug signature (pre-fix):** `esom_neighbor_ids=['unit_0323','unit_0098','unit_0082']`
  on every single tick, regardless of symbol or time.
- **Expected after fix:** different symbol → different slate; consecutive
  ticks of the same symbol → mostly different slates (some overlap is fine).
- **Red flag:** identical neighbor lists across more than 3 consecutive
  ticks or across different symbols.

### 4. `conflict_score` (from `memory_retrieval`)
No longer locked at 1.0 on symbols with small per-symbol calibration
buckets.

- **Bug signature (pre-fix):** MSFT conflict_score = 1.0 forever (binary
  rule killed the bucket).
- **Expected after fix:** values in (0, 1), typically 0.3–0.7 on weak
  matches, 0.0–0.3 on strong matches.
- **Red flag:** still exactly 1.0 for any symbol.

### 5. `reason_code` (from `memory_retrieval`)
Previously alternated between `calibrated_weak_match` and
`conflicting_outcomes`. After the fixes the bank should sometimes emit
`strong_match`, `weak_match`, `aligned_with_memory`, etc.

- **Expected after fix:** variety — at least some ticks should NOT be
  `calibrated_weak_match` or `conflicting_outcomes`.
- **Red flag:** same reason code on every tick for hours.

### 6. `advisory_effect` + `memory_vetoed/dampened` (from `memory_retrieval`)
Memory is advisory-only, but its advice should actually be useful. With
a healthier bank, you want to see at least SOME ticks where memory
AGREES with the brain (not just dampens every proposal).

- **Expected after fix:** mix of `dampened`, `supported`, `neutral`. If
  every tick is `dampened` → the calibration is still weak.
- **Red flag:** 100% `dampened`, 100% `memory_vetoed=False` but every
  trade still gets pushed to flat.

## Quick grep commands

Run any of these on the live log to audit a run in 10 seconds:

```bash
# Is the soma respecting confidence? Pull branch_confidences and soma_weights
grep model_decision logs/live.log | tail -20

# Are memory matches varying?
grep calibrated_match_strength logs/live.log | awk -F'calibrated_match_strength=' '{print $2}' | awk '{print $1}' | sort -u | head

# Are ESOM neighbors varying?
grep esom_neighbor_ids logs/live.log | grep -oE "'unit_[0-9]+'" | sort -u | wc -l

# Reason code distribution
grep memory_retrieval logs/live.log | grep -oE 'reason_code=\S+' | sort | uniq -c | sort -rn
```

## If everything still looks degenerate

The bank and the runtime code are only as good as the brain checkpoint.
If after rebuilding with the fixes the logs still show pinned
`calibrated_match_strength` or dominant `regime` soma weights, the
next step is a full retrain with the new `gating_confidence` loss — that's
what pushes bucket agreement above chance and fixes the "brain collapses to
regime" problem at its source.
