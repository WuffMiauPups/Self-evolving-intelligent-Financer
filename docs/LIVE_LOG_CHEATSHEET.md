# Live Log Cheat Sheet

Quick reference for reading `logs/seif_runtime.log` (or the GUI log tail)
during a paper run. These are the log events the runtime actually emits
today (`src/execution/tick_processor.py`); anything else you may remember
from older docs (soma_weights analysis, hopfield_top_indices,
esom_neighbor_ids, calibrated_match_strength) belongs to deleted machinery
and no longer appears.

## Per-tick events, in loop order

| Event | What to watch |
| --- | --- |
| `broker_sync_failed` (WARN) | Broker unreachable this tick; `consecutive_failures` climbing toward `failure_limit=30` means trading is about to be blocked, not just skipped. |
| `tick_trading_skipped reason=broker_sync_failed` | Data lane still ran but no proposals were generated this tick; occasional is fine, constant means broker/network trouble. |
| `order_transition` | Order lifecycle from broker sync and after each submit; watch `status` reach a terminal state and `filled_qty`/`remaining_qty` add up. |
| `broker_fill_captured` | A broker fill was folded into the ledger/portfolio; check `realized_pnl` and that the `client_order_id` matches a prior submit. |
| `bar_backfill` / `bar_backfill_failed` | Gap-repair after an outage or session boundary; repeated `bar_backfill_failed` for a symbol means the session guard will start blocking on data gaps. |
| `warmup_pending` | Symbol buffer still below `runtime.warmup_bars`; logged once per symbol — normal at startup, suspicious hours in. |
| `symbol_quarantined` | Symbol excluded with `reason_code` + `release_mode`; verify it is a symbol you expected to be blocked. |
| `memory_update` | Passive memory finalized outcomes (`finalized_count`) and appended to the bank (`bank_items`); the bank should grow slowly during a run with memory updates enabled. |
| `session_guard` | One per eligible symbol per tick; `allowed=False` with a `reason_code` (stale data, session window, data gap) is the most common reason nothing trades. |
| `state_reset` | Policy state reset for a symbol (data gap / non-monotonic timestamps); frequent resets point at a data-quality problem. |
| `model_decision` | The proposal: `action` (0=short 1=flat 2=long), `confidence`, `target_fraction`. Ignore `soma_weights`/`soma_activation`/`branch_confidences` — legacy placeholder fields, constant since the soma/branch machinery was removed. |
| `risk_decision` | `approved` + `reasons`; watch for `resized_to_cap` (approved but capped) and halt reasons (`kill_switch`, `daily_loss_limit`, `spread_too_wide`). |
| `order_submission_blocked` | Session guard failed between decision and submit; should be rare. |
| `order_submit_attempt` | The intent actually sent to the router; every one should be followed by an `order_transition`. |

Note: a `memory_retrieval` log call still exists in the code but never fires
with the current passive memory — retrieval diagnostics are recorded as
`memory_knn` inside `runs/decisions.jsonl` instead of the log stream.

## Quick greps

```bash
# Why is nothing trading?  Reason-code distribution of blocked ticks
grep session_guard logs/seif_runtime.log | grep "allowed=False" | grep -oE 'reason_code=\S+' | sort | uniq -c | sort -rn

# What is the risk layer doing to proposals?
grep risk_decision logs/seif_runtime.log | grep -oE 'reasons=\[[^]]*\]' | sort | uniq -c | sort -rn

# Did submits complete?
grep -c order_submit_attempt logs/seif_runtime.log; grep order_transition logs/seif_runtime.log | grep -c terminal=True

# Fills and realized PnL
grep broker_fill_captured logs/seif_runtime.log | tail -20
```

For per-decision analysis (executed action, resized flag, memory kNN
neighbors), read `runs/decisions.jsonl`; for fills, `runs/fills.jsonl`.
The runtime status snapshot also exposes `intervention_rate` — the fraction
of proposals the risk layer rejected or resized.
