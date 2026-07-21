# Confidence Audit — 2026-07-21

Trigger: live paper trades (day 2 of the paper week) showed
Spearman(confidence, realized PnL) = **-0.428** (p < 0.001, n = 207) with a
monotone quartile gradient — the LEAST-confident quartile made all the money
(+$124, 70.6% win) and the most-confident quartile won 18.5%. If real, an
inversion of that size means either a sign-class bug (like the EV-leg swap,
550d6de) or anti-predictive alpha. This audit tested it OOS and traced the
confidence path end-to-end.

## 1. OOS reproduction test: the inversion is NOT real

`scripts/test_confidence_inversion.py` runs the incumbent's OOS backtest
(shared risk path, EV-leg fix active, next_open fills) and joins each round
trip's **entry-decision confidence** to its realized net PnL. The join is
exact: decisions log `bar_index`, next_open fills land at `bar_index + 1`,
matched on action sign (0 unmatched entries out of 1,313 RTs).

Result (`runs/confidence_inversion_oos.json`, incumbent d748f28c):

| quantity | n | Spearman rho | p |
|---|---|---|---|
| entry confidence vs RT net PnL | 1,313 | **-0.009** | 0.744 |
| entry conf, long RTs only | 912 | -0.028 | 0.397 |
| entry conf, short RTs only | 401 | +0.004 | 0.934 |
| close confidence vs RT net PnL | 1,313 | +0.026 | 0.338 |

Per-symbol entry rho scatters in [-0.22, +0.15] around zero. Winners' and
losers' mean confidence differ by < 0.001. With n = 1,313 the 95% CI is
roughly ±0.054, so |true rho| < 0.06.

**Verdict: confidence is UNINFORMATIVE, not inverted.** The live -0.428 was
a 207-sample artifact amplified by two measurement problems: (a) the live
join keyed on the fill's `client_order_id`, i.e. the CLOSING decision's
confidence, not the entry's; (b) the live confidence band is 5 points wide
(0.485-0.529), so quartile boundaries sit inside noise. The OOS close-conf
variant is also null (+0.026), confirming there is nothing there either.

## 2. What "confidence" actually is for the incumbent

The incumbent (d748f28c, `checkpoints/brain.pt`) predates Phase 2 §6.1
disentanglement: its state dict contains `policy_head.confidence.weight`,
i.e. the **legacy sigmoid head**, NOT the decoupled P(chosen class). Under
the non-decoupled loss wiring (`src/training/losses.py`), that head is
trained by regression onto **MFE / volatility / stability** path statistics
— it was never trained to mean "P(this trade wins)". The code comments say
exactly this ("regressing a stability statistic onto it was the reason
calibrated confidence never approximated win probability"); the fix
(`decoupled_heads: true`) applies only to checkpoints trained after it.

Observable consequence: incumbent confidence reaches 0.32 OOS —
impossible for a 3-class softmax max (>= 1/3), consistent with a sigmoid
stability regressor. Its narrow live band (0.485-0.537 across 81,099
decisions) is a calm-regime slice of the wider OOS range (0.32-0.55).

So for the incumbent, ALL THREE inputs of the EV gate
(`conf*gain - (1-conf)*loss - cost`) are now measured uninformative:
gain/loss legs are a vol prior + cost floor (§7.0.4, Spearman ~0 per
symbol), and conf is a stability regressor with zero outcome rank signal
(this audit). The gate's trade selection is economically empty for this
checkpoint. Mercy: sizing does NOT scale with confidence (decoupled sizing
default fraction), so the empty signal mispicks but does not mis-size.

## 3. Consequence for §9.4 arm 1 (Platt + confidence threshold)

Platt scaling is monotone: it cannot create rank signal that is not in the
input. Entry confidence has rho ~= 0 against outcomes over 1,313 OOS round
trips, per side as well — **arm 1's mechanism has nothing to work with on
the incumbent**, regardless of how many paper trades are banked. The
300-resolved-trade data gate was waiting to fit a curve onto a flat input.
(The per-side split requirement is moot for the same reason: both sides are
individually null.)

Follow-up, MEASURED same day on all three noise-floor seed checkpoints
(sweep_20260717_142927, decoupled heads, conf = P(chosen class); results in
`runs/confidence_inversion_oos_decoupled{,_s2,_s3}.json`):

| seed | long n | long rho (p) | short rho (p) | long top-half $/trade |
|---|---|---|---|---|
| 1 | 1,028 | +0.063 (0.044) | +0.005 (0.885) | +1.01 |
| 2 | 1,104 | +0.021 (0.487) | -0.027 (0.426) | -0.62 |
| 3 |   684 | +0.066 (0.085) | -0.014 (0.703) | +5.80 |

**The decoupled long-side confidence is the only measured rank signal in
any EV-gate input**: 3/3 positive rho (suggestive tier per §2.0), win-rate
quartile gradients ~41%→51%. Shorts are null on every seed and both head
types. Converting the signal to money with a median-confidence floor on
longs gives +1.01/-0.62/+5.80 per trade — 2/3 positive, mixed signs,
**does NOT clear §2.0** and does not meet the §9 step-5 milestone. Status:
pivot-discussion evidence (it survives any pivot that retrains under
`decoupled_heads: true`), not a promotable result.

## 4. Live shorts: blocked by broker physics, not by the EV gate

Separate finding while ground-truthing the day: the 2026-07-21 "short"
alerts were FALSE — a monitoring artifact. `watch_opens.py` replayed the
day's fills from an assumed-flat book, but 9 long positions carried over
from Friday; the "SHORT OPENED" events were sells CLOSING/trimming those
carried longs. Broker truth at 14:10 UTC: 6 positions, all long, zero
shorts today. The watcher now seeds from `logs/.open_watch_seed`
(start-of-day book = broker qty minus the day's net fills).

The EV-leg fix has therefore NOT yet been exercised live. Today's ledger
shows 5 short proposals, all zeroed by `fractional_short_unsupported`
(risk_manager.py:83-122, the whole-share short constraint): with model
target fractions around 0.9-2% of equity (~$880-1,800), short targets on
high-priced symbols (SPY $745, META $645, MSFT $400, TSLA $375) floor to
sub-share and are unexecutable. Live shorts are only possible on
low-priced symbols (NFLX ~$67, MSTR ~$103, NVDA ~$207) — and none has
been proposed short at a whole-share size from a flat book yet. The OOS
backtest books short RTs freely (401) because backtest equity/prices allow
whole-share sizes on the cheap symbols; the live universe skews expensive.

## 5. THE FIX (2026-07-21, same day)

**Core** (`src/strategies/brain_policy.py`, `_build_proposal` — the shared
funnel, same site as the EV-leg swap): proposal confidence is now
**softmax(action_logits)[resolved_action] = P(resolved class)**, computed
exactly once where the resolved action is known. Legacy checkpoints stop
emitting the stability regressor; decoupled checkpoints are numerically
unchanged (flip off) and CORRECTED under a forced flip (P(resolved), not
P(flat)). The replaced head value stays observable as
`diagnostics["head_confidence"]`. Non-finite logits fail CLOSED
(confidence forced to 0.0; the EV gate additionally rejects non-finite
EV — `NaN <= 0` is False, a fail-open found by adversarial review).

**Semantics boundary machinery** (the guard that keeps old and new
quantities from ever mixing — three converging adversarial-review findings):

- `CONFIDENCE_SEMANTICS = "p_resolved_class"` (src/core/constants.py).
- Ledger rows (`DecisionEvent`) carry `confidence_semantics`; pre-fix rows
  lack the key and are thereby identifiable forever.
- `PlattCalibrator.fit()` **never self-stamps** — the caller must attest
  the quantity; unattested artifacts are refused at load.
- `load_calibrator_for_checkpoint` refuses artifacts without the current
  stamp (the canonical pre-fix sidecar next to brain.pt is now inert) and
  honors its never-raises contract on corrupt artifacts.
- `scripts/refit_platt_live.py` admits ONLY current-semantics ledger rows
  (exclusion count surfaced in stats) — the ~81k pre-fix rows and all
  pre-fix resolved trades can never poison a refit.
- `src/training/supervised.py` fits train-time Platt on P(argmax) computed
  FROM THE LOGITS (identical for decoupled heads; correct even if a legacy
  config is ever retrained) and attests it.
- `runtime_bootstrap` warns loudly when `platt_calibration_enabled` is a
  no-op (memory-off bare-policy path never applies calibration — this was
  silently true for the entire paper run).

**Paired OOS impact** (incumbent, EV-leg fix constant, only conf changed;
`runs/conf_fix_impact_post.json` vs `runs/ev_leg_fix_post.json`):

|  | pre (head conf) | post (P(resolved)) |
|---|---|---|
| fills | 5,573 | 6,716 |
| short round trips | 400 | 604 |
| expectancy/trade | -0.215 | **-0.152** |
| total return | -1.21% | **-1.02%** |
| EV pass share long/short | 76.9% / 48.4% | 83.8% / 78.5% |

Post-fix inversion re-test (`runs/confidence_inversion_oos_postfix.json`):
entry-conf rho **still null** (-0.015 overall) — the fix makes the
quantity semantically correct; the incumbent's LOGITS carry no rank
signal, so no confidence derived from them can rank outcomes. The fix is
correctness + future-proofing (decoupled retrains DO show the long-side
gradient in this quantity), not incumbent edge.

Tests: `tests/test_confidence_semantics.py` (15; behavioral ones verified
failing pre-fix), platt loader contract updated, refit boundary test.
Suite 1,084. **Deployment: the paper trader picks this up at its next
launch** — confidence distribution at the EV gate shifts (day-1/day-2
throughput baselines are not comparable across the boundary; ledger rows
self-describe via the semantics tag from the first post-fix launch).

Deferred (recorded, not fixed): seed-ensemble `_merge` averages members'
own-class probabilities (violates the contract if ever revived — arm 2 is
crossed off); GUI `platt_fitted` badge is existence-only and will show the
refused stale sidecar as active; calibrator save() is non-atomic;
force-directional-flip runtimes would apply calibration out-of-support
(flag false everywhere).

## 6. Post-fix live session (2026-07-21 16:11-19:22 UTC, session 20260721p5324)

First session on the fixed semantics. 468 post-fix decisions, all tagged
`confidence_semantics=p_resolved_class`; the stale canonical sidecar was
refused in the LIVE path (`platt_calibrator_stale_semantics` on
`checkpoints\platt_calibration.json`) — note memory IS enabled, so
calibration was previously being APPLIED to live confidence, making the
guard consequential rather than theoretical.

| direction | n | conf median | EV-pass |
|---|---|---|---|
| LONG  | 178 | **0.864** | 98% |
| FLAT  | 230 | 0.613 | 100% |
| SHORT |  60 | **0.510** | 80% |

**The model is confident long and coin-flip short**, held over 468
decisions. Pre-fix this was invisible: every proposal got ~0.49, so the
gate treated a 0.86-conviction long and a 0.51-guess short identically.
This independently corroborates §1's OOS finding (long-side rank signal
3/3 seeds, short-side null 3/3) from a completely different measurement.
Backtest prediction P1 long-side confirmed and exceeded (77%→84%
predicted, 85%→98% live). P1 short-side is NOT cleanly testable live —
the pre/post symbol mixes differ (pre: NVDA/AAPL/MSTR/NFLX; post:
TSLA/QQQ/AAPL/SPY), and spread/legs are per-symbol.

### 6.1 The real short-side ceiling: SIZING, not price

Investigating a false-alarm divergence produced the sharper finding:

| direction | median target_fraction | notional @ ~$97.7k equity |
|---|---|---|
| LONG  | 2.03%  | **$1,987** |
| SHORT | 0.134% | **$131** |

**The model requests shorts ~15x smaller than its longs** — not
confidence-scaled (sizing is decoupled from confidence); the sizing head
simply emits tiny magnitudes on bearish states. At $131, only **2 of 13**
symbols can host a whole-share short (NFLX $67 → 1.95sh, MSTR $102 →
1.28sh; PLTR $134 misses at 0.98sh; everything else 0.18–0.63sh). That is
exactly and only why the day's two live shorts were NFLX and MSTR.

Consequence: the whole-share broker constraint is NOT the binding limit —
the model's own short sizing is. Any universe below ~$130/share unlocks
short expression; above it, shorts are structurally unreachable regardless
of signal quality. This is independent of the confidence fix and is now a
second, stronger argument for pivot option (c) (cost-viable universe).
Open question worth a measurement before acting: WHY the sizing head is
asymmetric — it may be a training-target artifact rather than a view.

### 6.2 Monitoring lesson: measure rates within a direction

Two false alarms this session, both from the same mistake — measuring a
reason-code as a share of ALL decisions when the per-direction proposal
MIX is set by `argmax(logits)`, which no gate change touches and which
drifts with market regime. `scripts/check_postfix_signature.py` now
measures within a direction, treats a whole-share block on an unaffordable
symbol as arithmetic rather than divergence, and dedupes standing
conditions so they announce once.

## 7. Tooling lesson (third fills-ledger artifact this week)

1. Line-count cursor re-read the tail (fixed: timestamp cursor).
2. `fills.jsonl` is not time-ordered (fixed: high-water-mark max()).
3. Flat-start replay across a carried book (fixed: seed file).

Any future consumer of `runs/fills.jsonl` that reconstructs positions MUST
sort by timestamp AND seed with the start-of-day book.
