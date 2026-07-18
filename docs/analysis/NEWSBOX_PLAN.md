# Newsbox Plan — news & event information as a measured input

**Created:** 2026-07-17. **Parent:** `MASTER_PLAN.md` (§9 step 5 pivot option (b) made concrete; §7.2.10 pointer). **Status:** N0 ungated (start anytime); N1 offline-anytime; N2–N4 gated as marked. All claims subject to the master plan's §2.0 noise-floor rule and §5 standing rules.

## 0. Verdict and the two hard rules

The idea (news, policy headlines, geopolitics as bot input) is the most plausible member of the §9.5 pivot menu — it is genuinely orthogonal information to OHLCV, and event bars are exactly where the §7.0.2 audit says moves clear costs. The version where **an LLM reads text and says buy/sell/short is rejected** — not because LLMs can't read, but because an LLM-as-trader is unbacktestable, uncalibratable, and un-promotable under this project's own evidence standards. Two rules are therefore constitutional for everything below:

1. **News becomes FEATURES, never orders.** Whatever reads the text (keywords, FinBERT, a local LLM) emits numbers into the existing feature contract; the existing model + EV gate + deterministic risk layer decide, exactly as for every other feature. "Learned proposes, deterministic risk disposes" is unchanged.
2. **Point-in-time or it didn't happen.** Every archived article carries BOTH `published_at` (source timestamp) and `ingested_at` (our wall clock). Backtests may only show an article to the model at `published_at + modeled_latency`, with latency modeled from our own measured `ingested_at − published_at` distribution — never at bar-close convenience. An unbacktestable news signal is a story, not a signal.

## 1. Assumptions (surfaced, and verified where possible)

- **[VERIFIED 2026-07-17]** The existing Alpaca account serves Benzinga news via `alpaca-py` 0.43.2: real-time (headline observed 19 min old, incl. geopolitics tagged to SPY/USO) and historical to ≥ 2022-01 — matching the bar cache range. Free, symbol-tagged, second-precision. The data layer costs nothing new.
- **Retail latency is seconds-to-minutes, not milliseconds.** Headline-race alpha (the first 1–2 bars after a shock) belongs to professional feeds and is assumed CAPTURED BY OTHERS before we see it. What can survive at our latency: post-publication drift/underreaction at 15 min–hours (aligned with the §9.4 horizon-extension direction) and defensive standing-aside. If N1 shows otherwise, update this.
- **Speeches are headlines.** "Trump/China says X" reaches the feed as headlines within minutes; a separate transcript/speech pipeline adds complexity for information that arrives later and less tradeably. No transcript pipeline — policy-actor coverage = entity/topic flags on headlines.
- **Prior on net-of-cost edge is LOW at 5-min horizons** (the market prices public headlines fast) and MODEST at 15–30 min+. The defensive consumer (N2) needs no alpha at all — only that post-news volatility is elevated, which is near-certain and is measured anyway.

## 2. Phases

### N0 — Point-in-time news archive *(ungated; start now — data accrues while everything else runs)*

`scripts/collect_news.py`: (a) one-time historical backfill for the 13-symbol universe + market-wide, 2022-01 → now; (b) an incremental poller (REST, ~60 s cadence is plenty at our latency assumption) appending to `runs/news/news_YYYYMM.jsonl`. Each row: article id, `published_at`, `ingested_at`, symbols, headline, summary, source, url. Dedup by article id; append-only; no interpretation of any kind in this phase. Runs on the paper-week box (tiny CPU/network footprint) via the same hidden-launcher pattern as the sweep.
**→ verify:** backfill covers ≥ 2022 for the universe with monotone timestamps; the poller's measured `ingested_at − published_at` distribution is reported (this IS the latency model N1/N2 use); a torn-line-tolerant loader with tests, matching the ledger style.

### N1 — Event study, offline, BEFORE any runtime wiring *(the §7.0.2 pattern: one script, one table, one go/no-go)*

`scripts/audit_news_events.py` over the archive × the bar cache: classify articles by CHEAP deterministic buckets first (topic keywords: rates/tariffs/geopolitics/earnings/legal/product; plus FinBERT headline sentiment — a pinned ~110M local model, deterministic, no LLM yet). For each bucket × symbol: forward |move| and signed move at H ∈ {1, 3, 6, 15, 30, 78} bars after `published_at + modeled_latency`, against a matched no-news baseline, with the 6 bps cost line drawn exactly as in `HORIZON_COST_AUDIT.md`.
**→ verify:** `docs/analysis/NEWS_EVENT_AUDIT.md` with the table and an explicit go/no-go PER CONSUMER: (a) "post-news vol is elevated at our latency" → gates N2; (b) "any bucket shows signed drift clearing costs at any H" → gates N3. **Kill rule:** if (a) fails, N2 dies; if (b) fails, N3/N4 die; the archive keeps running regardless (it is nearly free and the dataset appreciates).

### N2 — Defensive consumer: event-risk gate *(gate: N1(a); shrink-only, the §6.2/§7.2.2 rails pattern)*

On fresh high-severity news for a symbol (or market-wide bucket), the risk layer blocks/shrinks NEW risk-increasing trades for that symbol for a cooldown window. Exits, the position guard, and halts are untouched — de-risking is never blocked. This consumer needs zero alpha: it monetizes only "don't add exposure into a volatility spike you cannot price." Config-gated (`risk.news_gate_*`), default off; evaluated by replaying the archive against the backtest (gate on/off) on the shared OOS window.
**→ verify:** paired backtest rows: adverse excursion around event windows drops with net expectancy not degrading; the §7.2.6 adverse-fill stress run improves or holds. Live wiring only after the backtest verdict, and it enters the paper week as an observed-first feature like every other rail.

### N3 — Offensive consumer: news FEATURES through the contract *(gate: N1(b) go + the master plan's current §9.4 arms resolved)*

Per (symbol, bar): decayed sentiment aggregates, topic one-hots, article count/novelty vs trailing baseline, time-since-last-event. Enters exactly like `market_context` did: feature-contract bump, ⟲ retrain, **3-seed paired protocol vs a no-news baseline on OOS net expectancy** (§2.0 verbatim — and note the seed-ensemble side-finding: expectancy noise dwarfs BA noise, so the bar is high and the metric is money, not accuracy). Extractor = the same pinned FinBERT + keyword buckets from N1 — deliberately boring.
**→ verify:** the §2.0 recipe on OOS net expectancy; promotion only through the normal gate.

### N4 — Local LLM as a BETTER EXTRACTOR *(gate: N3 shows the cheap extractor moves the needle; never before)*

Only if boring features show signal does nuance earn a chance: a local instruction model (7–8B class, fits the 4070 Ti) replacing FinBERT as the text→features function. Constraints: temperature 0; model weights hash + prompt version pinned INTO the feature fingerprint (a prompt edit = a feature-contract event); structured JSON output validated against the same feature schema; hard timeout with FinBERT fallback; **output is the feature vector — fields like action/buy/sell do not exist in the schema.** The schema MAY include a **directional-expectation column** (`direction_score ∈ [−1, +1]` plus the LLM's stated strength) — the "probabilistically up/down/stay" reading — but it enters ONLY as a feature the trained model consumes and weighs (and may learn to ignore). It never reaches the EV gate or sizing directly: an LLM's stated probability is not a calibrated frequency; calibration happens where it is measurable — the trained model's output plus Platt fitted on realized outcomes. No cloud LLM APIs on any hot path (cost, latency, nondeterminism, and headlines are third-party content we don't re-ship). Evidence caveat to carry into the design: published LLM-direction-score results are daily-horizon and pre-cost — this pairs with the horizon-extension direction, not with 5-minute bars.
**→ verify:** determinism check (same article → identical features across runs); latency p99 < bar period; the same 3-seed paired protocol, FinBERT-features vs LLM-features, on OOS net expectancy. **Kill rule:** LLM features that don't beat FinBERT features under the protocol are deleted, not tuned.

## 3. Do-NOT list (binding, additive to ROADMAP §5)

- No LLM emitting actions, sizes, tickers-to-trade, or veto/approve decisions — at any phase, ever.
- No scraping fragile/unlicensed sources as load-bearing inputs; the licensed API feed is the substrate.
- No transcript/speech pipeline (headlines carry the tradeable content at our latency).
- No news feature without the point-in-time latency model; no backtest that shows an article earlier than measured ingestion allows.
- No cloud LLM calls on the decision path.

## 4. Cost & sequencing honesty

| Phase | Build cost | Compute | Blocks on |
|---|---|---|---|
| N0 archive | ~1 day | negligible | nothing — can start now |
| N1 event study | ~1–2 days | CPU + FinBERT (trivial) | N0 backfill |
| N2 event-risk gate | ~2–3 days | backtest replays | N1(a) |
| N3 news features | ⟲-retrain program (~1 wk with protocol) | 3-seed GPU retrains | N1(b) + current §9.4 arms resolved |
| N4 LLM extractor | ~3–5 days | 7B local inference | N3 positive |

N0 deliberately starts before the current master-plan sequence resolves: archives appreciate with time and cost nothing, so the option is cheap to hold. N2+ must NOT preempt the current §9.4 arms — this program is the concrete form of pivot option (b), entered on the §9 step-5 stopping rule or after the current arms resolve, whichever comes first.
