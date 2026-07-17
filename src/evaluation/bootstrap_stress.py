"""Bootstrap-only synthetic stress (MASTER_PLAN §7.2.6).

Regime-conditioned SESSION-block bootstrap: synthetic price paths are
stitched from whole real trading sessions sampled (with replacement)
from a target regime cluster's session pool (§7.2.2 labels).  No
generative models of any kind (do-NOT #2: no GANs at all).

Design constraints, in order:
- Within-session structure is REAL by construction (whole sessions are
  copied), so intraday stylized facts survive verbatim.
- Every symbol receives the SAME sampled session sequence, preserving
  cross-symbol correlation.
- Sessions are rebased at each join (next session's prices scaled so its
  first open continues from the previous stitched close) — splices can
  therefore never create price jumps.
- The stylized-facts detector gates the OUTPUT: flags inherited from the
  source data (same symbol, same bar return — e.g. real crash bars in a
  stress cluster) are allowed; any flag NOT present in the source is a
  splice artifact and raises.

Outputs are informational stress inputs (adverse-fill backtests,
promotion stress deltas) — NEVER a promotion criterion.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np

from src.core.logging import get_logger
from src.data.schemas import MarketBar

# Synthetic timeline base: clearly outside real history so bootstrapped
# bars can never be mistaken for (or merged with) real data. A Monday —
# the stitcher only advances across weekdays.
_SYNTHETIC_BASE_DATE = date(2030, 1, 7)

_MIN_SESSION_BARS = 12


class BootstrapStressError(RuntimeError):
    """Raised when the bootstrapped path fails the stylized-facts gate."""


def _sessions_by_date(bars: list[MarketBar]) -> dict[date, list[MarketBar]]:
    by_day: dict[date, list[MarketBar]] = {}
    for bar in bars:
        by_day.setdefault(bar.timestamp.date(), []).append(bar)
    return {
        day: sorted(day_bars, key=lambda b: b.timestamp)
        for day, day_bars in by_day.items()
        if len(day_bars) >= _MIN_SESSION_BARS
    }


def _next_weekday(day: date) -> date:
    day = day + timedelta(days=1)
    while day.weekday() >= 5:
        day = day + timedelta(days=1)
    return day


def generate_bootstrap_paths(
    bars_by_symbol: dict[str, list[MarketBar]],
    labels_by_date: dict[str, int],
    *,
    target_clusters: set[int],
    n_sessions: int,
    seed: int,
) -> dict[str, list[MarketBar]]:
    """Stitch a synthetic multi-symbol path from target-regime sessions.

    The session pool is the set of dates labeled with a target cluster
    that every symbol has a full session for; ``n_sessions`` dates are
    drawn with replacement (seeded) and the SAME sequence is applied to
    all symbols.  Raises BootstrapStressError when the result introduces
    stylized-facts flags absent from the source data.
    """
    logger = get_logger("bootstrap_stress")
    sessions = {
        symbol: _sessions_by_date(bars) for symbol, bars in bars_by_symbol.items()
    }
    pool: list[date] = []
    for day_iso, label in labels_by_date.items():
        if int(label) not in target_clusters:
            continue
        day = date.fromisoformat(day_iso)
        if all(day in by_day for by_day in sessions.values()):
            pool.append(day)
    pool.sort()
    if not pool:
        raise BootstrapStressError(
            f"No sessions shared by all symbols carry a target cluster {sorted(target_clusters)}"
        )
    rng = np.random.default_rng(seed)
    sampled = [pool[int(i)] for i in rng.integers(0, len(pool), size=n_sessions)]

    paths: dict[str, list[MarketBar]] = {}
    for symbol, by_day in sessions.items():
        stitched: list[MarketBar] = []
        scale = 1.0
        synthetic_day = _SYNTHETIC_BASE_DATE
        for source_day in sampled:
            day_bars = by_day[source_day]
            first_open = float(day_bars[0].open)
            if stitched:
                scale = float(stitched[-1].close) / max(first_open, 1.0e-8)
            for bar in day_bars:
                stitched.append(
                    replace(
                        bar,
                        timestamp=datetime.combine(
                            synthetic_day, bar.timestamp.timetz()
                        ),
                        open=bar.open * scale,
                        high=bar.high * scale,
                        low=bar.low * scale,
                        close=bar.close * scale,
                        spread=0.0,
                    )
                )
            synthetic_day = _next_weekday(synthetic_day)
        paths[symbol] = stitched

    _stylized_facts_gate(bars_by_symbol, paths, logger=logger)
    logger.info(
        "bootstrap_paths_generated",
        symbols=sorted(paths),
        sessions=n_sessions,
        pool_size=len(pool),
        target_clusters=sorted(target_clusters),
    )
    return paths


def _stylized_facts_gate(
    source: dict[str, list[MarketBar]],
    paths: dict[str, list[MarketBar]],
    *,
    logger: object,
) -> None:
    """Allow inherited flags (real crash bars), refuse splice artifacts.

    Session rebasing multiplies a whole session by one scale, so bar
    RETURNS are scale-invariant: a legitimate inherited flag matches a
    source flag on (symbol, rounded return); anything else was created
    by the stitching and fails the gate.
    """
    from src.evaluation.stylized_facts import compute_stylized_facts

    source_report = compute_stylized_facts(source)
    inherited = {
        (symbol, round(float(ret), 6)) for symbol, _, ret in source_report.flagged
    }
    path_report = compute_stylized_facts(paths)
    introduced = [
        (symbol, ts, ret)
        for symbol, ts, ret in path_report.flagged
        if (symbol, round(float(ret), 6)) not in inherited
    ]
    if introduced:
        raise BootstrapStressError(
            f"Bootstrap stitching introduced stylized-facts flags: {introduced[:5]}"
        )
    if path_report.flagged:
        logger.info(
            "bootstrap_inherited_flags",
            count=len(path_report.flagged),
            note="real crash bars carried over from the source sessions — allowed",
        )
