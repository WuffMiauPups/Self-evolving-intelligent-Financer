"""Regime snapshots over the full bar history (MASTER_PLAN §7.2.2, offline half).

Fits the per-session K-means regime snapshot on the exact bars the
training path loads, writes the snapshot artifact, and prints the two
§7.2.2 verification views:

1. Separation: per-year cluster composition — 2022 (bear) and 2024
   (bull) sessions must not share a dominant cluster.
2. Novelty: a second snapshot fit EXCLUDING the last ~90 sessions
   scores the held-out tail; the most-novel sessions are listed.

Usage:
    python scripts/build_regime_snapshots.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger

DEFAULT_REPORT_PATH = Path("runs/regime_snapshots.json")
HOLDOUT_SESSIONS = 90


def main() -> int:
    parser = argparse.ArgumentParser(description="Build regime snapshots (§7.2.2)")
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT_PATH), help="Where to write the snapshot JSON"
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml", "configs/training.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("build_regime_snapshots")

    from src.evaluation.regime_snapshots import (
        build_regime_snapshot,
        build_session_features,
    )
    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    kwargs = dict(
        k=int(config.evaluation.regime_snapshot_k),
        seed=int(config.evaluation.seed),
        short_window=int(config.features.windows["short"]),
        medium_window=int(config.features.windows["medium"]),
        long_window=int(config.features.windows["long"]),
        epsilon=float(config.features.epsilon),
    )
    snapshot = build_regime_snapshot(bars_by_symbol, **kwargs)
    if snapshot is None:
        logger.warning("regime_snapshot_not_built")
        return 2

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")

    labels = snapshot.labels_by_date
    print(f"\n  REGIME SNAPSHOT  (k={snapshot.k}, {len(labels)} sessions)")
    print("\n  Cluster composition by year (sessions):")
    years = sorted({day[:4] for day in labels})
    header = "  year | " + " | ".join(f"r{c:<4}" for c in range(snapshot.k))
    print(header)
    dominant: dict[str, int] = {}
    for year in years:
        counts = Counter(label for day, label in labels.items() if day.startswith(year))
        dominant[year] = counts.most_common(1)[0][0] if counts else -1
        row = " | ".join(f"{counts.get(c, 0):<5}" for c in range(snapshot.k))
        print(f"  {year} | {row}")
    separation_ok = dominant.get("2022", -1) != dominant.get("2024", -2)
    print(
        f"\n  2022-bear dominant cluster: r{dominant.get('2022')}  vs  "
        f"2024-bull dominant cluster: r{dominant.get('2024')}  -> "
        f"{'SEPARATED' if separation_ok else 'NOT SEPARATED'}"
    )

    # Held-out novelty: fit excluding the last N sessions, score the tail.
    all_dates = sorted(labels)
    cut = all_dates[-HOLDOUT_SESSIONS]
    train_bars = {
        symbol: [bar for bar in bars if bar.timestamp.date().isoformat() < cut]
        for symbol, bars in bars_by_symbol.items()
    }
    holdout_bars = {
        symbol: [bar for bar in bars if bar.timestamp.date().isoformat() >= cut]
        for symbol, bars in bars_by_symbol.items()
    }
    train_snapshot = build_regime_snapshot(train_bars, **kwargs)
    if train_snapshot is not None:
        dates, features = build_session_features(
            holdout_bars,
            short_window=kwargs["short_window"],
            medium_window=kwargs["medium_window"],
            long_window=kwargs["long_window"],
            epsilon=kwargs["epsilon"],
        )
        scored = [(day, train_snapshot.score(features[i])) for i, day in enumerate(dates)]
        novel = [(day, s) for day, s in scored if s.novel]
        print(
            f"\n  Held-out novelty (fit < {cut}, scored {len(scored)} sessions, "
            f"threshold {train_snapshot.novelty_threshold:.3f}): {len(novel)} novel"
        )
        for day, s in sorted(scored, key=lambda t: -t[1].distance)[:5]:
            marker = "NOVEL" if s.novel else "     "
            print(f"    {marker} {day}  cluster r{s.cluster}  distance {s.distance:.3f}")

    print(f"\n  Snapshot -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
