"""Path-label oracle test — decide whether the 5-class path_label head is worth keeping.

Runs a data-side analysis (no trained model required): for each bar in the
dataset, compute the ORACLE path_label (from ground-truth future returns via
``classify_path_label``) and the direction label. Then check whether knowing
the path_label adds directional information.

Decision rule:
- If ``chop`` dominates heavily (>60% of labels) and minority classes are
  <5%, the labeling scheme itself is skewed — the head will collapse to
  majority regardless. Recommend killing or recalibrating path_label_rules.
- If path_label values cleanly partition direction (e.g., ``breakout_up``
  is >90% LONG in realized outcomes), the head has real information. A
  3-class simplification (bearish/neutral/bullish) would be more efficient.
- If the cross-tab is flat (path_label tells you nothing about direction
  beyond chance), kill the head.

Usage:
    python scripts/path_label_oracle_test.py [--bars 5000] [--symbol AAPL]
    python scripts/path_label_oracle_test.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.memory.outcomes import compute_outcome_summary
from src.training.targets import PATH_LABEL_VOCABULARY

_DIRECTIONS = ("short", "flat", "long")


def _direction_name(primary_return: float, flat_threshold: float) -> str:
    if primary_return > flat_threshold:
        return "long"
    if primary_return < -flat_threshold:
        return "short"
    return "flat"


def analyze(bars, horizons, path_label_rules, flat_threshold: float) -> dict:
    """Walk every bar where a full outcome is computable; build labels."""
    horizons = list(horizons)
    max_horizon = max(horizons)
    path_counter: Counter[str] = Counter()
    direction_counter: Counter[str] = Counter()
    joint_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for index in range(len(bars) - max_horizon - 1):
        try:
            outcome = compute_outcome_summary(
                bars,
                index,
                horizons=horizons,
                path_label_rules=path_label_rules,
            )
        except (IndexError, ValueError):
            continue
        path_label = outcome.path_label
        if path_label not in PATH_LABEL_VOCABULARY:
            path_label = "chop"
        direction_label = _direction_name(outcome.primary_return, flat_threshold)
        path_counter[path_label] += 1
        direction_counter[direction_label] += 1
        joint_counter[path_label][direction_label] += 1

    return {
        "path_counts": dict(path_counter),
        "direction_counts": dict(direction_counter),
        "joint_counts": {k: dict(v) for k, v in joint_counter.items()},
    }


def _format_table(results: dict) -> str:
    total = sum(results["path_counts"].values())
    out: list[str] = []
    out.append(f"Total samples: {total}")
    out.append("")
    out.append("Path label distribution:")
    for name in PATH_LABEL_VOCABULARY:
        count = results["path_counts"].get(name, 0)
        pct = count / total * 100 if total else 0.0
        out.append(f"  {name:22s} {count:>7d}  {pct:5.1f}%")

    out.append("")
    out.append("Direction distribution:")
    for d in _DIRECTIONS:
        count = results["direction_counts"].get(d, 0)
        pct = count / total * 100 if total else 0.0
        out.append(f"  {d:6s} {count:>7d}  {pct:5.1f}%")

    out.append("")
    out.append("P(direction | path_label) -- oracle cross-tab:")
    out.append(f"  {'path_label':22s}  {'short':>7s}  {'flat':>7s}  {'long':>7s}  {'n':>7s}")
    for name in PATH_LABEL_VOCABULARY:
        row_total = results["path_counts"].get(name, 0)
        if row_total == 0:
            out.append(f"  {name:22s}  {'-':>7s}  {'-':>7s}  {'-':>7s}  {0:>7d}")
            continue
        joint = results["joint_counts"].get(name, {})
        short_pct = joint.get("short", 0) / row_total
        flat_pct = joint.get("flat", 0) / row_total
        long_pct = joint.get("long", 0) / row_total
        out.append(
            f"  {name:22s}  {short_pct:>7.2%}  {flat_pct:>7.2%}  {long_pct:>7.2%}  {row_total:>7d}"
        )
    return "\n".join(out)


def _verdict(results: dict) -> str:
    total = sum(results["path_counts"].values())
    if total == 0:
        return "VERDICT: no samples — inconclusive."

    chop_share = results["path_counts"].get("chop", 0) / total
    minority_shares = [
        results["path_counts"].get(name, 0) / total
        for name in ("breakout_down", "breakout_up")
    ]

    # Informativeness: largest P(direction | path_label) for directional paths.
    def _directional_prob(name: str, expected_direction: str) -> float:
        row_total = results["path_counts"].get(name, 0)
        if row_total == 0:
            return 0.0
        return results["joint_counts"].get(name, {}).get(expected_direction, 0) / row_total

    p_long_given_breakout_up = _directional_prob("breakout_up", "long")
    p_short_given_breakout_down = _directional_prob("breakout_down", "short")

    lines: list[str] = []
    lines.append("")
    lines.append("VERDICT:")
    lines.append(
        f"  chop share = {chop_share:.1%};  "
        f"breakout_down = {minority_shares[0]:.1%};  breakout_up = {minority_shares[1]:.1%}"
    )
    lines.append(
        f"  P(long | breakout_up)   = {p_long_given_breakout_up:.2%}"
    )
    lines.append(
        f"  P(short | breakout_down) = {p_short_given_breakout_down:.2%}"
    )
    if chop_share > 0.60 and min(minority_shares) < 0.05:
        lines.append(
            "  ->Path labels are collapsing to 'chop'. Head will learn the "
            "majority and produce noise elsewhere. KILL the head OR tighten "
            "breakout_return_threshold in memory.path_label_rules."
        )
    elif p_long_given_breakout_up > 0.80 and p_short_given_breakout_down > 0.80:
        lines.append(
            "  ->Path labels carry real directional information. Keep the "
            "head; consider 3-class simplification (bearish / neutral / "
            "bullish) to reduce minority-class noise."
        )
    else:
        lines.append(
            "  ->Path labels only weakly predict direction. Head learns "
            "noise or overlaps the direction classifier. Candidate for KILL."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--bars", type=int, default=5000, help="Synthetic bars to analyze")
    args = parser.parse_args()

    config = load_config(args.config)
    horizons = list(config.memory.forward_horizons)
    rules = dict(config.memory.path_label_rules)
    flat_threshold = float(rules.get("flat_return_threshold", 0.00075))

    bars = generate_synthetic_bars(args.symbol, args.bars)
    results = analyze(bars, horizons, rules, flat_threshold)
    print(_format_table(results))
    print(_verdict(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
