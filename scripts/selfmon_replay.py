"""Replay the self-monitor over a banked ledger day and print its verdict.

Validates the L0 monitor against real history before it ever runs live:
"would it have flagged what we found by hand?" It reconstructs a day from
runs/decisions.jsonl + runs/fills.jsonl following the two binding ledger
rules (sort fills by timestamp, seed the start-of-day book;
CONFIDENCE_AUDIT.md §7).

Examples:
    # 2026-07-20 was pre-fix (confidence pinned ~0.49) -> expect S3.
    python scripts/selfmon_replay.py --day 2026-07-20
    # A post-fix day, with the start-of-day book seeded from the watcher seed.
    python scripts/selfmon_replay.py --day 2026-07-22 --seed logs/.open_watch_seed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config.monitoring import SelfMonitorConfig
from src.monitoring.self_monitor import SelfMonitor


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the self-monitor over a ledger day")
    parser.add_argument("--day", required=True, help="YYYY-MM-DD (matches the ts prefix)")
    parser.add_argument("--decisions", default="runs/decisions.jsonl")
    parser.add_argument("--fills", default="runs/fills.jsonl")
    parser.add_argument("--seed", default=None, help="start-of-day book JSON (symbol->qty)")
    args = parser.parse_args()

    seed = None
    if args.seed and Path(args.seed).exists():
        seed = {k: float(v) for k, v in json.loads(Path(args.seed).read_text(encoding="utf-8")).items()}

    monitor = SelfMonitor.replay_from_ledgers(
        decisions_path=args.decisions,
        fills_path=args.fills,
        day=args.day,
        config=SelfMonitorConfig(enabled=True),
        seed_positions=seed,
    )
    report = monitor.health(now=f"{args.day}T20:00:00+00:00")

    print(f"\n  SELF-MONITOR REPLAY — {args.day}")
    print(f"    decisions {report.decisions}  resolved {report.resolved}")
    print(f"    win_rate {report.win_rate}  z {report.win_rate_z}")
    print(f"    confidence p10-p90 {report.conf_p10} - {report.conf_p90}")
    print(f"    realized_slippage_bps {report.realized_slippage_bps}")
    if report.calibration:
        print("    calibration:")
        for bucket, stats in report.calibration.items():
            print(f"      {bucket}: n={stats['n']} claims {stats['claimed']} wins {stats['realized']}")
    print(f"    FLAGS: {report.flags or '(none)'}")
    print(f"    damping_multiplier {report.damping_multiplier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
