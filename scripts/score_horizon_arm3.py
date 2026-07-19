"""Score MASTER_PLAN 9.4 arm-3 (horizon extension): H=15 vs H=30.

Benchmarks the three H=15 and three H=30 sweep checkpoints strictly out of
sample, each at ITS OWN native horizon (the benchmark holds
effective_primary_horizon bars), on ONE shared OOS window, then pairs per
seed and reports the named metric — OOS net expectancy — under the 2.0
three-seed protocol.

Usage:
    python scripts/score_horizon_arm3.py \
        --h15-dir checkpoints/sweep_YYYYMMDD_HHMMSS \
        --h30-dir checkpoints/sweep_YYYYMMDD_HHMMSS \
        [--seeds 7 13 29] [--append-scorecard]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.core.logging import configure_logging  # noqa: E402

# Full-data noise floor (2.0), for the "clears the floor?" comparison.
VAL_BA_FLOOR = 0.0092  # 0.92 pt val balanced-accuracy spread


def _matching_config(arm: str):
    stack = ["configs/default.yaml", "configs/training.yaml"]
    if arm == "h30":
        stack.append("configs/experiments/horizon_30.yaml")
    return load_config(*stack)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score arm-3 horizon extension")
    parser.add_argument("--h15-dir", required=True, help="Sweep dir with the 3 H=15 checkpoints")
    parser.add_argument("--h30-dir", required=True, help="Sweep dir with the 3 H=30 checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29],
                        help="Seed order matching run001/run002/run003")
    parser.add_argument("--append-scorecard", action="store_true",
                        help="Append the 6 rows to docs/analysis/SCORECARD.md")
    parser.add_argument(
        "--h30-hold-bars", type=int, default=0,
        help=(
            "7.1.6 combined arm: hold the H=30 candidate's positions for this "
            "many bars (sets risk.horizon_hold_sec = N x backtest.seconds_per_bar "
            "on the h30 config only). 0 = plain arm-3 scoring."
        ),
    )
    args = parser.parse_args()

    base = _matching_config("h15")
    configure_logging(base.logging.level, base.logging.json_logs,
                      file_enabled=False, file_path=base.logging.file_path)

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_backtest import _load_bars_by_symbol  # noqa: E402
    from src.evaluation.benchmark import (  # noqa: E402
        DEFAULT_SCORECARD_PATH,
        append_scorecard_row,
        run_benchmark,
    )
    from src.evaluation.oos import checkpoint_training_end  # noqa: E402

    # Bars are horizon-independent (same universe) — load once.
    bars_by_symbol = _load_bars_by_symbol(base, use_real_data=True, allow_mock=False)

    arms = {
        "h15": Path(args.h15_dir),
        "h30": Path(args.h30_dir),
    }
    # One shared OOS window for all six -> identical evaluation bars, so the
    # only thing that varies within a seed-pair is the horizon.
    ends = []
    for d in arms.values():
        for i in (1, 2, 3):
            ck = d / f"brain_run00{i}.pt"
            e = checkpoint_training_end(ck)
            if e is not None:
                ends.append(e)
    shared_cutoff = max(ends) if ends else None
    print(f"\nShared OOS cutoff (max train end across all 6): {shared_cutoff}\n")

    rows: dict[str, dict[int, object]] = {"h15": {}, "h30": {}}
    for arm, d in arms.items():
        cfg = _matching_config(arm)
        if arm == "h30" and args.h30_hold_bars > 0:
            cfg.risk.horizon_hold_sec = float(
                args.h30_hold_bars * cfg.backtest.seconds_per_bar
            )
            print(f"[h30] horizon hold ON: {args.h30_hold_bars} bars "
                  f"= {cfg.risk.horizon_hold_sec:.0f}s")
        horizon = cfg.training.effective_primary_horizon
        for idx, seed in enumerate(args.seeds, start=1):
            ck = d / f"brain_run00{idx}.pt"
            run_cfg = cfg.model_copy(deep=True)
            run_cfg.model.checkpoint_path = str(ck)
            row = run_benchmark(run_cfg, bars_by_symbol, oos_cutoff=shared_cutoff)
            rows[arm][seed] = row
            print(f"[{arm} H={horizon:>2} seed={seed:>2}] "
                  f"expectancy={row.expectancy:+.4f}  BA={row.balanced_accuracy:.4f}  "
                  f"trades={row.trade_count}  excess_vs_bh={row.excess_vs_buy_hold:+.4f}  "
                  f"sharpe={row.sharpe:+.3f}")
            if args.append_scorecard:
                append_scorecard_row(row, path=DEFAULT_SCORECARD_PATH)

    # ---- Paired analysis (named metric: OOS net expectancy) ----
    print("\n" + "=" * 78)
    print("ARM-3 PAIRED RESULT  (H=30 candidate  -  H=15 baseline), per seed")
    print("=" * 78)
    exp_deltas, ba_deltas = [], []
    print(f"{'seed':>5} | {'exp H15':>9} {'exp H30':>9} {'Δexp':>9} | "
          f"{'BA H15':>7} {'BA H30':>7} {'ΔBA':>8}")
    for seed in args.seeds:
        r15, r30 = rows["h15"].get(seed), rows["h30"].get(seed)
        if r15 is None or r30 is None:
            print(f"{seed:>5} | MISSING checkpoint(s) — cannot pair")
            continue
        de = r30.expectancy - r15.expectancy
        db = r30.balanced_accuracy - r15.balanced_accuracy
        exp_deltas.append(de)
        ba_deltas.append(db)
        print(f"{seed:>5} | {r15.expectancy:>+9.4f} {r30.expectancy:>+9.4f} {de:>+9.4f} | "
              f"{r15.balanced_accuracy:>7.4f} {r30.balanced_accuracy:>7.4f} {db:>+8.4f}")

    def _verdict(deltas, floor_note):
        if len(deltas) < 3:
            return "INCOMPLETE (need 3 pairs)"
        pos = sum(1 for x in deltas if x > 0)
        mean = sum(deltas) / len(deltas)
        signs = "".join("+" if x > 0 else "-" for x in deltas)
        line = f"signs={signs}  mean_delta={mean:+.4f}  ({pos}/3 positive)"
        if pos == 3:
            line += "  -> SUGGESTIVE (3/3, p=0.125)" + floor_note
        elif pos == 0:
            line += "  -> SUGGESTIVE NEGATIVE (0/3)"
        else:
            line += "  -> INCONCLUSIVE (mixed signs, within noise)"
        return line

    print("\nNamed metric — OOS net expectancy (H=30 - H=15):")
    print("  " + _verdict(exp_deltas, ""))
    print("Secondary — balanced accuracy (H=30 - H=15):")
    ba_mean = sum(ba_deltas) / len(ba_deltas) if ba_deltas else 0.0
    floor_note = (f"  [mean ΔBA {ba_mean:+.4f} vs {VAL_BA_FLOOR:.4f} floor: "
                  f"{'CLEARS' if abs(ba_mean) > VAL_BA_FLOOR else 'under floor'}]")
    print("  " + _verdict(ba_deltas, floor_note))
    print("\n(2.0: 3/3 same-sign paired deltas = 'suggestive'; a promotion still "
          "needs the mean delta to exceed the measured floor AND an OOS scorecard win.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
