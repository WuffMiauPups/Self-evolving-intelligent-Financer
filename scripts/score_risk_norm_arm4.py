"""Aggregate MASTER_PLAN 9.4 arm-4 (risk-target per-symbol norm) EV audits.

Reads the per-checkpoint JSON reports written by scripts/audit_ev_inputs.py
for the knob-OFF baseline (arm-3 H=15 checkpoints) and the knob-ON candidate,
seed-paired, and reports whether the PER-SYMBOL Spearman moved off ~0 — the
named metric (7.0.4 warns pooled inflates on cross-sectional vol, so only
per-symbol counts).

Usage:
    python scripts/score_risk_norm_arm4.py \
        --off runs/ev_audit_arm4_off_seed7.json runs/..._seed13.json runs/..._seed29.json \
        --on  runs/ev_audit_arm4_on_seed7.json  runs/..._seed13.json runs/..._seed29.json \
        --seeds 7 13 29
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _per_symbol_medians(report: dict) -> tuple[float, float, int, int]:
    """(median gain/MFE, median loss/MAE, n_symbols, n with both > 0.05)."""
    ps = report.get("per_symbol", {})
    gains = [v["spearman_gain_vs_mfe"] for v in ps.values() if v["spearman_gain_vs_mfe"] == v["spearman_gain_vs_mfe"]]
    losses = [v["spearman_loss_vs_mae"] for v in ps.values() if v["spearman_loss_vs_mae"] == v["spearman_loss_vs_mae"]]
    strong = sum(1 for v in ps.values()
                 if v["spearman_gain_vs_mfe"] > 0.05 and v["spearman_loss_vs_mae"] > 0.05)
    return (statistics.median(gains) if gains else float("nan"),
            statistics.median(losses) if losses else float("nan"),
            len(ps), strong)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate arm-4 EV audits")
    parser.add_argument("--off", nargs="+", required=True, help="knob-OFF report JSONs, seed-ordered")
    parser.add_argument("--on", nargs="+", required=True, help="knob-ON report JSONs, seed-ordered")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29])
    args = parser.parse_args()
    assert len(args.off) == len(args.on) == len(args.seeds), "off/on/seeds length mismatch"

    print("\n" + "=" * 92)
    print("ARM-4 PER-SYMBOL EV-AUDIT SPEARMAN  (named metric: per-symbol correlation moving OFF ~0)")
    print("=" * 92)
    print(f"{'seed':>4} | {'median gain/MFE':>26} | {'median loss/MAE':>26} | {'#sym both>0.05':>16}")
    print(f"{'':>4} | {'OFF':>8} {'ON':>8} {'d':>8} | {'OFF':>8} {'ON':>8} {'d':>8} | {'OFF':>7} {'ON':>7}")
    gain_deltas, loss_deltas = [], []
    for off_path, on_path, seed in zip(args.off, args.on, args.seeds):
        off = json.loads(Path(off_path).read_text())
        on = json.loads(Path(on_path).read_text())
        og, ol, on_nsym, os_strong = _per_symbol_medians(off)
        ng, nl, nn_sym, ns_strong = _per_symbol_medians(on)
        dg, dl = ng - og, nl - ol
        gain_deltas.append(dg)
        loss_deltas.append(dl)
        print(f"{seed:>4} | {og:>8.4f} {ng:>8.4f} {dg:>+8.4f} | "
              f"{ol:>8.4f} {nl:>8.4f} {dl:>+8.4f} | {os_strong:>7} {ns_strong:>7}")
        # pooled for context only
        print(f"{'':>4} |   pooled OFF gain {off['pooled']['spearman_gain_vs_mfe']:+.3f} "
              f"loss {off['pooled']['spearman_loss_vs_mae']:+.3f}  ->  ON gain "
              f"{on['pooled']['spearman_gain_vs_mfe']:+.3f} loss {on['pooled']['spearman_loss_vs_mae']:+.3f}")

    def _verdict(deltas, name):
        pos = sum(1 for x in deltas if x > 0)
        mean = sum(deltas) / len(deltas)
        signs = "".join("+" if x > 0 else "-" for x in deltas)
        tag = ("SUGGESTIVE UP (3/3)" if pos == 3 else
               "SUGGESTIVE FLAT/DOWN (0/3)" if pos == 0 else
               "INCONCLUSIVE (mixed)")
        return f"  {name}: signs={signs} mean d={mean:+.4f} ({pos}/3 up) -> {tag}"

    print("\nPer-symbol median Spearman, knob-ON minus knob-OFF, paired per seed:")
    print(_verdict(gain_deltas, "gain/MFE"))
    print(_verdict(loss_deltas, "loss/MAE"))
    print("\n(Named metric is per-symbol moving OFF ~0. 3/3 same-sign up = suggestive; a real")
    print(" risk-head-quality claim wants the per-symbol median clearly positive, not pooled.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
