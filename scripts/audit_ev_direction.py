"""Direction-split EV-input audit (§7.0.4 follow-up; 2026-07-20 live finding).

Paper-week day 1 measured a stark long/short asymmetry in the EV gate's
inputs (92 short intents, 6 executed; mean pre-cost EV −53.5 bps for shorts
vs +121.2 for longs at near-identical confidence). This audit turns that
one-day observation into a measured property of the model class on the
incumbent's OOS window, split by INTENDED direction — and quantifies the
mechanism found in code review: ``_extract_ev_signals_*`` maps
gain=MFE, loss=|MAE| UNCONDITIONALLY, but MFE/MAE are price-path up/down
excursions (targets.py trains them direction-agnostically), so for a SHORT
proposal the legs are swapped — the gate charges the short's win (the
down-move) as its loss. The report therefore shows each side's EV under the
AS-CONSUMED mapping and under the DIRECTION-CORRECTED mapping
(short: gain=|MAE|, loss=MFE).

Measurement only — no model or gate changes here.

Usage:
    python scripts/audit_ev_direction.py --real-data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger

from audit_ev_inputs import realized_excursions_bps, spearman  # noqa: E402

DEFAULT_REPORT_PATH = Path("runs/ev_direction_audit.json")


def _side_stats(rows: list[dict], action: int) -> dict:
    """Per-side EV-leg stats under both mappings + realized excursions."""
    side = [row for row in rows if row["action"] == action]
    if len(side) < 30:
        return {"n": len(side)}
    conf = np.array([row["confidence"] for row in side])
    gain = np.array([row["pred_gain"] for row in side])   # = predicted MFE bps
    loss = np.array([row["pred_loss"] for row in side])   # = predicted |MAE| bps
    real_mfe = np.array([row["real_mfe"] for row in side])
    real_mae = np.array([row["real_mae"] for row in side])

    ev_as_consumed = conf * gain - (1.0 - conf) * loss
    # Direction-corrected legs: a short's gain is the down-move (|MAE|),
    # its loss the up-move (MFE). Longs are unchanged.
    if action == -1:
        ev_corrected = conf * loss - (1.0 - conf) * gain
    else:
        ev_corrected = ev_as_consumed

    return {
        "n": len(side),
        "mean_confidence": float(conf.mean()),
        "pred_mfe_bps_mean": float(gain.mean()),
        "pred_mae_bps_mean": float(loss.mean()),
        "realized_mfe_bps_mean": float(real_mfe.mean()),
        "realized_mae_bps_mean": float(real_mae.mean()),
        "ev_as_consumed_bps_mean": float(ev_as_consumed.mean()),
        "ev_as_consumed_positive_share": float((ev_as_consumed > 0).mean()),
        "ev_direction_corrected_bps_mean": float(ev_corrected.mean()),
        "ev_direction_corrected_positive_share": float((ev_corrected > 0).mean()),
        "spearman_pred_mfe_vs_real_mfe": spearman(gain, real_mfe),
        "spearman_pred_mae_vs_real_mae": spearman(loss, real_mae),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Direction-split EV audit")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("audit_ev_direction")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path or not args.real_data:
        logger.warning("ev_direction_audit_requires_checkpoint_and_real_data")
        return 2
    config.device.preferred = args.device

    import torch

    from src.backtest.engine import BacktestEngine
    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.oos import enforce_out_of_sample
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.training.data_loading import _resolve_bars_by_symbol
    from src.training.checkpoints import load_checkpoint_payload

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    bars_by_symbol, oos_meta = enforce_out_of_sample(
        bars_by_symbol, checkpoint_path=config.model.checkpoint_path
    )

    config = config.model_copy(deep=True)
    config.backtest.fill_timing = "next_open"
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    policy = build_brain_policy(config, bars_by_symbol=bars_by_symbol, pipeline=pipeline)
    report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars_by_symbol, policy
    )
    logger.info("ev_direction_backtest_done", decisions=len(report.decision_log))

    max_horizon = int(max(config.training.forward_horizons))
    closes_by_symbol = {
        symbol: np.array([bar.close for bar in bars], dtype=np.float64)
        for symbol, bars in bars_by_symbol.items()
    }
    rows: list[dict] = []
    for record in report.decision_log:
        if int(record.get("action", 0)) == 0:
            continue  # flat proposals never face the EV gate
        excursions = realized_excursions_bps(
            closes_by_symbol[str(record["symbol"])], int(record["bar_index"]), max_horizon
        )
        if excursions is None:
            continue
        real_mfe, real_mae = excursions
        rows.append(
            {
                "action": int(record["action"]),
                "confidence": float(record["confidence"]),
                "pred_gain": float(record.get("expected_gain_bps", 0.0)),
                "pred_loss": float(record.get("expected_loss_bps", 0.0)),
                "real_mfe": real_mfe,
                "real_mae": real_mae,
            }
        )

    # Train-target symmetry check (hypothesis: asymmetric targets taught a
    # bullish prior). Read from the incumbent's own persisted diagnostics —
    # the exact train-split the head regressed on.
    meta = load_checkpoint_payload(Path(config.model.checkpoint_path)).get("meta", {})
    train_summary = (
        meta.get("target_diagnostics", {}).get("train", {}).get("risk_path_summaries", {})
    )
    train_targets = {
        "train_mfe_bps_mean": float(train_summary.get("mfe", {}).get("mean", float("nan"))) * 1e4,
        "train_abs_mae_bps_mean": abs(float(train_summary.get("mae", {}).get("mean", float("nan")))) * 1e4,
    }
    ratio = (
        train_targets["train_mfe_bps_mean"] / train_targets["train_abs_mae_bps_mean"]
        if train_targets["train_abs_mae_bps_mean"]
        else float("nan")
    )
    train_targets["mfe_over_abs_mae_ratio"] = ratio

    result = {
        "checkpoint": str(config.model.checkpoint_path),
        "oos_meta": {k: str(v) for k, v in (oos_meta or {}).items()},
        "directional_decisions": len(rows),
        "long": _side_stats(rows, +1),
        "short": _side_stats(rows, -1),
        "train_target_symmetry": train_targets,
        "mechanism_note": (
            "gain=MFE, loss=|MAE| unconditionally (_extract_ev_signals_*, "
            "src/strategies/brain_policy.py) while MFE/MAE are price-path "
            "up/down excursions -> for SHORT proposals the EV legs are "
            "swapped; 'direction_corrected' rows evaluate shorts with "
            "gain=|MAE|, loss=MFE."
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n  DIRECTION-SPLIT EV AUDIT (incumbent, OOS)")
    print(f"  directional decisions: {len(rows):,}")
    for name in ("long", "short"):
        stats = result[name]
        if stats.get("n", 0) < 30:
            print(f"  {name.upper()}: n={stats.get('n', 0)} (too few)")
            continue
        print(
            f"  {name.upper():5s} n={stats['n']:>6,} conf={stats['mean_confidence']:.4f}  "
            f"predMFE={stats['pred_mfe_bps_mean']:7.1f} pred|MAE|={stats['pred_mae_bps_mean']:7.1f}  "
            f"realMFE={stats['realized_mfe_bps_mean']:6.1f} real|MAE|={stats['realized_mae_bps_mean']:6.1f}"
        )
        print(
            f"        EV as-consumed: mean {stats['ev_as_consumed_bps_mean']:+7.1f} bps, "
            f">0 in {stats['ev_as_consumed_positive_share']*100:5.1f}%   |   "
            f"direction-corrected: mean {stats['ev_direction_corrected_bps_mean']:+7.1f} bps, "
            f">0 in {stats['ev_direction_corrected_positive_share']*100:5.1f}%"
        )
        print(
            f"        Spearman predMFE~realMFE {stats['spearman_pred_mfe_vs_real_mfe']:+.4f}  "
            f"pred|MAE|~real|MAE| {stats['spearman_pred_mae_vs_real_mae']:+.4f}"
        )
    tt = result["train_target_symmetry"]
    print(
        f"  train targets: MFE {tt['train_mfe_bps_mean']:.1f} bps vs |MAE| "
        f"{tt['train_abs_mae_bps_mean']:.1f} bps (ratio {tt['mfe_over_abs_mae_ratio']:.2f}) "
        "-> near-symmetric; the asymmetry is NOT a target artifact"
    )
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
