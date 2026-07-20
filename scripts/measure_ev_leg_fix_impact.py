"""Impact measurement for the short-side EV-leg fix (§7.0.4 addendum).

Runs the incumbent's OOS backtest and reports what the EV-leg semantics
produce: fills by side, short-position participation, per-side pre-cost EV
pass shares, and the headline backtest metrics. Run once BEFORE the fix and
once AFTER (same window, same checkpoint) to get the paired delta the fix
task requires. Measurement only — nothing is promoted or tuned.

Usage:
    python scripts/measure_ev_leg_fix_impact.py --real-data --out runs/ev_leg_fix_pre.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="EV-leg fix impact measurement")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("ev_leg_fix_impact")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path or not args.real_data:
        logger.warning("impact_requires_checkpoint_and_real_data")
        return 2
    config.device.preferred = "cpu"

    import torch

    from src.backtest.engine import BacktestEngine
    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.oos import enforce_out_of_sample
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.training.data_loading import _resolve_bars_by_symbol

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

    # Per-side proposal counts + pre-cost EV pass shares from the decision log
    # (legs exactly as the gate consumed them in THIS code state).
    sides = {}
    for action in (+1, -1):
        rows = [r for r in report.decision_log if int(r.get("action", 0)) == action]
        if rows:
            conf = np.array([float(r["confidence"]) for r in rows])
            gain = np.array([float(r.get("expected_gain_bps", 0.0)) for r in rows])
            loss = np.array([float(r.get("expected_loss_bps", 0.0)) for r in rows])
            ev = conf * gain - (1.0 - conf) * loss
            sides["long" if action == 1 else "short"] = {
                "proposals": len(rows),
                "mean_gain_bps": round(float(gain.mean()), 2),
                "mean_loss_bps": round(float(loss.mean()), 2),
                "precost_ev_mean_bps": round(float(ev.mean()), 2),
                "precost_ev_positive_share": round(float((ev > 0).mean()), 4),
            }

    # Fills by side + short-position participation (replay per symbol; the
    # backtest book starts flat, so the replay is exact here).
    position: dict[str, float] = {}
    short_opening_fills = 0
    short_closing_realized = 0.0
    short_round_trips = 0
    for fill in report.fills:
        qty = float(fill.quantity) * (1.0 if fill.side == "buy" else -1.0)
        before = position.get(fill.symbol, 0.0)
        after = before + qty
        position[fill.symbol] = after
        if qty < 0 and after < -1e-9 and before <= 1e-9:
            short_opening_fills += 1
        if qty > 0 and before < -1e-9:
            short_closing_realized += float(fill.realized_pnl)
            if after > -1e-9:
                short_round_trips += 1

    buys = sum(1 for f in report.fills if f.side == "buy")
    sells = len(report.fills) - buys
    metrics = report.metrics
    result = {
        "checkpoint": str(config.model.checkpoint_path),
        "oos_meta": {k: str(v) for k, v in (oos_meta or {}).items()},
        "per_side": sides,
        "fills": {
            "total": len(report.fills),
            "buys": buys,
            "sells": sells,
            "short_opening_fills": short_opening_fills,
            "short_round_trips": short_round_trips,
            "short_closing_realized": round(short_closing_realized, 4),
        },
        "metrics": {
            "expectancy": round(float(metrics.get("expectancy", 0.0)), 4),
            "round_trip_count": int(metrics.get("round_trip_count", 0)),
            "total_return": round(float(metrics.get("total_return", 0.0)), 6),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 4),
            "fill_count": int(metrics.get("fill_count", 0)),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
