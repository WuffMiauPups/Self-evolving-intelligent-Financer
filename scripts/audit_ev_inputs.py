"""EV-input quality audit (MASTER_PLAN §7.0.4).

The perturbation and cap-sweep results suggest the risk head's MFE/MAE
predictions are weak; this measures it directly.  An OOS backtest of the
checkpoint produces a decision log carrying the EV legs exactly as the EV
gate consumed them (predicted MFE bps, predicted |MAE| bps); realized
MFE/MAE per decision come from the forward bar window around each
decision, computed the way ``targets.py`` builds MFE/MAE targets
(``compute_outcome_summary``: max/min of the relative close path over
1..max(training.forward_horizons) bars).  Spearman rank correlation of
predicted vs realized answers whether the EV gate discriminates at all.

Runs the backtest on CPU by default so it never contends with a training
sweep on the GPU.

Usage:
    python scripts/audit_ev_inputs.py --real-data
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

DEFAULT_REPORT_PATH = Path("runs/ev_input_audit.json")


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), like scipy.stats.rankdata."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN when either side is constant."""
    if len(x) < 3:
        return float("nan")
    rx, ry = average_ranks(x), average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def realized_excursions_bps(
    closes: np.ndarray, index: int, max_horizon: int
) -> tuple[float, float] | None:
    """Realized (MFE, |MAE|) in bps for a decision at ``index``.

    Mirrors ``compute_outcome_summary``: relative close path over
    1..max_horizon bars from the decision bar's close; MFE = path max,
    MAE = path min (|.| taken here to match the EV-gate leg convention).
    Returns None when fewer than max_horizon forward bars exist.
    """
    if index + max_horizon >= len(closes):
        return None
    base = float(closes[index])
    if base <= 0:
        return None
    path = closes[index + 1 : index + max_horizon + 1] / base - 1.0
    return float(path.max() * 1e4), float(abs(path.min()) * 1e4)


def main() -> int:
    parser = argparse.ArgumentParser(description="EV-input quality audit (§7.0.4)")
    parser.add_argument("--real-data", action="store_true", help="Use cached real historical bars")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for the audit backtest (default cpu: never contend with a GPU training sweep)",
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT_PATH), help="Where to write the JSON report"
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("audit_ev_inputs")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path:
        logger.warning("ev_audit_requires_checkpoint")
        return 2
    if not args.real_data:
        logger.warning(
            "ev_audit_synthetic_refused",
            note="the audit is only meaningful on real OOS data; pass --real-data",
        )
        return 2
    config.device.preferred = args.device

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
    decision_log = report.decision_log
    logger.info("ev_audit_backtest_done", decisions=len(decision_log))

    max_horizon = int(max(config.training.forward_horizons))
    closes_by_symbol = {
        symbol: np.array([bar.close for bar in bars], dtype=np.float64)
        for symbol, bars in bars_by_symbol.items()
    }

    rows: list[dict[str, float | str]] = []
    skipped_tail = 0
    for record in decision_log:
        symbol = str(record["symbol"])
        excursions = realized_excursions_bps(
            closes_by_symbol[symbol], int(record["bar_index"]), max_horizon
        )
        if excursions is None:
            skipped_tail += 1
            continue
        realized_mfe, realized_mae = excursions
        pred_gain = float(record.get("expected_gain_bps", 0.0))
        pred_loss = float(record.get("expected_loss_bps", 0.0))
        # Since the 2026-07-21 EV-leg fix, SHORT proposals carry
        # TRADE-relative legs (gain = predicted down-move). This audit's
        # baseline (2026-07-17 artifact) correlates PRICE-relative
        # predictions vs price-path excursions, so un-swap shorts back to
        # price-relative here — keeping every run of this script comparable
        # with the recorded §7.0.4 baseline.
        if int(record.get("action", 0)) == -1:
            pred_gain, pred_loss = pred_loss, pred_gain
        rows.append(
            {
                "symbol": symbol,
                "pred_gain": pred_gain,
                "pred_loss": pred_loss,
                "real_mfe": realized_mfe,
                "real_mae": realized_mae,
            }
        )

    if len(rows) < 30:
        logger.warning("ev_audit_too_few_decisions", usable=len(rows))
        return 2

    pred_gain = np.array([row["pred_gain"] for row in rows])
    pred_loss = np.array([row["pred_loss"] for row in rows])
    real_mfe = np.array([row["real_mfe"] for row in rows])
    real_mae = np.array([row["real_mae"] for row in rows])

    def _stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "distinct": int(len(np.unique(np.round(values, 4)))),
        }

    per_symbol: dict[str, dict[str, float | int]] = {}
    for symbol in sorted({str(row["symbol"]) for row in rows}):
        mask = np.array([row["symbol"] == symbol for row in rows])
        if mask.sum() < 30:
            continue
        per_symbol[symbol] = {
            "n": int(mask.sum()),
            "spearman_gain_vs_mfe": spearman(pred_gain[mask], real_mfe[mask]),
            "spearman_loss_vs_mae": spearman(pred_loss[mask], real_mae[mask]),
        }

    result = {
        "checkpoint": str(config.model.checkpoint_path),
        "oos_meta": {k: str(v) for k, v in (oos_meta or {}).items()},
        "max_horizon_bars": max_horizon,
        "decisions_total": len(decision_log),
        "decisions_usable": len(rows),
        "skipped_no_forward_window": skipped_tail,
        "pooled": {
            "spearman_gain_vs_mfe": spearman(pred_gain, real_mfe),
            "spearman_loss_vs_mae": spearman(pred_loss, real_mae),
        },
        "prediction_stats": {
            "expected_gain_bps": _stats(pred_gain),
            "expected_loss_bps": _stats(pred_loss),
        },
        "realized_stats": {
            "mfe_bps": _stats(real_mfe),
            "mae_bps": _stats(real_mae),
        },
        "per_symbol": per_symbol,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n  EV-INPUT QUALITY AUDIT (predicted vs realized excursions, OOS)")
    print(f"  decisions: {len(rows):,} usable of {len(decision_log):,}")
    print(f"  Spearman gain vs MFE : {result['pooled']['spearman_gain_vs_mfe']:+.4f}")
    print(f"  Spearman loss vs MAE : {result['pooled']['spearman_loss_vs_mae']:+.4f}")
    print(
        "  predicted gain bps   : "
        f"mean {result['prediction_stats']['expected_gain_bps']['mean']:.2f}, "
        f"std {result['prediction_stats']['expected_gain_bps']['std']:.2f}, "
        f"p5-p95 [{result['prediction_stats']['expected_gain_bps']['p5']:.2f}, "
        f"{result['prediction_stats']['expected_gain_bps']['p95']:.2f}]"
    )
    print(
        "  predicted loss bps   : "
        f"mean {result['prediction_stats']['expected_loss_bps']['mean']:.2f}, "
        f"std {result['prediction_stats']['expected_loss_bps']['std']:.2f}, "
        f"p5-p95 [{result['prediction_stats']['expected_loss_bps']['p5']:.2f}, "
        f"{result['prediction_stats']['expected_loss_bps']['p95']:.2f}]"
    )
    print(
        "  realized MFE bps     : "
        f"mean {result['realized_stats']['mfe_bps']['mean']:.2f}, "
        f"std {result['realized_stats']['mfe_bps']['std']:.2f}"
    )
    print(
        "  realized MAE bps     : "
        f"mean {result['realized_stats']['mae_bps']['mean']:.2f}, "
        f"std {result['realized_stats']['mae_bps']['std']:.2f}"
    )
    for symbol, stats in per_symbol.items():
        print(
            f"    {symbol:<6} n={stats['n']:>6,}  gain/MFE {stats['spearman_gain_vs_mfe']:+.4f}  "
            f"loss/MAE {stats['spearman_loss_vs_mae']:+.4f}"
        )
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
