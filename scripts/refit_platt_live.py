"""Platt refit from realized paper outcomes (MASTER_PLAN §7.0.3).

Live confidence pins at ~0.485–0.515, so the EV gate can barely
discriminate. This joins the runtime's decision log (raw confidence,
client_order_id) with the outcome stream (decision_id = client_order_id,
realized net PnL) and refits Platt scaling on REALIZED trade outcomes:
win = realized_pnl > 0.

Pitfall handled per spec: ``realized_pnl`` is nonzero only on
closing/reducing legs — an opening fill's 0.0 means "unresolved", NOT a
loss. Only observations with |realized_pnl| > 0 enter the fit, and the
fit refuses below ``--min-samples`` (default 300) resolved trades — the
paper week supplies them.

The refit artifact is written to an ISOLATED path
(``runs/platt_live_refit.json``), fingerprint-stamped for the current
checkpoint. The canonical sidecar next to the checkpoint moves only
through the normal promotion gate — never from here.

Usage:
    python scripts/refit_platt_live.py
    python scripts/refit_platt_live.py --min-samples 50   # explicit experiments only
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

DEFAULT_REPORT_PATH = Path("runs/platt_live_refit.json")
DEFAULT_MIN_SAMPLES = 300


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def join_resolved_trades(
    decisions: list[dict], observations: list[dict]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """(confidences, realized_outcomes, stats) for RESOLVED trades only.

    Join key: decision.client_order_id == observation.decision_id.
    Opening fills (realized_pnl == 0.0) are unresolved and excluded —
    counting them as losses would poison the win labels.
    """
    confidence_by_order: dict[str, float] = {}
    for decision in decisions:
        order_id = decision.get("client_order_id")
        if order_id:
            confidence_by_order[str(order_id)] = float(decision.get("confidence", 0.0))

    confidences: list[float] = []
    outcomes: list[float] = []
    unresolved = 0
    unmatched = 0
    for observation in observations:
        realized = float(observation.get("realized_pnl", 0.0))
        if realized == 0.0:
            unresolved += 1
            continue
        confidence = confidence_by_order.get(str(observation.get("decision_id", "")))
        if confidence is None:
            unmatched += 1
            continue
        confidences.append(confidence)
        outcomes.append(realized)
    stats = {
        "decisions_with_orders": len(confidence_by_order),
        "observations": len(observations),
        "unresolved_opening_fills": unresolved,
        "unmatched_observations": unmatched,
        "resolved_trades": len(outcomes),
    }
    return np.asarray(confidences), np.asarray(outcomes), stats


def reliability_table(
    calibrator, confidences: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    labels = (outcomes > 0.0).astype(float)
    calibrated = np.array([calibrator.calibrate(float(c)) for c in confidences])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    table = []
    for i in range(n_bins):
        mask = (calibrated >= bins[i]) & (calibrated < bins[i + 1])
        if mask.sum() == 0:
            continue
        table.append(
            {
                "bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_calibrated": float(calibrated[mask].mean()),
                "win_rate": float(labels[mask].mean()),
            }
        )
    return table


def _checkpoint_fingerprint(checkpoint_path: str) -> tuple[str | None, str | None]:
    import torch

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        return meta.get("encoder_fingerprint"), Path(checkpoint_path).name
    except Exception:  # noqa: BLE001 - fingerprint is provenance, not a blocker
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Platt refit from realized outcomes (§7.0.3)")
    parser.add_argument("--decisions", default="runs/decisions.jsonl")
    parser.add_argument("--observations", default="runs/observations.jsonl")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Refuse to fit below this many RESOLVED trades (default 300; lower only for explicit experiments)",
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("refit_platt_live")

    from src.training.platt_calibration import PlattCalibrator

    decisions = _read_jsonl(Path(args.decisions))
    observations = _read_jsonl(Path(args.observations))
    confidences, outcomes, stats = join_resolved_trades(decisions, observations)
    logger.info("platt_live_join", **stats)
    print(f"\n  PLATT REFIT FROM REALIZED OUTCOMES (§7.0.3)")
    for key, value in stats.items():
        print(f"    {key:<28} {value}")

    if len(outcomes) < args.min_samples:
        print(
            f"\n  REFUSED: {len(outcomes)} resolved trades < required {args.min_samples}."
            f"\n  The unattended paper week supplies them — re-run after it."
        )
        return 2

    passthrough = PlattCalibrator()
    before = {
        "ece": passthrough.expected_calibration_error(confidences, outcomes),
        "brier": passthrough.brier_score(confidences, outcomes),
    }
    calibrator = PlattCalibrator()
    calibrator.fit(confidences, outcomes)
    fingerprint, checkpoint_name = _checkpoint_fingerprint(config.model.checkpoint_path)
    calibrator.encoder_fingerprint = fingerprint
    calibrator.checkpoint_filename = checkpoint_name
    after = {
        "ece": calibrator.expected_calibration_error(confidences, outcomes),
        "brier": calibrator.brier_score(confidences, outcomes),
    }
    raw_lo, raw_hi = float(confidences.min()), float(confidences.max())
    cal_lo, cal_hi = calibrator.calibrate(raw_lo), calibrator.calibrate(raw_hi)

    report = {
        "checkpoint": str(config.model.checkpoint_path),
        "encoder_fingerprint": fingerprint,
        "join_stats": stats,
        "fit": {"a": calibrator.a, "b": calibrator.b, "n_samples": calibrator.n_samples},
        "before": before,
        "after": after,
        "raw_confidence_range": [raw_lo, raw_hi],
        "calibrated_confidence_range": [cal_lo, cal_hi],
        "reliability_after": reliability_table(calibrator, confidences, outcomes),
        "note": (
            "isolated refit artifact — the canonical sidecar next to the "
            "checkpoint moves only through the promotion gate"
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    calibrator.save(report_path.with_suffix(".calibrator.json"))

    print(f"\n  fit: a={calibrator.a:+.4f} b={calibrator.b:+.4f} on {calibrator.n_samples} trades")
    print(f"  ECE   before {before['ece']:.4f} -> after {after['ece']:.4f}")
    print(f"  Brier before {before['brier']:.4f} -> after {after['brier']:.4f}")
    print(f"  confidence range: raw [{raw_lo:.4f}, {raw_hi:.4f}] -> calibrated [{cal_lo:.4f}, {cal_hi:.4f}]")
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
