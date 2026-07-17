"""Bootstrap synthetic stress run (MASTER_PLAN §7.2.6).

Builds regime labels (§7.2.2), stitches synthetic paths from the RARE
regime clusters' sessions, and backtests the checkpoint on them twice —
honest fills vs adverse fills — reporting the deltas.  The output is the
informational stress block a promotion record may attach
(``evaluate_promotion(stress_metrics=...)``); it is NEVER a criterion.

Memory is disabled for stress runs (synthetic bars must not seed the
bank) and inference runs on CPU so a GPU training sweep is never
contended.

Usage:
    python scripts/run_bootstrap_stress.py --real-data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger

DEFAULT_REPORT_PATH = Path("runs/bootstrap_stress.json")
STRESS_METRIC_KEYS = ("total_return", "sharpe_ratio", "max_drawdown", "expectancy", "fill_count")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap synthetic stress (§7.2.6)")
    parser.add_argument("--real-data", action="store_true", help="Use cached real historical bars")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override")
    parser.add_argument("--n-sessions", type=int, default=60, help="Synthetic sessions to stitch")
    parser.add_argument(
        "--clusters",
        nargs="*",
        type=int,
        default=None,
        help="Target regime clusters (default: the two smallest with >= 10 sessions)",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (default cpu)")
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
    logger = get_logger("run_bootstrap_stress")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not args.real_data:
        logger.warning("bootstrap_stress_synthetic_refused", note="pass --real-data")
        return 2
    config.device.preferred = args.device
    config.memory.enabled = False  # synthetic bars must not seed the bank

    import torch

    from src.backtest.engine import BacktestEngine
    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.bootstrap_stress import generate_bootstrap_paths
    from src.evaluation.regime_snapshots import build_regime_snapshot
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    snapshot = build_regime_snapshot(
        bars_by_symbol,
        k=int(config.evaluation.regime_snapshot_k),
        seed=int(config.evaluation.seed),
        short_window=int(config.features.windows["short"]),
        medium_window=int(config.features.windows["medium"]),
        long_window=int(config.features.windows["long"]),
        epsilon=float(config.features.epsilon),
    )
    if snapshot is None:
        logger.warning("bootstrap_stress_no_snapshot")
        return 2

    if args.clusters:
        target_clusters = set(args.clusters)
    else:
        sizes: dict[int, int] = {}
        for label in snapshot.labels_by_date.values():
            sizes[label] = sizes.get(label, 0) + 1
        eligible = sorted(
            (count, cluster) for cluster, count in sizes.items() if count >= 10
        )
        target_clusters = {cluster for _, cluster in eligible[:2]}
    paths = generate_bootstrap_paths(
        bars_by_symbol,
        snapshot.labels_by_date,
        target_clusters=target_clusters,
        n_sessions=int(args.n_sessions),
        seed=int(config.evaluation.seed),
    )

    def _run(adverse: bool) -> dict[str, float]:
        run_config = config.model_copy(deep=True)
        run_config.backtest.fill_timing = "next_open"
        run_config.backtest.adverse_fills = adverse
        pipeline = FeaturePipeline(run_config.features, device=torch.device("cpu"))
        policy = build_brain_policy(run_config, bars_by_symbol=paths, pipeline=pipeline)
        report = BacktestEngine(
            run_config, pipeline, RiskManager(run_config.risk.model_copy())
        ).run(paths, policy)
        return {key: float(report.metrics.get(key, 0.0)) for key in STRESS_METRIC_KEYS}

    baseline = _run(adverse=False)
    adverse = _run(adverse=True)
    stress_metrics = {
        **{f"stress_baseline_{key}": value for key, value in baseline.items()},
        **{f"stress_adverse_{key}": value for key, value in adverse.items()},
        **{
            f"stress_delta_{key}": adverse[key] - baseline[key]
            for key in STRESS_METRIC_KEYS
        },
    }
    result = {
        "checkpoint": str(config.model.checkpoint_path),
        "target_clusters": sorted(target_clusters),
        "n_sessions": int(args.n_sessions),
        "adverse_fill_range_fraction": float(config.backtest.adverse_fill_range_fraction),
        "baseline": baseline,
        "adverse": adverse,
        "stress_metrics": stress_metrics,
        "note": "informational only — never a promotion criterion (§7.2.6)",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n  BOOTSTRAP STRESS  (clusters {sorted(target_clusters)}, {args.n_sessions} sessions)")
    print("  | metric        | baseline   | adverse    | delta      |")
    print("  |---------------|------------|------------|------------|")
    for key in STRESS_METRIC_KEYS:
        print(
            f"  | {key:<13} | {baseline[key]:+10.4f} | {adverse[key]:+10.4f} "
            f"| {adverse[key] - baseline[key]:+10.4f} |"
        )
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
