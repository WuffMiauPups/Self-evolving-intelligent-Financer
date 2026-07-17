"""Seed-ensemble scorecard run (MASTER_PLAN §9.4 arm 2).

Backtests the majority-vote ensemble of the three noise-floor checkpoints
against the incumbent AND each seed individually, all on one shared OOS
window (bars after every checkpoint's train/val end). Zero training.

Evidence tier: SUGGESTIVE (one OOS window; the §2.0 paired protocol still
governs promotion claims). Named metric per the plan: OOS balanced
accuracy AND net expectancy vs the incumbent row.

Usage:
    python scripts/run_seed_ensemble.py --real-data
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

DEFAULT_MEMBERS = [
    "checkpoints/sweep_20260717_142927/brain_run001.pt",
    "checkpoints/sweep_20260717_142927/brain_run002.pt",
    "checkpoints/sweep_20260717_142927/brain_run003.pt",
]
DEFAULT_REPORT_PATH = Path("runs/seed_ensemble.json")
METRIC_KEYS = ("total_return", "sharpe_ratio", "expectancy", "max_drawdown", "fill_count")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed-ensemble scorecard (§9.4 arm 2)")
    parser.add_argument("--real-data", action="store_true", help="Use cached real historical bars")
    parser.add_argument("--members", nargs="*", default=None, help="Member checkpoints (default: the 3 floor seeds)")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("run_seed_ensemble")
    if not args.real_data:
        logger.warning("seed_ensemble_synthetic_refused", note="pass --real-data")
        return 2

    import torch

    from src.backtest.engine import BacktestEngine
    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.benchmark import directional_balanced_accuracy
    from src.evaluation.oos import enforce_out_of_sample
    from src.evaluation.seed_ensemble import SeedEnsemblePolicy
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.training.data_loading import _resolve_bars_by_symbol

    members = args.members or DEFAULT_MEMBERS
    incumbent = str(config.model.checkpoint_path)
    bars_by_symbol, _ = _resolve_bars_by_symbol(config)

    # Shared OOS window: strictly after EVERY checkpoint's train/val end.
    cutoff = None
    for checkpoint in [incumbent, *members]:
        oos_bars, _ = enforce_out_of_sample(bars_by_symbol, checkpoint_path=checkpoint)
        start = min(bars[0].timestamp for bars in oos_bars.values() if bars)
        cutoff = start if cutoff is None or start > cutoff else cutoff
    bars_oos, _ = enforce_out_of_sample(
        bars_by_symbol, checkpoint_path=incumbent, cutoff_override=cutoff
    )
    logger.info("seed_ensemble_oos_window", cutoff=str(cutoff))

    horizon = int(config.training.effective_primary_horizon)
    cost_band = float(config.training.transaction_cost_bps) / 10_000.0

    def _policy_for(checkpoint: str):
        run_config = config.model_copy(deep=True)
        run_config.model.checkpoint_path = checkpoint
        run_config.backtest.fill_timing = "next_open"
        pipeline = FeaturePipeline(run_config.features, device=torch.device("cpu"))
        return (
            build_brain_policy(run_config, bars_by_symbol=bars_oos, pipeline=pipeline),
            run_config,
            pipeline,
        )

    def _run(policy, run_config, pipeline) -> dict[str, float]:
        report = BacktestEngine(
            run_config, pipeline, RiskManager(run_config.risk.model_copy())
        ).run(bars_oos, policy)
        row = {key: float(report.metrics.get(key, 0.0)) for key in METRIC_KEYS}
        row["balanced_accuracy"] = directional_balanced_accuracy(
            report.decision_log, bars_oos, horizon=horizon, cost_band=cost_band
        )
        return row

    rows: dict[str, dict[str, float]] = {}
    member_policies = []
    for index, checkpoint in enumerate(members, start=1):
        policy, run_config, pipeline = _policy_for(checkpoint)
        member_policies.append(policy)
        rows[f"seed_run{index:03d}"] = _run(policy, run_config, pipeline)
        # Fresh recurrent state for the ensemble pass.
    ensemble_members = [_policy_for(checkpoint)[0] for checkpoint in members]
    ensemble_config = config.model_copy(deep=True)
    ensemble_config.backtest.fill_timing = "next_open"
    ensemble_pipeline = FeaturePipeline(ensemble_config.features, device=torch.device("cpu"))
    rows["ensemble"] = _run(
        SeedEnsemblePolicy(members=ensemble_members), ensemble_config, ensemble_pipeline
    )
    incumbent_policy, incumbent_config, incumbent_pipeline = _policy_for(incumbent)
    rows["incumbent"] = _run(incumbent_policy, incumbent_config, incumbent_pipeline)

    result = {
        "oos_cutoff": str(cutoff),
        "members": members,
        "incumbent": incumbent,
        "rows": rows,
        "evidence_tier": "suggestive (single shared OOS window; §2.0 governs claims)",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n  SEED-ENSEMBLE SCORECARD (shared OOS from {cutoff})")
    header = f"  | {'row':<12} | {'bal_acc':>7} | {'expectancy':>10} | {'total_ret':>9} | {'sharpe':>8} | {'fills':>6} |"
    print(header)
    print("  |" + "-" * (len(header) - 4) + "|")
    for name, row in rows.items():
        print(
            f"  | {name:<12} | {row['balanced_accuracy']:>7.4f} | {row['expectancy']:>+10.4f} "
            f"| {row['total_return']:>+9.4f} | {row['sharpe_ratio']:>+8.3f} | {int(row['fill_count']):>6} |"
        )
    print(f"  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
