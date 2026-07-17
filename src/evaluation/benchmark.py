"""The accuracy scorecard (MASTER_PLAN §2.1).

One fixed protocol that grades a checkpoint out of sample with honest
fills against tradeable baselines, and appends one deterministic row to
``docs/analysis/SCORECARD.md``.  From the first row onward, "accuracy
improved" means a new row beats the incumbent on balanced accuracy AND
net expectancy — not a feeling.

The row is deterministic for (checkpoint, data range): no wall-clock
values enter it, so re-running the same evaluation reproduces the row
byte-for-byte.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from src.backtest.engine import BacktestEngine
from src.backtest.policy_factory import (
    build_baseline_policy,
    build_brain_policy,
    build_buy_and_hold_policy,
)
from src.core.config import AppConfig
from src.core.logging import get_logger
from src.evaluation.oos import enforce_out_of_sample
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager

logger = get_logger("evaluation.benchmark")

DEFAULT_SCORECARD_PATH = Path("docs/analysis/SCORECARD.md")

_SCORECARD_HEADER = """# SETAI Accuracy Scorecard

One row per benchmark run (MASTER_PLAN §2.1). Protocol: bars strictly
after the checkpoint's train/val end, next-bar-open fills, real costs,
vs tradeable baselines. "Accuracy improved" = a new row beats the
incumbent on balanced accuracy AND net expectancy. Rows are
deterministic for (checkpoint, data range).

| checkpoint | fingerprint | data start | data end | bal. acc | expectancy | sharpe | max DD | CVaR95 | vs buy&hold | trades | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""


@dataclass(frozen=True)
class BenchmarkRow:
    checkpoint: str
    fingerprint: str
    data_start: str
    data_end: str
    balanced_accuracy: float
    expectancy: float
    sharpe: float
    max_drawdown: float
    cvar_95: float
    excess_vs_buy_hold: float
    trade_count: int
    tag: str

    def to_markdown(self) -> str:
        return (
            f"| {self.checkpoint} | {self.fingerprint} | {self.data_start} | {self.data_end} "
            f"| {self.balanced_accuracy:.4f} | {self.expectancy:+.2f} | {self.sharpe:+.3f} "
            f"| {self.max_drawdown:+.4f} | {self.cvar_95:+.5f} | {self.excess_vs_buy_hold:+.4f} "
            f"| {self.trade_count} | {self.tag} |"
        )


def directional_balanced_accuracy(
    decision_log: list[dict[str, object]],
    bars_by_symbol: dict[str, list],
    *,
    horizon: int,
    cost_band: float,
) -> float:
    """3-class balanced accuracy of proposals vs realized forward returns.

    Realized direction uses the same cost-aware band as training labels:
    a move inside the band is FLAT (not worth trading either way).
    Balanced accuracy = mean recall over the classes actually present.
    """
    per_class_hits: dict[int, int] = {-1: 0, 0: 0, 1: 0}
    per_class_total: dict[int, int] = {-1: 0, 0: 0, 1: 0}
    for record in decision_log:
        symbol = str(record["symbol"])
        index = int(record["bar_index"])
        bars = bars_by_symbol.get(symbol)
        if bars is None or index + horizon >= len(bars):
            continue
        entry = float(bars[index].close)
        if entry <= 0:
            continue
        realized_return = float(bars[index + horizon].close) / entry - 1.0
        if realized_return > cost_band:
            realized = 1
        elif realized_return < -cost_band:
            realized = -1
        else:
            realized = 0
        per_class_total[realized] += 1
        if int(record["action"]) == realized:
            per_class_hits[realized] += 1
    recalls = [
        per_class_hits[cls] / per_class_total[cls]
        for cls in (-1, 0, 1)
        if per_class_total[cls] > 0
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def run_benchmark(
    config: AppConfig,
    bars_by_symbol: dict[str, list],
    *,
    allow_in_sample: bool = False,
    oos_cutoff: datetime | None = None,
) -> BenchmarkRow:
    checkpoint_path = config.model.checkpoint_path
    if not checkpoint_path:
        raise RuntimeError("benchmark requires model.checkpoint_path (a trained checkpoint)")
    bars_by_symbol, in_sample_tagged = enforce_out_of_sample(
        bars_by_symbol,
        checkpoint_path=checkpoint_path,
        allow_in_sample=allow_in_sample,
        cutoff_override=oos_cutoff,
    )
    # The protocol is non-negotiable regardless of the loaded config:
    # honest fills, always.
    config = config.model_copy(deep=True)
    config.backtest.fill_timing = "next_open"

    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    brain_policy = build_brain_policy(config, bars_by_symbol=bars_by_symbol, pipeline=pipeline)
    brain_report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars_by_symbol, brain_policy
    )
    buyhold_report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars_by_symbol, build_buy_and_hold_policy()
    )
    momentum_report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars_by_symbol, build_baseline_policy()
    )

    horizon = int(config.training.effective_primary_horizon)
    cost_band = float(config.training.transaction_cost_bps) / 10_000.0
    balanced_accuracy = directional_balanced_accuracy(
        brain_report.decision_log,
        bars_by_symbol,
        horizon=horizon,
        cost_band=cost_band,
    )

    from src.models.factory import load_model_artifacts

    fingerprint = "unknown"
    try:
        artifacts = load_model_artifacts(checkpoint_path, device=torch.device("cpu"))
        fingerprint = (artifacts.encoder_fingerprint or "unknown")[:12]
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark_fingerprint_unreadable", error=str(exc))

    all_timestamps = [bar.timestamp for bars in bars_by_symbol.values() for bar in (bars[0], bars[-1])]
    metrics = brain_report.metrics
    row = BenchmarkRow(
        checkpoint=Path(checkpoint_path).name,
        fingerprint=fingerprint,
        data_start=min(all_timestamps).date().isoformat(),
        data_end=max(all_timestamps).date().isoformat(),
        balanced_accuracy=balanced_accuracy,
        expectancy=float(metrics.get("expectancy", 0.0)),
        sharpe=float(metrics.get("sharpe_ratio", 0.0)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        cvar_95=float(metrics.get("cvar_95", 0.0)),
        excess_vs_buy_hold=float(metrics.get("total_return", 0.0))
        - float(buyhold_report.metrics.get("total_return", 0.0)),
        trade_count=int(metrics.get("round_trip_count", 0)),
        tag="IN-SAMPLE" if in_sample_tagged else "oos",
    )
    logger.info(
        "benchmark_complete",
        checkpoint=row.checkpoint,
        balanced_accuracy=row.balanced_accuracy,
        sharpe=row.sharpe,
        excess_vs_buy_hold=row.excess_vs_buy_hold,
        buyhold_return=buyhold_report.metrics.get("total_return", 0.0),
        momentum_return=momentum_report.metrics.get("total_return", 0.0),
        buyhold_fills=buyhold_report.metrics.get("fill_count", 0.0),
    )
    # Guard against the old baselines-cannot-trade defect resurfacing:
    # a benchmark where buy-and-hold produced zero fills is not a
    # benchmark at all.
    if float(buyhold_report.metrics.get("fill_count", 0.0)) <= 0:
        raise RuntimeError(
            "buy-and-hold produced zero fills — the EV gate is rejecting baselines "
            "again; benchmark comparisons would be fiction"
        )
    if math.isnan(row.balanced_accuracy):
        raise RuntimeError("balanced accuracy is NaN — decision log or bars are inconsistent")
    return row


def _brain_decision_log(config: AppConfig, bars_by_symbol: dict[str, list]) -> list[dict[str, object]]:
    """One honest-fill brain backtest; returns its per-decision log."""
    config = config.model_copy(deep=True)
    config.backtest.fill_timing = "next_open"
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    policy = build_brain_policy(config, bars_by_symbol=bars_by_symbol, pipeline=pipeline)
    report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars_by_symbol, policy
    )
    return report.decision_log


def slice_decision_log(
    decision_log: list[dict[str, object]],
    *,
    window_start: datetime,
    window_end: datetime,
    n_slices: int,
) -> list[tuple[str, list[dict[str, object]]]]:
    """Bucket decisions into equal-duration chronological slices.

    Boundaries come from the WINDOW (not the decisions), so two decision
    logs over the same window land in identical buckets — the pairing the
    backward-transfer gate needs.
    """
    if n_slices <= 0 or window_end <= window_start:
        return []
    span = (window_end - window_start) / n_slices
    buckets: list[tuple[str, list[dict[str, object]]]] = []
    for index in range(n_slices):
        start = window_start + span * index
        end = window_end if index == n_slices - 1 else window_start + span * (index + 1)
        # Index prefix guarantees unique names for intraday windows where
        # several slices share a calendar date.
        name = f"s{index}:{start.date().isoformat()}..{end.date().isoformat()}"
        members = [
            record
            for record in decision_log
            if start <= record["timestamp"] and (record["timestamp"] < end or index == n_slices - 1)
        ]
        buckets.append((name, members))
    return buckets


def benchmark_backward_transfer_slices(
    config: AppConfig,
    bars_by_symbol: dict[str, list],
    *,
    candidate_checkpoint: str,
    incumbent_checkpoint: str,
    oos_cutoff: datetime | None,
    n_slices: int | None = None,
    min_decisions: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Per-slice (candidate, incumbent) balanced accuracy for the
    backward-transfer gate (§4.5.7).

    The shared OOS window is cut into ``n_slices`` equal-duration
    chronological regime slices; each checkpoint runs ONE honest-fill
    backtest over the window and its decisions are graded per slice with
    the same cost-aware banding as the scorecard. Slices where either
    checkpoint decided fewer than ``min_decisions`` times are skipped
    (logged): a handful of trades cannot evidence forgetting.

    Note: truly pre-training "frozen historical" slices are not honestly
    gradable while every retrain trains on the full history — the slices
    here cover regime variation across the honest OOS window. Frozen
    pre-training slices become possible with incremental fine-tuning.
    """
    promotion = config.evaluation.promotion
    if n_slices is None:
        n_slices = int(promotion.backward_transfer_slices)
    if min_decisions is None:
        min_decisions = int(promotion.backward_transfer_min_decisions)
    if n_slices <= 0:
        return {}
    bars_oos, _ = enforce_out_of_sample(
        bars_by_symbol,
        checkpoint_path=candidate_checkpoint,
        cutoff_override=oos_cutoff,
    )
    logs: dict[str, list[dict[str, object]]] = {}
    for label, checkpoint in (
        ("candidate", candidate_checkpoint),
        ("incumbent", incumbent_checkpoint),
    ):
        run_config = config.model_copy(deep=True)
        run_config.model.checkpoint_path = str(checkpoint)
        logs[label] = _brain_decision_log(run_config, bars_oos)
    all_bar_timestamps = [
        bar.timestamp for bars in bars_oos.values() for bar in (bars[0], bars[-1])
    ]
    window_start, window_end = min(all_bar_timestamps), max(all_bar_timestamps)
    horizon = int(config.training.effective_primary_horizon)
    cost_band = float(config.training.transaction_cost_bps) / 10_000.0
    # §7.2.2: regime-labeled slices by default — sessions grouped by
    # K-means regime cluster (one shared clustering, so both logs bucket
    # identically); chronological equal-duration cut as fallback/opt-in.
    regime_labels: dict[str, int] | None = None
    if str(promotion.backward_transfer_slicing) == "regime":
        from src.evaluation.regime_snapshots import build_regime_snapshot

        snapshot = build_regime_snapshot(
            bars_oos,
            k=int(config.evaluation.regime_snapshot_k),
            seed=int(config.evaluation.seed),
            short_window=int(config.features.windows["short"]),
            medium_window=int(config.features.windows["medium"]),
            long_window=int(config.features.windows["long"]),
            epsilon=float(config.features.epsilon),
        )
        if snapshot is None:
            logger.warning("backward_transfer_regime_fallback_chronological")
        else:
            regime_labels = snapshot.labels_by_date
    if regime_labels is not None:
        from src.evaluation.regime_snapshots import slice_decision_log_by_regime

        candidate_by_name, dropped_candidate = slice_decision_log_by_regime(
            logs["candidate"], regime_labels
        )
        incumbent_by_name, dropped_incumbent = slice_decision_log_by_regime(
            logs["incumbent"], regime_labels
        )
        if dropped_candidate or dropped_incumbent:
            logger.info(
                "backward_transfer_unlabeled_decisions_dropped",
                candidate=dropped_candidate,
                incumbent=dropped_incumbent,
            )
        # Align by slice NAME: a cluster can be empty for one log only
        # (unlike the chronological cut, whose buckets pair positionally).
        candidate_map = dict(candidate_by_name)
        incumbent_map = dict(incumbent_by_name)
        names = sorted(set(candidate_map) | set(incumbent_map))
        paired = [
            (name, candidate_map.get(name, []), incumbent_map.get(name, []))
            for name in names
        ]
    else:
        candidate_slices = slice_decision_log(
            logs["candidate"], window_start=window_start, window_end=window_end, n_slices=n_slices
        )
        incumbent_slices = slice_decision_log(
            logs["incumbent"], window_start=window_start, window_end=window_end, n_slices=n_slices
        )
        paired = [
            (name, candidate_records, incumbent_records)
            for (name, candidate_records), (_, incumbent_records) in zip(
                candidate_slices, incumbent_slices
            )
        ]
    slice_metrics: dict[str, tuple[float, float]] = {}
    for name, candidate_records, incumbent_records in paired:
        if min(len(candidate_records), len(incumbent_records)) < min_decisions:
            logger.info(
                "backward_transfer_slice_skipped",
                slice=name,
                candidate_decisions=len(candidate_records),
                incumbent_decisions=len(incumbent_records),
                min_decisions=min_decisions,
            )
            continue
        slice_metrics[name] = (
            directional_balanced_accuracy(
                candidate_records, bars_oos, horizon=horizon, cost_band=cost_band
            ),
            directional_balanced_accuracy(
                incumbent_records, bars_oos, horizon=horizon, cost_band=cost_band
            ),
        )
    logger.info(
        "backward_transfer_slices_produced",
        slices={name: (round(c, 4), round(i, 4)) for name, (c, i) in slice_metrics.items()},
        n_requested=n_slices,
    )
    return slice_metrics


def append_scorecard_row(row: BenchmarkRow, *, path: Path | str = DEFAULT_SCORECARD_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_SCORECARD_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as fp:
        fp.write(row.to_markdown() + "\n")
    return path
