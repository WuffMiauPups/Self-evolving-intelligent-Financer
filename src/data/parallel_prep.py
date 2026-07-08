"""Parallel dataset preparation for multi-symbol training data.

Parallelizes the expensive per-symbol feature/target computation
across a thread pool while preserving FeatureContract consistency
and temporal split integrity (no leakage).
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.core.config import AppConfig
from src.data.dataset import TrainingSample
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeatureContract, FeaturePipeline
from src.memory.outcomes import compute_outcome_summary
from src.training.splits import SymbolSplit, build_symbol_splits
from src.training.targets import build_training_targets, compute_symbol_epsilon

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolPrepResult:
    """Result of preparing samples for one symbol."""

    symbol: str
    samples_by_split: dict[str, list[TrainingSample]]
    split_ranges: dict[str, SymbolSplit]
    feature_contract: FeatureContract | None
    sample_count: int
    error: str = ""


def _prepare_symbol(
    symbol: str,
    bars: list[MarketBar],
    pipeline: FeaturePipeline,
    config: AppConfig,
) -> SymbolPrepResult:
    """Build all training samples for a single symbol. Thread-safe."""
    max_horizon = max(config.training.forward_horizons)
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    eligible_start = max(config.runtime.warmup_bars, config.training.sequence_length - 1)
    eligible_stop = len(sorted_bars) - max_horizon

    if eligible_stop <= eligible_start:
        return SymbolPrepResult(
            symbol=symbol,
            samples_by_split={"train": [], "val": [], "test": []},
            split_ranges={},
            feature_contract=None,
            sample_count=0,
        )

    eligible_indices = list(range(eligible_start, eligible_stop))
    eligible_timestamps = [sorted_bars[i].timestamp for i in eligible_indices]

    symbol_splits = build_symbol_splits(
        symbol=symbol,
        eligible_indices=eligible_indices,
        timestamps=eligible_timestamps,
        train_split=config.training.train_split,
        val_split=config.training.val_split,
        test_split=config.training.test_split,
    )

    samples_by_split: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}
    split_ranges: dict[str, SymbolSplit] = {}
    feature_contract: FeatureContract | None = None
    total_samples = 0

    # Adaptive epsilon: per-symbol return volatility.
    # Use training-period bars ONLY to prevent future volatility leakage.
    if config.training.adaptive_label_epsilon:
        train_split_obj = next((s for s in symbol_splits if s.split == "train"), None)
        train_end_idx = train_split_obj.end_index if train_split_obj else len(sorted_bars)
        symbol_epsilon = compute_symbol_epsilon(
            sorted_bars[:train_end_idx],
            multiplier=config.training.adaptive_epsilon_multiplier,
            min_epsilon=config.training.adaptive_epsilon_min,
            max_epsilon=config.training.adaptive_epsilon_max,
            min_bars=config.training.label_epsilon_min_bars,
        )
    else:
        symbol_epsilon = config.training.label_epsilon

    cursor = 0
    for split in symbol_splits:
        split_indices = eligible_indices[cursor: cursor + split.sample_count]
        cursor += split.sample_count
        split_ranges[split.split] = split

        for index in split_indices:
            window_start = max(0, index + 1 - config.training.sequence_length)
            snapshot = pipeline.build(sorted_bars[window_start: index + 1], symbol=symbol)
            if feature_contract is None:
                feature_contract = snapshot.contract

            outcome = compute_outcome_summary(
                sorted_bars,
                index,
                horizons=config.training.forward_horizons,
                path_label_rules=config.training.path_label_rules,
            )
            targets = build_training_targets(
                outcome,
                label_epsilon=symbol_epsilon,
                regression_targets=config.training.regression_targets,
                transaction_cost_bps=config.training.transaction_cost_bps,
            )
            samples_by_split[split.split].append(
                TrainingSample(
                    features=snapshot.features,
                    branch_inputs=dict(snapshot.branch_inputs),
                    regime_input=snapshot.regime_input,
                    temporal_tensor=snapshot.temporal_tensor,
                    label=torch.tensor(targets.direction_label, dtype=torch.long),
                    symbol=symbol,
                    timestamp=snapshot.timestamp,
                    direction_name=targets.direction_name,
                    path_label=torch.tensor(targets.path_label, dtype=torch.long),
                    path_name=targets.path_name,
                    regression_targets=dict(targets.regression_targets),
                    future_returns=dict(targets.future_returns),
                )
            )
            total_samples += 1

    return SymbolPrepResult(
        symbol=symbol,
        samples_by_split=samples_by_split,
        split_ranges=split_ranges,
        feature_contract=feature_contract,
        sample_count=total_samples,
    )


def parallel_prepare_samples(
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
    config: AppConfig,
    max_workers: int = 4,
) -> tuple[dict[str, list[TrainingSample]], dict[str, dict[str, SymbolSplit]], FeatureContract]:
    """Prepare training samples for all symbols in parallel.

    Uses ThreadPoolExecutor because FeaturePipeline uses numpy/torch
    which release the GIL during heavy computation.

    Returns:
        (samples_by_split, split_ranges, feature_contract)
    """
    symbols = sorted(bars_by_symbol.keys())
    logger.info(
        "parallel_prep_start symbols=%d workers=%d",
        len(symbols), min(max_workers, len(symbols)),
    )

    effective_workers = min(max_workers, len(symbols))
    results: list[SymbolPrepResult] = []

    if effective_workers <= 1:
        # Sequential fallback for single symbol or single worker
        for symbol in symbols:
            result = _prepare_symbol(symbol, bars_by_symbol[symbol], pipeline, config)
            results.append(result)
            logger.info(
                "symbol_prepared symbol=%s samples=%d",
                result.symbol, result.sample_count,
            )
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(_prepare_symbol, sym, bars_by_symbol[sym], pipeline, config): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(
                    "symbol_prepared symbol=%s samples=%d",
                    result.symbol, result.sample_count,
                )

    # Merge results
    merged_samples: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}
    merged_splits: dict[str, dict[str, SymbolSplit]] = {"train": {}, "val": {}, "test": {}}
    feature_contract: FeatureContract | None = None

    for result in sorted(results, key=lambda r: r.symbol):
        if result.error:
            logger.error("symbol_prep_error symbol=%s error=%s", result.symbol, result.error)
            continue
        for split_name in ("train", "val", "test"):
            merged_samples[split_name].extend(result.samples_by_split.get(split_name, []))
        for split_name, split_range in result.split_ranges.items():
            merged_splits[split_name][result.symbol] = split_range
        if feature_contract is None and result.feature_contract is not None:
            feature_contract = result.feature_contract

    if feature_contract is None:
        raise ValueError("No training samples generated from any symbol")

    total = sum(len(s) for s in merged_samples.values())
    logger.info(
        "parallel_prep_complete total_samples=%d train=%d val=%d test=%d",
        total,
        len(merged_samples["train"]),
        len(merged_samples["val"]),
        len(merged_samples["test"]),
    )
    return merged_samples, merged_splits, feature_contract


# ---------------------------------------------------------------------------
# Prepared artifact cache (optional precompute)
# ---------------------------------------------------------------------------


def _bars_fingerprint(bars_by_symbol: dict[str, list[MarketBar]]) -> str:
    """Compute a fast fingerprint of the input bars for cache invalidation."""
    hasher = hashlib.sha1()
    for symbol in sorted(bars_by_symbol.keys()):
        bars = bars_by_symbol[symbol]
        hasher.update(symbol.encode())
        hasher.update(str(len(bars)).encode())
        if bars:
            hasher.update(bars[0].timestamp.isoformat().encode())
            hasher.update(bars[-1].timestamp.isoformat().encode())
    return hasher.hexdigest()[:16]


def save_prepared_samples(
    path: str | Path,
    samples_by_split: dict[str, list[TrainingSample]],
    feature_contract: FeatureContract,
    bars_fingerprint: str,
) -> Path:
    """Save precomputed samples to disk for fast reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bars_fingerprint": bars_fingerprint,
        "feature_contract": feature_contract.to_dict(),
        "splits": {},
    }
    for split_name, samples in samples_by_split.items():
        payload["splits"][split_name] = [
            {
                "features": s.features,
                "branch_inputs": s.branch_inputs,
                "regime_input": s.regime_input,
                "temporal_tensor": s.temporal_tensor,
                "label": s.label,
                "symbol": s.symbol,
                "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, "isoformat") else str(s.timestamp),
                "direction_name": s.direction_name,
                "path_label": s.path_label,
                "path_name": s.path_name,
                "regression_targets": s.regression_targets,
                "future_returns": s.future_returns,
            }
            for s in samples
        ]
    torch.save(payload, path)
    logger.info("prepared_samples_saved path=%s fingerprint=%s", path, bars_fingerprint)
    return path


def load_prepared_samples(
    path: str | Path,
    expected_fingerprint: str,
) -> tuple[dict[str, list[TrainingSample]], FeatureContract] | None:
    """Load precomputed samples if fingerprint matches. Returns None on mismatch."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning("cache_load_legacy_format path=%s", str(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("bars_fingerprint") != expected_fingerprint:
        logger.info(
            "prepared_samples_stale path=%s expected=%s got=%s",
            path, expected_fingerprint, payload.get("bars_fingerprint"),
        )
        return None

    from datetime import datetime, UTC

    feature_contract = FeatureContract.from_dict(payload["feature_contract"])
    samples_by_split: dict[str, list[TrainingSample]] = {}
    for split_name, sample_list in payload.get("splits", {}).items():
        samples = []
        for s in sample_list:
            ts = s["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            samples.append(
                TrainingSample(
                    features=s["features"],
                    branch_inputs=s["branch_inputs"],
                    regime_input=s["regime_input"],
                    temporal_tensor=s["temporal_tensor"],
                    label=s["label"],
                    symbol=s["symbol"],
                    timestamp=ts,
                    direction_name=s["direction_name"],
                    path_label=s.get("path_label"),
                    path_name=s.get("path_name", ""),
                    regression_targets=s.get("regression_targets", {}),
                    future_returns=s.get("future_returns", {}),
                )
            )
        samples_by_split[split_name] = samples
    logger.info("prepared_samples_loaded path=%s fingerprint=%s", path, expected_fingerprint)
    return samples_by_split, feature_contract
