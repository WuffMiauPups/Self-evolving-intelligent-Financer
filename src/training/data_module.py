from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.data.dataset import HistoricalSequenceDataset, TrainingSample, collate_training_samples
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeatureContract, FeaturePipeline
from src.memory.outcomes import compute_outcome_summary
from src.training.class_balance import (
    TargetDiagnostics,
    build_class_weights,
    build_weighted_sampler,
    compute_target_diagnostics,
)
from src.training.sample_cache import (
    CachedTrainingData,
    compute_fingerprint,
    load_cached_samples,
    save_cached_samples,
)
from src.training.splits import SymbolSplit, build_symbol_splits
from src.training.targets import build_training_targets


@dataclass(frozen=True)
class TrainingDataBundle:
    feature_contract: FeatureContract
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    diagnostics: dict[str, TargetDiagnostics]
    split_ranges: dict[str, dict[str, SymbolSplit]]
    class_weights: object | None
    samples_by_split: dict[str, list[TrainingSample]]


def build_real_data_training_bundle(
    *,
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
) -> TrainingDataBundle:
    logger = get_logger("training_data")
    feature_contract: FeatureContract | None = None
    samples_by_split: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}
    split_ranges: dict[str, dict[str, SymbolSplit]] = {"train": {}, "val": {}, "test": {}}
    estimated_total_samples = 0

    # --- Sample cache fast-path ------------------------------------------------
    # Per-sample FeaturePipeline.build() is the single largest cold-start cost.
    # The samples are deterministic given a handful of config fields + the raw
    # bars, so cache the whole materialized list on disk. Miss / disable /
    # corruption falls through to the rebuild below.
    try:
        fingerprint = compute_fingerprint(config=config, bars_by_symbol=bars_by_symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sample_cache_fingerprint_failed", error=str(exc))
        fingerprint = None
    cached: CachedTrainingData | None = None
    if fingerprint is not None:
        cached = load_cached_samples(fingerprint=fingerprint)
    if cached is not None:
        feature_contract = cached.feature_contract
        samples_by_split = cached.samples_by_split
        split_ranges = cached.split_ranges
        logger.info(
            "training_samples_loaded_from_cache",
            fingerprint=fingerprint,
            train=len(samples_by_split.get("train", [])),
            val=len(samples_by_split.get("val", [])),
            test=len(samples_by_split.get("test", [])),
        )
        return _finalize_training_bundle(
            config=config,
            feature_contract=feature_contract,
            samples_by_split=samples_by_split,
            split_ranges=split_ranges,
            logger=logger,
        )

    all_horizons = config.training.all_horizons
    max_horizon = max(all_horizons)

    for symbol, raw_bars in bars_by_symbol.items():
        eligible_start = max(config.runtime.warmup_bars, config.training.sequence_length - 1)
        eligible_stop = len(raw_bars) - max_horizon
        estimated_total_samples += max(eligible_stop - eligible_start, 0)
    logger.info(
        "training_dataset_planned",
        symbol_count=len(bars_by_symbol),
        estimated_total_samples=estimated_total_samples,
        sequence_length=config.training.sequence_length,
        forward_horizons=list(config.training.forward_horizons),
        auxiliary_horizons=list(config.training.auxiliary_horizons),
        primary_direction_horizon=config.training.effective_primary_horizon,
        use_real_historical_data=config.training.use_real_historical_data,
    )
    if estimated_total_samples >= 200_000:
        logger.warning(
            "training_dataset_large",
            estimated_total_samples=estimated_total_samples,
            note=(
                "Real-data dataset build may take a long time on 1Min multi-year bars; "
                "this is expected until a precomputed feature cache is added."
            ),
        )

    for symbol, raw_bars in bars_by_symbol.items():
        bars = sorted(raw_bars, key=lambda bar: bar.timestamp)
        eligible_start = max(config.runtime.warmup_bars, config.training.sequence_length - 1)
        eligible_stop = len(bars) - max_horizon
        if eligible_stop <= eligible_start:
            continue
        eligible_indices = list(range(eligible_start, eligible_stop))
        eligible_timestamps = [bars[index].timestamp for index in eligible_indices]
        logger.info(
            "training_symbol_prepare",
            symbol=symbol,
            bar_count=len(bars),
            eligible_samples=len(eligible_indices),
        )
        symbol_splits = build_symbol_splits(
            symbol=symbol,
            eligible_indices=eligible_indices,
            timestamps=eligible_timestamps,
            train_split=config.training.train_split,
            val_split=config.training.val_split,
            test_split=config.training.test_split,
        )
        cursor = 0
        for split in symbol_splits:
            split_indices = eligible_indices[cursor: cursor + split.sample_count]
            cursor += split.sample_count
            split_ranges[split.split][symbol] = split
            progress_interval = 25_000 if len(split_indices) >= 100_000 else 10_000
            for index in split_indices:
                processed_in_split = index - split_indices[0] + 1
                if processed_in_split == 1 or processed_in_split % progress_interval == 0:
                    logger.info(
                        "training_dataset_progress",
                        symbol=symbol,
                        split=split.split,
                        processed=processed_in_split,
                        split_samples=len(split_indices),
                        last_timestamp=bars[index].timestamp.isoformat(),
                    )
                window_start = max(0, index + 1 - config.training.sequence_length)
                snapshot = pipeline.build(bars[window_start: index + 1], symbol=symbol)
                if feature_contract is None:
                    feature_contract = snapshot.contract
                else:
                    feature_contract.validate_snapshot(snapshot, context=f"real_training[{symbol}:{split.split}]")
                outcome = compute_outcome_summary(
                    bars,
                    index,
                    horizons=all_horizons,
                    path_label_rules=config.training.path_label_rules,
                )
                targets = build_training_targets(
                    outcome,
                    label_epsilon=config.training.label_epsilon,
                    regression_targets=config.training.regression_targets,
                    primary_horizon=config.training.effective_primary_horizon,
                    enable_future_return_targets=config.training.enable_future_return_targets,
                    enable_risk_path_targets=config.training.enable_risk_path_targets,
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
            logger.info(
                "training_split_materialized",
                symbol=symbol,
                split=split.split,
                sample_count=len(split_indices),
            )
    if feature_contract is None:
        raise ValueError("No real-data training samples were generated")

    # Write to cache BEFORE fast-mode capping / diagnostics so the cache holds
    # the canonical (full) sample set; fast-mode only affects the subset that
    # gets trained on, it should not force a different cache key.
    if fingerprint is not None:
        save_cached_samples(
            fingerprint=fingerprint,
            data=CachedTrainingData(
                feature_contract=feature_contract,
                samples_by_split=samples_by_split,
                split_ranges=split_ranges,
            ),
        )

    return _finalize_training_bundle(
        config=config,
        feature_contract=feature_contract,
        samples_by_split=samples_by_split,
        split_ranges=split_ranges,
        logger=logger,
    )


def _finalize_training_bundle(
    *,
    config: AppConfig,
    feature_contract: FeatureContract,
    samples_by_split: dict[str, list[TrainingSample]],
    split_ranges: dict[str, dict[str, SymbolSplit]],
    logger,
) -> TrainingDataBundle:
    # Fast-mode: deterministic sample caps per split (applied post-cache so the
    # cached artifact stays full-size and reusable across fast/non-fast runs).
    fast = config.training.fast_mode
    if fast.enabled:
        rng = random.Random(fast.sample_seed)
        # Do not mutate the input dict (it may be the cached payload).
        subset: dict[str, list[TrainingSample]] = dict(samples_by_split)
        for split_name, cap in [
            ("train", fast.max_train_samples),
            ("val", fast.max_val_samples),
            ("test", fast.max_test_samples),
        ]:
            original_count = len(subset[split_name])
            if cap is not None and original_count > cap:
                subset[split_name] = rng.sample(subset[split_name], cap)
                logger.info(
                    "fast_mode_sample_cap_applied",
                    split=split_name,
                    original_count=original_count,
                    capped_count=cap,
                    sample_seed=fast.sample_seed,
                )
        samples_by_split = subset

    diagnostics = {
        split_name: compute_target_diagnostics(
            samples,
            split=split_name,
            minority_class_floor_warning=config.training.minority_class_floor_warning,
            label_epsilon=config.training.label_epsilon,
            inspect_target_boundaries=config.training.inspect_target_boundaries,
        )
        for split_name, samples in samples_by_split.items()
    }
    class_weights = build_class_weights(
        samples_by_split["train"],
        mode=config.training.class_weight_mode,
        cap=config.training.class_weight_cap,
    )
    sampler = (
        build_weighted_sampler(samples_by_split["train"], mode=config.training.weighted_sampler_mode)
        if config.training.use_weighted_sampler
        else None
    )
    # DataLoader tuning (Phase 3a):
    #   - num_workers=0 because samples are already materialized tensors in RAM;
    #     forking would pickle-copy the entire list per worker (memory blow-up).
    #   - pin_memory=True when a CUDA device is available so the main-thread
    #     host->device copy in the trainer can use async non_blocking=True.
    #     Harmless on CPU-only runs.
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        HistoricalSequenceDataset(samples_by_split["train"], feature_contract=feature_contract, split="train"),
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_training_samples,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        HistoricalSequenceDataset(samples_by_split["val"], feature_contract=feature_contract, split="val"),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_training_samples,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        HistoricalSequenceDataset(samples_by_split["test"], feature_contract=feature_contract, split="test"),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_training_samples,
        pin_memory=pin_memory,
    )
    return TrainingDataBundle(
        feature_contract=feature_contract,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        diagnostics=diagnostics,
        split_ranges=split_ranges,
        class_weights=class_weights,
        samples_by_split=samples_by_split,
    )
