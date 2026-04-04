from __future__ import annotations

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
    max_horizon = max(config.training.forward_horizons)
    feature_contract: FeatureContract | None = None
    samples_by_split: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}
    split_ranges: dict[str, dict[str, SymbolSplit]] = {"train": {}, "val": {}, "test": {}}
    estimated_total_samples = 0

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
                    horizons=config.training.forward_horizons,
                    path_label_rules=config.training.path_label_rules,
                )
                targets = build_training_targets(
                    outcome,
                    label_epsilon=config.training.label_epsilon,
                    regression_targets=config.training.regression_targets,
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
    train_loader = DataLoader(
        HistoricalSequenceDataset(samples_by_split["train"], feature_contract=feature_contract, split="train"),
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_training_samples,
    )
    val_loader = DataLoader(
        HistoricalSequenceDataset(samples_by_split["val"], feature_contract=feature_contract, split="val"),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_training_samples,
    )
    test_loader = DataLoader(
        HistoricalSequenceDataset(samples_by_split["test"], feature_contract=feature_contract, split="test"),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_training_samples,
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
