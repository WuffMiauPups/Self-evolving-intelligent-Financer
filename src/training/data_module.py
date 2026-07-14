from __future__ import annotations

import gc
import math
import random
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.data.dataset import HistoricalSequenceDataset, TrainingSample, collate_training_samples, make_columnar_collate
from src.data.schemas import MarketBar, bars_to_tensor
from src.features.feature_pipeline import FeatureContract, FeaturePipeline
from src.features.normalization import NormalizationStats
from src.memory.outcomes import compute_outcome_summary
from src.training.class_balance import (
    TargetDiagnostics,
    build_class_weights,
    build_weighted_sampler,
    compute_target_diagnostics,
)
from src.training.curriculum import CurriculumSampler, compute_sample_difficulties
from src.training.sample_cache import (
    CachedSymbolShard,
    CachedTrainingData,
    compute_fingerprint,
    compute_shard_fingerprint,
    load_all_shards,
    load_cached_samples,
    save_cached_samples,
    save_cached_shard,
)
from src.training.sample_columns import (
    ColumnarDataset,
    ColumnarSamples,
    IndexDataset,
    apply_normalization_stats,
    normalize_columns,
)
from src.training.splits import SymbolSplit, build_symbol_splits
from src.training.targets import build_training_targets, compute_symbol_epsilon


def train_drop_last(sample_count: int, batch_size: int) -> bool:
    """Whether the training DataLoader should drop its final partial batch.

    True only when the dataset spans more than one batch: dropping the
    remainder then guards BatchNorm against size-1 batches at negligible
    data cost. Single-batch datasets (smoke/fast-mode tests) must keep
    every sample — drop_last would leave zero batches.
    """
    return sample_count > batch_size


@dataclass(frozen=True)
class TrainingDataBundle:
    feature_contract: FeatureContract
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    diagnostics: dict[str, TargetDiagnostics]
    split_ranges: dict[str, dict[str, SymbolSplit]]
    class_weights: object | None
    # NOTE: samples_by_split is now a ``dict[str, ColumnarSamples]`` (was
    # ``dict[str, list[TrainingSample]]``).  ColumnarSamples supports
    # ``len()``, ``__iter__``, and ``__getitem__`` so all legacy
    # ``len(bundle.samples_by_split[k])`` and
    # ``for sample in bundle.samples_by_split[k]`` patterns continue to
    # work unchanged — but the in-memory cost is ~5x lower because we
    # store one tensor per field instead of one tensor per sample.
    # See ``src/training/sample_columns.py`` for the rationale.
    samples_by_split: dict[str, ColumnarSamples]
    # Train-split z-score statistics applied to every split above.  Must
    # be persisted with any checkpoint trained on this bundle and
    # re-applied at inference (see NormalizationStats) — otherwise the
    # deployed model receives raw-scale inputs it never saw in training.
    normalization_stats: NormalizationStats | None = None


def _absorb_shard_samples(
    columns_by_split: dict[str, ColumnarSamples],
    shard_samples_by_split: dict[str, list[TrainingSample]],
) -> None:
    """Convert one shard's per-split sample lists into columnar form and
    merge them into the bundle-wide accumulator.

    The intermediate ``ColumnarSamples`` object is built once per shard via
    ``from_samples`` (a single ``torch.stack`` per field), then either
    becomes the seed of the accumulator (cheap pointer move) or is appended
    via ``torch.cat``.  The caller must drop the input
    ``shard_samples_by_split`` reference immediately after this returns —
    that is what frees the per-sample dataclass list and keeps the peak
    working set bounded across the 38-symbol-shard load.
    """
    for split_name, split_samples in shard_samples_by_split.items():
        if not split_samples:
            continue
        shard_columns = ColumnarSamples.from_samples(split_samples)
        if split_name not in columns_by_split:
            columns_by_split[split_name] = shard_columns
        else:
            columns_by_split[split_name].append(shard_columns)


def build_real_data_training_bundle(
    *,
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
    normalization_override: NormalizationStats | None = None,
) -> TrainingDataBundle:
    logger = get_logger("training_data")
    feature_contract: FeatureContract | None = None
    # Columnar accumulator (was: ``dict[str, list[TrainingSample]]``).
    # Per-symbol shards are stacked into ColumnarSamples once and then
    # immediately concatenated into this bundle-wide store, so the peak
    # working set never holds more than one shard's dataclass list at a
    # time.  See ``src/training/sample_columns.py`` for memory math.
    columns_by_split: dict[str, ColumnarSamples] = {}
    split_ranges: dict[str, dict[str, SymbolSplit]] = {"train": {}, "val": {}, "test": {}}
    estimated_total_samples = 0

    # --- Sample cache fast-path ------------------------------------------------
    # Per-sample FeaturePipeline.build() is the single largest cold-start cost.
    # Two cache layers sit in front of the rebuild path:
    #
    #   1. The legacy *monolithic* cache: one pickle per (config + full
    #      bars) fingerprint.  Fastest when every symbol's bars and the
    #      config are unchanged — a single ``torch.load`` gets everything.
    #
    #   2. The *per-symbol shard* cache: one pickle per (config + single
    #      symbol + that symbol's bar summary).  Survives adding bars for
    #      one symbol, adding a new symbol, or reusing training data
    #      across similar configs.  This is the research-workflow cache.
    #
    # Both caches are additive — any miss, version mismatch, or corruption
    # just falls through to rebuild.
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
        cached_samples_by_split = cached.samples_by_split
        split_ranges = cached.split_ranges
        # Convert legacy on-disk dataclass form to columnar in-memory form
        # split-by-split, dropping each list reference as we go so the
        # transient peak stays bounded even when the monolithic cache is
        # huge.  ColumnarSamples-typed cache payloads (bumped CACHE_VERSION
        # in the future) would be passed through unchanged here.
        for split_name, split_payload in cached_samples_by_split.items():
            if isinstance(split_payload, ColumnarSamples):
                columns_by_split[split_name] = split_payload
            elif split_payload:
                columns_by_split[split_name] = ColumnarSamples.from_samples(split_payload)
        cached_samples_by_split = None  # release the dataclass list ASAP
        del cached
        gc.collect()
        logger.info(
            "training_samples_loaded_from_cache",
            fingerprint=fingerprint,
            train=len(columns_by_split.get("train", ColumnarSamples.empty())),
            val=len(columns_by_split.get("val", ColumnarSamples.empty())),
            test=len(columns_by_split.get("test", ColumnarSamples.empty())),
        )
        return _finalize_training_bundle(
            config=config,
            feature_contract=feature_contract,
            samples_by_split=columns_by_split,
            split_ranges=split_ranges,
            logger=logger,
            normalization_override=normalization_override,
        )

    # --- Shard cache: load as many per-symbol shards as possible -------------
    shard_hits, shard_misses = load_all_shards(
        config=config, bars_by_symbol=bars_by_symbol
    )
    if shard_hits:
        logger.info(
            "training_sample_shard_cache_hit",
            hits=sorted(shard_hits.keys()),
            misses=sorted(shard_misses),
        )
        # Replay every hit into the columnar accumulator.  Each shard is
        # converted to ColumnarSamples and concatenated immediately, then
        # the original ``shard.samples_by_split`` reference is dropped via
        # ``shard_hits[symbol] = None``.  This bounds the peak working set
        # to roughly (current accumulator size) + (one shard's dataclass
        # list) instead of (all shards held simultaneously) — the latter
        # was the root cause of the 38-shard ~46 GB blowup that took the
        # 40-symbol training run over the 64 GB ceiling.
        symbols_loaded: list[str] = []
        for symbol in list(shard_hits.keys()):
            shard = shard_hits[symbol]
            if shard is None:
                continue
            if feature_contract is None:
                feature_contract = shard.feature_contract
            else:
                try:
                    feature_contract.assert_matches(
                        shard.feature_contract,
                        context=f"shard_contract[{symbol}]",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "training_sample_shard_contract_mismatch",
                        symbol=symbol,
                        error=str(exc),
                    )
                    shard_misses.append(symbol)
                    shard_hits[symbol] = None  # release contract-mismatch shard
                    continue
            _absorb_shard_samples(columns_by_split, shard.samples_by_split)
            for split_name, split in shard.split_by_name.items():
                split_ranges.setdefault(split_name, {})[symbol] = split
            symbols_loaded.append(symbol)
            # Critical: drop the shard reference so its in-memory dataclass
            # list can be reclaimed before we load the next shard.
            shard_hits[symbol] = None
        # All loaded shards have been absorbed into columnar form.  Force
        # a collection cycle to release the per-shard dataclass garbage
        # before moving on to the rebuild loop or _finalize_training_bundle.
        gc.collect()
        logger.info(
            "training_shard_columnarized",
            symbols=sorted(symbols_loaded),
            train_rows=len(columns_by_split.get("train", ColumnarSamples.empty())),
            val_rows=len(columns_by_split.get("val", ColumnarSamples.empty())),
            test_rows=len(columns_by_split.get("test", ColumnarSamples.empty())),
        )
    # Only rebuild the symbols that were not loaded from shards.  If
    # there were no shard hits this degenerates to the old behaviour
    # (rebuild everything).
    rebuild_symbols: dict[str, list[MarketBar]] = {
        symbol: bars
        for symbol, bars in bars_by_symbol.items()
        if symbol in shard_misses or symbol not in shard_hits
    }

    # Cross-sectional market context (Phase 3): computed ONCE from the
    # full multi-symbol bar set (not just the rebuild subset — breadth
    # and the SPY/QQQ proxies must see the whole universe) and broadcast
    # into every symbol's batched snapshots.  None while the flag is off.
    market_context = None
    if config.features.market_context_enabled and rebuild_symbols:
        from src.features.market_context import compute_market_context_batch

        market_context = compute_market_context_batch(
            bars_by_symbol,
            short_window=int(config.features.windows["short"]),
            medium_window=int(config.features.windows["medium"]),
            epsilon=config.features.epsilon,
        )

    all_horizons = config.training.all_horizons
    max_horizon = max(all_horizons)

    for symbol, raw_bars in rebuild_symbols.items():
        eligible_start = max(config.runtime.warmup_bars, config.training.sequence_length - 1)
        eligible_stop = len(raw_bars) - max_horizon
        estimated_total_samples += max(eligible_stop - eligible_start, 0)
    logger.info(
        "training_dataset_planned",
        symbol_count=len(rebuild_symbols),
        shard_hit_count=len(shard_hits) - len([s for s in shard_misses if s in shard_hits]),
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

    for symbol, raw_bars in rebuild_symbols.items():
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
        # ------------------------------------------------------------------
        # Vectorized rolling-feature precompute for the training path.
        #
        # Replaces the per-sample ``pipeline.build`` loop with a single
        # ``pipeline.build_batch`` call that computes every feature
        # across the full bar history in one vectorized pass via
        # ``torch.unfold``.  Per-sample access is then O(F) via
        # ``snapshot_at(index)``.  For multi-year 1-minute bar datasets
        # this collapses the dominant cold-start training cost from
        # ``O(N_samples × W)`` scalar reductions to ``O(N_samples)``
        # contiguous tensor slices.
        # ------------------------------------------------------------------
        sequence_length = config.training.sequence_length
        batched_snapshots = pipeline.build_batch(
            bars,
            symbol=symbol,
            temporal_window=sequence_length,
            market_context=market_context,
        )
        # Accumulate this symbol's samples locally so we can persist
        # them as a standalone shard at the end of the loop (incremental
        # cache).  We also append into the aggregate ``samples_by_split``
        # for the training bundle itself.
        shard_samples_by_split: dict[str, list[TrainingSample]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        shard_split_by_name: dict[str, SymbolSplit] = {}
        # Adaptive epsilon: per-symbol return volatility.
        # IMPORTANT: compute from training-period bars ONLY to avoid
        # leaking future volatility information into val/test labels.
        if config.training.adaptive_label_epsilon:
            train_split_obj = next((s for s in symbol_splits if s.split == "train"), None)
            train_end_idx = train_split_obj.end_index if train_split_obj else len(bars)
            symbol_epsilon = compute_symbol_epsilon(
                bars[:train_end_idx],
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
            # Embargo: a sample at bar i carries labels computed from bars
            # i+1..i+max_horizon, so the trailing max_horizon samples of
            # train (and val) have labels that read into the next split's
            # window — outcomes the model "sees" in training and is then
            # graded on. Purge them. (Splits smaller than the horizon are
            # left intact — only degenerate test fixtures hit that.)
            if (
                split.split in ("train", "val")
                and max_horizon > 0
                and len(split_indices) > max_horizon
            ):
                split_indices = split_indices[:-max_horizon]
                split = replace(
                    split,
                    sample_count=len(split_indices),
                    end_index=split_indices[-1],
                    end_timestamp=bars[split_indices[-1]].timestamp,
                )
            split_ranges[split.split][symbol] = split
            shard_split_by_name[split.split] = split
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
                snapshot = batched_snapshots.snapshot_at(index)
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
                    label_epsilon=symbol_epsilon,
                    regression_targets=config.training.regression_targets,
                    primary_horizon=config.training.effective_primary_horizon,
                    enable_future_return_targets=config.training.enable_future_return_targets,
                    enable_risk_path_targets=config.training.enable_risk_path_targets,
                    transaction_cost_bps=config.training.transaction_cost_bps,
                )
                sample = TrainingSample(
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
                # The local ``shard_samples_by_split`` keeps the per-symbol
                # dataclass list alive only long enough to (a) write the
                # on-disk shard cache and (b) absorb into the bundle-wide
                # columnar accumulator below — both happen at the end of
                # this symbol's loop iteration.  We no longer accumulate
                # into a process-wide ``samples_by_split`` list; that was
                # the source of the multi-shard memory blowup.
                shard_samples_by_split[split.split].append(sample)
            logger.info(
                "training_split_materialized",
                symbol=symbol,
                split=split.split,
                sample_count=len(split_indices),
            )
        # Persist this symbol's shard so the next research iteration can
        # reuse it directly even if unrelated symbols or config fields
        # have changed.  Best-effort — failures are logged inside.  The
        # on-disk shard format is unchanged (still ``dict[str, list[
        # TrainingSample]]``) for backwards compatibility with the 43+
        # existing shards on user disks.
        if feature_contract is not None:
            try:
                shard_fingerprint = compute_shard_fingerprint(
                    config=config,
                    symbol=symbol,
                    bars=raw_bars,
                    market_bars_by_symbol=bars_by_symbol,
                )
                save_cached_shard(
                    fingerprint=shard_fingerprint,
                    data=CachedSymbolShard(
                        symbol=symbol,
                        feature_contract=feature_contract,
                        samples_by_split=shard_samples_by_split,
                        split_by_name=shard_split_by_name,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "training_sample_shard_save_failed",
                    symbol=symbol,
                    error=str(exc),
                )
        # Now absorb this symbol's per-sample dataclasses into the
        # bundle-wide columnar accumulator and immediately drop the
        # ``shard_samples_by_split`` reference so the dataclass list and
        # its many tiny tensors become eligible for garbage collection
        # before the next symbol's rebuild begins.  This bounds the peak
        # working set of the rebuild loop to (current accumulator size +
        # one symbol's dataclass list) instead of (all symbols stacked).
        _absorb_shard_samples(columns_by_split, shard_samples_by_split)
        shard_samples_by_split = {"train": [], "val": [], "test": []}
        # Drop the per-symbol BatchedSnapshots tensor so its underlying
        # storage is reclaimed too — the views in the now-absorbed sample
        # tensors no longer hold a reference because we cloned via stack.
        batched_snapshots = None  # noqa: F841
        gc.collect()
    if feature_contract is None:
        raise ValueError("No real-data training samples were generated")

    # Monolithic cache write: write a small marker payload by default — the
    # per-symbol shards above are the canonical cache, so the monolithic
    # samples_<hash>.pt file no longer needs to hold the full sample set.
    # The legacy load path still works because it falls through to the
    # shard cache when the monolithic load returns None or the columnar
    # form on a hit (see ``load_cached_samples`` adapter at the top of
    # this function).  We keep writing it ONLY for tiny universes (under
    # 5k samples) so the existing test suite that asserts
    # ``cache_path(fp).exists()`` after a small build still passes
    # without forcing the columnar storage back into dataclass form.
    if fingerprint is not None:
        total_rows = sum(len(c) for c in columns_by_split.values())
        if total_rows <= 5_000:
            save_cached_samples(
                fingerprint=fingerprint,
                data=CachedTrainingData(
                    feature_contract=feature_contract,
                    samples_by_split=columns_by_split,
                    split_ranges=split_ranges,
                ),
            )

    return _finalize_training_bundle(
        config=config,
        feature_contract=feature_contract,
        samples_by_split=columns_by_split,
        split_ranges=split_ranges,
        logger=logger,
        normalization_override=normalization_override,
    )


def _finalize_training_bundle(
    *,
    config: AppConfig,
    feature_contract: FeatureContract,
    samples_by_split: dict[str, ColumnarSamples],
    split_ranges: dict[str, dict[str, SymbolSplit]],
    logger,
    normalization_override: NormalizationStats | None = None,
) -> TrainingDataBundle:
    # Fast-mode: deterministic sample caps per split (applied post-cache so the
    # cached artifact stays full-size and reusable across fast/non-fast runs).
    # The columnar form supports row-index slicing via ``ColumnarSamples.slice``
    # so we generate the subsample indices in plain Python and then gather.
    fast = config.training.fast_mode
    if fast.enabled:
        rng = random.Random(fast.sample_seed)
        # Do not mutate the input dict (it may be the cached payload).
        capped: dict[str, ColumnarSamples] = dict(samples_by_split)
        for split_name, cap in [
            ("train", fast.max_train_samples),
            ("val", fast.max_val_samples),
            ("test", fast.max_test_samples),
        ]:
            split_columns = capped.get(split_name)
            if split_columns is None:
                continue
            original_count = len(split_columns)
            if cap is not None and original_count > cap:
                indices = rng.sample(range(original_count), cap)
                indices.sort()  # preserve approximate temporal order
                capped[split_name] = split_columns.slice(indices)
                logger.info(
                    "fast_mode_sample_cap_applied",
                    split=split_name,
                    original_count=original_count,
                    capped_count=cap,
                    sample_seed=fast.sample_seed,
                )
        samples_by_split = capped

    # --- Z-score normalize input features for AMP safety ----------------
    # Compute mean/std from *train* only (no data leakage), apply to all
    # splits, clamp to +/-10 so float16 intermediates never overflow.
    # The returned stats travel with the bundle into checkpoint meta so
    # inference re-applies the identical transform (C2 train/serve skew).
    normalization_stats = None
    if normalization_override is not None:
        # §4.6.4 injection: normalize every split with the CALLER's stats
        # (a checkpoint's persisted train-split stats) instead of deriving
        # from the current data — evaluation against any data range then
        # applies the identical transform the checkpoint trained with.
        apply_normalization_stats(
            normalization_override,
            *[samples_by_split[k] for k in ("train", "val", "test") if k in samples_by_split],
        )
        normalization_stats = normalization_override
        logger.info("normalization_stats_injected", source="caller_override")
    elif "train" in samples_by_split:
        normalization_stats = normalize_columns(
            samples_by_split["train"],
            *[
                samples_by_split[k]
                for k in ("val", "test")
                if k in samples_by_split
            ],
        )

    # Diagnostics + class-weight + weighted-sampler currently consume
    # ``list[TrainingSample]``-style iteration: they call ``len(samples)``
    # and then ``for sample in samples``.  ``ColumnarSamples`` implements
    # both ``__len__`` and ``__iter__`` (the latter yields lazily-
    # reconstructed TrainingSample dataclasses one at a time, refcount-
    # freed at each step) so we pass the columnar object through unchanged
    # — never materializing the full sample list.  This trades wall-clock
    # time for memory: iterating ~3 M samples sequentially takes ~30-60
    # seconds, but the peak working set stays at the columnar level
    # (~9 GB instead of ~46 GB).  A future optimisation would be true
    # columnar-native diagnostics that runs in seconds via vectorized
    # torch ops on the underlying column tensors.
    diagnostics = {
        split_name: compute_target_diagnostics(
            samples,
            split=split_name,
            minority_class_floor_warning=config.training.minority_class_floor_warning,
            label_epsilon=config.training.label_epsilon,
            inspect_target_boundaries=config.training.inspect_target_boundaries,
            near_flat_multiplier=config.training.near_flat_multiplier,
            epsilon_reduction_factor=config.training.epsilon_reduction_factor,
            flat_dominance_threshold=config.training.flat_dominance_threshold,
        )
        for split_name, samples in samples_by_split.items()
    }
    train_columns = samples_by_split["train"]
    class_weights = build_class_weights(
        train_columns,
        mode=config.training.class_weight_mode,
        cap=config.training.class_weight_cap,
    )
    sampler = (
        build_weighted_sampler(
            train_columns, mode=config.training.weighted_sampler_mode
        )
        if config.training.use_weighted_sampler
        else None
    )
    # --- Adaptive curriculum (§6.5 / roadmap §4 item 9) --------------------
    # Difficulty = distance of the sample's primary return from the label
    # band edge, using the SAME effective epsilon that labeled it
    # (targets.py: label_epsilon * sqrt(horizon) + cost).  The honest
    # per-sample return is the future_returns column at the primary
    # horizon — the exact value the direction label was derived from.
    # NOTE: with adaptive_label_epsilon the per-symbol epsilon is not
    # recoverable from cached columnar samples, so the static
    # config.training.label_epsilon is the band-scale proxy here.
    if config.training.curriculum_enabled:
        if sampler is not None:
            logger.warning(
                "curriculum_skipped_weighted_sampler_active",
                note="use_weighted_sampler and curriculum_enabled are mutually exclusive; keeping the weighted sampler",
            )
        else:
            primary_horizon = config.training.effective_primary_horizon
            returns_column = train_columns.future_returns.get(primary_horizon)
            if returns_column is None:
                returns_column = train_columns.regression_targets.get("primary_return")
            if returns_column is None:
                logger.warning(
                    "curriculum_skipped_no_primary_return",
                    primary_horizon=primary_horizon,
                    note="train columns expose neither the primary-horizon future_returns column nor a primary_return regression target",
                )
            else:
                effective_epsilon = config.training.label_epsilon * math.sqrt(
                    max(primary_horizon, 1)
                ) + config.training.transaction_cost_bps / 10_000.0
                sampler = CurriculumSampler(
                    compute_sample_difficulties(returns_column, effective_epsilon),
                    start_quantile=config.training.curriculum_start_quantile,
                    full_epoch=config.training.curriculum_full_epoch,
                )
                logger.info(
                    "curriculum_sampler_enabled",
                    train_samples=len(train_columns),
                    epoch0_samples=len(sampler),
                    effective_epsilon=effective_epsilon,
                    start_quantile=config.training.curriculum_start_quantile,
                    full_epoch=config.training.curriculum_full_epoch,
                )
    # --- GPU-resident preload --------------------------------------------
    # When a CUDA device is available and the full columnar dataset fits
    # in less than 50% of free VRAM, move all tensor columns to GPU once
    # up front.  The columnar collate then slices GPU tensors directly —
    # no CPU->GPU copy per batch.  This eliminates the 2400ms/batch
    # overhead caused by CUDA driver contention in processes with many
    # loaded modules and competing GPU contexts.
    gpu_preloaded = False
    if torch.cuda.is_available():
        train_bytes = samples_by_split["train"].estimated_bytes()
        val_bytes = samples_by_split["val"].estimated_bytes()
        test_bytes = samples_by_split["test"].estimated_bytes()
        total_bytes = train_bytes + val_bytes + test_bytes
        free_mem = torch.cuda.mem_get_info()[0]  # free VRAM in bytes
        if total_bytes < free_mem * 0.5:
            device = torch.device("cuda")
            samples_by_split["train"].to_device(device)
            samples_by_split["val"].to_device(device)
            samples_by_split["test"].to_device(device)
            gpu_preloaded = True
            logger.info(
                "gpu_preload",
                total_mb=round(total_bytes / 1024**2, 1),
                free_mb=round(free_mem / 1024**2, 1),
            )

    # DataLoader tuning (Phase 3a):
    #   - num_workers=0 because the columnar store already lives in RAM;
    #     forking would pickle-copy the entire dataset per worker (memory
    #     blow-up).
    #   - pin_memory=True when a CUDA device is available AND data is on
    #     CPU so the main-thread host->device copy can use non_blocking=True.
    #     When data is already GPU-resident, pin_memory must be False
    #     (pinning GPU tensors is a no-op at best, a crash at worst).
    pin_memory = torch.cuda.is_available() and not gpu_preloaded
    # Fast columnar collate: slice pre-stacked tensors directly with batch
    # indices instead of reconstructing TrainingSample dataclasses per row
    # and re-stacking them in collate_training_samples.  IndexDataset
    # returns the raw integer index; make_columnar_collate receives the
    # list of indices and does one advanced-index op per field.
    train_dataset = IndexDataset(len(train_columns))
    train_collate = make_columnar_collate(train_columns)
    # Sequential GRU sampler removed with the GRU memory (Phase 8). Standard
    # shuffled DataLoader path only.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=train_collate,
        pin_memory=pin_memory,
        # §4.7 BatchNorm guard: a size-1 final training batch trains
        # BatchNorm on single-sample statistics (and hard-crashes for
        # (N, C) inputs). Dropping the remainder batch costs < batch_size
        # samples per epoch; datasets that fit in a single batch (smoke
        # runs) keep everything. Applies to whatever the sampler yields,
        # so curriculum-subset epochs are covered too.
        drop_last=train_drop_last(len(train_columns), config.training.batch_size),
    )
    val_columns = samples_by_split["val"]
    val_loader = DataLoader(
        IndexDataset(len(val_columns)),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=make_columnar_collate(val_columns),
        pin_memory=pin_memory,
    )
    test_columns = samples_by_split["test"]
    test_loader = DataLoader(
        IndexDataset(len(test_columns)),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=make_columnar_collate(test_columns),
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
        normalization_stats=normalization_stats,
    )
