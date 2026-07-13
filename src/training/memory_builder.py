from __future__ import annotations

import torch

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.storage import save_memory_bank
from src.memory.updater import build_memory_bank_from_bars
from src.models.state_encoder import ModelStateEncoder
from src.training.data_module import TrainingDataBundle


def _select_bars_for_memory_build(
    *,
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    split_ranges: dict[str, dict[str, object]],
) -> dict[str, list[MarketBar]]:
    if config.training.memory_build_split == "all":
        return {
            symbol: list(bars)
            for symbol, bars in bars_by_symbol.items()
        }
    max_horizon = max(config.training.forward_horizons)
    history_padding = max(config.runtime.warmup_bars, config.training.sequence_length - 1)
    selected: dict[str, list[MarketBar]] = {}
    for symbol, bars in bars_by_symbol.items():
        split = split_ranges[config.training.memory_build_split].get(symbol)
        if split is None or getattr(split, "sample_count", 0) == 0:
            continue
        left = max(0, int(getattr(split, "start_index")) - history_padding)
        right = min(len(bars), int(getattr(split, "end_index")) + max_horizon + 1)
        selected[symbol] = list(bars[left:right])
    return selected


def build_post_training_memory(
    config: AppConfig,
    model: torch.nn.Module,
    bundle: TrainingDataBundle,
    bars_by_symbol: dict[str, list[MarketBar]],
    device: torch.device,
    encoder_fingerprint: str | None = None,
) -> str | None:
    logger = get_logger("training")
    memory_path: str | None = None
    if config.training.save_memory_after_training:
        selected_bars = _select_bars_for_memory_build(
            config=config,
            bars_by_symbol=bars_by_symbol,
            split_ranges=bundle.split_ranges,
        )
        memory_config = config.memory.model_copy(
            update={
                "build_from_real_checkpoint": config.training.use_real_historical_data,
                "forward_horizons": list(config.training.forward_horizons),
                "path_label_rules": dict(config.training.path_label_rules),
            }
        )
        encoder = ModelStateEncoder(
            model=model,
            device=device,
            feature_contract=bundle.feature_contract,
            normalization=bundle.normalization_stats,
            expected_latent_dim=memory_config.latent_dim,
            # Without the fingerprint the bank is saved unstamped and every
            # later require_encoder_match check is silently bypassed.
            encoder_fingerprint=encoder_fingerprint,
        )
        pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
        memory_bank = build_memory_bank_from_bars(
            bars_by_symbol=selected_bars,
            pipeline=pipeline,
            encoder=encoder,
            memory_config=memory_config,
            warmup_bars=max(config.runtime.warmup_bars, config.training.sequence_length - 1),
        )
        save_memory_bank(config.memory.persist_path, memory_bank)
        memory_path = config.memory.persist_path
        logger.info(
            "memory_prebuild_complete",
            path=config.memory.persist_path,
            item_count=len(memory_bank),
            split=config.training.memory_build_split,
        )
    return memory_path
