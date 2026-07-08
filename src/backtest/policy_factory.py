"""Shared policy builders for backtest entry points.

Both ``scripts/run_backtest.py`` (CLI) and ``src/gui_services/backtest_service.py``
(GUI) need to construct policies from a loaded AppConfig.  Prior to this
module the construction logic lived only in the CLI script, so the GUI
backtest could only run the baseline policy.  Pulling the factory into
``src/backtest/policy_factory.py`` lets both call sites share one
implementation.

Three builders:
    build_baseline_policy         — no model, no memory (pure momentum)
    build_brain_policy            — BrainPolicy over a loaded checkpoint
    build_brain_memory_policy     — MemoryAwarePolicy (brain + memory + fusion)

The builders intentionally take the same signature where possible so the
caller can switch between them by changing one string.  The memory-aware
variant needs sample bars + a feature pipeline to bootstrap the encoder
with a real latent-dim value, so those are passed as keyword args.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.core.config import AppConfig
from src.core.exceptions import ConfigurationError
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.storage import load_memory_bank, save_memory_bank
from src.memory.updater import AssociativeMemoryUpdater
from src.models.factory import resolve_inference_model
from src.models.state_encoder import ModelStateEncoder
from src.strategies.baseline_policy import BaselineMomentumPolicy
from src.strategies.baselines import BuyAndHoldPolicy, MeanReversionPolicy
from src.strategies.brain_policy import BrainPolicy
from src.strategies.memory_policy import MemoryAwarePolicy

logger = logging.getLogger(__name__)


def _resolve_backtest_device(config: AppConfig) -> torch.device:
    """Resolve the device for backtest inference.

    Reads ``config.device.preferred``:

    * ``"auto"`` (default) → CUDA if available else CPU.
    * ``"cuda"`` → CUDA (raise ``ConfigurationError`` if unavailable).
    * ``"cpu"`` → CPU.
    """
    preferred = (config.device.preferred or "auto").lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError(
                "device.preferred=cuda but CUDA is not available on this machine"
            )
        return torch.device("cuda")
    # "auto" or anything else
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_baseline_policy() -> BaselineMomentumPolicy:
    """Construct the parameter-free baseline momentum policy."""
    return BaselineMomentumPolicy()


def build_buy_and_hold_policy() -> BuyAndHoldPolicy:
    """Construct a buy-and-hold baseline policy."""
    return BuyAndHoldPolicy()


def build_mean_reversion_policy() -> MeanReversionPolicy:
    """Construct a simple mean-reversion baseline policy."""
    return MeanReversionPolicy()


def _resolve_artifacts(
    config: AppConfig,
    *,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
):
    """Build a sample snapshot (for feature contract) and resolve the model.

    ``resolve_inference_model`` requires a ``feature_contract`` to validate
    the loaded checkpoint against the live feature configuration, so we
    build a single sample snapshot from the first symbol's warm-up bars
    before calling the resolver.  This matches the CLI path in
    ``scripts/run_backtest.py`` (lines 285-296).
    """
    sample_symbol = next(iter(bars_by_symbol))
    sample_snapshot = pipeline.build(
        bars_by_symbol[sample_symbol][: config.runtime.warmup_bars + 1],
        symbol=sample_symbol,
    )
    device = _resolve_backtest_device(config)
    logger.info(
        "backtest_device=%s cuda_available=%s",
        device,
        torch.cuda.is_available(),
    )
    artifacts = resolve_inference_model(
        model_config=config.model,
        feature_config=config.features,
        feature_contract=sample_snapshot.contract,
        device=device,
        default_model_kind="brain",
        allow_untrained=False,  # backtest requires real trained weights
        device_config=config.device,
    )
    return artifacts, sample_snapshot


def build_brain_policy(
    config: AppConfig,
    *,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
) -> BrainPolicy:
    """Construct a BrainPolicy backed by the checkpoint in ``config.model``.

    Raises ``ConfigurationError`` (via ``resolve_inference_model``) if the
    checkpoint cannot be loaded.  The caller should catch and present the
    error to the operator.

    ``bars_by_symbol`` and ``pipeline`` are required because
    ``resolve_inference_model`` needs a ``feature_contract`` built from a
    sample snapshot to validate the checkpoint.
    """
    artifacts, _ = _resolve_artifacts(config, bars_by_symbol=bars_by_symbol, pipeline=pipeline)
    return BrainPolicy(
        model=artifacts.model,
        device=_resolve_backtest_device(config),
        feature_contract=artifacts.feature_contract,
        normalization=artifacts.normalization,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
        force_directional_flip=config.backtest.force_directional_flip_enabled,
    )


def build_brain_memory_policy(
    config: AppConfig,
    *,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
    enable_memory: bool = True,
    conservative_thresholds: bool = False,
    verbose_fusion: bool = False,
) -> MemoryAwarePolicy:
    """Construct a MemoryAwarePolicy wrapping a BrainPolicy.

    This is the logic that used to live in ``scripts/run_backtest.py``'s
    ``_make_memory_aware_policy``.  Behavior preserved exactly:
      - Load the persisted memory bank if it exists and rebuild_on_startup
        is false; otherwise start fresh.
      - Refresh organization once if dynamic growth is enabled and the
        bank has no units yet.
      - Force fusion_config.enabled=True whenever the policy is invoked
        (the callers that want fusion off should use build_brain_policy).
      - Apply conservative fusion thresholds when requested.
    """
    artifacts, _ = _resolve_artifacts(
        config, bars_by_symbol=bars_by_symbol, pipeline=pipeline
    )
    device = _resolve_backtest_device(config)
    brain_policy = BrainPolicy(
        model=artifacts.model,
        device=device,
        feature_contract=artifacts.feature_contract,
        normalization=artifacts.normalization,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
        force_directional_flip=config.backtest.force_directional_flip_enabled,
    )

    encoder = ModelStateEncoder(
        model=artifacts.model,
        device=device,
        feature_contract=artifacts.feature_contract,
        normalization=artifacts.normalization,
        expected_latent_dim=config.memory.latent_dim,
        encoder_fingerprint=artifacts.encoder_fingerprint,
    )

    # Sample the first symbol to discover the latent dim for a fresh bank.
    sample_symbol = next(iter(bars_by_symbol))
    sample_snapshot = pipeline.build(
        bars_by_symbol[sample_symbol][: config.runtime.warmup_bars + 1],
        symbol=sample_symbol,
    )
    sample_encoded = encoder.encode(sample_snapshot)

    memory_bank: MemoryBank
    persist_path = Path(config.memory.persist_path)
    if enable_memory and persist_path.exists() and not config.memory.rebuild_on_startup:
        # IMPORTANT: keep bank on CPU. The encoder intentionally stores
        # EncodedState.latent on CPU (state_encoder.py:143), and
        # similarity.pairwise_similarity (similarity.py:23) moves candidates
        # to the query device. If bank is loaded on GPU, every retrieval
        # pays a GPU->CPU transfer of the full latent_matrix (~46 MB per
        # call for a 20K/576-dim bank) — ruins the backtest runtime.
        # Model forward still runs on GPU via the encoder; only the bank
        # stays on CPU alongside the CPU query.
        memory_bank = load_memory_bank(persist_path, map_location="cpu")
        memory_bank.feature_contract.assert_matches(
            artifacts.feature_contract,
            context=f"memory_bank[{persist_path.name}]",
        )
        # Fail fast if the bank was built with a different encoder than
        # the current checkpoint — retrieval over a stale latent space
        # silently produces garbage neighbours.
        if memory_bank.encoder_fingerprint and artifacts.encoder_fingerprint:
            if memory_bank.encoder_fingerprint != artifacts.encoder_fingerprint:
                raise ConfigurationError(
                    f"Memory bank encoder fingerprint mismatch: "
                    f"bank={memory_bank.encoder_fingerprint} model={artifacts.encoder_fingerprint}. "
                    f"Rebuild memory bank with the current model checkpoint."
                )
        # Warn when the builder did not stamp an embargo boundary — the
        # bank may contain items whose outcomes leak into the backtest
        # window.
        embargo_meta = memory_bank.meta.get("builder_embargo_end_ts")
        if embargo_meta is None:
            logger.warning(
                "memory_bank_embargo_unknown path=%s — no embargo metadata; "
                "backtest results may include future-outcome leakage",
                persist_path,
            )
    else:
        memory_bank = MemoryBank(
            feature_contract=artifacts.feature_contract,
            latent_dim=sample_encoded.latent.shape[0],
            similarity_metric=config.memory.similarity_metric,
            max_items=config.memory.max_items,
            require_encoder_match=config.memory.require_encoder_match,
            meta={"source": "backtest_runtime"},
        )

    # Backtest must not mutate the memory bank at runtime — online updates
    # would inject future-outcome information into retrieval. Leave the
    # updater None so ``record_snapshot`` and ``prepare_symbol_history``
    # become no-ops.  The offline memory builder is the only legitimate
    # writer for a backtest bank.
    updater: AssociativeMemoryUpdater | None = None
    _ = AssociativeMemoryUpdater  # keep import referenced for type hints

    # Platt calibrator (load from disk if available)
    _platt_calibrator = None
    if config.training.platt_calibration_enabled:
        from src.training.platt_calibration import PlattCalibrator
        _platt_calibrator = PlattCalibrator()
        _platt_calibrator.load(Path("runs/platt_calibration.json"))

    policy = MemoryAwarePolicy(
        base_policy=brain_policy,
        encoder=encoder,
        bank=memory_bank,
        updater=updater,
        platt_calibrator=_platt_calibrator,
    )
    _ = conservative_thresholds  # parameter reserved; fusion layer removed
    _ = verbose_fusion  # parameter reserved; fusion layer removed
    return policy
