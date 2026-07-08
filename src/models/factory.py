from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.core.config import DeviceConfig, FeatureConfig, ModelConfig
from src.core.exceptions import ConfigurationError
from src.features.feature_pipeline import FeatureContract
from src.memory.memory_bank import compute_encoder_fingerprint
from src.models.baseline import BaselineMLP
from src.models.brain_model import BrainTradingModel
from src.training.checkpoints import load_checkpoint_payload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModelArtifacts:
    model: torch.nn.Module
    feature_contract: FeatureContract
    meta: dict[str, Any]
    model_kind: str

    @property
    def encoder_fingerprint(self) -> str | None:
        fp = self.meta.get("encoder_fingerprint")
        if fp is not None:
            return str(fp)
        return compute_encoder_fingerprint(self.meta) if self.meta else None


def build_model_from_contract(
    *,
    model_kind: str,
    model_config: ModelConfig,
    feature_contract: FeatureContract,
) -> torch.nn.Module:
    normalized_kind = model_kind.lower()
    if normalized_kind == "baseline":
        return BaselineMLP(
            input_dim=feature_contract.feature_dim,
            hidden_dim=model_config.baseline_hidden_dim,
            dropout=model_config.dropout,
        )
    if normalized_kind == "brain":
        resolved_model_config = model_config.model_copy(update={"regime_input_dim": feature_contract.regime_dim})
        return BrainTradingModel(
            branch_input_dims=feature_contract.branch_dims,
            config=resolved_model_config,
            temporal_input_dim=feature_contract.temporal_dim,
        )
    raise ConfigurationError(f"Unsupported model kind: {model_kind}")


def build_checkpoint_meta(
    *,
    model_kind: str,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    feature_contract: FeatureContract,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        # 2.0: Great Simplification — dendritic branches, soma, regime encoder,
        # GRU memory all removed; CNN + MLP + heads only. Old 1.x checkpoints
        # are structurally incompatible with the new model and will error on
        # load_state_dict.
        "checkpoint_version": "2.0",
        "model_kind": model_kind,
        "model_config": model_config.model_dump(mode="json"),
        "feature_config": feature_config.model_dump(mode="json"),
        "feature_contract": feature_contract.to_dict(),
    }
    if extra:
        meta.update(extra)
    meta["encoder_fingerprint"] = compute_encoder_fingerprint(meta)
    return meta


def load_model_artifacts(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> LoadedModelArtifacts:
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    meta = checkpoint.get("meta", {})
    model_kind = str(meta.get("model_kind", "brain"))
    if "model_config" not in meta or "feature_contract" not in meta:
        raise ConfigurationError(f"Checkpoint {checkpoint_path} is missing model reconstruction metadata")
    model_config = ModelConfig.model_validate(meta["model_config"])
    feature_contract = FeatureContract.from_dict(meta["feature_contract"])
    if model_kind == "brain" and model_config.use_cnn_encoder:
        model_state = checkpoint.get("model_state", {})
        has_temporal_weights = any(str(name).startswith("temporal_encoder.") for name in model_state)
        if not has_temporal_weights:
            model_config = model_config.model_copy(update={"use_cnn_encoder": False})
    model = build_model_from_contract(
        model_kind=model_kind,
        model_config=model_config,
        feature_contract=feature_contract,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return LoadedModelArtifacts(
        model=model,
        feature_contract=feature_contract,
        meta=meta,
        model_kind=model_kind,
    )


def _maybe_compile_inference_model(
    model: torch.nn.Module,
    device_config: DeviceConfig | None,
) -> torch.nn.Module:
    """Wrap ``model`` with ``torch.compile`` when enabled, with graceful fallback.

    Intended for the inference path only. ``mode='reduce-overhead'`` uses CUDA
    graphs for stable small-batch shapes (we run 1 sample per symbol).  If
    compilation fails for any reason (unsupported control flow, missing backend,
    older torch version), fall back to the uncompiled model so the backtest /
    paper run can still proceed.

    Important: ``torch.compile`` is LAZY — the backend (Inductor) only runs on
    the first forward pass, so wrapping-time try/except doesn't catch backend
    errors like "cl.exe not found" on Windows without Visual Studio build
    tools.  ``torch._dynamo.config.suppress_errors = True`` makes Dynamo fall
    back to eager mode silently when the backend fails at first-call time.
    """
    if not getattr(device_config, "torch_compile_enabled", False):
        return model
    try:
        # Silently fall back to eager on backend failures (e.g. missing cl.exe
        # on Windows, missing Triton on CPU-only, etc.). Without this flag the
        # first forward pass raises and kills the backtest.
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
        except Exception:  # noqa: BLE001
            pass
        compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
        logger.info("torch_compile_applied mode=reduce-overhead suppress_errors=True")
        return compiled
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "torch_compile_failed fallback=uncompiled error=%s",
            exc,
        )
        return model


def resolve_inference_model(
    *,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    feature_contract: FeatureContract,
    device: torch.device,
    default_model_kind: str = "brain",
    allow_untrained: bool = True,
    device_config: DeviceConfig | None = None,
) -> LoadedModelArtifacts:
    """Resolve a model for inference.

    If ``model_config.checkpoint_path`` is set, load and validate the
    checkpoint against the feature contract.  Otherwise this returns a
    randomly-initialized model — that path is only safe for tests and CLI
    smoke checks.  When invoked without a checkpoint, a clear warning is
    emitted so the operator knows the inference model has no learned
    weights.  Set ``allow_untrained=False`` to make the missing-checkpoint
    case raise instead of falling through to a fresh init (use this on the
    paper/live runtime path).
    """
    if model_config.checkpoint_path:
        artifacts = load_model_artifacts(model_config.checkpoint_path, device=device)
        checkpoint_feature_config = artifacts.meta.get("feature_config")
        if checkpoint_feature_config is not None and checkpoint_feature_config != feature_config.model_dump(mode="json"):
            raise ConfigurationError(
                f"Checkpoint feature configuration mismatch for {Path(model_config.checkpoint_path)}"
            )
        artifacts.feature_contract.assert_matches(
            feature_contract,
            context=f"checkpoint_contract[{Path(model_config.checkpoint_path).name}]",
        )
        compiled_model = _maybe_compile_inference_model(artifacts.model, device_config)
        if compiled_model is artifacts.model:
            return artifacts
        return LoadedModelArtifacts(
            model=compiled_model,
            feature_contract=artifacts.feature_contract,
            meta=artifacts.meta,
            model_kind=artifacts.model_kind,
        )
    if not allow_untrained:
        raise ConfigurationError(
            "model.checkpoint_path is empty but a trained model is required for this entry point"
        )
    logger.warning(
        "resolve_inference_model_untrained",
    )
    logger.warning(
        "no checkpoint_path configured; returning a randomly-initialized %s model. "
        "This produces meaningless predictions and must NOT be used for live or paper trading.",
        default_model_kind,
    )
    model = build_model_from_contract(
        model_kind=default_model_kind,
        model_config=model_config,
        feature_contract=feature_contract,
    ).to(device)
    model.eval()
    model = _maybe_compile_inference_model(model, device_config)
    return LoadedModelArtifacts(
        model=model,
        feature_contract=feature_contract,
        meta={"untrained": True},
        model_kind=default_model_kind,
    )
