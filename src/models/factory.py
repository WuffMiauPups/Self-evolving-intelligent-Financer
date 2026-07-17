from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.core.config import DeviceConfig, FeatureConfig, ModelConfig
from src.core.exceptions import ConfigurationError
from src.features.feature_pipeline import FeatureContract
from src.features.normalization import NormalizationStats
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

    @property
    def risk_target_scales(self) -> dict[str, dict[str, float]] | None:
        """Per-symbol mfe/mae target scales (§7.0.4 normalization).

        ``None`` = checkpoint trained on raw excursions (all current
        checkpoints); the policy then applies no de-normalization.
        """
        scales = self.meta.get("risk_target_scales")
        if not isinstance(scales, dict) or not scales:
            return None
        return {
            str(symbol): {str(t): float(v) for t, v in targets.items()}
            for symbol, targets in scales.items()
            if isinstance(targets, dict)
        }

    @property
    def normalization(self) -> NormalizationStats | None:
        """Train-split z-score stats persisted at save time (C2 fix).

        ``None`` means a pre-fix checkpoint trained on normalized inputs
        with no recorded stats — its live behavior is undefined relative
        to its validation metrics and it should be retrained.
        """
        stats = NormalizationStats.from_meta(self.meta)
        if stats is None and self.meta and not self.meta.get("untrained"):
            logger.warning(
                "checkpoint has no normalization stats; inference inputs will be "
                "raw-scale while the model was trained on z-scored inputs — retrain "
                "this checkpoint (pre-C2-fix artifact)."
            )
        return stats


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


def state_dict_digest(state_dict: dict[str, Any]) -> str:
    """sha1 over parameter names + tensor bytes: the weights' identity."""
    import hashlib

    digest = hashlib.sha1()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        if isinstance(tensor, torch.Tensor):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()[:16]


def build_checkpoint_meta(
    *,
    model_kind: str,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    feature_contract: FeatureContract,
    extra: dict[str, Any] | None = None,
    weights_digest: str | None = None,
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
    # Weights identity (2026-07-14 fingerprint-collision fix): without it,
    # two same-config trainings share an encoder fingerprint and stale
    # banks/calibrators silently pass their binding checks.
    if weights_digest is not None:
        meta["weights_digest"] = weights_digest
    if extra:
        meta.update(extra)
    meta["encoder_fingerprint"] = compute_encoder_fingerprint(meta)
    return meta


def _parse_checkpoint_model_config(
    raw_model_config: dict[str, Any],
    checkpoint_path: str | Path,
) -> ModelConfig:
    """Parse a checkpoint's stored model_config with forward-compat filtering.

    ``ModelConfig`` validates under ``extra='forbid'``, which is correct for
    user-authored YAML but wrong at this ONE boundary: a checkpoint saved by
    a newer code version (or before a field was *removed*) may carry keys the
    current ``ModelConfig`` no longer knows. Rejecting the whole checkpoint
    for an unused key would brick every old artifact on a field removal, so
    unknown keys are dropped here with a warning instead.
    """
    known_fields = set(ModelConfig.model_fields)
    unknown_keys = sorted(set(raw_model_config) - known_fields)
    if unknown_keys:
        logger.warning(
            "checkpoint %s meta.model_config contains unknown keys %s; "
            "dropping them for forward-compatibility (saved by a different "
            "code version?)",
            checkpoint_path,
            unknown_keys,
        )
        raw_model_config = {
            key: value for key, value in raw_model_config.items() if key in known_fields
        }
    return ModelConfig.model_validate(raw_model_config)


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
    model_config = _parse_checkpoint_model_config(dict(meta["model_config"]), checkpoint_path)
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
        if checkpoint_feature_config is not None:
            expected_feature_config = feature_config.model_dump(mode="json")
            if checkpoint_feature_config != expected_feature_config:
                # Round-trip the stored dict through FeatureConfig so a
                # checkpoint saved BEFORE a new defaulted field existed
                # (e.g. market_context_enabled) still compares equal while
                # the current config keeps the default.  A checkpoint that
                # is genuinely different — including one trained with the
                # other value of such a flag — still refuses here, and the
                # feature-contract check below refuses on name/dim drift.
                try:
                    normalized_checkpoint_config = FeatureConfig(
                        **checkpoint_feature_config
                    ).model_dump(mode="json")
                except Exception:  # noqa: BLE001 — unparseable == mismatch
                    normalized_checkpoint_config = checkpoint_feature_config
                if normalized_checkpoint_config != expected_feature_config:
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
