from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.core.config import FeatureConfig, ModelConfig
from src.core.exceptions import ConfigurationError
from src.features.feature_pipeline import FeatureContract
from src.memory.memory_bank import compute_encoder_fingerprint
from src.models.baseline import BaselineMLP
from src.models.brain_model import BrainTradingModel
from src.training.checkpoints import load_checkpoint_payload


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
        "checkpoint_version": "1.4",
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


def resolve_inference_model(
    *,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    feature_contract: FeatureContract,
    device: torch.device,
    default_model_kind: str = "brain",
) -> LoadedModelArtifacts:
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
        return artifacts
    model = build_model_from_contract(
        model_kind=default_model_kind,
        model_config=model_config,
        feature_contract=feature_contract,
    ).to(device)
    model.eval()
    return LoadedModelArtifacts(
        model=model,
        feature_contract=feature_contract,
        meta={},
        model_kind=default_model_kind,
    )
