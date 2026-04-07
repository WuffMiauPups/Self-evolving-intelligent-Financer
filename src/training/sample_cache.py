"""On-disk cache for pre-materialized training samples.

Cold-start bottleneck root cause:
    ``build_real_data_training_bundle`` recomputes ``FeaturePipeline.build``
    per eligible bar (hundreds of thousands of windows on 2y × 1m bars).
    This dominates cold-start by 5–7 minutes per run. The samples themselves
    are deterministic given the input bars + a handful of config fields, so
    they can safely be cached to disk and reloaded across training runs.

Design rules:
- The cache is *additive*. Any unknown state (missing file, version mismatch,
  fingerprint miss, corruption) falls back to the slow rebuild path. The
  training behavior never regresses — a broken cache just means the old
  timing.
- The cache key is a content-addressed fingerprint of the exact inputs that
  affect sample content: symbol list, bar ranges, warmup/sequence/horizon
  fields, label rules, target flags, and the feature-pipeline-relevant
  FeatureConfig subset. Anything that changes sample content MUST be in the
  fingerprint.
- Cache files live under ``artifacts/training_cache/`` and are named
  ``samples_<fingerprint>.pt``. They are safe to delete at any time.
- The cache never serializes optimizer/model state or PII.
- Loader and writer are best-effort; failures log a warning and fall through
  to the non-cached path.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.data.dataset import TrainingSample
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeatureContract
from src.training.splits import SymbolSplit

# Bump this when the *structure* of cached payloads changes in a way that
# existing pickles can no longer be used safely.
CACHE_VERSION = 2

DEFAULT_CACHE_DIR = Path("artifacts/training_cache")
CACHE_DISABLED_ENV = "SETAI_DISABLE_SAMPLE_CACHE"


@dataclass(frozen=True)
class CachedTrainingData:
    feature_contract: FeatureContract
    samples_by_split: dict[str, list[TrainingSample]]
    split_ranges: dict[str, dict[str, SymbolSplit]]


def _feature_config_fingerprint(cfg: AppConfig) -> dict[str, Any]:
    f = cfg.features
    return {
        "windows": dict(f.windows),
        "volatility_windows": list(f.volatility_windows),
        "momentum_windows": list(f.momentum_windows),
        "volume_windows": list(f.volume_windows),
        "epsilon": float(f.epsilon),
        "branch_feature_prefixes": {
            k: list(v) for k, v in f.branch_feature_prefixes.items()
        },
    }


def _training_config_fingerprint(cfg: AppConfig) -> dict[str, Any]:
    t = cfg.training
    return {
        "sequence_length": int(t.sequence_length),
        "forward_horizons": list(t.forward_horizons),
        "auxiliary_horizons": list(t.auxiliary_horizons),
        "primary_direction_horizon": (
            None if t.primary_direction_horizon is None else int(t.primary_direction_horizon)
        ),
        "effective_primary_horizon": int(t.effective_primary_horizon),
        "label_epsilon": float(t.label_epsilon),
        "path_label_rules": {k: float(v) for k, v in t.path_label_rules.items()},
        "regression_targets": list(t.regression_targets),
        "enable_future_return_targets": bool(t.enable_future_return_targets),
        "enable_risk_path_targets": bool(t.enable_risk_path_targets),
        "train_split": float(t.train_split),
        "val_split": float(t.val_split),
        "test_split": float(t.test_split),
        # Fast-mode content-affecting knobs:
        "fast_mode": {
            "enabled": bool(t.fast_mode.enabled),
            "symbols": list(t.fast_mode.symbols),
            "sample_seed": int(t.fast_mode.sample_seed),
            "max_train_samples": t.fast_mode.max_train_samples,
            "max_val_samples": t.fast_mode.max_val_samples,
            "max_test_samples": t.fast_mode.max_test_samples,
        },
    }


def _runtime_config_fingerprint(cfg: AppConfig) -> dict[str, Any]:
    return {"warmup_bars": int(cfg.runtime.warmup_bars)}


def _bars_fingerprint(bars_by_symbol: dict[str, list[MarketBar]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for symbol in sorted(bars_by_symbol.keys()):
        bars = bars_by_symbol[symbol]
        if not bars:
            out.append({"symbol": symbol, "count": 0})
            continue
        sorted_bars = sorted(bars, key=lambda b: b.timestamp)
        first = sorted_bars[0]
        last = sorted_bars[-1]
        out.append(
            {
                "symbol": symbol,
                "count": len(sorted_bars),
                "first_ts": first.timestamp.isoformat(),
                "last_ts": last.timestamp.isoformat(),
                # Include close prices to detect silent cache reshape /
                # re-download of the same range with different contents.
                "first_close": float(first.close),
                "last_close": float(last.close),
            }
        )
    return out


def compute_fingerprint(
    *, config: AppConfig, bars_by_symbol: dict[str, list[MarketBar]]
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "features": _feature_config_fingerprint(config),
        "training": _training_config_fingerprint(config),
        "runtime": _runtime_config_fingerprint(config),
        "bars": _bars_fingerprint(bars_by_symbol),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def cache_is_disabled() -> bool:
    return bool(os.environ.get(CACHE_DISABLED_ENV))


def cache_path(fingerprint: str, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"samples_{fingerprint}.pt"


def load_cached_samples(
    *, fingerprint: str, cache_dir: Path | None = None
) -> CachedTrainingData | None:
    """Best-effort load. Returns None on any failure / miss."""
    logger = get_logger("training_sample_cache")
    if cache_is_disabled():
        logger.info("sample_cache_disabled_by_env", env=CACHE_DISABLED_ENV)
        return None
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    path = cache_path(fingerprint, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sample_cache_load_failed", path=str(path), error=str(exc))
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != CACHE_VERSION:
        logger.info(
            "sample_cache_version_mismatch",
            path=str(path),
            expected=CACHE_VERSION,
            found=payload.get("version"),
        )
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    try:
        contract = payload["feature_contract"]
        samples_by_split = payload["samples_by_split"]
        split_ranges = payload["split_ranges"]
    except KeyError as exc:
        logger.warning("sample_cache_payload_missing_key", path=str(path), error=str(exc))
        return None
    if not isinstance(contract, FeatureContract):
        return None
    if not isinstance(samples_by_split, dict):
        return None
    total = sum(len(v) for v in samples_by_split.values() if isinstance(v, list))
    logger.info(
        "sample_cache_hit",
        path=str(path),
        fingerprint=fingerprint,
        total_samples=total,
        split_sizes={k: len(v) for k, v in samples_by_split.items()},
    )
    return CachedTrainingData(
        feature_contract=contract,
        samples_by_split=samples_by_split,
        split_ranges=split_ranges,
    )


def save_cached_samples(
    *,
    fingerprint: str,
    data: CachedTrainingData,
    cache_dir: Path | None = None,
) -> None:
    """Best-effort write. Failures are logged and never raise."""
    logger = get_logger("training_sample_cache")
    if cache_is_disabled():
        return
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    path = cache_path(fingerprint, cache_dir=cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "feature_contract": data.feature_contract,
            "samples_by_split": data.samples_by_split,
            "split_ranges": data.split_ranges,
        }
        torch.save(payload, tmp)
        tmp.replace(path)
        total = sum(len(v) for v in data.samples_by_split.values())
        logger.info(
            "sample_cache_write",
            path=str(path),
            fingerprint=fingerprint,
            total_samples=total,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sample_cache_write_failed", path=str(path), error=str(exc))
