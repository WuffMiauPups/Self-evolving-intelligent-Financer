"""On-disk cache for FeaturePipeline.build_batch() output.

Reasoning: build_batch() is deterministic given (bars, config, temporal_window).
Feature computation takes ~30-60s per symbol on full-history backtests.
For repeat backtests (ablation studies), this is pure overhead.

Cache structure:
  artifacts/features_cache/<fingerprint>.pt
where <fingerprint> = sha256 of:
  - bars_signature (first_ts, last_ts, count, first_close, last_close)
  - symbol
  - FeatureConfig fields (windows, epsilon, branch_feature_prefixes)
  - temporal_window

Environment:
  SETAI_DISABLE_FEATURE_CACHE=1 disables cache (fall-through to rebuild).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

DEFAULT_CACHE_DIR = Path("artifacts/features_cache")
ENV_DISABLE = "SETAI_DISABLE_FEATURE_CACHE"


def cache_is_disabled() -> bool:
    return bool(os.environ.get(ENV_DISABLE))


def compute_fingerprint(
    *,
    bars_signature: dict[str, Any],
    symbol: str,
    feature_config_snapshot: dict[str, Any],
    temporal_window: int,
) -> str:
    payload = {
        "bars": bars_signature,
        "symbol": symbol,
        "features": feature_config_snapshot,
        "window": int(temporal_window),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def cache_path(fingerprint: str, cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    return cache_dir / f"features_{fingerprint}.pt"


def load_cached(fingerprint: str, cache_dir: Path | None = None):
    if cache_is_disabled():
        return None
    path = cache_path(fingerprint, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def save_cached(fingerprint: str, payload: Any, cache_dir: Path | None = None) -> None:
    if cache_is_disabled():
        return
    path = cache_path(fingerprint, cache_dir=cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
    except OSError:
        pass  # best-effort cache; silent failure
