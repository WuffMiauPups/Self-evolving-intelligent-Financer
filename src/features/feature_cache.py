"""On-disk cache for FeaturePipeline.build_batch() output.

Reasoning: build_batch() is deterministic given (bars, config, temporal_window).
Feature computation takes ~30-60s per symbol on full-history backtests.
For repeat backtests (ablation studies), this is pure overhead.

Cache structure:
  artifacts/features_cache/<fingerprint>.pt
where <fingerprint> = sha256 of:
  - FEATURE_SCHEMA_VERSION (code/formula version salt)
  - bars_signature (endpoints + content hash: close/volume sums,
    sampled interior closes)
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

# Contract: bump this constant whenever a feature FORMULA changes — any
# edit to src/features/* that alters computed values for the same bars
# and config (new/removed features, changed math, changed zero-fill or
# window semantics, changed epsilon handling).  The version is part of
# the cache fingerprint, so a bump makes every previously cached entry
# miss and forces a rebuild; forgetting to bump silently serves feature
# tensors computed by the OLD code to training/backtest.
# History:
#   1 — implicit pre-versioning behavior (no salt in the fingerprint).
#   2 — §4.3 parity fixes: RSI/momentum warmup zeroing, epsilon-guarded
#       return denominators, volatility_proxy_t window reset.
#   3 — Phase 3 cross-sectional market context: market_* feature family
#       (SPY/QQQ return+vol, breadth, relative strength) behind
#       features.market_context_enabled; fingerprint additionally keys
#       on the flag and on the market-context data signature.
FEATURE_SCHEMA_VERSION = 3

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
    market_context_signature: dict[str, Any] | None = None,
) -> str:
    payload = {
        # Read at call time (not bound at import) so a version bump —
        # including a monkeypatched one in tests — always changes the
        # fingerprint of subsequent computations.
        "schema_version": int(FEATURE_SCHEMA_VERSION),
        "bars": bars_signature,
        "symbol": symbol,
        "features": feature_config_snapshot,
        "window": int(temporal_window),
    }
    if market_context_signature is not None:
        # Only present when a market-context batch was supplied: the
        # cached market_* columns depend on OTHER symbols' bars, which
        # bars_signature alone cannot see.
        payload["market_context"] = market_context_signature
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
