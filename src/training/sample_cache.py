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
CACHE_VERSION = 3  # v3: label-affecting knobs added to the fingerprint

# Bump this when the on-disk *shard* payload structure changes.  Shards
# are a sibling cache format to the monolithic ``samples_<hash>.pt``
# file — they store one symbol's training samples per file, keyed on a
# fingerprint that combines (config, symbol, bar-content-summary).  A
# change to only one symbol's bars invalidates that one shard instead of
# forcing a full rebuild across the whole universe.
SHARD_CACHE_VERSION = 2  # v2: split-boundary embargo purge changes shard contents

DEFAULT_CACHE_DIR = Path("artifacts/training_cache")
DEFAULT_SHARD_DIR = DEFAULT_CACHE_DIR / "shards"
CACHE_DISABLED_ENV = "SETAI_DISABLE_SAMPLE_CACHE"


@dataclass(frozen=True)
class CachedTrainingData:
    feature_contract: FeatureContract
    samples_by_split: dict[str, list[TrainingSample]]
    split_ranges: dict[str, dict[str, SymbolSplit]]


@dataclass(frozen=True)
class CachedSymbolShard:
    """On-disk payload for a single symbol's training samples.

    The shard is keyed on a fingerprint that combines the config
    fingerprint (features + training + runtime) with the symbol name
    and a summary of the symbol's bar content.  A change to one
    symbol's bars invalidates only that shard; all other symbols are
    reloaded from disk unchanged.  This is the single biggest research-
    workflow win — tweaking a threshold or adding bars for one symbol
    no longer forces a rebuild across a multi-year multi-symbol
    universe.
    """
    symbol: str
    feature_contract: FeatureContract
    samples_by_split: dict[str, list[TrainingSample]]
    split_by_name: dict[str, SymbolSplit]


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
        # Label-affecting knobs that were missing from the fingerprint:
        # changing any of these alters cached labels, so omitting them
        # served stale samples across config changes.
        "transaction_cost_bps": float(t.transaction_cost_bps),
        "adaptive_label_epsilon": bool(t.adaptive_label_epsilon),
        "adaptive_epsilon_multiplier": float(t.adaptive_epsilon_multiplier),
        "adaptive_epsilon_min": float(t.adaptive_epsilon_min),
        "adaptive_epsilon_max": float(t.adaptive_epsilon_max),
        "label_epsilon_min_bars": int(t.label_epsilon_min_bars),
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
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("cache_load_legacy_format", path=str(path))
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


# ---------------------------------------------------------------------------
# Incremental per-symbol shard cache
# ---------------------------------------------------------------------------


def _config_fingerprint_payload(config: AppConfig) -> dict[str, Any]:
    """Config-only fingerprint (no bars).  Used as the prefix for every
    per-symbol shard fingerprint.  A change to features / training /
    runtime invalidates every shard."""
    return {
        "shard_version": SHARD_CACHE_VERSION,
        "cache_version": CACHE_VERSION,
        "features": _feature_config_fingerprint(config),
        "training": _training_config_fingerprint(config),
        "runtime": _runtime_config_fingerprint(config),
    }


def _symbol_bars_fingerprint_payload(bars: list[MarketBar]) -> dict[str, Any]:
    if not bars:
        return {"count": 0}
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    first = sorted_bars[0]
    last = sorted_bars[-1]
    return {
        "count": len(sorted_bars),
        "first_ts": first.timestamp.isoformat(),
        "last_ts": last.timestamp.isoformat(),
        "first_close": float(first.close),
        "last_close": float(last.close),
    }


def compute_shard_fingerprint(
    *, config: AppConfig, symbol: str, bars: list[MarketBar]
) -> str:
    """Per-symbol shard key.  Invariant: identical (config, symbol, bar
    summary) → identical fingerprint → the same shard file."""
    payload = {
        "config": _config_fingerprint_payload(config),
        "symbol": symbol,
        "bars": _symbol_bars_fingerprint_payload(bars),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def shard_cache_path(
    fingerprint: str, *, cache_dir: Path = DEFAULT_SHARD_DIR
) -> Path:
    return cache_dir / f"shard_{fingerprint}.pt"


def load_cached_shard(
    *, fingerprint: str, cache_dir: Path | None = None
) -> CachedSymbolShard | None:
    """Best-effort load of a single per-symbol shard.  Returns None on
    any mismatch / miss so callers always fall through to rebuild."""
    logger = get_logger("training_sample_cache")
    if cache_is_disabled():
        return None
    if cache_dir is None:
        cache_dir = DEFAULT_SHARD_DIR
    path = shard_cache_path(fingerprint, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("cache_load_legacy_format", path=str(path))
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sample_cache_shard_load_failed", path=str(path), error=str(exc)
        )
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("shard_version", -1)) != SHARD_CACHE_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    try:
        symbol = str(payload["symbol"])
        contract = payload["feature_contract"]
        samples_by_split = payload["samples_by_split"]
        split_by_name = payload["split_by_name"]
    except KeyError:
        return None
    if not isinstance(contract, FeatureContract):
        return None
    if not isinstance(samples_by_split, dict):
        return None
    if not isinstance(split_by_name, dict):
        return None
    return CachedSymbolShard(
        symbol=symbol,
        feature_contract=contract,
        samples_by_split=samples_by_split,
        split_by_name=split_by_name,
    )


def save_cached_shard(
    *,
    fingerprint: str,
    data: CachedSymbolShard,
    cache_dir: Path | None = None,
) -> None:
    """Best-effort write of a single per-symbol shard."""
    logger = get_logger("training_sample_cache")
    if cache_is_disabled():
        return
    if cache_dir is None:
        cache_dir = DEFAULT_SHARD_DIR
    path = shard_cache_path(fingerprint, cache_dir=cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "shard_version": SHARD_CACHE_VERSION,
            "fingerprint": fingerprint,
            "symbol": data.symbol,
            "feature_contract": data.feature_contract,
            "samples_by_split": data.samples_by_split,
            "split_by_name": data.split_by_name,
        }
        torch.save(payload, tmp)
        tmp.replace(path)
        total = sum(len(v) for v in data.samples_by_split.values())
        logger.info(
            "sample_cache_shard_write",
            path=str(path),
            fingerprint=fingerprint,
            symbol=data.symbol,
            total_samples=total,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sample_cache_shard_write_failed", path=str(path), error=str(exc)
        )


# ---------------------------------------------------------------------------
# Automatic cache cleanup
# ---------------------------------------------------------------------------

# Default size limit: 10 GB.  When the training cache exceeds this, the
# oldest files are deleted until it fits.  Cache files are safe to delete
# at any time — a miss just triggers a slower rebuild on the next training
# run.  Set SETAI_TRAINING_CACHE_MAX_GB to override, or 0 to disable.
DEFAULT_CACHE_MAX_GB = 10.0
CACHE_MAX_GB_ENV = "SETAI_TRAINING_CACHE_MAX_GB"


def _get_cache_max_bytes() -> int | None:
    """Read the max cache size from env, or use the default."""
    raw = os.environ.get(CACHE_MAX_GB_ENV)
    if raw is not None:
        try:
            gb = float(raw)
            if gb <= 0:
                return None  # disabled
            return int(gb * 1_073_741_824)
        except ValueError:
            pass
    return int(DEFAULT_CACHE_MAX_GB * 1_073_741_824)


def _list_cache_files(cache_dir: Path) -> list[tuple[Path, float, int]]:
    """List all .pt files under cache_dir (recursively), with mtime and size."""
    files: list[tuple[Path, float, int]] = []
    if not cache_dir.exists():
        return files
    for path in cache_dir.rglob("*.pt"):
        if path.is_file():
            try:
                stat = path.stat()
                files.append((path, stat.st_mtime, stat.st_size))
            except OSError:
                continue
    # Also pick up .pt.tmp files (leftover from interrupted writes)
    for path in cache_dir.rglob("*.pt.tmp"):
        if path.is_file():
            try:
                stat = path.stat()
                files.append((path, stat.st_mtime, stat.st_size))
            except OSError:
                continue
    return files


def enforce_cache_limit(
    *, cache_dir: Path | None = None, max_bytes: int | None = None
) -> int:
    """Delete oldest cache files until the total is under the limit.

    Returns the number of bytes freed.  Skips entirely when the limit
    is not exceeded or when cleanup is disabled (max_bytes=0/None via env).

    Called automatically at the start of every training run via
    ``run_supervised_training`` and at startup of the GUI training service.
    """
    logger = get_logger("training_sample_cache")
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    if max_bytes is None:
        max_bytes = _get_cache_max_bytes()
    if max_bytes is None:
        return 0  # cleanup disabled

    files = _list_cache_files(cache_dir)
    total_size = sum(size for _, _, size in files)
    if total_size <= max_bytes:
        return 0

    # Sort by modification time ascending (oldest first)
    files.sort(key=lambda t: t[1])

    freed = 0
    deleted_count = 0
    for path, _mtime, size in files:
        if total_size - freed <= max_bytes:
            break
        try:
            path.unlink()
            freed += size
            deleted_count += 1
        except OSError:
            continue

    if freed > 0:
        logger.info(
            "training_cache_cleanup",
            deleted_files=deleted_count,
            freed_bytes=freed,
            freed_gb=round(freed / 1_073_741_824, 2),
            remaining_bytes=total_size - freed,
            limit_bytes=max_bytes,
        )
    return freed


def load_all_shards(
    *,
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    cache_dir: Path | None = None,
) -> tuple[dict[str, CachedSymbolShard], list[str]]:
    """Attempt to load every symbol's shard.

    Returns ``(hits, misses)`` where ``hits`` maps symbol -> loaded
    shard and ``misses`` is the list of symbols that need to be
    rebuilt.  A shard hit requires a matching fingerprint AND a
    feature-contract check on the loaded contract.
    """
    hits: dict[str, CachedSymbolShard] = {}
    misses: list[str] = []
    reference_contract: FeatureContract | None = None
    for symbol in bars_by_symbol:
        fingerprint = compute_shard_fingerprint(
            config=config, symbol=symbol, bars=bars_by_symbol[symbol]
        )
        shard = load_cached_shard(fingerprint=fingerprint, cache_dir=cache_dir)
        if shard is None:
            misses.append(symbol)
            continue
        # All shards must agree on the feature contract; mix-and-match
        # across contract versions is never safe.  On mismatch, drop
        # the stale shard into misses.
        if reference_contract is None:
            reference_contract = shard.feature_contract
        else:
            try:
                reference_contract.assert_matches(
                    shard.feature_contract, context=f"shard[{symbol}]"
                )
            except Exception:  # noqa: BLE001 - mismatch falls through to rebuild
                misses.append(symbol)
                continue
        hits[symbol] = shard
    return hits, misses
