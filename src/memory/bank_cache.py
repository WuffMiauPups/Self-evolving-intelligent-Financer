"""On-disk cache for fully-built memory banks.

Rebuilding the associative memory bank is expensive: per-bar feature
computation + per-chunk encoder forward + per-symbol outcome summaries +
post-build organization (calibration, dynamic units, ESOM topology). On
2-year / 1-minute bars across multiple symbols this runs into minutes.

The cache is deterministic: given the same inputs (bars + memory config +
model checkpoint + feature-pipeline config + warmup), the produced bank
is byte-identical up to floating-point associativity in the encoder. So
we fingerprint those inputs and save the produced bank under
``artifacts/memory_bank_cache/bank_<fingerprint>.pt``.

Rules:
- Additive only. Miss / corruption / version mismatch falls through to
  the slow rebuild.
- No silent staleness: the fingerprint includes a checkpoint hash (if a
  path is resolvable) and the FeatureConfig + MemoryConfig fields that
  affect the bank contents. Changing any of them invalidates the cache.
- Disable via ``SETAI_DISABLE_BANK_CACHE=1``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.data.schemas import MarketBar
from src.memory.memory_bank import MemoryBank
from src.memory.storage import load_memory_bank, save_memory_bank

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path("artifacts/memory_bank_cache")
CACHE_DISABLED_ENV = "SETAI_DISABLE_BANK_CACHE"


def _memory_config_fingerprint(cfg: AppConfig) -> dict[str, Any]:
    m = cfg.memory
    return {
        "latent_dim": m.latent_dim,
        "top_k": int(m.top_k),
        "similarity_metric": str(m.similarity_metric),
        "min_similarity": float(m.min_similarity),
        "cluster_method": str(m.cluster_method),
        "max_items": int(m.max_items),
        "forward_horizons": list(m.forward_horizons),
        "path_label_rules": {k: float(v) for k, v in m.path_label_rules.items()},
        "min_cluster_size": int(m.min_cluster_size),
        "attention_temperature": float(m.attention_temperature),
        "use_hopfield_retrieval": bool(m.use_hopfield_retrieval),
        "use_esom_map": bool(m.use_esom_map),
        "dynamic_growth_enabled": bool(m.dynamic_growth_enabled),
        "max_units": int(m.max_units),
        "calibration_enabled": bool(m.calibration_enabled),
        "symbol_specific_profiles": bool(m.symbol_specific_profiles),
        "regime_specific_profiles": bool(m.regime_specific_profiles),
        "require_encoder_match": bool(m.require_encoder_match),
    }


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


def _bars_fingerprint(bars_by_symbol: dict[str, Sequence[MarketBar]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for symbol in sorted(bars_by_symbol.keys()):
        bars = list(bars_by_symbol[symbol])
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
                "first_close": float(first.close),
                "last_close": float(last.close),
            }
        )
    return out


def _checkpoint_fingerprint(checkpoint_path: str | None) -> str:
    """Hash the first 1MB of the checkpoint file. That's enough to detect a
    checkpoint swap without reading hundreds of megabytes every build. If
    the path is missing or unreadable we return a sentinel so the cache key
    changes and we fall through to a rebuild."""
    if not checkpoint_path:
        return "no_checkpoint"
    try:
        p = Path(checkpoint_path)
        if not p.exists():
            return "missing_checkpoint"
        h = hashlib.sha256()
        with p.open("rb") as handle:
            # 1 MB prefix + file size. File size is cheap and catches "same
            # first MB but longer file" edits.
            h.update(handle.read(1024 * 1024))
            h.update(str(p.stat().st_size).encode("utf-8"))
        return h.hexdigest()[:24]
    except OSError:
        return "checkpoint_io_error"


def compute_bank_fingerprint(
    *,
    config: AppConfig,
    bars_by_symbol: dict[str, Sequence[MarketBar]],
    warmup_bars: int,
    checkpoint_path: str | None = None,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "memory": _memory_config_fingerprint(config),
        "features": _feature_config_fingerprint(config),
        "warmup_bars": int(warmup_bars),
        "bars": _bars_fingerprint(bars_by_symbol),
        "checkpoint": _checkpoint_fingerprint(checkpoint_path),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def cache_is_disabled() -> bool:
    return bool(os.environ.get(CACHE_DISABLED_ENV))


def bank_cache_path(fingerprint: str, *, cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    return cache_dir / f"bank_{fingerprint}.pt"


def load_cached_bank(
    *, fingerprint: str, cache_dir: Path | None = None
) -> MemoryBank | None:
    logger = get_logger("memory_bank_cache")
    if cache_is_disabled():
        logger.info("bank_cache_disabled_by_env", env=CACHE_DISABLED_ENV)
        return None
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    path = bank_cache_path(fingerprint, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        bank = load_memory_bank(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bank_cache_load_failed", path=str(path), error=str(exc))
        return None
    logger.info(
        "bank_cache_hit",
        path=str(path),
        fingerprint=fingerprint,
        items=len(bank),
        units=len(bank.memory_units),
    )
    return bank


def save_cached_bank(
    *,
    fingerprint: str,
    bank: MemoryBank,
    cache_dir: Path | None = None,
) -> None:
    logger = get_logger("memory_bank_cache")
    if cache_is_disabled():
        return
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    path = bank_cache_path(fingerprint, cache_dir=cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_memory_bank(tmp, bank)
        tmp.replace(path)
        logger.info(
            "bank_cache_write",
            path=str(path),
            fingerprint=fingerprint,
            items=len(bank),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("bank_cache_write_failed", path=str(path), error=str(exc))
