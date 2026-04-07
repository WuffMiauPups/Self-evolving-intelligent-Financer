"""Tests for the training sample disk cache.

Covers:
 - fingerprint stability across calls with identical inputs
 - fingerprint sensitivity to content-affecting config changes
 - roundtrip: write then read yields an equivalent bundle
 - cache miss returns None (no raise)
 - env-var disable flag is honored
 - corrupt cache file falls back gracefully (no raise)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.training.data_module import build_real_data_training_bundle
from src.training.sample_cache import (
    CACHE_DISABLED_ENV,
    CachedTrainingData,
    cache_path,
    compute_fingerprint,
    load_cached_samples,
    save_cached_samples,
)


def _tiny_config_and_bars():
    config = load_config("configs/default.yaml")
    config.training.use_real_historical_data = True
    config.training.fast_mode.enabled = False
    config.data.symbols = ["AAPL"]
    config.runtime.symbols = ["AAPL"]
    # Tiny dataset so the test is fast but still exercises the full path.
    bars = {"AAPL": generate_synthetic_bars("AAPL", 400)}
    return config, bars


def test_fingerprint_is_deterministic() -> None:
    config, bars = _tiny_config_and_bars()
    fp1 = compute_fingerprint(config=config, bars_by_symbol=bars)
    fp2 = compute_fingerprint(config=config, bars_by_symbol=bars)
    assert fp1 == fp2
    assert len(fp1) == 24


def test_fingerprint_changes_when_horizon_changes() -> None:
    config, bars = _tiny_config_and_bars()
    fp_before = compute_fingerprint(config=config, bars_by_symbol=bars)
    config.training.forward_horizons = [1, 2, 3]
    fp_after = compute_fingerprint(config=config, bars_by_symbol=bars)
    assert fp_before != fp_after


def test_fingerprint_changes_when_bars_change() -> None:
    config, bars = _tiny_config_and_bars()
    fp_before = compute_fingerprint(config=config, bars_by_symbol=bars)
    bars["AAPL"] = generate_synthetic_bars("AAPL", 420)  # different length
    fp_after = compute_fingerprint(config=config, bars_by_symbol=bars)
    assert fp_before != fp_after


def test_roundtrip_write_then_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.training.sample_cache as sc
    monkeypatch.setattr(sc, "DEFAULT_CACHE_DIR", tmp_path)
    config, bars = _tiny_config_and_bars()
    pipeline = FeaturePipeline(config.features)

    # First build: miss -> write
    monkeypatch.delenv(CACHE_DISABLED_ENV, raising=False)
    fp = compute_fingerprint(config=config, bars_by_symbol=bars)
    assert load_cached_samples(fingerprint=fp, cache_dir=tmp_path) is None

    bundle = build_real_data_training_bundle(
        config=config, bars_by_symbol=bars, pipeline=pipeline
    )
    # Manually write through the public API (data_module already did this via
    # its own DEFAULT_CACHE_DIR; this is a direct contract test).
    save_cached_samples(
        fingerprint=fp,
        data=CachedTrainingData(
            feature_contract=bundle.feature_contract,
            samples_by_split=bundle.samples_by_split,
            split_ranges=bundle.split_ranges,
        ),
        cache_dir=tmp_path,
    )
    assert cache_path(fp, cache_dir=tmp_path).exists()

    # Reload: hit
    cached = load_cached_samples(fingerprint=fp, cache_dir=tmp_path)
    assert cached is not None
    assert cached.feature_contract == bundle.feature_contract
    assert set(cached.samples_by_split.keys()) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert len(cached.samples_by_split[split]) == len(bundle.samples_by_split[split])


def test_miss_returns_none(tmp_path: Path) -> None:
    assert load_cached_samples(fingerprint="doesnotexist0123456789ab", cache_dir=tmp_path) is None


def test_env_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_DISABLED_ENV, "1")
    # Create a file that would otherwise be a hit; loader must still return None.
    fp = "a" * 24
    (tmp_path / f"samples_{fp}.pt").write_bytes(b"garbage")
    assert load_cached_samples(fingerprint=fp, cache_dir=tmp_path) is None


def test_corrupt_cache_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHE_DISABLED_ENV, raising=False)
    fp = "b" * 24
    (tmp_path / f"samples_{fp}.pt").write_bytes(b"not a torch payload")
    assert load_cached_samples(fingerprint=fp, cache_dir=tmp_path) is None


def test_data_module_hits_cache_on_second_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: building the bundle twice should use the cache the second
    time and produce an identical split layout."""
    import src.training.sample_cache as sc
    monkeypatch.setattr(sc, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.delenv(CACHE_DISABLED_ENV, raising=False)
    config, bars = _tiny_config_and_bars()
    pipeline = FeaturePipeline(config.features)

    bundle1 = build_real_data_training_bundle(
        config=config, bars_by_symbol=bars, pipeline=pipeline
    )
    # Cache file should now exist.
    fp = compute_fingerprint(config=config, bars_by_symbol=bars)
    assert cache_path(fp, cache_dir=tmp_path).exists()

    bundle2 = build_real_data_training_bundle(
        config=config, bars_by_symbol=bars, pipeline=pipeline
    )
    for split in ("train", "val", "test"):
        assert len(bundle2.samples_by_split[split]) == len(bundle1.samples_by_split[split])
    assert bundle2.feature_contract == bundle1.feature_contract
