"""Feature-cache salting and content signature (MASTER_PLAN §4.3.2).

The build_batch on-disk cache fingerprint must change when either the
feature CODE changes (``FEATURE_SCHEMA_VERSION`` bump — the
bump-on-formula-change contract documented at the constant) or the bar
CONTENT changes anywhere in the history, not just at the endpoints.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features import feature_cache
from src.features.feature_pipeline import FeaturePipeline


@pytest.fixture()
def pipeline() -> FeaturePipeline:
    config = load_config("configs/default.yaml")
    return FeaturePipeline(config.features, device=torch.device("cpu"))


def test_schema_version_bump_invalidates_warm_cache(
    pipeline: FeaturePipeline,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(feature_cache.ENV_DISABLE, raising=False)
    monkeypatch.setattr(feature_cache, "DEFAULT_CACHE_DIR", tmp_path)

    calls = {"count": 0}
    real_impl = FeaturePipeline._build_batch_impl

    def counting_impl(self, *args, **kwargs):
        calls["count"] += 1
        return real_impl(self, *args, **kwargs)

    monkeypatch.setattr(FeaturePipeline, "_build_batch_impl", counting_impl)

    bars = generate_synthetic_bars("AAPL", 90)
    pipeline.build_batch(bars, symbol="AAPL", temporal_window=48)
    assert calls["count"] == 1

    # Warm cache: identical inputs short-circuit the rebuild.
    pipeline.build_batch(bars, symbol="AAPL", temporal_window=48)
    assert calls["count"] == 1

    # A schema-version bump (formula change) must miss the warm cache.
    monkeypatch.setattr(
        feature_cache,
        "FEATURE_SCHEMA_VERSION",
        feature_cache.FEATURE_SCHEMA_VERSION + 1,
    )
    pipeline.build_batch(bars, symbol="AAPL", temporal_window=48)
    assert calls["count"] == 2


def test_interior_bar_edit_changes_signature_and_fingerprint(
    pipeline: FeaturePipeline,
) -> None:
    """Endpoint fields (count / first / last) are identical for the two
    histories — only the content hash separates them.  Pre-§4.3.2 the
    signature was endpoint-only, so a corrected interior bar (e.g. a
    re-fetch with fixed adjusted prices) kept serving stale features."""
    bars = generate_synthetic_bars("AAPL", 90)
    edited = list(bars)
    mid = len(edited) // 2
    edited[mid] = replace(edited[mid], close=edited[mid].close + 1.0)

    original = pipeline._compute_bars_signature(bars)
    changed = pipeline._compute_bars_signature(edited)

    for endpoint_key in ("count", "first_ts", "last_ts", "first_close", "last_close"):
        assert original[endpoint_key] == changed[endpoint_key]
    assert original != changed

    snapshot = pipeline._feature_config_snapshot()
    fingerprint_original = feature_cache.compute_fingerprint(
        bars_signature=original,
        symbol="AAPL",
        feature_config_snapshot=snapshot,
        temporal_window=48,
    )
    fingerprint_changed = feature_cache.compute_fingerprint(
        bars_signature=changed,
        symbol="AAPL",
        feature_config_snapshot=snapshot,
        temporal_window=48,
    )
    assert fingerprint_original != fingerprint_changed
