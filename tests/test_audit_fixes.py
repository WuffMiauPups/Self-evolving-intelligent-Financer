"""Targeted tests for the architecture-audit fixes.

These tests cover the specific safety / truthfulness / performance fixes
made during the audit pass:

- Default config: ``training.fast_mode.enabled`` is False.
- ``PollingStreamManager``: a single all-symbols-failed poll yields an
  empty dict instead of raising a permanent ``RuntimeError`` that kills the
  loop.
- ``MemoryBank._validate_item``: encoder-fingerprint mismatch with
  ``require_encoder_match=False`` is now logged AND skipped (the previous
  behaviour was log + silently append into an incompatible latent space).
- ``build_memory_bank_from_bars``: the bank's encoder fingerprint is
  stamped at construction (was previously left as None).
- ``resolve_inference_model``: missing checkpoint logs a warning and is
  rejected when ``allow_untrained=False`` (paper/live runtime path).
- ``RuntimeService._fill_from_runtime_status``: per-tick fusion / memory /
  frozen-brain flags are read from the runtime snapshot and override the
  static config view.
- ``HistoricalCacheRecord.from_payload``: the new columnar tensor format is
  read back correctly AND the legacy per-bar dict format is still readable.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

from src.core.config import FeatureConfig, ModelConfig, load_config
from src.core.exceptions import ConfigurationError
from src.data.cache import (
    CACHE_FORMAT_TENSOR_V1,
    HistoricalBarCache,
    HistoricalCacheMetadata,
    HistoricalCacheRecord,
    serialize_bar,
)
from src.data.schemas import MarketBar
from src.data.stream_manager import PollingStreamManager
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank, feature_contract_fingerprint
from src.memory.outcomes import build_risk_summary
from src.memory.schemas import MemoryItem, OutcomeSummary
from src.models.factory import resolve_inference_model


# ---------------------------------------------------------------------------
# Default config truthfulness
# ---------------------------------------------------------------------------


def test_default_config_fast_mode_disabled() -> None:
    """fast_mode.enabled must be False in the canonical default config."""
    cfg = load_config("configs/default.yaml")
    assert cfg.training.fast_mode.enabled is False


def test_default_paper_runtime_unbounded() -> None:
    """The default paper runtime is unbounded; max_runtime_steps must be 0."""
    cfg = load_config("configs/default.yaml", "configs/paper_trade.yaml")
    assert cfg.runtime.max_runtime_steps == 0


# ---------------------------------------------------------------------------
# PollingStreamManager: no hidden runtime stop
# ---------------------------------------------------------------------------


class _AlwaysFailClient:
    def get_latest_bar(self, *, symbol: str, timeframe: str) -> MarketBar:  # noqa: ARG002
        raise RuntimeError(f"broker outage for {symbol}")


class _OneShotClient:
    """First N polls fail for all symbols, then return a single bar."""

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.calls = 0

    def get_latest_bar(self, *, symbol: str, timeframe: str) -> MarketBar:  # noqa: ARG002
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError("transient broker glitch")
        return MarketBar(
            symbol=symbol,
            timestamp=datetime(2026, 4, 7, 14, 30, tzinfo=UTC),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
            spread=0.0,
        )


def test_stream_manager_yields_empty_on_total_failure() -> None:
    """A poll where every symbol fails must yield an empty dict, not raise."""
    mgr = PollingStreamManager(
        client=_AlwaysFailClient(),
        symbols=["AAPL", "MSFT"],
        timeframe="1Min",
        poll_interval_sec=0.0,
    )

    async def _take_one() -> dict[str, MarketBar]:
        async for bars in mgr.stream():
            return bars
        return {}

    result = asyncio.run(_take_one())
    assert result == {}
    assert set(mgr.last_errors.keys()) == {"AAPL", "MSFT"}
    assert mgr.consecutive_empty_polls == 1


def test_stream_manager_recovers_after_transient_failure() -> None:
    """After several transient failures the stream must keep yielding."""
    client = _OneShotClient(fail_count=3)
    mgr = PollingStreamManager(
        client=client,
        symbols=["AAPL"],
        timeframe="1Min",
        poll_interval_sec=0.0,
    )

    async def _drive() -> list[dict[str, MarketBar]]:
        out: list[dict[str, MarketBar]] = []
        n = 0
        async for bars in mgr.stream():
            out.append(bars)
            n += 1
            if n >= 5:
                return out
        return out

    polls = asyncio.run(_drive())
    # First three polls are empty, then we recover.
    empties = [p for p in polls if not p]
    non_empty = [p for p in polls if p]
    assert len(empties) == 3
    assert len(non_empty) == 2
    assert mgr.consecutive_empty_polls == 0


# ---------------------------------------------------------------------------
# MemoryBank encoder fingerprint enforcement
# ---------------------------------------------------------------------------


def _build_contract_and_item(
    *,
    encoder_fp: str | None,
) -> tuple[FeaturePipeline, MemoryItem, str]:
    bars = [
        MarketBar(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 7, 14, 30, tzinfo=UTC) + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(80)
    ]
    pipeline = FeaturePipeline(FeatureConfig(), device=torch.device("cpu"))
    snapshot = pipeline.build(bars, symbol="AAPL")
    contract_id = feature_contract_fingerprint(snapshot.contract)
    outcome = OutcomeSummary(
        future_returns={1: 0.001},
        max_favorable_excursion=0.002,
        max_adverse_excursion=-0.001,
        realized_volatility=0.01,
        drawdown=0.0,
        path_label="flat",
        direction_class="flat",
        time_to_move=10,
    )
    item = MemoryItem(
        item_id="x",
        symbol="AAPL",
        timestamp=bars[-1].timestamp,
        latent=torch.tensor([1.0, 0.0]),
        feature_contract_id=contract_id,
        regime_weights=torch.tensor([0.5, 0.5], dtype=torch.float32),
        branch_confidences={"trend": 0.7},
        branch_norms={"trend": 1.0},
        outcome=outcome,
        risk=build_risk_summary(outcome),
        encoder_fingerprint=encoder_fp,
    )
    return pipeline, item, contract_id


def test_memory_bank_skip_on_fingerprint_mismatch_when_not_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mismatched item must be SKIPPED, not silently appended into the bank."""
    pipeline, item, contract_id = _build_contract_and_item(encoder_fp="OTHER_FP")
    snapshot = pipeline.build(
        [
            MarketBar(
                symbol="AAPL",
                timestamp=datetime(2026, 4, 7, 14, 30, tzinfo=UTC) + timedelta(minutes=i),
                open=100.0 + i * 0.01,
                high=100.5 + i * 0.01,
                low=99.5 + i * 0.01,
                close=100.2 + i * 0.01,
                volume=1000.0 + i,
                spread=0.05,
            )
            for i in range(80)
        ],
        symbol="AAPL",
    )
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=8,
        encoder_fingerprint="BANK_FP",
        require_encoder_match=False,
    )
    with caplog.at_level(logging.WARNING):
        added = bank.append(item)
    assert added is False
    assert len(bank) == 0
    assert any("Encoder fingerprint mismatch" in rec.message for rec in caplog.records)


def test_memory_bank_extend_skips_mismatched_items() -> None:
    """``extend`` returns the count of items actually added."""
    pipeline, mismatched_item, contract_id = _build_contract_and_item(encoder_fp="OTHER_FP")
    matching_item = MemoryItem(
        item_id="y",
        symbol="AAPL",
        timestamp=datetime(2026, 4, 7, 15, 30, tzinfo=UTC),
        latent=torch.tensor([0.0, 1.0]),
        feature_contract_id=contract_id,
        regime_weights=torch.tensor([0.5, 0.5], dtype=torch.float32),
        branch_confidences={"trend": 0.7},
        branch_norms={"trend": 1.0},
        outcome=mismatched_item.outcome,
        risk=mismatched_item.risk,
        encoder_fingerprint="BANK_FP",
    )
    snapshot = pipeline.build(
        [
            MarketBar(
                symbol="AAPL",
                timestamp=datetime(2026, 4, 7, 14, 30, tzinfo=UTC) + timedelta(minutes=i),
                open=100.0 + i * 0.01,
                high=100.5 + i * 0.01,
                low=99.5 + i * 0.01,
                close=100.2 + i * 0.01,
                volume=1000.0 + i,
                spread=0.05,
            )
            for i in range(80)
        ],
        symbol="AAPL",
    )
    bank = MemoryBank(
        feature_contract=snapshot.contract,
        latent_dim=2,
        similarity_metric="cosine",
        max_items=8,
        encoder_fingerprint="BANK_FP",
        require_encoder_match=False,
    )
    added = bank.extend([mismatched_item, matching_item])
    assert added == 1
    assert len(bank) == 1
    assert bank.items[0].item_id == "y"


# ---------------------------------------------------------------------------
# resolve_inference_model: untrained-model rejection
# ---------------------------------------------------------------------------


def test_resolve_inference_model_warns_on_missing_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    feature_cfg = FeatureConfig()
    pipeline = FeaturePipeline(feature_cfg, device=torch.device("cpu"))
    bars = [
        MarketBar(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 7, 14, 30, tzinfo=UTC) + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(80)
    ]
    snapshot = pipeline.build(bars, symbol="AAPL")
    model_cfg = ModelConfig(checkpoint_path=None, regime_input_dim=snapshot.contract.regime_dim)
    with caplog.at_level(logging.WARNING):
        artifacts = resolve_inference_model(
            model_config=model_cfg,
            feature_config=feature_cfg,
            feature_contract=snapshot.contract,
            device=torch.device("cpu"),
            default_model_kind="brain",
            allow_untrained=True,
        )
    assert artifacts.meta.get("untrained") is True
    assert any("untrained" in rec.message.lower() for rec in caplog.records)


def test_resolve_inference_model_rejects_missing_checkpoint_when_required() -> None:
    feature_cfg = FeatureConfig()
    pipeline = FeaturePipeline(feature_cfg, device=torch.device("cpu"))
    bars = [
        MarketBar(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 7, 14, 30, tzinfo=UTC) + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(80)
    ]
    snapshot = pipeline.build(bars, symbol="AAPL")
    model_cfg = ModelConfig(checkpoint_path=None, regime_input_dim=snapshot.contract.regime_dim)
    with pytest.raises(ConfigurationError, match="checkpoint_path is empty"):
        resolve_inference_model(
            model_config=model_cfg,
            feature_config=feature_cfg,
            feature_contract=snapshot.contract,
            device=torch.device("cpu"),
            default_model_kind="brain",
            allow_untrained=False,
        )


# ---------------------------------------------------------------------------
# Runtime status: per-tick truth wins over static config
# ---------------------------------------------------------------------------


def test_runtime_service_consumes_per_tick_truth_for_safety_flags(tmp_path) -> None:
    """RuntimeService must read fusion/memory/frozen_brain from the snapshot.

    The previous behaviour reported the *config view* (captured at spawn time)
    even when the runtime had silently downgraded those flags. The fix wires
    the per-tick snapshot fields straight into the GUI status object so the
    operator sees real runtime truth.
    """
    from src.gui_services.runtime_service import RuntimeService
    from src.gui_services.state import RuntimeState, RuntimeStatus
    from src.monitoring.runtime_status import (
        RuntimeStatusSnapshot,
        RuntimeStatusWriter,
        DEFAULT_RUNTIME_STATUS_PATH,
    )

    # Stage a fake runtime status snapshot at the default path.
    snap_path = Path(DEFAULT_RUNTIME_STATUS_PATH)
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    backup = snap_path.read_bytes() if snap_path.exists() else None
    try:
        writer = RuntimeStatusWriter(snap_path)
        # Static config view says everything is on, but the snapshot says
        # the runtime has actually disabled fusion and memory.
        snap = RuntimeStatusSnapshot(
            ts=datetime(2026, 4, 7, 14, 30, tzinfo=UTC),
            step=42,
            session_id="2026-04-07",
            symbol="AAPL",
            equity=100_000.0,
            fills=0,
            safe_to_trade=True,
            memory_enabled=False,
            frozen_brain=True,
        )
        writer.write(snap)

        svc = RuntimeService()
        status = RuntimeStatus(
            state=RuntimeState.STOPPED,
            pid=None,
            started_at=None,
            symbols=["AAPL"],
            frozen_brain=False,  # static config disagreed
            memory_enabled=True,  # static config disagreed
            memory_updates_enabled=True,
            error_message="",
        )
        ok = svc._fill_from_runtime_status(status)
        assert ok is True
        # Per-tick truth wins over static config view.
        assert status.memory_enabled is False
        assert status.frozen_brain is True
    finally:
        if backup is None:
            try:
                snap_path.unlink()
            except FileNotFoundError:
                pass
        else:
            snap_path.write_bytes(backup)


# ---------------------------------------------------------------------------
# Historical bar cache: tensor format + legacy fallback
# ---------------------------------------------------------------------------


def _make_bars(n: int = 64) -> list[MarketBar]:
    t0 = datetime(2026, 4, 7, 14, 30, tzinfo=UTC)
    return [
        MarketBar(
            symbol="AAPL",
            timestamp=t0 + timedelta(minutes=i),
            open=100.0 + i * 0.01,
            high=100.5 + i * 0.01,
            low=99.5 + i * 0.01,
            close=100.2 + i * 0.01,
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(n)
    ]


def test_historical_cache_writes_columnar_tensor_format() -> None:
    bars = _make_bars(64)
    metadata = HistoricalCacheMetadata(
        symbol="AAPL",
        timeframe="1Min",
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
        source="synthetic",
        timezone="UTC",
        row_count=len(bars),
        generated_at=bars[0].timestamp,
        updated_at=bars[-1].timestamp,
    )
    record = HistoricalCacheRecord(metadata=metadata, bars=bars)
    payload = record.to_payload()
    assert "bars_columnar" in payload
    assert payload["bars_columnar"]["cache_format"] == CACHE_FORMAT_TENSOR_V1
    assert payload["bars_columnar"]["close"].shape[0] == len(bars)


def test_historical_cache_round_trip_tensor_format(tmp_path: Path) -> None:
    bars = _make_bars(128)
    cache = HistoricalBarCache(tmp_path)
    metadata = HistoricalCacheMetadata(
        symbol="AAPL",
        timeframe="1Min",
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
        source="synthetic",
        timezone="UTC",
        row_count=len(bars),
        generated_at=bars[0].timestamp,
        updated_at=bars[-1].timestamp,
    )
    cache.save(HistoricalCacheRecord(metadata=metadata, bars=bars))
    loaded = cache.load("AAPL", "1Min")
    assert loaded is not None
    assert len(loaded.bars) == len(bars)
    for original, restored in zip(bars, loaded.bars):
        assert restored.symbol == original.symbol
        assert restored.timestamp.tzinfo is not None
        assert abs(restored.open - original.open) < 1e-9
        assert abs(restored.close - original.close) < 1e-9
        assert abs(restored.volume - original.volume) < 1e-9


def test_historical_cache_legacy_format_still_readable(tmp_path: Path) -> None:
    """Legacy per-bar dict caches must still load."""
    bars = _make_bars(32)
    legacy_payload = {
        "metadata": HistoricalCacheMetadata(
            symbol="AAPL",
            timeframe="1Min",
            start=bars[0].timestamp,
            end=bars[-1].timestamp,
            source="synthetic",
            timezone="UTC",
            row_count=len(bars),
            generated_at=bars[0].timestamp,
            updated_at=bars[-1].timestamp,
        ).to_payload(),
        # Note: NO bars_columnar key — pure legacy.
        "bars": [serialize_bar(b) for b in bars],
    }
    path = tmp_path / "AAPL_1Min.pt"
    torch.save(legacy_payload, path)

    cache = HistoricalBarCache(tmp_path)
    loaded = cache.load("AAPL", "1Min")
    assert loaded is not None
    assert len(loaded.bars) == len(bars)
    assert abs(loaded.bars[0].close - bars[0].close) < 1e-9
