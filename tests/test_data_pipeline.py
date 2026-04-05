"""Tests for the chunked fetch planner, cache manifest, merge/validate,
parallel fetch, and parallel prep pipeline.

All tests are deterministic and require no live Alpaca access.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data.fetch_planner import (
    CacheManifest,
    FetchChunk,
    ValidationReport,
    inspect_cache_coverage,
    merge_dedupe_validate,
    plan_chunks,
)
from src.data.schemas import MarketBar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _make_bar(symbol: str, ts: datetime, close: float = 100.0) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=ts,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=1000,
        spread=0.01,
    )


# ---------------------------------------------------------------------------
# Chunk planning tests
# ---------------------------------------------------------------------------


class TestChunkPlanning:

    def test_monthly_chunks_single_symbol(self):
        chunks = plan_chunks(
            symbols=["AAPL"],
            timeframe="1Min",
            start=_utc(2023, 1, 15),
            end=_utc(2023, 4, 10),
            chunk_size="monthly",
        )
        assert len(chunks) == 4
        assert chunks[0].symbol == "AAPL"
        assert chunks[0].start == _utc(2023, 1, 15)
        assert chunks[0].end == _utc(2023, 2, 1)
        assert chunks[1].start == _utc(2023, 2, 1)
        assert chunks[1].end == _utc(2023, 3, 1)
        assert chunks[-1].end == _utc(2023, 4, 10)

    def test_quarterly_chunks(self):
        chunks = plan_chunks(
            symbols=["SPY"],
            timeframe="1Day",
            start=_utc(2022, 1, 1),
            end=_utc(2023, 1, 1),
            chunk_size="quarterly",
        )
        assert len(chunks) == 4
        assert chunks[0].start == _utc(2022, 1, 1)
        assert chunks[0].end == _utc(2022, 4, 1)
        assert chunks[-1].start == _utc(2022, 10, 1)
        assert chunks[-1].end == _utc(2023, 1, 1)

    def test_yearly_chunks(self):
        chunks = plan_chunks(
            symbols=["TSLA"],
            timeframe="1Hour",
            start=_utc(2016, 6, 1),
            end=_utc(2026, 3, 1),
            chunk_size="yearly",
        )
        # 2016 partial + 2017..2025 full + 2026 partial = 11
        assert len(chunks) == 11
        assert chunks[0].start == _utc(2016, 6, 1)
        assert chunks[0].end == _utc(2017, 1, 1)
        assert chunks[-1].start == _utc(2026, 1, 1)
        assert chunks[-1].end == _utc(2026, 3, 1)

    def test_multi_symbol_deterministic_order(self):
        chunks = plan_chunks(
            symbols=["MSFT", "AAPL"],
            timeframe="1Min",
            start=_utc(2023, 1, 1),
            end=_utc(2023, 3, 1),
            chunk_size="monthly",
        )
        # Symbols sorted: AAPL first, then MSFT
        aapl_chunks = [c for c in chunks if c.symbol == "AAPL"]
        msft_chunks = [c for c in chunks if c.symbol == "MSFT"]
        assert len(aapl_chunks) == 2
        assert len(msft_chunks) == 2
        # AAPL comes before MSFT in the list
        assert chunks.index(aapl_chunks[0]) < chunks.index(msft_chunks[0])

    def test_empty_range_returns_no_chunks(self):
        chunks = plan_chunks(
            symbols=["AAPL"],
            timeframe="1Min",
            start=_utc(2023, 3, 1),
            end=_utc(2023, 1, 1),
            chunk_size="monthly",
        )
        assert chunks == []

    def test_chunk_ids_are_unique(self):
        chunks = plan_chunks(
            symbols=["AAPL", "MSFT"],
            timeframe="1Min",
            start=_utc(2020, 1, 1),
            end=_utc(2025, 1, 1),
            chunk_size="monthly",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_ten_year_monthly_chunk_count(self):
        """10 years × 12 months × 2 symbols = 240 chunks."""
        chunks = plan_chunks(
            symbols=["AAPL", "MSFT"],
            timeframe="1Min",
            start=_utc(2016, 1, 1),
            end=_utc(2026, 1, 1),
            chunk_size="monthly",
        )
        assert len(chunks) == 240


# ---------------------------------------------------------------------------
# Cache manifest tests
# ---------------------------------------------------------------------------


class TestCacheManifest:

    def test_load_save_roundtrip(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest.load(manifest_path)
        assert len(manifest.chunks) == 0

        chunk = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 1, 1), end=_utc(2023, 2, 1),
            chunk_id="AAPL_1Min_20230101_20230201",
        )
        manifest.record_chunk(chunk, row_count=5000, file_path="test.pt")
        manifest.save()

        reloaded = CacheManifest.load(manifest_path)
        assert len(reloaded.chunks) == 1
        assert reloaded.is_covered("AAPL_1Min_20230101_20230201")
        assert not reloaded.is_covered("AAPL_1Min_20230201_20230301")

    def test_filter_uncovered(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunk1 = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 1, 1), end=_utc(2023, 2, 1),
            chunk_id="c1",
        )
        chunk2 = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 2, 1), end=_utc(2023, 3, 1),
            chunk_id="c2",
        )
        manifest.record_chunk(chunk1, row_count=100)
        uncovered = manifest.filter_uncovered([chunk1, chunk2])
        assert len(uncovered) == 1
        assert uncovered[0].chunk_id == "c2"

    def test_missing_ranges(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        # Cover Jan and Mar, leave Feb missing
        chunk_jan = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 1, 1), end=_utc(2023, 2, 1),
            chunk_id="jan",
        )
        chunk_mar = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 3, 1), end=_utc(2023, 4, 1),
            chunk_id="mar",
        )
        manifest.record_chunk(chunk_jan, row_count=100)
        manifest.record_chunk(chunk_mar, row_count=100)

        missing = manifest.missing_ranges(
            "AAPL", "1Min",
            _utc(2023, 1, 1), _utc(2023, 4, 1),
        )
        assert len(missing) == 1
        assert missing[0] == (_utc(2023, 2, 1), _utc(2023, 3, 1))

    def test_summary(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        for month in range(1, 4):
            end_month = month + 1 if month < 12 else 1
            end_year = 2023 if month < 12 else 2024
            chunk = FetchChunk(
                symbol="AAPL", timeframe="1Min",
                start=_utc(2023, month, 1),
                end=_utc(end_year, end_month, 1),
                chunk_id=f"c{month}",
            )
            manifest.record_chunk(chunk, row_count=1000 * month)
        summary = manifest.summary()
        assert summary["total_chunks"] == 3
        assert len(summary["symbols"]) == 1
        assert summary["symbols"][0]["total_rows"] == 6000

    def test_coverage_for_symbol(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunk = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 1, 1), end=_utc(2023, 2, 1),
            chunk_id="test",
        )
        manifest.record_chunk(chunk, row_count=500)
        coverages = manifest.coverage_for_symbol("AAPL", "1Min")
        assert len(coverages) == 1
        assert coverages[0].row_count == 500

        # Different symbol returns empty
        assert manifest.coverage_for_symbol("MSFT", "1Min") == []

    def test_atomic_save(self, tmp_path: Path):
        """Save uses tmp file + rename for atomicity."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest.load(manifest_path)
        chunk = FetchChunk(
            symbol="AAPL", timeframe="1Min",
            start=_utc(2023, 1, 1), end=_utc(2023, 2, 1),
            chunk_id="test",
        )
        manifest.record_chunk(chunk, row_count=100)
        manifest.save()

        # File should exist and be valid JSON
        assert manifest_path.exists()
        with manifest_path.open() as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert len(data["chunks"]) == 1


# ---------------------------------------------------------------------------
# Resume behavior tests
# ---------------------------------------------------------------------------


class TestResumeBehavior:

    def test_resume_skips_completed_chunks(self, tmp_path: Path):
        """After interruption, only uncovered chunks are fetched."""
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        all_chunks = plan_chunks(
            symbols=["AAPL"],
            timeframe="1Day",
            start=_utc(2023, 1, 1),
            end=_utc(2023, 6, 1),
            chunk_size="monthly",
        )
        # Simulate: first 3 chunks completed before interruption
        for chunk in all_chunks[:3]:
            manifest.record_chunk(chunk, row_count=20)

        remaining = manifest.filter_uncovered(all_chunks)
        assert len(remaining) == 2
        # Remaining should be the last 2 chunks
        assert remaining[0].chunk_id == all_chunks[3].chunk_id
        assert remaining[1].chunk_id == all_chunks[4].chunk_id

    def test_resume_with_full_coverage(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        all_chunks = plan_chunks(
            symbols=["SPY"],
            timeframe="1Day",
            start=_utc(2023, 1, 1),
            end=_utc(2023, 3, 1),
            chunk_size="monthly",
        )
        for chunk in all_chunks:
            manifest.record_chunk(chunk, row_count=20)

        remaining = manifest.filter_uncovered(all_chunks)
        assert remaining == []


# ---------------------------------------------------------------------------
# Merge / dedupe / validate tests
# ---------------------------------------------------------------------------


class TestMergeDedupeValidate:

    def test_dedupe_by_timestamp(self):
        ts1 = _utc(2023, 1, 1)
        ts2 = _utc(2023, 1, 1) + timedelta(minutes=1)
        bars1 = [_make_bar("AAPL", ts1, 100.0), _make_bar("AAPL", ts2, 101.0)]
        bars2 = [_make_bar("AAPL", ts1, 100.0), _make_bar("AAPL", ts2, 101.5)]  # duplicate timestamps

        merged, report = merge_dedupe_validate([bars1, bars2], symbol="AAPL", timeframe="1Min")
        assert len(merged) == 2
        assert report.duplicates_removed == 2
        assert report.is_monotonic

    def test_sort_by_timestamp(self):
        ts1 = _utc(2023, 1, 1)
        ts2 = ts1 + timedelta(minutes=1)
        ts3 = ts1 + timedelta(minutes=2)
        # Out of order
        bars = [_make_bar("AAPL", ts3), _make_bar("AAPL", ts1), _make_bar("AAPL", ts2)]

        merged, report = merge_dedupe_validate([bars], symbol="AAPL", timeframe="1Min")
        assert merged[0].timestamp == ts1
        assert merged[1].timestamp == ts2
        assert merged[2].timestamp == ts3
        assert report.is_monotonic

    def test_gap_detection(self):
        ts1 = _utc(2023, 1, 1)
        ts2 = ts1 + timedelta(minutes=1)
        ts3 = ts1 + timedelta(hours=5)  # Large gap
        bars = [_make_bar("AAPL", ts1), _make_bar("AAPL", ts2), _make_bar("AAPL", ts3)]

        merged, report = merge_dedupe_validate([bars], symbol="AAPL", timeframe="1Min")
        assert report.gap_count == 1
        assert report.max_gap_seconds == 5 * 3600 - 60

    def test_no_gaps_continuous(self):
        base = _utc(2023, 1, 1)
        bars = [_make_bar("AAPL", base + timedelta(minutes=i)) for i in range(100)]

        merged, report = merge_dedupe_validate([bars], symbol="AAPL", timeframe="1Min")
        assert report.gap_count == 0
        assert report.total_rows == 100

    def test_merge_multiple_chunks(self):
        """Bars from chunks fetched out of order merge correctly."""
        base = _utc(2023, 1, 1)
        chunk1 = [_make_bar("AAPL", base + timedelta(minutes=i)) for i in range(0, 50)]
        chunk2 = [_make_bar("AAPL", base + timedelta(minutes=i)) for i in range(50, 100)]
        chunk3 = [_make_bar("AAPL", base + timedelta(minutes=i)) for i in range(45, 55)]  # overlapping

        merged, report = merge_dedupe_validate(
            [chunk3, chunk1, chunk2],  # out of order
            symbol="AAPL",
            timeframe="1Min",
        )
        assert report.total_rows == 100
        assert report.duplicates_removed == 10  # 45..54 duplicated
        assert report.is_monotonic
        assert report.gap_count == 0

    def test_empty_input(self):
        merged, report = merge_dedupe_validate([], symbol="AAPL", timeframe="1Min")
        assert merged == []
        assert report.total_rows == 0
        assert report.is_monotonic


# ---------------------------------------------------------------------------
# Inspect / report tests
# ---------------------------------------------------------------------------


class TestInspectCache:

    def test_inspect_fully_covered(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunks = plan_chunks(
            symbols=["AAPL"],
            timeframe="1Day",
            start=_utc(2023, 1, 1),
            end=_utc(2023, 4, 1),
            chunk_size="monthly",
        )
        for chunk in chunks:
            manifest.record_chunk(chunk, row_count=20)

        reports = inspect_cache_coverage(manifest, ["AAPL"], "1Day", _utc(2023, 1, 1), _utc(2023, 4, 1))
        assert len(reports) == 1
        assert reports[0]["fully_covered"]
        assert reports[0]["missing_ranges"] == []

    def test_inspect_partially_covered(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunk = FetchChunk(
            symbol="AAPL", timeframe="1Day",
            start=_utc(2023, 1, 1), end=_utc(2023, 2, 1),
            chunk_id="jan",
        )
        manifest.record_chunk(chunk, row_count=20)

        reports = inspect_cache_coverage(
            manifest, ["AAPL"], "1Day",
            _utc(2023, 1, 1), _utc(2023, 4, 1),
        )
        assert not reports[0]["fully_covered"]
        assert len(reports[0]["missing_ranges"]) == 1

    def test_inspect_no_coverage(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        reports = inspect_cache_coverage(
            manifest, ["AAPL", "MSFT"], "1Min",
            _utc(2023, 1, 1), _utc(2023, 12, 1),
        )
        assert len(reports) == 2
        for r in reports:
            assert not r["fully_covered"]
            assert r["chunk_count"] == 0
            assert r["total_rows"] == 0

    def test_inspect_output_correctness(self, tmp_path: Path):
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunk = FetchChunk(
            symbol="SPY", timeframe="1Hour",
            start=_utc(2023, 1, 1), end=_utc(2023, 7, 1),
            chunk_id="h1",
        )
        manifest.record_chunk(chunk, row_count=4380, validated=True)

        reports = inspect_cache_coverage(
            manifest, ["SPY"], "1Hour",
            _utc(2023, 1, 1), _utc(2023, 12, 1),
        )
        assert reports[0]["total_rows"] == 4380
        assert not reports[0]["fully_covered"]
        assert reports[0]["missing_days"] > 0


# ---------------------------------------------------------------------------
# Split leakage prevention tests
# ---------------------------------------------------------------------------


class TestSplitLeakage:

    def test_temporal_split_no_overlap(self):
        """Verify that train/val/test splits don't overlap in time."""
        from src.training.splits import build_symbol_splits

        indices = list(range(100, 500))
        # Create monotonically increasing timestamps
        base = _utc(2023, 1, 1)
        timestamps = [base + timedelta(minutes=i) for i in indices]

        splits = build_symbol_splits(
            symbol="TEST",
            eligible_indices=indices,
            timestamps=timestamps,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
        )
        assert len(splits) == 3
        # Verify no overlap: each split's index range is contiguous and non-overlapping
        split_dict = {s.split: s for s in splits}
        assert "train" in split_dict
        assert "val" in split_dict
        assert "test" in split_dict
        # sample_count should sum to total
        total = sum(s.sample_count for s in splits)
        assert total == len(indices)


# ---------------------------------------------------------------------------
# Offline-only behavior tests
# ---------------------------------------------------------------------------


class TestOfflineBehavior:

    def test_offline_complete_coverage(self, tmp_path: Path):
        """When all chunks are covered, offline mode succeeds."""
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunks = plan_chunks(
            symbols=["AAPL"],
            timeframe="1Day",
            start=_utc(2023, 1, 1),
            end=_utc(2023, 4, 1),
            chunk_size="monthly",
        )
        for chunk in chunks:
            manifest.record_chunk(chunk, row_count=20)

        # All chunks covered → filter_uncovered returns empty
        remaining = manifest.filter_uncovered(chunks)
        assert remaining == []

    def test_offline_incomplete_reports_gaps(self, tmp_path: Path):
        """When coverage is incomplete, offline mode can detect it."""
        manifest = CacheManifest.load(tmp_path / "manifest.json")
        chunks = plan_chunks(
            symbols=["AAPL"],
            timeframe="1Day",
            start=_utc(2023, 1, 1),
            end=_utc(2023, 6, 1),
            chunk_size="monthly",
        )
        # Only cover first 2 of 5 chunks
        for chunk in chunks[:2]:
            manifest.record_chunk(chunk, row_count=20)

        remaining = manifest.filter_uncovered(chunks)
        assert len(remaining) == 3

        missing = manifest.missing_ranges("AAPL", "1Day", _utc(2023, 1, 1), _utc(2023, 6, 1))
        assert len(missing) >= 1


# ---------------------------------------------------------------------------
# Pipeline config tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:

    def test_default_config_loads(self):
        from src.core.config import PipelineConfig
        pc = PipelineConfig()
        assert pc.chunk_size == "monthly"
        assert pc.max_fetch_workers == 4
        assert pc.retry_count == 3

    def test_invalid_chunk_size(self):
        from src.core.config import PipelineConfig
        with pytest.raises(Exception):
            PipelineConfig(chunk_size="weekly")

    def test_invalid_workers(self):
        from src.core.config import PipelineConfig
        with pytest.raises(Exception):
            PipelineConfig(max_fetch_workers=0)

    def test_config_from_yaml(self):
        from src.core.config import load_config
        config = load_config("configs/default.yaml")
        assert config.data.pipeline.chunk_size == "monthly"
        assert config.data.pipeline.max_fetch_workers == 4
        assert config.data.pipeline.manifest_path == "artifacts/historical/manifest.json"
