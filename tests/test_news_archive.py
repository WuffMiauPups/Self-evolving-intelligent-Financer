"""NEWSBOX N0 archive: point-in-time rows, month partitioning, torn-line
tolerance, and the poll-only latency model (NEWSBOX_PLAN §N0 verify clause)."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from src.data.news_archive import (
    append_rows,
    archive_summary,
    article_row,
    known_ids,
    latency_stats,
    load_news_jsonl,
    month_path,
)

T = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _article(article_id: str, published: datetime) -> dict:
    return {
        "id": article_id,
        "created_at": published,
        "symbols": ["AAPL", "SPY"],
        "headline": "Apple announces things",
        "summary": "A summary.",
        "source": "benzinga",
        "url": "https://example.com/a",
    }


def test_article_row_maps_created_at_to_published_at_and_stamps_mode() -> None:
    row = article_row(_article("42", T), ingested_at=T.replace(minute=3), mode="poll")
    assert row["id"] == "42"
    assert row["published_at"] == T.isoformat()
    assert row["ingested_at"] == T.replace(minute=3).isoformat()
    assert row["symbols"] == ["AAPL", "SPY"]
    assert row["mode"] == "poll"


def test_rows_partition_by_publication_month(tmp_path) -> None:
    early = article_row(
        _article("1", datetime(2022, 1, 5, tzinfo=UTC)), ingested_at=T, mode="backfill"
    )
    late = article_row(_article("2", T), ingested_at=T, mode="poll")
    assert append_rows([early, late], tmp_path) == 2
    assert (tmp_path / "news_202201.jsonl").exists()
    assert (tmp_path / "news_202607.jsonl").exists()
    assert month_path(early["published_at"], tmp_path).name == "news_202201.jsonl"


def test_loader_is_torn_line_tolerant_ledger_style(tmp_path) -> None:
    good = article_row(_article("1", T), ingested_at=T, mode="poll")
    path = tmp_path / "news_202607.jsonl"
    path.write_text(
        json.dumps(good)
        + "\n"
        + "{not json at all\n"
        + json.dumps({"no_id": True})
        + "\n"
        + json.dumps(good)[: len(json.dumps(good)) // 2],  # torn final line
        encoding="utf-8",
    )
    rows = load_news_jsonl(path)
    assert len(rows) == 1 and rows[0]["id"] == "1"


def test_known_ids_deduplicates_across_appends(tmp_path) -> None:
    row = article_row(_article("7", T), ingested_at=T, mode="poll")
    append_rows([row], tmp_path)
    seen = known_ids(tmp_path)
    assert seen == {"7"}
    fresh = [
        article_row(_article(article_id, T), ingested_at=T, mode="poll")
        for article_id in ("7", "8")
        if article_id not in seen
    ]
    append_rows(fresh, tmp_path)
    assert known_ids(tmp_path) == {"7", "8"}
    assert len(load_news_jsonl(tmp_path / "news_202607.jsonl")) == 2


def test_latency_model_uses_poll_rows_only() -> None:
    poll_fast = article_row(_article("1", T), ingested_at=T.replace(second=30), mode="poll")
    poll_slow = article_row(_article("2", T), ingested_at=T.replace(minute=5), mode="poll")
    stale = article_row(
        _article("3", datetime(2022, 1, 5, tzinfo=UTC)), ingested_at=T, mode="backfill"
    )
    stats = latency_stats([poll_fast, poll_slow, stale])
    assert stats["n"] == 2  # the backfill row (years of "latency") is excluded
    assert stats["p50_sec"] == 30.0
    assert stats["max_sec"] == 300.0
    assert latency_stats([stale]) == {"n": 0}


def test_archive_summary_reports_months_and_latency(tmp_path) -> None:
    rows = [
        article_row(_article("1", datetime(2022, 1, 5, tzinfo=UTC)), ingested_at=T, mode="backfill"),
        article_row(_article("2", T), ingested_at=T.replace(second=10), mode="poll"),
    ]
    append_rows(rows, tmp_path)
    summary = archive_summary(tmp_path)
    assert summary["total"] == 2
    assert set(summary["months"]) == {"news_202201", "news_202607"}
    assert summary["months"]["news_202201"]["first"].startswith("2022-01-05")
    assert summary["poll_latency"]["n"] == 1
