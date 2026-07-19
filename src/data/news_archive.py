"""Point-in-time news archive (NEWSBOX_PLAN N0).

Append-only JSONL archive of symbol-tagged news headlines, partitioned by
publication month (``runs/news/news_YYYYMM.jsonl``). No interpretation of
any kind lives here — this layer only records what was published and WHEN
WE SAW IT, because the point-in-time rule (NEWSBOX_PLAN §0 rule 2) makes
`ingested_at - published_at` the latency model every downstream consumer
must respect.

Row schema (spec + one honesty field):
    id, published_at, ingested_at, symbols, headline, summary, source, url,
    mode — "backfill" rows were ingested long after publication (their
    latency is meaningless); "poll" rows were observed live and are the
    ONLY rows the latency model may be fitted on.

Loader is ledger-style (torn/corrupt lines skipped), matching
``load_observations`` in src/training/replay_buffer.py.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from src.core.logging import get_logger

DEFAULT_ARCHIVE_DIR = Path("runs/news")

_ROW_FIELDS = (
    "id",
    "published_at",
    "ingested_at",
    "symbols",
    "headline",
    "summary",
    "source",
    "url",
    "mode",
)


def _get(article: object, name: str, default: object = None) -> object:
    """Read a field from an alpaca News model OR a plain dict."""
    if isinstance(article, dict):
        return article.get(name, default)
    return getattr(article, name, default)


def _iso(value: object) -> str:
    """ISO-8601 UTC string from a datetime or an already-ISO string."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def article_row(article: object, *, ingested_at: datetime, mode: str) -> dict:
    """Normalize one article to the archive row schema.

    ``published_at`` maps from the feed's ``created_at`` (Benzinga stamps
    creation = publication); ``ingested_at`` is OUR wall clock at receipt.
    """
    return {
        "id": str(_get(article, "id")),
        "published_at": _iso(_get(article, "created_at")),
        "ingested_at": _iso(ingested_at),
        "symbols": list(_get(article, "symbols") or []),
        "headline": str(_get(article, "headline") or ""),
        "summary": str(_get(article, "summary") or ""),
        "source": str(_get(article, "source") or ""),
        "url": str(_get(article, "url") or ""),
        "mode": mode,
    }


def month_path(published_at: str, base_dir: Path = DEFAULT_ARCHIVE_DIR) -> Path:
    """news_YYYYMM.jsonl by PUBLICATION month (point-in-time partitioning)."""
    stamp = datetime.fromisoformat(published_at)
    return Path(base_dir) / f"news_{stamp.year:04d}{stamp.month:02d}.jsonl"


def load_news_jsonl(path: str | Path) -> list[dict]:
    """Read one archive month (ledger-style: torn/corrupt lines skipped)."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id") and row.get("published_at"):
            rows.append(row)
    return rows


def known_ids(base_dir: Path = DEFAULT_ARCHIVE_DIR) -> set[str]:
    """All article ids already archived (dedup index across month files)."""
    ids: set[str] = set()
    base = Path(base_dir)
    if not base.exists():
        return ids
    for path in sorted(base.glob("news_*.jsonl")):
        for row in load_news_jsonl(path):
            ids.add(str(row["id"]))
    return ids


def append_rows(rows: list[dict], base_dir: Path = DEFAULT_ARCHIVE_DIR) -> int:
    """Append rows to their publication-month files. Append-only, flushed.

    Caller owns dedup (via ``known_ids``); this function only writes.
    """
    if not rows:
        return 0
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    by_path: dict[Path, list[dict]] = {}
    for row in rows:
        by_path.setdefault(month_path(row["published_at"], base), []).append(row)
    written = 0
    for path, path_rows in sorted(by_path.items()):
        with path.open("a", encoding="utf-8") as fp:
            for row in path_rows:
                fp.write(json.dumps({k: row.get(k) for k in _ROW_FIELDS}) + "\n")
                written += 1
            fp.flush()
    return written


def latency_stats(rows: list[dict]) -> dict:
    """ingested_at - published_at distribution over POLL rows only (seconds).

    Backfill rows are excluded by construction: they were ingested years
    after publication, so including them would poison the latency model
    N1/N2 backtests must use (NEWSBOX_PLAN §0 rule 2).
    """
    latencies: list[float] = []
    for row in rows:
        if row.get("mode") != "poll":
            continue
        try:
            published = datetime.fromisoformat(str(row["published_at"]))
            ingested = datetime.fromisoformat(str(row["ingested_at"]))
        except ValueError:
            continue
        latencies.append((ingested - published).total_seconds())
    if not latencies:
        return {"n": 0}
    latencies.sort()

    def _pct(q: float) -> float:
        index = min(len(latencies) - 1, max(0, math.ceil(q * len(latencies)) - 1))
        return latencies[index]

    return {
        "n": len(latencies),
        "p50_sec": _pct(0.50),
        "p90_sec": _pct(0.90),
        "p99_sec": _pct(0.99),
        "max_sec": latencies[-1],
    }


def archive_summary(base_dir: Path = DEFAULT_ARCHIVE_DIR) -> dict:
    """Per-month article counts + publication-time ranges (the N0 verify view)."""
    logger = get_logger("news_archive")
    base = Path(base_dir)
    months: dict[str, dict] = {}
    total = 0
    poll_rows: list[dict] = []
    for path in sorted(base.glob("news_*.jsonl")) if base.exists() else []:
        rows = load_news_jsonl(path)
        if not rows:
            continue
        stamps = sorted(str(row["published_at"]) for row in rows)
        months[path.stem] = {"count": len(rows), "first": stamps[0], "last": stamps[-1]}
        total += len(rows)
        poll_rows.extend(row for row in rows if row.get("mode") == "poll")
    summary = {"months": months, "total": total, "poll_latency": latency_stats(poll_rows)}
    logger.info("news_archive_summary", total=total, month_count=len(months))
    return summary
