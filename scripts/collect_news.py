"""NEWSBOX N0 point-in-time news collector (NEWSBOX_PLAN §N0).

Archives symbol-tagged Benzinga news (the existing Alpaca account, no new
credentials) into ``runs/news/news_YYYYMM.jsonl`` — append-only, dedup by
article id, no interpretation. Two ingestion modes plus a report:

    python scripts/collect_news.py --backfill            # 2022-01 -> now, once
    python scripts/collect_news.py --poll                # incremental, ~60 s cadence, forever
    python scripts/collect_news.py --summary             # per-month counts + poll latency model

Rows carry BOTH ``published_at`` (feed timestamp) and ``ingested_at`` (our
wall clock): the poller's measured latency distribution IS the model that
N1/N2 backtests must respect (§0 rule 2). Backfill rows are marked
``mode=backfill`` so they can never poison that distribution.

Run unattended via scripts/launch_news_collector_hidden.vbs (§10.10).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.core.logging import configure_logging, get_logger  # noqa: E402
from src.data.news_archive import (  # noqa: E402
    DEFAULT_ARCHIVE_DIR,
    archive_summary,
    article_row,
    append_rows,
    known_ids,
    latency_stats,
)

# alpaca-py paginates INTERNALLY (get_news passes page_size=50 to
# _get_marketdata and consumes every page token itself): NewsRequest.limit
# is the TOTAL article cap per call and NewsSet.next_page_token is always
# None (verified against the live API 2026-07-19). Batching therefore
# advances a publication-time cursor, not a page token.
BATCH_LIMIT = 2000
POLL_OVERLAP = timedelta(minutes=15)  # re-request window; dedup absorbs it


def _news_client():
    import os

    from alpaca.data.historical.news import NewsClient

    return NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])


def _fetch_batches(client, *, symbols: str, start: datetime, logger):
    """Yield article batches ascending in publication time until exhausted.

    Cursor loop: each call returns up to BATCH_LIMIT articles from
    ``cursor`` forward; the next call starts at the newest timestamp seen
    (id-dedup absorbs the boundary overlap). A short batch = caught up.
    """
    from alpaca.data.requests import NewsRequest

    cursor = start
    while True:
        request = NewsRequest(
            symbols=symbols,
            start=cursor,
            sort="asc",
            limit=BATCH_LIMIT,
            include_content=False,
        )
        news_set = client.get_news(request)
        articles = news_set.data.get("news", [])
        if articles:
            # The feed is only near-sorted within a response; order the batch
            # so month files stay locally ordered.
            articles = sorted(articles, key=lambda a: a.created_at)
            yield articles
            newest = articles[-1].created_at
            if newest <= cursor:
                return  # no forward progress — degenerate guard
            cursor = newest
        if len(articles) < BATCH_LIMIT:
            return


def run_backfill(config, *, start: datetime, archive_dir: Path, logger) -> int:
    symbols = ",".join(config.data.symbols)
    seen = known_ids(archive_dir)
    logger.info(
        "news_backfill_start", start=start.isoformat(), symbols=symbols, already_archived=len(seen)
    )
    client = _news_client()
    total_new, batches = 0, 0
    for articles in _fetch_batches(client, symbols=symbols, start=start, logger=logger):
        ingested = datetime.now(UTC)
        rows = [
            article_row(article, ingested_at=ingested, mode="backfill")
            for article in articles
            if str(article.id) not in seen
        ]
        for row in rows:
            seen.add(row["id"])
        total_new += append_rows(rows, archive_dir)
        batches += 1
        logger.info(
            "news_backfill_progress",
            batches=batches,
            new_rows=total_new,
            at=str(articles[-1].created_at),
        )
    logger.info("news_backfill_done", batches=batches, new_rows=total_new)
    return 0


def run_poll(config, *, interval_sec: float, archive_dir: Path, logger) -> int:
    from src.data.news_archive import load_news_jsonl

    symbols = ",".join(config.data.symbols)
    seen = known_ids(archive_dir)
    # Resume from the newest ARCHIVED publication time (bounded to 7 days
    # back) so a machine-off gap self-heals on restart; id-dedup absorbs
    # the overlap. Fresh archive: start from the recent past.
    latest = max(
        (
            row["published_at"]
            for path in sorted(Path(archive_dir).glob("news_*.jsonl"))[-2:]
            for row in load_news_jsonl(path)
        ),
        default=None,
    )
    floor = datetime.now(UTC) - timedelta(days=7)
    since = max(datetime.fromisoformat(latest), floor) if latest else datetime.now(UTC) - POLL_OVERLAP
    logger.info(
        "news_poll_start",
        symbols=symbols,
        interval_sec=interval_sec,
        known=len(seen),
        resume_from=since.isoformat(),
    )
    client = _news_client()
    session_rows: list[dict] = []
    cycles = 0
    while True:
        cycles += 1
        try:
            fetched = 0
            for articles in _fetch_batches(
                client, symbols=symbols, start=since - POLL_OVERLAP, logger=logger
            ):
                ingested = datetime.now(UTC)
                rows = [
                    article_row(article, ingested_at=ingested, mode="poll")
                    for article in articles
                    if str(article.id) not in seen
                ]
                for row in rows:
                    seen.add(row["id"])
                fetched += append_rows(rows, archive_dir)
                session_rows.extend(rows)
                for row in rows:
                    published = datetime.fromisoformat(row["published_at"])
                    if published > since:
                        since = published
            if fetched:
                logger.info("news_poll_ingested", new_rows=fetched, cycle=cycles)
            if cycles % 30 == 0:
                logger.info(
                    "news_poll_latency", **latency_stats(session_rows), session_rows=len(session_rows)
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — a poller must outlive transient API errors
            logger.warning("news_poll_error", error=str(exc), cycle=cycles)
        time.sleep(interval_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="NEWSBOX N0 news collector")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true", help="Historical backfill, then exit")
    mode.add_argument("--poll", action="store_true", help="Incremental poller (runs forever)")
    mode.add_argument("--summary", action="store_true", help="Print archive summary + latency model")
    parser.add_argument("--start", default="2022-01-01", help="Backfill start date (UTC)")
    parser.add_argument("--interval", type=float, default=60.0, help="Poll cadence seconds")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    args = parser.parse_args()

    config = load_config("configs/default.yaml")  # also loads .env credentials
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("collect_news")
    archive_dir = Path(args.archive_dir)

    if args.summary:
        print(json.dumps(archive_summary(archive_dir), indent=2))
        return 0

    import os

    if not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET")):
        logger.error("news_collector_no_credentials")
        return 2
    if args.backfill:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
        return run_backfill(config, start=start, archive_dir=archive_dir, logger=logger)
    return run_poll(config, interval_sec=args.interval, archive_dir=archive_dir, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
