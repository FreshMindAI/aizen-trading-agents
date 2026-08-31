"""Backfill CLI: populate the news_snapshot table from Alpaca for a date range.

Spec 003 / T020.

Usage:
    python -m src.agents.cli.backfill_news \
        --start 2026-07-29 --end 2026-08-28 \
        --universe NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,AMD,SPY,QQQ \
        [--per-day-limit 50] [--db-path PATH]

Per spec US1 acceptance scenario 1:
  - One row per (timestamp, symbol) where timestamp = article's published_at
  - Non-null raw_json for replay

Per spec US1 acceptance scenario 4:
  - On HTTP 429/5xx: retry with exponential backoff (handled by AlpacaDataClient)
  - On unrecoverable failure: single non-fatal warning, exit code 0
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.db import connect, init_db, utc_now_iso
from src.agents.alpaca_data import AlpacaDataClient, AlpacaDataError
from src.agents.nodes._lexicon import per_article_sentiment

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _articles_to_cells(
    articles: list[dict], universe: list[str], day_start: str, day_end: str,
) -> list[tuple]:
    """Aggregate articles to (timestamp, symbol) cells; one row per cell.

    Filters:
      - Only symbols in the universe
      - Published within [day_start, day_end] (UTC)

    For each (article_published_at, symbol) cell:
      - sentiment = weighted mean (weight = len(summary)+1) of per-article sentiments
      - article_count = count of articles contributing
      - topics_json = top-3 most-frequent tokens from headlines (deterministic)
      - raw_json = JSON string of the contributing articles
    """
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for art in articles:
        published_at = art.get("published_at") or art.get("created_at")
        if not published_at:
            continue
        if not (day_start <= published_at <= day_end):
            continue
        symbols = [s.upper() for s in (art.get("symbols") or []) if s]
        for sym in symbols:
            if sym not in universe:
                continue
            by_cell.setdefault((published_at, sym), []).append(art)

    rows: list[tuple] = []
    created_at = _utcnow()
    for (ts, sym), arts in by_cell.items():
        # Weighted-mean sentiment: longer summaries count more
        weights = []
        sents = []
        for a in arts:
            s = per_article_sentiment(a)
            w = len(a.get("summary") or "") + len(a.get("headline") or "") + 1
            weights.append(w)
            sents.append(s)
        wsum = sum(weights) or 1
        sentiment = sum(s * w for s, w in zip(sents, weights)) / wsum
        sentiment = max(-1.0, min(1.0, sentiment))
        # Topics: most-frequent tokens from headlines
        from collections import Counter
        counter: Counter[str] = Counter()
        for a in arts:
            for tok in (a.get("headline") or "").lower().split():
                tok = tok.strip(".,;:()[]{}\"'?!-")
                if len(tok) >= 5 and tok.isalpha():
                    counter[tok] += 1
        topics = [w for w, _ in counter.most_common(3)]
        raw_json = json.dumps(arts, sort_keys=True, default=str)
        rows.append((
            ts, sym, float(sentiment), len(arts), json.dumps(topics),
            raw_json, created_at,
        ))
    return rows


def _date_range(start: str, end: str) -> list[tuple[str, str]]:
    """Return list of (day_start_iso, day_end_iso) UTC for the inclusive range."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    days: list[tuple[str, str]] = []
    d = s
    while d <= e:
        ds = d.strftime("%Y-%m-%dT00:00:00Z")
        de = d.strftime("%Y-%m-%dT23:59:59Z")
        days.append((ds, de))
        d += timedelta(days=1)
    return days


def run(args: argparse.Namespace) -> int:
    universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
    conn = connect(args.db_path) if args.db_path else connect()
    try:
        init_db(conn)
        client = AlpacaDataClient()
        days = _date_range(args.start, args.end)
        total_inserted = 0
        total_skipped = 0
        for i, (ds, de) in enumerate(days, 1):
            try:
                # Pass the full ISO timestamp, not the date-only slice.
                # Alpaca's /v1beta1/news returns 0 articles when start == end
                # even if both have a time component on the same day; the
                # per-day iteration already guarantees a non-empty window
                # when ds is the day's start and de is the day's end-of-day.
                articles = client.fetch_news(
                    universe, ds, de, limit=args.per_day_limit,
                )
            except AlpacaDataError as exc:
                logger.warning("alpaca news fetch failed for %s: %s", ds, exc)
                continue
            rows = _articles_to_cells(articles, universe, ds, de)
            if not rows:
                continue
            inserted, skipped = _upsert_rows(conn, rows)
            total_inserted += inserted
            total_skipped += skipped
            print(
                f"[{i}/{len(days)}] {ds[:10]}  "
                f"articles={len(articles):>3}  rows={len(rows):>3}  "
                f"inserted={inserted}  skipped={skipped}"
            )
        print(
            f"[backfill_news] done: days={len(days)}  "
            f"total_inserted={total_inserted}  total_skipped={total_skipped}"
        )
    finally:
        conn.close()
    return 0


def _upsert_rows(conn: sqlite3.Connection, rows: list[tuple]) -> tuple[int, int]:
    """Insert rows into news_snapshot, ignoring duplicates. Returns (inserted, skipped)."""
    if not rows:
        return 0, 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO news_snapshot "
        "(timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    inserted = conn.total_changes - before
    return inserted, len(rows) - inserted


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill news_snapshot from Alpaca")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--universe", required=True, help="comma-separated tickers")
    p.add_argument("--per-day-limit", type=int, default=50)
    p.add_argument("--db-path", default=None)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
