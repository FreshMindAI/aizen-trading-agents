"""Pre-tick news refresh: fetch the last 15 min of news + rebuild the
news-driven GNN topology snapshot.

Why this loop
-------------
The trade tick (cron-loop.yml) has a 5-minute timeout. Adding a network
call to Alpaca's News API there risks the tick being cancelled mid-
order, so the news fetch is decoupled and runs 5 minutes BEFORE the
trade tick (the workflow schedule is 5/20/35/50 minutes past the hour).

What it does (per invocation)
-----------------------------
1. Read the 15-symbol universe from ``src.config.DEFAULT_UNIVERSE``.
2. Fetch the last 15 minutes of news from Alpaca's ``/v1beta1/news``
   endpoint via :class:`AlpacaDataClient`.
3. Aggregate the raw articles to per-(timestamp, symbol) cells and
   upsert into the ``news_snapshot`` table (the same path the
   backfill CLI uses — deterministic lexicon sentiment, top-3 topics,
   INSERT OR IGNORE on (timestamp, symbol)).
4. Rebuild the news-driven GNN topology snapshot for the universe at
   the current timestamp by calling
   :func:`build_option_payload` + :func:`write_option_snapshot`
   (both already cutoff-aware — they read news_snapshot rows whose
   timestamp <= now, so the rows we just inserted are immediately
   visible).
5. Print a 3-line summary: ``articles_fetched``, ``rows_inserted``,
   ``topology_snapshot_id``.

The loop intentionally does NOT call ``AIZEN_MARKET_HOURS_ONLY`` — news
arrives 24/7 and pre-market / after-hours articles are valuable signal
for the next open. The trade-tick loop keeps the gate; the news loop
does not.

CLI::

    python -m src.agents.cli.news_pre_tick [--universe NVDA,AAPL,...] \\
        [--lookback-minutes 15] [--db-path PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Pin env BEFORE importing anything that reads it.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("AIZEN_TRACE", "0")

from src.config import DEFAULT_UNIVERSE  # noqa: E402
from src.db import connect, init_db, utc_now_iso  # noqa: E402

logger = logging.getLogger("news_pre_tick")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ---------------------------------------------------------------------------
# News fetch + upsert
# ---------------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _window_iso(lookback_minutes: int) -> tuple[str, str]:
    """Return (start_iso, end_iso) for the news lookback window."""
    end = _now_utc()
    start = end - timedelta(minutes=lookback_minutes)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def _articles_to_rows(
    articles: list[dict], universe: list[str],
    window_start: str, window_end: str,
) -> list[tuple]:
    """Aggregate Alpaca articles to (timestamp, symbol) cell rows.

    Re-implemented locally (not imported from backfill_news) so this
    loop is self-contained and won't break if backfill_news is moved.
    The logic is the same: weighted-mean sentiment, top-3 topics from
    headlines, INSERT OR IGNORE on (timestamp, symbol).

    Returns a list of 7-tuples ready for INSERT OR IGNORE into
    ``news_snapshot``:
        (timestamp, symbol, sentiment, article_count, topics_json,
         raw_json, created_at)
    """
    from collections import Counter
    from src.agents.nodes._lexicon import per_article_sentiment

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for art in articles:
        published_at = art.get("published_at") or art.get("created_at")
        if not published_at:
            continue
        if not (window_start <= published_at <= window_end):
            continue
        symbols = [s.upper() for s in (art.get("symbols") or []) if s]
        for sym in symbols:
            if sym not in universe:
                continue
            by_cell.setdefault((published_at, sym), []).append(art)

    rows: list[tuple] = []
    created_at = utc_now_iso()
    for (ts, sym), arts in by_cell.items():
        weights: list[float] = []
        sents: list[float] = []
        for a in arts:
            s = per_article_sentiment(a)
            w = len(a.get("summary") or "") + len(a.get("headline") or "") + 1
            weights.append(float(w))
            sents.append(float(s))
        wsum = sum(weights) or 1
        sentiment = sum(s * w for s, w in zip(sents, weights)) / wsum
        sentiment = max(-1.0, min(1.0, sentiment))
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


def refresh_news(
    universe: list[str], *,
    lookback_minutes: int = 15,
    client: Any | None = None,
    conn_factory: Any | None = None,
) -> dict[str, int]:
    """Fetch Alpaca news for the lookback window and upsert into
    ``news_snapshot``.

    Args:
        universe: list of tickers to filter on.
        lookback_minutes: how far back to pull news (default 15).
        client: an injected :class:`AlpacaDataClient` for tests; if
            None, a fresh client is constructed.
        conn_factory: an injected zero-arg callable returning a
            :class:`sqlite3.Connection`; if None, :func:`src.db.connect`
            is used. Used by tests so they can pass a connection that
            won't be closed by the function's ``finally`` block.

    Returns:
        A dict with ``articles_fetched`` and ``rows_inserted``.
    """
    from src.agents.alpaca_data import AlpacaDataClient, AlpacaDataError

    window_start, window_end = _window_iso(lookback_minutes)
    if client is None:
        client = AlpacaDataClient()
    try:
        articles = client.fetch_news(
            universe, window_start, window_end, limit=50,
        )
    except AlpacaDataError as exc:
        logger.warning("alpaca news fetch failed: %s", exc)
        return {"articles_fetched": 0, "rows_inserted": 0}
    if not articles:
        return {"articles_fetched": 0, "rows_inserted": 0}

    rows = _articles_to_rows(articles, universe, window_start, window_end)
    if not rows:
        return {"articles_fetched": len(articles), "rows_inserted": 0}

    # Apply schema (idempotent) and upsert. When an injected
    # conn_factory is provided, the function does NOT close the
    # connection (the caller owns it). Otherwise a fresh connection
    # is opened and closed in the finally block.
    own_conn = conn_factory is None
    conn = conn_factory() if conn_factory else connect()
    try:
        init_db(conn)
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO news_snapshot "
            "(timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        inserted = conn.total_changes - before
    finally:
        if own_conn:
            conn.close()
    return {"articles_fetched": len(articles), "rows_inserted": inserted}


# ---------------------------------------------------------------------------
# Topology rebuild
# ---------------------------------------------------------------------------
def rebuild_news_topology(
    conn: sqlite3.Connection, universe: list[str],
    *, now_iso: str | None = None,
) -> dict[str, Any]:
    """Rebuild the news-driven GNN topology snapshot for the universe.

    Reuses :func:`build_option_payload` and :func:`write_option_snapshot`
    from ``src.gnn.option_graph`` — both already cutoff-aware (they
    read ``news_snapshot WHERE timestamp <= news_cutoff``), so the rows
    inserted by :func:`refresh_news` are immediately visible.

    Returns a dict with ``snapshot_id``, ``node_count``, ``edge_count``,
    ``topology_version``. Returns an empty dict when no contracts match
    (e.g. on a fresh DB with no option_contracts).
    """
    from src.gnn.option_graph import write_option_snapshot

    ts = now_iso or utc_now_iso()
    result = write_option_snapshot(
        conn, ts, underlying=tuple(universe),
        news_enabled=True, news_cutoff=ts,
    )
    return {
        "snapshot_id": result.get("snapshot_id"),
        "node_count": result.get("node_count", 0),
        "edge_count": result.get("edge_count", 0),
        "topology_version": (
            "option-v2-news" if (result.get("edge_count") or 0) > 0 else "option-v2"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--universe", default=DEFAULT_UNIVERSE,
                   help="Comma-separated tickers (default: src.config.DEFAULT_UNIVERSE)")
    p.add_argument("--lookback-minutes", type=int, default=15,
                   help="How far back to fetch news (default 15)")
    p.add_argument("--db-path", default=None)
    p.add_argument("--skip-topology", action="store_true",
                   help="Don't rebuild the topology snapshot (news-only).")
    args = p.parse_args(argv)

    universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
    if not universe:
        logger.warning("empty universe; nothing to do")
        return 0

    news_stats = refresh_news(universe, lookback_minutes=args.lookback_minutes)
    logger.info(
        "news refresh: articles=%d  rows_inserted=%d",
        news_stats["articles_fetched"], news_stats["rows_inserted"],
    )

    topology: dict[str, Any] = {}
    if not args.skip_topology:
        conn = connect(args.db_path)
        try:
            init_db(conn)
            topology = rebuild_news_topology(conn, universe)
            logger.info(
                "topology rebuild: snapshot_id=%s  nodes=%d  edges=%d  version=%s",
                topology.get("snapshot_id"),
                topology.get("node_count", 0),
                topology.get("edge_count", 0),
                topology.get("topology_version"),
            )
        finally:
            conn.close()

    print("== news_pre_tick summary ==")
    print(f"  universe           : {','.join(universe)}")
    print(f"  lookback_minutes   : {args.lookback_minutes}")
    print(f"  articles_fetched   : {news_stats['articles_fetched']}")
    print(f"  rows_inserted      : {news_stats['rows_inserted']}")
    if topology:
        print(f"  topology_snapshot  : {topology.get('snapshot_id')}")
        print(f"  topology_version   : {topology.get('topology_version')}")
        print(f"  topology_nodes     : {topology.get('node_count')}")
        print(f"  topology_edges     : {topology.get('edge_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
