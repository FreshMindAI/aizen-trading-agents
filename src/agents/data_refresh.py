"""Lightweight per-tick data refresh.

Pulls the latest 1-min/15-min bar for each underlying from Alpaca and
upserts into ``underlying_bars``. Option chains are *not* refreshed
here — they need a separate pre-market job (Alpaca options snapshot
endpoints have stricter rate limits and the synthetic option_contracts
table is already populated for the hackathon window).

Failure mode: if Alpaca is unreachable, the function returns an empty
dict and logs a warning. The cron job continues and runs a cycle on
the latest data we have. The fallback is preferable to failing the
whole tick.

This module deliberately has no LLM dependency; it's the "I/O" half
of the cron tick.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _alpaca_stock_client():
    """Return an Alpaca historical-data client. Imported lazily so
    tests / dry-runs that don't hit the broker don't pay the import."""
    try:
        from src.alpaca_client import AlpacaClient
        return AlpacaClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alpaca client import failed: %s", exc)
        return None


def _fetch_bars(client, symbol: str, timeframe: str, start: datetime, end: datetime, *, feed: str = "iex") -> list[dict]:
    """Hit the v2 stock bars endpoint for one symbol and return a
    flat list of bar dicts (one per row). Returns an empty list on
    error or empty response (errors are logged by the client).

    `feed` is the Alpaca data subscription: "iex" (free, 15-min delayed
    intraday) or "sip" (paid, real-time). Default to iex so the free
    paper account works out of the box; the cron env can override to
    "sip" if the account is upgraded.
    """
    # Alpaca v2 bars API:
    # GET {data_base_url}/v2/stocks/{symbol}/bars
    #   ?timeframe=15Min&start=ISO&end=ISO&limit=10000&feed=iex
    # Pagination: response includes "next_page_token" which we pass
    # back as `page_token` on the next request.
    settings = client.settings
    base = settings.data_base_url.rstrip("/")
    url = f"{base}/v2/stocks/{symbol}/bars"
    out: list[dict] = []
    page_token: str | None = None
    # Cap at 5 pages of 10k bars each = 50k bars per symbol (way more
    # than 15-min bars in a hackathon window).
    for _ in range(5):
        params: dict[str, str | int] = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "feed": feed,
        }
        if page_token:
            params["page_token"] = page_token
        try:
            payload = client.get(url, params=params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpaca get_bars(%s) failed: %s", symbol, exc)
            return out
        if not isinstance(payload, dict):
            break
        out.extend(payload.get("bars", []) or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return out


def refresh_one_bar(
    universe: Iterable[str],
    *,
    db_path: Path | str,
    timeframe: str = "1Hour",
    lookback_minutes: int = 60 * 4,  # 4 hours of hourly bars
    feed: str = "iex",
) -> dict[str, int]:
    """Pull the most recent bar for each symbol and upsert into
    ``underlying_bars``. Returns {symbol: n_rows_written}.

    Silently no-ops on broker errors; the cron job continues on stale
    data rather than failing the whole tick.
    """
    import sqlite3
    symbols = [s for s in universe if s]
    if not symbols:
        return {}
    client = _alpaca_stock_client()
    if client is None:
        return {}
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    written: dict[str, int] = {}
    p = Path(db_path)
    conn = sqlite3.connect(p)
    conn.row_factory = None  # tuple rows are fine for the bulk insert
    try:
        for sym in symbols:
            bars = _fetch_bars(client, sym, timeframe, start, end, feed=feed)
            if not bars:
                continue
            # Normalize to the underlying_bars schema.
            rows = []
            for r in bars:
                ts = r.get("t") or r.get("timestamp")
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                rows.append((
                    sym,
                    str(ts),
                    float(r.get("o", 0.0)),
                    float(r.get("h", 0.0)),
                    float(r.get("l", 0.0)),
                    float(r.get("c", 0.0)),
                    int(r.get("v", 0)),
                ))
            if not rows:
                continue
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO underlying_bars "
                    "(symbol, timestamp, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                written[sym] = len(rows)
            except sqlite3.OperationalError as exc:
                logger.warning("underlying_bars upsert for %s failed: %s", sym, exc)
    finally:
        conn.close()
    return written
