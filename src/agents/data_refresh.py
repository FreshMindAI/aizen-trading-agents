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

import pandas as pd

logger = logging.getLogger(__name__)


def _alpaca_stock_client():
    """Return an Alpaca historical-data client. Imported lazily so
    tests / dry-runs that don't hit the broker don't pay the import."""
    try:
        from src.alpaca_client import AlpacaHistoricalClient
        return AlpacaHistoricalClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alpaca client import failed: %s", exc)
        return None


def refresh_one_bar(
    universe: Iterable[str],
    *,
    db_path: Path | str,
    timeframe: str = "15Min",
    lookback_minutes: int = 60,
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
            try:
                df = client.get_bars(sym, timeframe=timeframe, start=start, end=end)
            except Exception as exc:  # noqa: BLE001
                logger.warning("alpaca get_bars(%s) failed: %s", sym, exc)
                continue
            if df is None or df.empty:
                continue
            # Normalize to the underlying_bars schema.
            rows = []
            for _, r in df.iterrows():
                ts = r.get("timestamp") or r.name
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                rows.append((
                    sym,
                    str(ts),
                    float(r.get("open", 0.0)),
                    float(r.get("high", 0.0)),
                    float(r.get("low", 0.0)),
                    float(r.get("close", 0.0)),
                    int(r.get("volume", 0)),
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
