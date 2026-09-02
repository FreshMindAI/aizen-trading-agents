"""Lightweight per-tick data refresh.

Pulls the latest 1-min/15-min bar for each underlying from Alpaca and
upserts into ``underlying_bars``. For options, picks the ATM subset of
``option_contracts`` already in the DB and pulls the last few hours of
delayed-feed option bars into ``option_bars`` so the option ML model
has a non-empty per-contract bar history to score on each tick.

Failure mode: if Alpaca is unreachable, the function returns an empty
dict and logs a warning. The cron job continues and runs a cycle on
the latest data we have. The fallback is preferable to failing the
whole tick.

This module deliberately has no LLM dependency; it's the "I/O" half
of the cron tick.
"""
from __future__ import annotations

import logging
import sqlite3
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
    timeframe: str = "15Min",
    lookback_minutes: int = 4 * 60,  # 4 hours of 15-minute bars = 16 bars
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


# ---- Option chain refresh (live tick) ---------------------------------------
# The option ML model (``option_h4_xgb_clf`` / ``option_h4_xgb_reg``) needs
# a per-contract bar history to build its 27-feature vector. Without a live
# refresh of ``option_bars`` the snapshot's option path sees an empty table
# and falls through to the heuristic - which is exactly the bug we are
# fixing. This function is the live-tick analog of
# ``src.download_option_bars``: it picks the ATM subset of contracts already
# in ``option_contracts`` (populated once by
# ``src.download_option_contracts``) and pulls the last few hours of
# delayed-feed bars from the same Alpaca endpoint.
#
# Why the historical-bars endpoint and not /v2/options/snapshots? The
# snapshots endpoint only carries the latest trade / quote / greeks - no
# bar history, so it cannot produce a row for ``option_bars``. The free
# tier's ``/v1beta1/options/bars`` with ``feed=indicative`` returns bars
# with a ~15-min delay, which is fine for a 15-min cron tick and for the
# option_h4 model (which predicts 4 bars ahead, not 0 bars).
#
# Like ``refresh_one_bar`` this function:
#   * is no-op on broker errors (returns ``{}``);
#   * never raises into the cron tick;
#   * is parameterized by the universe (caller controls which symbols).
def _alpaca_data_client():
    """Return an Alpaca data-host client. Imported lazily so tests /
    dry-runs that don't hit the broker don't pay the import."""
    try:
        from src.alpaca_client import AlpacaClient
        return AlpacaClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alpaca data client import failed: %s", exc)
        return None


def _pick_atm_contracts(conn: sqlite3.Connection, symbol: str,
                        spot: float, max_contracts: int = 6,
                        max_dte: int = 45) -> list[str]:
    """Pick up to ``max_contracts`` ATM contracts (calls AND puts) for
    ``symbol`` from ``option_contracts`` whose DTE is within
    ``[1, max_dte]``.

    Bug C fix: the previous version filtered on
    ``option_type = 'call'`` only, so the put side of the chain
    never had live bars and the option_h4 model would score them
    with a stale (or empty) bar history. Both calls and puts are
    equally important - a long-put strategy on META needs the
    ``META260902P00572500`` bar series, not just the call side.

    Ranking: the order matters because ``LIMIT max_contracts``
    will otherwise grab all-calls (sorted alphabetically) on the
    earliest expiry and starve the put side. We sort by ATM
    distance first so the result naturally balances calls and
    puts at the nearest strikes, with nearest-expiry as a
    tiebreaker. The list is bounded so a single symbol's full
    chain doesn't fan out to hundreds of HTTP calls.
    """
    rows = conn.execute(
        """
        SELECT contract_symbol,
               option_type,
               CAST(julianday(expiration_date) - julianday('now') AS INTEGER) AS dte
        FROM   option_contracts
        WHERE  underlying_symbol = ?
          AND  option_type IN ('call', 'put')
          AND  tradable = 1
          AND  CAST(julianday(expiration_date) - julianday('now') AS INTEGER) BETWEEN 1 AND ?
        ORDER BY ABS(strike_price - ?) ASC,
                 expiration_date ASC,
                 option_type ASC
        LIMIT  ?
        """,
        (symbol, max_dte, spot, max_contracts),
    ).fetchall()
    return [r["contract_symbol"] for r in rows]


def refresh_option_chains(
    universe: Iterable[str],
    *,
    db_path: Path | str,
    lookback_minutes: int = 4 * 60,
    max_contracts_per_symbol: int = 6,
    timeframe: str = "15Min",
    feed: str = "indicative",
    max_dte: int = 45,
) -> dict[str, int]:
    """Pull the most recent option bars for the ATM subset of the
    universe's contracts and upsert into ``option_bars``.

    Returns ``{contract_symbol: n_rows_written}``. Returns ``{}`` on
    broker errors or if no contracts are present. This is the live-tick
    analog of ``src.download_option_bars`` - the difference is that it
    reads the contract list from the local DB (not from
    ``--contract-symbols`` on the CLI) and uses the delayed
    ``indicative`` feed by default so it works on a free paper account.
    """
    symbols = [s for s in universe if s]
    if not symbols:
        return {}
    client = _alpaca_data_client()
    if client is None:
        logger.warning("refresh_option_chains: no alpaca client, returning empty")
        return {}
    # Data host. We deliberately use ``/v1beta1/options/snapshots``
    # (NOT ``/v1beta1/options/bars``) because the bars endpoint
    # requires an OPRA agreement on Alpaca (free paper accounts get
    # HTTP 403 "OPRA agreement is not signed"). The snapshots endpoint
    # serves the same daily OHLCV bar for the current session on the
    # data host without an OPRA subscription, which is what the
    # option_h4 XGBoost model needs to build its per-contract feature
    # vector. We write the snapshot's ``dailyBar`` to ``option_bars``
    # as a single row per contract (the option_h4 feature builder
    # takes the LAST bar per contract via
    # ``ROW_NUMBER() OVER (PARTITION BY contract_symbol ORDER BY
    # timestamp DESC)`` so one fresh row per tick is sufficient).
    from src.config import DATA_HOST
    base = DATA_HOST.rstrip("/")
    url = f"{base}/v1beta1/options/snapshots"
    p = Path(db_path)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    written: dict[str, int] = {}
    contracts_picked = 0
    broker_calls = 0
    try:
        for sym in symbols:
            spot_row = conn.execute(
                "SELECT close AS s FROM underlying_bars WHERE symbol=? "
                "ORDER BY timestamp DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if spot_row is None or spot_row["s"] is None:
                logger.debug("refresh_option_chains: no spot for %s, skipping", sym)
                continue
            spot = float(spot_row["s"])
            # Re-stamp the option bar to the latest underlying_bars
            # timestamp for this symbol. The snapshot endpoint returns
            # the daily bar with a fixed UTC-midnight-style timestamp
            # (e.g. ``2026-09-02T04:00:00Z``) but ``underlying_bars``
            # are stamped at the 15Min cadence during the trading
            # session (e.g. ``2026-09-02T15:30:00Z``). The
            # ``_OPTION_LATEST_SQL`` in src/ml/predict.py joins on
            # ``ub.timestamp = g.timestamp`` and would otherwise return
            # 0 rows, leaving the option_h4 XGBoost model without
            # features and forcing every contract to fall through to
            # ``model_version = "heuristic-1"``. We re-stamp to the
            # underlying's latest timestamp so the join succeeds and
            # the real ML path runs.
            ub_ts_row = conn.execute(
                "SELECT timestamp AS t FROM underlying_bars WHERE symbol=? "
                "ORDER BY timestamp DESC LIMIT 1",
                (sym,),
            ).fetchone()
            aligned_ts = ub_ts_row["t"] if ub_ts_row else None
            contracts = _pick_atm_contracts(
                conn, sym, spot,
                max_contracts=max_contracts_per_symbol, max_dte=max_dte,
            )
            if not contracts:
                logger.debug("refresh_option_chains: no contracts in option_contracts for %s", sym)
                continue
            contracts_picked += len(contracts)
            # Chunk to honor the ~100-symbols-per-request cap that the
            # snapshots endpoint accepts.
            chunk_size = 100
            for i in range(0, len(contracts), chunk_size):
                chunk = contracts[i : i + chunk_size]
                params: dict[str, str] = {
                    "symbols": ",".join(chunk),
                }
                try:
                    payload = client.get(url, params=params)
                    broker_calls += 1
                    if not isinstance(payload, dict):
                        break
                    snapshots = payload.get("snapshots") or {}
                    bars_in_page = 0
                    for csym, snap in snapshots.items():
                        if not isinstance(snap, dict):
                            continue
                        daily = snap.get("dailyBar") or snap.get("minuteBar")
                        if not isinstance(daily, dict):
                            continue
                        raw_ts = _bar_ts(daily)
                        if not raw_ts:
                            continue
                        # Use the underlying-aligned timestamp when
                        # available so the option_bars.underlying_bars
                        # join in _OPTION_LATEST_SQL succeeds.
                        ts = aligned_ts or raw_ts
                        try:
                            conn.execute(
                                "INSERT OR REPLACE INTO option_bars "
                                "(contract_symbol, timestamp, open, high, low, close, "
                                " volume, vwap, trade_count, feed) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (
                                    csym, ts,
                                    _bar_float(daily, "o", "open"),
                                    _bar_float(daily, "h", "high"),
                                    _bar_float(daily, "l", "low"),
                                    _bar_float(daily, "c", "close"),
                                    _bar_int(daily, "v", "volume"),
                                    _bar_float(daily, "vw", "vwap"),
                                    _bar_int(daily, "n", "trade_count"),
                                    "indicative",
                                ),
                            )
                            written[csym] = written.get(csym, 0) + 1
                            bars_in_page += 1
                        except sqlite3.OperationalError as exc:
                            logger.warning(
                                "option_bars upsert for %s failed: %s", csym, exc,
                            )
                    if bars_in_page == 0:
                        # 200 OK with no snapshots for these contracts.
                        # Don't retry - move on.
                        logger.debug(
                            "refresh_option_chains: 0 snapshots returned for %s chunk %s",
                            sym, chunk[:3],
                        )
                    conn.commit()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("alpaca options snapshots for %s failed: %s", sym, exc)
                    continue
    finally:
        # Bug C fix: structured summary at INFO level so the cron log
        # shows whether the refresh actually wrote anything. Without
        # this, a silent failure (broker returned 200 OK with empty
        # bars, or no ATM contracts match the 1-45 DTE filter) looks
        # identical to a successful refresh.
        logger.info(
            "refresh_option_chains summary: universe=%d, contracts_picked=%d, "
            "broker_calls=%d, rows_written=%d, n_contracts_with_bars=%d",
            len(symbols), contracts_picked, broker_calls,
            sum(written.values()), len(written),
        )
        conn.close()
    return written


# ---- Option contracts refresh (live tick) -------------------------------
# Bug: on the cloud cron the runner is ephemeral - the SQLite DB is
# wiped on every invocation, which means ``option_contracts`` is empty
# at the start of each tick. ``refresh_option_chains`` only READS from
# ``option_contracts`` to pick the ATM subset, so it returns ``{}``
# every cloud cycle, ``option_bars`` stays empty, and the option_h4
# XGBoost model has no per-contract bar history to score. The cycle
# then falls through to equity-only and the user's hackathon mandate
# ("trade options, not just stocks") is silently unmet.
#
# Fix: ``populate_option_contracts`` runs at the TOP of every cron
# tick and re-fetches the option chain from
# ``PAPER_HOST/v2/options/contracts`` for each underlying, bounded by
# a 1-30 DTE window and a +/-10% strike band. This mirrors
# ``src.download_option_contracts.py:main`` (the historical one-shot
# downloader) but is a function (not a CLI) and is silent on broker
# errors. The contract set is small (<= 100 rows per symbol) so the
# network cost per tick is ~1 HTTP call per underlying.
#
# ``refresh_option_chains`` then has a non-empty ``option_contracts``
# table to read, populates ``option_bars`` for the ATM subset, and
# the option_h4 model can score the contracts on this tick.
def _fetch_option_contracts(client, symbol: str, spot: float,
                            *, min_dte: int, max_dte: int,
                            band_pct: float) -> list[tuple]:
    """Hit ``PAPER_HOST/v2/options/contracts`` for one symbol and
    return rows ready to upsert into ``option_contracts``.

    The API takes ``expiration_date_gte`` / ``expiration_date_lte``
    as ISO date strings and ``strike_price_gte`` / ``strike_price_lte``
    as numerics. We bound the strike band to +/-10% of the latest
    stored spot (mirroring the ``download_option_contracts`` default)
    so we don't pull a thousand OTM contracts per symbol.

    Returns a list of tuples in the column order of the
    ``option_contracts`` table.
    """
    from datetime import date, timedelta
    from src.config import PAPER_HOST
    base = PAPER_HOST.rstrip("/")
    url = f"{base}/v2/options/contracts"
    lo = round(spot * (1 - band_pct), 2)
    hi = round(spot * (1 + band_pct), 2)
    min_exp = (date.today() + timedelta(days=min_dte)).isoformat()
    max_exp = (date.today() + timedelta(days=max_dte)).isoformat()
    params: dict = {
        "underlying_symbols": symbol,
        "status": "active",
        "expiration_date_gte": min_exp,
        "expiration_date_lte": max_exp,
        "strike_price_gte": lo,
        "strike_price_lte": hi,
        "limit": 10000,
    }
    rows: list[tuple] = []
    seen: set[str] = set()
    try:
        for payload in client.paginate(url, params):
            contracts = payload.get("option_contracts") or payload.get("contracts") or []
            if not isinstance(contracts, list):
                continue
            for c in contracts:
                if not isinstance(c, dict):
                    continue
                sym = c.get("symbol")
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                rows.append((
                    sym,
                    c.get("id"),
                    c.get("underlying_symbol") or symbol,
                    c.get("expiration_date"),
                    c.get("strike_price"),
                    c.get("type"),
                    c.get("style"),
                    c.get("status"),
                    1 if c.get("tradable") else 0,
                    c.get("root_symbol"),
                    datetime.now(timezone.utc).isoformat(),
                ))
    except Exception as exc:  # noqa: BLE001
        # Treat any per-symbol broker failure as a no-op for that
        # symbol. The cron tick should not fail because one underlying
        # hit a 5xx.
        logger.warning("populate_option_contracts: %s failed: %s", symbol, exc)
        return []
    return rows


def populate_option_contracts(
    universe: Iterable[str],
    *,
    db_path: Path | str,
    min_dte: int = 1,
    max_dte: int = 30,
    band_pct: float = 0.10,
) -> dict[str, int]:
    """Populate ``option_contracts`` for every symbol in the universe
    with the 1-30 DTE, +/-10% strike-band subset of the chain.

    Returns ``{symbol: n_contracts_upserted}``. Silent on broker
    errors: returns ``{}`` if Alpaca is unreachable, and logs
    per-symbol warnings for any 5xx/timeout. The function is the
    live-tick analog of ``src.download_option_contracts.py`` -
    designed to be called on every cron tick so the option ML
    model has a non-empty contract list to score, even on the
    ephemeral GitHub Actions runner where the DB is wiped on
    every invocation.
    """
    symbols = [s for s in universe if s]
    if not symbols:
        return {}
    client = _alpaca_data_client()
    if client is None:
        logger.warning("populate_option_contracts: no alpaca client, returning empty")
        return {}
    p = Path(db_path)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    written: dict[str, int] = {}
    contracts_by_type: dict[str, int] = {}
    try:
        for sym in symbols:
            spot_row = conn.execute(
                "SELECT close AS s FROM underlying_bars WHERE symbol=? "
                "ORDER BY timestamp DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if spot_row is None or spot_row["s"] is None:
                logger.debug("populate_option_contracts: no spot for %s, skipping", sym)
                continue
            spot = float(spot_row["s"])
            rows = _fetch_option_contracts(
                client, sym, spot,
                min_dte=min_dte, max_dte=max_dte, band_pct=band_pct,
            )
            if not rows:
                continue
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO option_contracts "
                    "(contract_symbol, contract_id, underlying_symbol, "
                    " expiration_date, strike_price, option_type, style, "
                    " status, tradable, root_symbol, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                written[sym] = len(rows)
                # Track the type balance so a one-sided chain (calls
                # only or puts only) is visible in the cron log.
                for r in rows:
                    t = r[5] or "unknown"
                    contracts_by_type[t] = contracts_by_type.get(t, 0) + 1
            except sqlite3.OperationalError as exc:
                logger.warning("option_contracts upsert for %s failed: %s", sym, exc)
    finally:
        logger.info(
            "populate_option_contracts summary: universe=%d, contracts=%d, "
            "by_type=%s",
            len(symbols), sum(written.values()), contracts_by_type,
        )
        conn.close()
    return written


def _iter_option_bars(payload: dict) -> Iterable[tuple[str, dict]]:
    """Yield (contract_symbol, bar) whether the bars dict is keyed by
    symbol or carries the symbol on each row."""
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if isinstance(bars, dict):
        for csym, items in bars.items():
            for bar in items or []:
                yield csym, bar
    elif isinstance(bars, list):
        for bar in bars:
            if isinstance(bar, dict):
                yield bar.get("S") or bar.get("symbol"), bar


def _bar_ts(bar: dict) -> str | None:
    raw = bar.get("t") or bar.get("timestamp")
    if raw is None:
        return None
    if hasattr(raw, "isoformat"):
        return raw.isoformat()
    return str(raw)


def _bar_float(bar: dict, *names: str) -> float | None:
    for n in names:
        if n in bar and bar[n] is not None:
            try:
                return float(bar[n])
            except (TypeError, ValueError):
                return None
    return None


def _bar_int(bar: dict, *names: str) -> int | None:
    for n in names:
        if n in bar and bar[n] is not None:
            try:
                return int(float(bar[n]))
            except (TypeError, ValueError):
                return None
    return None
