"""Daily trade P&L analyzer (Loop 3 of the autonomous-loops plan).

Why this loop
-------------
The cron loop fires every 15 min and submits orders through the
multi-agent orchestrator. Until now, there was no automated way to
grade yesterday's trades by **realized** P&L. This module closes
that feedback loop by:

1. Fetching the last 24h of fills from the Alpaca paper-trading
   account.
2. FIFO-pairing them per symbol (buy → sell for long, sell → buy for
   short-cover).
3. Linking each fill back to the originating ``decision_journal``
   row via ``client_order_id`` (which we already stamp on
   ``submit_order``).
4. Classifying each paired trade as ``win`` / ``loss`` / ``breakeven``
   at a $5 threshold.
5. Upserting into the new ``decision_pnl`` table.

Why daily, not per-tick
-----------------------
- Realized P&L requires a **closing** fill; most option structures
  open in the morning and close before close. A per-tick check would
  always be in the middle of a trade.
- Alpaca's order history endpoint is rate-limited; one daily call
  keeps the budget low.

CLI::

    python -m src.agents.cli.daily_pnl [--since-hours 24] \\
        [--threshold 5.0] [--db-path PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Pin env BEFORE importing anything that reads it.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("AIZEN_TRACE", "0")

from src.db import connect, init_db, utc_now_iso  # noqa: E402

logger = logging.getLogger("daily_pnl")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EQUITY_MULTIPLIER = 1.0   # equity: 1 share per share
OPTION_MULTIPLIER = 100.0 # options: 100 shares per contract

CLASSIFY_THRESHOLD_DEFAULT = 5.0  # USD; |pnl| < threshold => breakeven


# ---------------------------------------------------------------------------
# Fill pairing
# ---------------------------------------------------------------------------
def _is_option_symbol(symbol: str) -> bool:
    """OCC option symbols are 21 chars and end in a digit (the
    strike's last digit, after a letter). Equities are shorter and
    all-uppercase. Heuristic, but good enough to classify."""
    if not symbol:
        return False
    s = symbol.upper()
    if len(s) != 21:
        return False
    if not s[-1].isdigit():
        return False
    return True


def _classify(asset_class: str) -> bool:
    return asset_class == "option"


def _fill_key(fill: dict) -> tuple:
    """Pair fills on (symbol, asset_class). Same symbol + same asset
    class = same "position bucket"."""
    symbol = (fill.get("symbol") or "").upper()
    asset_class = "option" if _is_option_symbol(symbol) else "equity"
    return (symbol, asset_class)


def pair_fills_into_trades(
    fills: list[dict],
    *,
    threshold: float = CLASSIFY_THRESHOLD_DEFAULT,
) -> list[dict]:
    """FIFO-pair fills on the same symbol.

    Each output row has::

        {
            "symbol": "AAPL" | "OCC210917C00150000",
            "asset_class": "equity" | "option",
            "side": "buy" | "sell",  # the side of the closing fill
            "quantity": int,
            "entry_price": float,   # avg price of the opening fill(s)
            "exit_price": float,    # avg price of the closing fill(s)
            "realized_pnl": float,  # signed USD
            "open_broker_id": str,
            "close_broker_id": str,
            "client_order_id": str, # of the closing fill
            "decision_id": str | None,  # linked via journal lookup
            "classification": "win" | "loss" | "breakeven",
        }

    Unpaired fills (open positions with no matching close) are
    returned as a single row with ``classification="open"`` and
    ``realized_pnl=0.0`` so the caller can record them as "still
    open" without losing the original fill data.

    Multipliers: options = $100/contract × qty × (exit - entry);
    equities = qty × (exit - entry). Short positions (sell → buy
    cover) flip the sign.
    """
    buckets: dict[tuple, deque] = {}
    paired: list[dict] = []
    unmatched: list[dict] = []

    # Sort ascending by filled_at so the FIFO order is correct.
    sorted_fills = sorted(
        fills, key=lambda f: f.get("filled_at") or f.get("submitted_at") or ""
    )

    for fill in sorted_fills:
        key = _fill_key(fill)
        if key not in buckets:
            buckets[key] = deque()
        # Normalize to the shape the rest of the function expects.
        side = (fill.get("side") or "").lower()
        qty = int(fill.get("filled_qty") or fill.get("qty") or 0)
        price = float(fill.get("filled_avg_price") or fill.get("limit_price") or 0.0)
        symbol, asset_class = key
        if qty <= 0 or price <= 0:
            continue

        bucket = buckets[key]
        while qty > 0 and bucket:
            open_fill = bucket[0]
            open_qty = open_fill["remaining_qty"]
            open_side = open_fill["side"]
            # Same side = additive; opposite side = closing.
            if open_side == side:
                # Stack the same-direction fill into the existing open
                # by averaging the price.
                total_qty = open_fill["filled_qty"] + qty
                open_fill["entry_price"] = (
                    open_fill["entry_price"] * open_fill["filled_qty"]
                    + price * qty
                ) / total_qty
                open_fill["filled_qty"] = total_qty
                open_fill["remaining_qty"] = total_qty
                qty = 0
            else:
                # Opposite side = closing.
                close_qty = min(open_qty, qty)
                entry_price = open_fill["entry_price"]
                multiplier = (
                    OPTION_MULTIPLIER if _classify(asset_class) else EQUITY_MULTIPLIER
                )
                # Long (buy then sell): pnl = (exit - entry) × mult × qty
                # Short (sell then buy): pnl = (entry - exit) × mult × qty
                if open_side == "buy":
                    pnl = (price - entry_price) * multiplier * close_qty
                else:
                    pnl = (entry_price - price) * multiplier * close_qty
                if pnl > threshold:
                    classification = "win"
                elif pnl < -threshold:
                    classification = "loss"
                else:
                    classification = "breakeven"
                paired.append({
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "side": side,
                    "quantity": close_qty,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "realized_pnl": pnl,
                    "open_broker_id": open_fill["broker_order_id"],
                    "close_broker_id": fill.get("id", ""),
                    "open_filled_at": open_fill["filled_at"],
                    "close_filled_at": fill.get("filled_at", ""),
                    "open_client_order_id": open_fill["client_order_id"],
                    "client_order_id": fill.get("client_order_id", ""),
                    "open_decision_id": open_fill["decision_id"],
                    "decision_id": fill.get("_decision_id"),  # set by link_to_decision
                    "classification": classification,
                })
                open_fill["remaining_qty"] -= close_qty
                if open_fill["remaining_qty"] <= 0:
                    bucket.popleft()
                qty -= close_qty
        if qty > 0:
            # Carry the remainder as an opening fill.
            bucket.append({
                "side": side,
                "filled_qty": qty,
                "remaining_qty": qty,
                "entry_price": price,
                "broker_order_id": fill.get("id", ""),
                "client_order_id": fill.get("client_order_id", ""),
                "decision_id": fill.get("_decision_id"),
                "filled_at": fill.get("filled_at", ""),
            })

    # Anything left in the buckets is unpaired (open).
    for (symbol, asset_class), bucket in buckets.items():
        for open_fill in bucket:
            unmatched.append({
                "symbol": symbol,
                "asset_class": asset_class,
                "side": open_fill["side"],
                "quantity": open_fill["remaining_qty"],
                "entry_price": open_fill["entry_price"],
                "exit_price": 0.0,
                "realized_pnl": 0.0,
                "open_broker_id": open_fill["broker_order_id"],
                "close_broker_id": "",
                "open_filled_at": open_fill["filled_at"],
                "close_filled_at": "",
                "open_client_order_id": open_fill["client_order_id"],
                "client_order_id": open_fill["client_order_id"],
                "open_decision_id": open_fill["decision_id"],
                "decision_id": open_fill["decision_id"],
                "classification": "open",
            })

    return paired + unmatched


# ---------------------------------------------------------------------------
# Decision linking
# ---------------------------------------------------------------------------
def link_to_decision(
    conn: sqlite3.Connection, fills: list[dict],
) -> dict[str, str | None]:
    """For each fill, set ``_decision_id`` by JSON-extracting
    ``client_order_id`` from ``decision_journal.execution_result_json``
    (a soft FK; no ALTER TABLE needed).

    Returns a mapping of ``client_order_id -> decision_id`` for all
    fills. Missing rows (no matching journal entry) are mapped to
    None — those fills are still recorded but ``decision_id`` is
    NULL in the resulting ``decision_pnl`` row.
    """
    client_ids: set[str] = set()
    for f in fills:
        coid = f.get("client_order_id")
        if coid:
            client_ids.add(coid)
    if not client_ids:
        return {}

    # JSON1 is enabled in modern SQLite by default. We use json_extract
    # on the text column; SQLite will JSON-parse on the fly.
    placeholders = ",".join("?" * len(client_ids))
    sql = (
        "SELECT decision_id, json_extract(execution_result_json, "
        " '$.client_order_id') AS coid "
        "FROM decision_journal "
        f"WHERE json_extract(execution_result_json, '$.client_order_id') "
        f"IN ({placeholders})"
    )
    rows = conn.execute(sql, list(client_ids)).fetchall()
    mapping: dict[str, str | None] = {coid: None for coid in client_ids}
    for r in rows:
        if r["coid"]:
            mapping[r["coid"]] = r["decision_id"]
    return mapping


def attach_decision_ids(
    fills: list[dict], mapping: dict[str, str | None],
) -> list[dict]:
    """Set ``_decision_id`` on each fill from the mapping (in place;
    also returns the list for convenience)."""
    for f in fills:
        coid = f.get("client_order_id")
        if coid and coid in mapping:
            f["_decision_id"] = mapping[coid]
    return fills


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def upsert_pnl_rows(
    conn: sqlite3.Connection, rows: list[dict], *, now_iso: str | None = None,
) -> int:
    """INSERT OR IGNORE into decision_pnl; return number inserted."""
    if not rows:
        return 0
    now = now_iso or utc_now_iso()
    payload = []
    for r in rows:
        payload.append((
            r.get("decision_id"),
            r.get("client_order_id") or "",
            r.get("close_broker_id") or r.get("open_broker_id") or "",
            r.get("close_filled_at") or r.get("open_filled_at") or "",
            r.get("symbol", ""),
            r.get("asset_class", "equity"),
            r.get("side", "buy"),
            int(r.get("quantity", 0)),
            float(r.get("entry_price", 0.0)),
            float(r.get("exit_price", 0.0)),
            float(r.get("realized_pnl", 0.0)),
            r.get("classification", "open"),
            now,
        ))
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO decision_pnl "
        "(decision_id, client_order_id, broker_order_id, filled_at, "
        " symbol, asset_class, side, quantity, entry_price, exit_price, "
        " realized_pnl, classification, computed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        payload,
    )
    conn.commit()
    return conn.total_changes - before


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_fills(
    after_iso: str, until_iso: str,
    *, client: Any | None = None,
) -> list[dict]:
    """Pull filled orders from Alpaca between ``after_iso`` and
    ``until_iso`` (inclusive). Returns raw order dicts."""
    from src.agents.alpaca_trading import AlpacaTradingClient
    if client is None:
        client = AlpacaTradingClient()
    return client.list_fills(after=after_iso, until=until_iso, status="filled")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--since-hours", type=float, default=24.0,
                   help="How many hours back to look (default 24).")
    p.add_argument("--threshold", type=float, default=CLASSIFY_THRESHOLD_DEFAULT,
                   help="USD threshold for win/loss/breakeven (default 5.0).")
    p.add_argument("--db-path", default=None)
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip the Alpaca fetch; pair whatever is already in the DB.")
    p.add_argument("--client", default=None,
                   help="Path to a JSON file with fill dicts (test/dev use).")
    args = p.parse_args(argv)

    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(hours=args.since_hours)
    since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    until_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.client:
        with open(args.client, "r", encoding="utf-8") as fh:
            fills = json.load(fh)
        logger.info("loaded %d fills from %s", len(fills), args.client)
    elif args.no_fetch:
        fills = []
        logger.info("--no-fetch: no fills to pair")
    else:
        fills = fetch_fills(since_iso, until_iso)
        logger.info("alpaca returned %d fills between %s and %s",
                    len(fills), since_iso, until_iso)

    conn = connect(args.db_path)
    try:
        init_db(conn)
        mapping = link_to_decision(conn, fills)
        attach_decision_ids(fills, mapping)
        rows = pair_fills_into_trades(fills, threshold=args.threshold)
        inserted = upsert_pnl_rows(conn, rows)
    finally:
        conn.close()

    n_paired = sum(1 for r in rows if r["classification"] != "open")
    n_unpaired = sum(1 for r in rows if r["classification"] == "open")
    total_realized = sum(r["realized_pnl"] for r in rows if r["classification"] != "open")
    wins = sum(1 for r in rows if r["classification"] == "win")
    win_rate = (wins / n_paired) if n_paired else 0.0
    print("== daily_pnl summary ==")
    print(f"  since            : {since_iso}")
    print(f"  until            : {until_iso}")
    print(f"  n_fills          : {len(fills)}")
    print(f"  n_paired         : {n_paired}")
    print(f"  n_unpaired       : {n_unpaired}")
    print(f"  rows_inserted    : {inserted}")
    print(f"  total_realized   : {total_realized:.2f} USD")
    print(f"  wins             : {wins}")
    print(f"  win_rate         : {win_rate:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
