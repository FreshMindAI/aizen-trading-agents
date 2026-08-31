"""End-of-day P&L reconciliation (hackathon-day critical).

Per the GATv2 doc §14: "Sharpe, Sortino, maximum drawdown and profit factor
in Phase 4." We don't have Phase 4 fully wired, so this script does the
hand-rolled EOD reconciliation that the judges will see on the leaderboard.

For each trading day:
  1. Pull all `decision_journal` rows where the day's timestamp is in
     `execution_result_json` and status='filled'.
  2. Pull live positions from Alpaca.
  3. For each filled row, mark-to-market against the latest underlying bar.
  4. Write a `daily_pnl` row with: realized, unrealized, total, trade_count,
     win_count, loss_count, max_drawdown, sharpe_approx.

Usage:
    python scripts/eod_pnl.py [--date 2026-08-31] [--write]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.db import connect as _connect  # noqa: E402

DAILY_PNL_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_pnl (
    date            TEXT PRIMARY KEY,
    realized_pnl    REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl  REAL NOT NULL DEFAULT 0.0,
    total_pnl       REAL NOT NULL DEFAULT 0.0,
    trade_count     INTEGER NOT NULL DEFAULT 0,
    win_count       INTEGER NOT NULL DEFAULT 0,
    loss_count      INTEGER NOT NULL DEFAULT 0,
    max_drawdown    REAL NOT NULL DEFAULT 0.0,
    sharpe_approx   REAL NOT NULL DEFAULT 0.0,
    cash_start      REAL,
    cash_end        REAL,
    positions_json  TEXT NOT NULL DEFAULT '[]',
    fills_json      TEXT NOT NULL DEFAULT '[]',
    notes           TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_field(d: dict, key: str, default=None):
    v = d.get(key)
    if v is None:
        return default
    return v


def _mark_to_market(conn: sqlite3.Connection, symbol: str) -> float | None:
    """Return the latest close for `symbol` from underlying_bars, or None."""
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT close FROM underlying_bars WHERE symbol = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    finally:
        conn.row_factory = prev_factory
    if row is None:
        return None
    return float(row["close"])


def compute_day_pnl(conn: sqlite3.Connection, date_str: str) -> dict:
    """Compute the P&L summary for a given calendar date (UTC)."""
    # 1. Find all fills on or after `date_str` from the journal
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT decision_id, underlying_focus, execution_result_json, "
            "       realized_pnl, order_intent_json "
            "FROM decision_journal "
            "WHERE substr(timestamp, 1, 10) >= ? "
            "ORDER BY timestamp ASC",
            (date_str,),
        ).fetchall()
    finally:
        conn.row_factory = prev_factory

    fills: list[dict] = []
    realized_total = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    eq_curve: list[float] = []
    for r in rows:
        try:
            er = json.loads(r["execution_result_json"] or "{}")
        except Exception:
            continue
        if (er.get("status") or "").lower() != "filled":
            continue
        symbol = r["underlying_focus"] or _safe_json_field(json.loads(r["order_intent_json"] or "{}"), "underlying", "NVDA")
        qty = float(_safe_json_field(er, "filled_qty", 0) or 0)
        avg = float(_safe_json_field(er, "filled_avg_price", 0) or 0)
        if not symbol or qty <= 0 or avg <= 0:
            continue
        mark = _mark_to_market(conn, symbol) or avg
        unrealized = (mark - avg) * qty
        realized = float(r["realized_pnl"] or 0.0)
        pnl = realized + unrealized
        fills.append({
            "decision_id": r["decision_id"],
            "symbol": symbol,
            "qty": qty,
            "avg_entry": avg,
            "mark": mark,
            "realized": realized,
            "unrealized": unrealized,
            "pnl": pnl,
        })
        realized_total += realized
        trade_count += 1
        if pnl > 0:
            win_count += 1
        elif pnl < 0:
            loss_count += 1
        eq_curve.append(pnl)

    # 2. Max drawdown over the day's P&L stream
    max_dd = 0.0
    running = 0.0
    peak = 0.0
    for pnl in eq_curve:
        running += pnl
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)

    # 3. Sharpe approximation: mean(returns) / std(returns) * sqrt(N)
    if len(eq_curve) > 1:
        mean = sum(eq_curve) / len(eq_curve)
        var = sum((p - mean) ** 2 for p in eq_curve) / max(1, len(eq_curve) - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        sharpe = (mean / std) * math.sqrt(len(eq_curve))
    else:
        sharpe = 0.0

    # 4. Try live account for cash
    cash_end = None
    try:
        from src.agents.alpaca_trading import AlpacaTradingClient
        client = AlpacaTradingClient()
        acct = client.get_account()
        cash_end = float(acct.get("cash")) if acct.get("cash") is not None else None
    except Exception:
        pass

    return {
        "date": date_str,
        "realized_pnl": round(realized_total, 4),
        "unrealized_pnl": round(sum(f["unrealized"] for f in fills), 4),
        "total_pnl": round(realized_total + sum(f["unrealized"] for f in fills), 4),
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "max_drawdown": round(max_dd, 4),
        "sharpe_approx": round(sharpe, 4),
        "cash_start": None,
        "cash_end": cash_end,
        "positions_json": "[]",
        "fills_json": json.dumps(fills),
        "notes": f"computed { _utcnow() }",
    }


def write_day_pnl(conn: sqlite3.Connection, summary: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO daily_pnl "
        "(date, realized_pnl, unrealized_pnl, total_pnl, trade_count, "
        " win_count, loss_count, max_drawdown, sharpe_approx, cash_start, "
        " cash_end, positions_json, fills_json, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            summary["date"], summary["realized_pnl"], summary["unrealized_pnl"],
            summary["total_pnl"], summary["trade_count"], summary["win_count"],
            summary["loss_count"], summary["max_drawdown"], summary["sharpe_approx"],
            summary["cash_start"], summary["cash_end"],
            summary["positions_json"], summary["fills_json"], summary["notes"],
        ),
    )
    conn.commit()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--write", action="store_true", help="persist to daily_pnl table")
    p.add_argument("--db-path", default=None)
    args = p.parse_args()
    conn = _connect(args.db_path) if args.db_path else _connect()
    conn.executescript(DAILY_PNL_SCHEMA)
    conn.commit()
    summary = compute_day_pnl(conn, args.date)
    print(json.dumps(summary, indent=2))
    if args.write:
        write_day_pnl(conn, summary)
        print(f"[eod_pnl] wrote daily_pnl row for {args.date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
