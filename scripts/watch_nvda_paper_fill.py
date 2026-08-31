"""Watch the NVDA paper-trade order and log the fill into decision_journal.

Polls Alpaca for the order every ``--poll-seconds`` seconds. On terminal
state (filled / canceled / expired / rejected), updates the
``decision_journal`` row with the final execution data, appends the
poll trail to ``agent_messages_json``, and (if filled) records the
realized P&L into ``realized_pnl`` using the most recent underlying
bar as the mark.

Usage:
    python scripts/watch_nvda_paper_fill.py [--order-id <id>] [--poll-seconds 60]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.alpaca_trading import AlpacaTradingClient  # noqa: E402
from src.db import connect as _connect  # noqa: E402

TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _update_journal_row(
    conn: sqlite3.Connection,
    decision_id: str,
    order_payload: dict,
    *,
    message: str,
) -> None:
    """Merge the latest order payload into ``execution_result_json`` and
    append a free-text ``message`` to ``agent_messages_json``."""
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT execution_result_json, agent_messages_json "
            "FROM decision_journal WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
    finally:
        conn.row_factory = prev_factory
    if row is None:
        print(f"[watch] journal row {decision_id} not found - skip")
        return
    existing = json.loads(row["execution_result_json"] or "{}")
    existing.update(order_payload)
    msgs = json.loads(row["agent_messages_json"] or "[]")
    msgs.append({
        "ts": _utcnow(),
        "actor": "fill_watcher",
        "kind": "order_poll",
        "message": message,
    })
    conn.execute(
        "UPDATE decision_journal SET execution_result_json = ?, "
        "agent_messages_json = ? WHERE decision_id = ?",
        (json.dumps(existing), json.dumps(msgs), decision_id),
    )
    conn.commit()
    print(f"[watch] journal row updated -> {decision_id}")


def _try_record_realized_pnl(
    conn: sqlite3.Connection,
    decision_id: str,
    order: dict,
) -> None:
    """Mark the fill against the latest underlying bar close and store
    the result in ``realized_pnl``. Best-effort only - skipped if any
    of the inputs are missing."""
    symbol = order.get("symbol")
    try:
        fill_price = float(order.get("filled_avg_price") or 0.0)
        qty = float(order.get("filled_qty") or 0.0)
    except (TypeError, ValueError):
        return
    if not symbol or not fill_price or not qty:
        return
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
        return
    mark = float(row["close"])
    pnl = round((mark - fill_price) * qty, 4)
    conn.execute(
        "UPDATE decision_journal SET realized_pnl = ?, "
        "outcome_label = 'filled' WHERE decision_id = ?",
        (pnl, decision_id),
    )
    conn.commit()
    print(f"[watch] mark recorded: fill={fill_price} mark={mark} qty={qty} pnl={pnl}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--order-id", default="e876e7b2-c0cc-4917-b810-2c83c3399691")
    p.add_argument("--decision-id", default="paper-trade-20260829-NVDA-001")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-polls", type=int, default=600)  # 10 hours at 60s
    args = p.parse_args()

    client = AlpacaTradingClient()
    conn = _connect()
    print(f"[watch] starting poll: order={args.order_id} decision={args.decision_id} "
          f"interval={args.poll_seconds}s max_polls={args.max_polls}")

    for i in range(args.max_polls):
        try:
            o = client._request("GET", f"/v2/orders/{args.order_id}")
        except Exception as e:
            print(f"[watch] poll {i}: API error: {e}")
            time.sleep(args.poll_seconds)
            continue
        status = o.get("status")
        filled_qty = o.get("filled_qty")
        filled_avg = o.get("filled_avg_price")
        ts = _utcnow()
        print(f"[watch] poll {i:03d} {ts} status={status} "
              f"filled_qty={filled_qty} filled_avg_price={filled_avg}")
        _update_journal_row(
            conn,
            args.decision_id,
            o,
            message=f"poll {i} status={status} filled_qty={filled_qty} "
                    f"filled_avg_price={filled_avg}",
        )
        if status in TERMINAL_STATUSES:
            print(f"[watch] terminal status: {status} -> exit")
            if status == "filled":
                _try_record_realized_pnl(conn, args.decision_id, o)
            return 0
        time.sleep(args.poll_seconds)
    print(f"[watch] max polls reached without terminal status")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
