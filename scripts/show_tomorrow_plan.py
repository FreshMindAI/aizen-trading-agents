"""Print the system's plan for the next market open (CLI version).

The dashboard shows the same info in HTML; this script is for the
operator who wants to read the plan in the terminal before pre-market
opens. Output is a single greppable block:

  ==
  next_day_plan:
    decision_id   : <uuid>
    cycle_at      : <iso>
    underlying    : <symbol>
    asset_class   : option | equity
    score         : <float>
    thesis        : <text>
    legs          : <count>
    side / qty    : <buy Nx symbol>
    limit_price   : <price>   (equity only)
  ==

Usage:
    python scripts/show_tomorrow_plan.py
    python scripts/show_tomorrow_plan.py --db /var/data/aizen/trading.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.db import connect as _connect  # noqa: E402


def _load_plan(conn) -> dict | None:
    cur = conn.execute(
        "SELECT decision_id, timestamp, underlying_focus, "
        "       selected_strategy_json, order_intent_json "
        "FROM decision_journal "
        "WHERE final_action = 'PROCEED' "
        "ORDER BY timestamp DESC LIMIT 1"
    )
    row = cur.fetchone()
    if row is None:
        return None
    sel = json.loads(row["selected_strategy_json"] or "{}") or {}
    intent = json.loads(row["order_intent_json"] or "{}") or {}
    if not isinstance(sel, dict):
        sel = {}
    if not isinstance(intent, dict):
        intent = {}
    return {
        "decision_id": row["decision_id"],
        "timestamp": row["timestamp"],
        "underlying": row["underlying_focus"],
        "score": sel.get("score"),
        "thesis": sel.get("thesis"),
        "legs": sel.get("legs", []),
        "intent": intent,
    }


def _print_plan(plan: dict | None) -> None:
    if plan is None:
        print("==")
        print("next_day_plan: NONE")
        print("  no PROCEED cycle yet — system has not produced a plan")
        print("==")
        return
    legs = plan.get("legs") or []
    if legs and isinstance(legs[0], dict):
        asset_class = legs[0].get("asset_class") or "option"  # legacy default
    else:
        asset_class = "?"
    side_qty = "—"
    limit_price = "—"
    if legs:
        leg = legs[0]
        side_qty = f"{leg.get('side','?')} {leg.get('quantity','?')}x {leg.get('contract_symbol','?')}"
        if leg.get("limit_price") is not None:
            limit_price = f"${leg['limit_price']}"
        elif leg.get("asset_class") == "option":
            limit_price = f"${leg.get('limit_price', '?')}"
    print("==")
    print("next_day_plan:")
    print(f"  decision_id   : {plan.get('decision_id')}")
    print(f"  cycle_at      : {plan.get('timestamp')}")
    print(f"  underlying    : {plan.get('underlying')}")
    print(f"  asset_class   : {asset_class}")
    score = plan.get("score")
    print(f"  score         : {score:.3f}" if isinstance(score, (int, float)) else f"  score         : —")
    print(f"  thesis        : {plan.get('thesis') or '—'}")
    print(f"  legs          : {len(legs)}")
    print(f"  side / qty    : {side_qty}")
    if asset_class == "equity":
        print(f"  limit_price   : {limit_price}")
    elif asset_class == "option":
        if legs:
            leg = legs[0]
            print(f"  option_type   : {leg.get('option_type', '—')}")
            print(f"  strike        : {leg.get('strike', '—')}")
            print(f"  expiry        : {leg.get('expiry', '—')}")
            print(f"  limit_price   : {limit_price}")
    print("==")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.getenv("AIZEN_DB_PATH", str(REPO / "data" / "trading.db")))
    args = p.parse_args(argv)
    if not Path(args.db).exists():
        print(f"[show_tomorrow_plan] db not found: {args.db}")
        return 1
    conn = _connect(args.db)
    try:
        plan = _load_plan(conn)
        _print_plan(plan)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
