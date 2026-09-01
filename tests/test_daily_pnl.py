"""Tests for the daily P&L analyzer (Loop 3 of the autonomous-loops plan).

Covers:
- pair_fills_into_trades:
    * Long option round-trip: buy qty=1 @ $2.00 → sell qty=1 @ $2.50
      → pnl = +$50.00 (100 multiplier).
    * Equity round-trip: buy 100 @ $150 → sell 100 @ $152
      → pnl = +$200.00.
    * Unpaired opening fill → "open" classification, pnl = 0.0.
    * Multi-leg short: sell then buy (short-cover) → pnl flips sign.
    * Partial close: open qty=10, close qty=3 → one paired row +
      one open row.
- link_to_decision: client_order_id matched to decision_journal via
  JSON-extract on execution_result_json.
- attach_decision_ids: sets _decision_id on each fill.
- upsert_pnl_rows: idempotent (second call inserts 0).
- classify: thresholds at $5.00 / $5.01 / -$5.01.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("AIZEN_TRACE", "0")

from src.agents.cli.daily_pnl import (  # noqa: E402
    _is_option_symbol,
    attach_decision_ids,
    link_to_decision,
    main,
    pair_fills_into_trades,
    upsert_pnl_rows,
)
from src.db import connect, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path: Path):
    p = tmp_path / "test.db"
    c = connect(str(p))
    init_db(c, sql_dir=REPO / "sql")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# _is_option_symbol
# ---------------------------------------------------------------------------
def test_is_option_symbol_classifies_occ():
    # OCC option symbols are 21 chars, last char is a digit. Tickers
    # <6 chars are left-padded with spaces (AAPL + 2 spaces + YYMMDD
    # + P/C + 8-digit strike = 21 chars).
    assert _is_option_symbol("AAPL  210917C00150000") is True
    assert _is_option_symbol("NVDA  260116P00400000") is True
    # Equities are short.
    assert _is_option_symbol("AAPL") is False
    assert _is_option_symbol("NVDA") is False
    # 21 chars but ending in a letter (not an option) is False.
    assert _is_option_symbol("ABCDEFGHIJKLMNOPQRST") is False


# ---------------------------------------------------------------------------
# pair_fills_into_trades
# ---------------------------------------------------------------------------
def _opt_fill(id: str, side: str, qty: int, price: float, ts: str,
              coid: str = "co-1") -> dict:
    return {
        "id": id,
        "symbol": "AAPL  210917C00150000",  # 21-char OCC symbol
        "side": side,
        "filled_qty": qty,
        "filled_avg_price": price,
        "client_order_id": coid,
        "filled_at": ts,
    }


def _equity_fill(id: str, side: str, qty: int, price: float, ts: str,
                 coid: str = "co-1") -> dict:
    return {
        "id": id,
        "symbol": "AAPL",
        "side": side,
        "filled_qty": qty,
        "filled_avg_price": price,
        "client_order_id": coid,
        "filled_at": ts,
    }


def test_pair_long_option_round_trip():
    fills = [
        _opt_fill("o1", "buy", 1, 2.00, "2026-08-30T13:30:00Z"),
        _opt_fill("o2", "sell", 1, 2.50, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills)
    assert len(rows) == 1
    assert rows[0]["asset_class"] == "option"
    assert rows[0]["realized_pnl"] == pytest.approx(50.0)  # 0.50 * 100 * 1
    assert rows[0]["classification"] == "win"


def test_pair_long_equity_round_trip():
    fills = [
        _equity_fill("e1", "buy", 100, 150.0, "2026-08-30T13:30:00Z"),
        _equity_fill("e2", "sell", 100, 152.0, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills)
    assert len(rows) == 1
    assert rows[0]["asset_class"] == "equity"
    assert rows[0]["realized_pnl"] == pytest.approx(200.0)
    assert rows[0]["classification"] == "win"


def test_pair_short_option_round_trip_flips_sign():
    fills = [
        _opt_fill("o1", "sell", 1, 3.00, "2026-08-30T13:30:00Z"),
        _opt_fill("o2", "buy", 1, 2.00, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills)
    assert len(rows) == 1
    # Short: pnl = (entry - exit) * 100 * 1 = (3 - 2) * 100 = 100
    assert rows[0]["realized_pnl"] == pytest.approx(100.0)
    assert rows[0]["classification"] == "win"


def test_pair_unpaired_open_returns_open_row():
    fills = [
        _equity_fill("e1", "buy", 50, 150.0, "2026-08-30T13:30:00Z"),
    ]
    rows = pair_fills_into_trades(fills)
    assert len(rows) == 1
    assert rows[0]["classification"] == "open"
    assert rows[0]["realized_pnl"] == 0.0
    assert rows[0]["exit_price"] == 0.0


def test_pair_partial_close_splits_one_open_one_paired():
    fills = [
        _equity_fill("e1", "buy", 10, 100.0, "2026-08-30T13:30:00Z"),
        _equity_fill("e2", "sell", 3, 110.0, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills)
    # 1 paired row (qty=3) + 1 open row (remaining qty=7).
    classified = [r for r in rows if r["classification"] != "open"]
    opened = [r for r in rows if r["classification"] == "open"]
    assert len(classified) == 1
    assert len(opened) == 1
    assert classified[0]["quantity"] == 3
    assert classified[0]["realized_pnl"] == pytest.approx(30.0)  # 3 * (110 - 100)
    assert opened[0]["quantity"] == 7


def test_pair_threshold_breakeven():
    # pnl = $4.99 < $5 threshold → breakeven
    fills = [
        _equity_fill("e1", "buy", 1, 100.0, "2026-08-30T13:30:00Z"),
        _equity_fill("e2", "sell", 1, 104.99, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills, threshold=5.0)
    assert rows[0]["classification"] == "breakeven"


def test_pair_threshold_just_above_wins():
    fills = [
        _equity_fill("e1", "buy", 1, 100.0, "2026-08-30T13:30:00Z"),
        _equity_fill("e2", "sell", 1, 105.01, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills, threshold=5.0)
    assert rows[0]["classification"] == "win"


def test_pair_threshold_just_below_loss():
    fills = [
        _equity_fill("e1", "buy", 1, 100.0, "2026-08-30T13:30:00Z"),
        _equity_fill("e2", "sell", 1, 94.99, "2026-08-30T14:00:00Z"),
    ]
    rows = pair_fills_into_trades(fills, threshold=5.0)
    assert rows[0]["classification"] == "loss"


def test_pair_zero_qty_or_price_skipped():
    fills = [
        {"id": "x1", "symbol": "AAPL", "side": "buy", "filled_qty": 0,
         "filled_avg_price": 100, "filled_at": "2026-08-30T13:30:00Z"},
        _equity_fill("e1", "buy", 1, 100.0, "2026-08-30T13:31:00Z"),
        _equity_fill("e2", "sell", 1, 105.0, "2026-08-30T13:32:00Z"),
    ]
    rows = pair_fills_into_trades(fills)
    # The zero-qty fill is dropped; the other two pair.
    assert len(rows) == 1
    assert rows[0]["quantity"] == 1


# ---------------------------------------------------------------------------
# link_to_decision + attach_decision_ids
# ---------------------------------------------------------------------------
def _insert_journal_row(
    conn: sqlite3.Connection, decision_id: str, client_order_id: str,
):
    """Insert a decision_journal row whose execution_result_json
    contains the given client_order_id."""
    er = json.dumps({"client_order_id": client_order_id, "status": "submitted"})
    conn.execute(
        "INSERT INTO decision_journal (decision_id, timestamp, completed_at, "
        "market_state_hash, schema_version, run_mode, underlying_focus, "
        "final_action, market_snapshot_json, ml_prediction_json, "
        "gnn_output_json, agent_messages_json, agent_observations_json, "
        "strategy_proposal_json, selected_strategy_json, "
        "risk_decision_json, order_intent_json, execution_result_json, "
        "model_versions, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (decision_id, "2026-08-30T13:30:00Z", "2026-08-30T13:31:00Z",
         "h", "1.0", "paper", "AAPL", "PROCEED",
         "{}", "[]", "{}", "[]", "[]", "null", "null",
         "null", "null", er, "[]", "2026-08-30T13:30:00Z"),
    )
    conn.commit()


def test_link_to_decision_maps_client_order_id(conn: sqlite3.Connection):
    _insert_journal_row(conn, "d1", "co-abc")
    _insert_journal_row(conn, "d2", "co-xyz")
    fills = [
        {"client_order_id": "co-abc"},
        {"client_order_id": "co-xyz"},
        {"client_order_id": "co-missing"},  # no journal row
    ]
    mapping = link_to_decision(conn, fills)
    assert mapping == {"co-abc": "d1", "co-xyz": "d2", "co-missing": None}
    attach_decision_ids(fills, mapping)
    assert fills[0]["_decision_id"] == "d1"
    assert fills[1]["_decision_id"] == "d2"
    assert fills[2]["_decision_id"] is None


def test_link_to_decision_empty_fills(conn: sqlite3.Connection):
    assert link_to_decision(conn, []) == {}


# ---------------------------------------------------------------------------
# upsert_pnl_rows
# ---------------------------------------------------------------------------
def test_upsert_pnl_rows_inserts_then_idempotent(conn: sqlite3.Connection):
    rows = [{
        "decision_id": "d1",
        "client_order_id": "co-1",
        "close_broker_id": "broker-1",
        "close_filled_at": "2026-08-30T14:00:00Z",
        "symbol": "AAPL",
        "asset_class": "equity",
        "side": "sell",
        "quantity": 1,
        "entry_price": 100.0,
        "exit_price": 110.0,
        "realized_pnl": 10.0,
        "classification": "win",
    }]
    n1 = upsert_pnl_rows(conn, rows, now_iso="2026-08-30T14:01:00Z")
    n2 = upsert_pnl_rows(conn, rows, now_iso="2026-08-30T14:02:00Z")
    assert n1 == 1
    assert n2 == 0  # PK collision
    # DB has exactly one row.
    n = conn.execute("SELECT COUNT(*) AS n FROM decision_pnl").fetchone()["n"]
    assert n == 1


def test_upsert_pnl_rows_handles_empty(conn: sqlite3.Connection):
    assert upsert_pnl_rows(conn, []) == 0


# ---------------------------------------------------------------------------
# main() smoke test (--no-fetch)
# ---------------------------------------------------------------------------
def test_main_no_fetch_prints_zero_summary(capsys, tmp_path: Path, monkeypatch):
    rc = main([
        "--since-hours", "24",
        "--db-path", str(tmp_path / "x.db"),
        "--no-fetch",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "daily_pnl summary" in out
    assert "n_fills          : 0" in out
    assert "n_paired         : 0" in out


def test_main_with_fake_client(capsys, tmp_path: Path, monkeypatch):
    db_path = tmp_path / "trading.db"
    monkeypatch.setenv("AIZEN_DB_PATH", str(db_path))

    # Stub fetch_fills to return a deterministic pair.
    import src.agents.cli.daily_pnl as dp
    monkeypatch.setattr(
        dp, "fetch_fills",
        lambda after, until, **kw: [
            _equity_fill("e1", "buy", 1, 100.0, "2026-08-30T13:30:00Z",
                         coid="co-test"),
            _equity_fill("e2", "sell", 1, 110.0, "2026-08-30T14:00:00Z",
                         coid="co-test"),
        ],
    )
    rc = main(["--since-hours", "24", "--db-path", str(db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_fills          : 2" in out
    assert "n_paired         : 1" in out
    assert "n_unpaired       : 0" in out
    assert "total_realized   : 10.00 USD" in out
    assert "win_rate         : 100.00%" in out
