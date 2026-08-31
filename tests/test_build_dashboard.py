"""Tests for the static HTML dashboard generator.

The dashboard is a single-file static page dropped into ``dashboard/`` and
pushed to ``gh-pages`` on every Render cron tick. The tests below pin
the contract:

  * Missing DB → placeholder HTML, no exceptions, exit 0.
  * Empty DB → "No cycles yet" / "No PROCEED cycle yet" empties.
  * Populated DB → KPI numbers, trade rows, signals, equity curve.
  * All sections present in the rendered HTML.
  * Output is valid HTML (has <!doctype html> at the start).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from scripts.build_dashboard import render, main as dash_main  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path):
    """A SQLite DB with the decision_journal schema + a few sample rows."""
    db = tmp_path / "trading.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE decision_journal (
            decision_id          TEXT PRIMARY KEY,
            timestamp            TEXT NOT NULL,
            completed_at         TEXT,
            market_state_hash    TEXT NOT NULL DEFAULT '',
            schema_version       TEXT NOT NULL DEFAULT '1.0',
            run_mode             TEXT NOT NULL DEFAULT 'paper',
            underlying_focus     TEXT,
            final_action         TEXT NOT NULL DEFAULT 'NO_TRADE',
            outcome_label        TEXT,
            realized_pnl         REAL,
            market_snapshot_json TEXT NOT NULL DEFAULT '{}',
            ml_prediction_json   TEXT NOT NULL DEFAULT '[]',
            gnn_output_json      TEXT NOT NULL DEFAULT '{}',
            topology_version     TEXT,
            agent_messages_json  TEXT NOT NULL DEFAULT '[]',
            agent_observations_json TEXT NOT NULL DEFAULT '[]',
            strategy_proposal_json TEXT NOT NULL DEFAULT 'null',
            selected_strategy_json TEXT NOT NULL DEFAULT 'null',
            risk_decision_json   TEXT NOT NULL DEFAULT 'null',
            order_intent_json    TEXT NOT NULL DEFAULT 'null',
            execution_result_json TEXT NOT NULL DEFAULT 'null',
            model_versions       TEXT NOT NULL DEFAULT '[]',
            created_at           TEXT NOT NULL DEFAULT ''
        );
    """)
    # Insert three sample cycles: one PROCEED win, one PROCEED loss, one NO_TRADE.
    strategy_eq = {
        "strategy_id": "strat-eq-1",
        "underlying": "AAPL",
        "score": 0.62,
        "thesis": "Long AAPL on bullish ML signal (dp=0.78)",
        "legs": [{
            "asset_class": "equity",
            "contract_symbol": "AAPL",
            "side": "buy",
            "quantity": 5,
            "limit_price": 180.0,
        }],
        "expected_return": 0.5, "probability_profit": 0.7, "confidence": 0.7,
        "max_loss": 900.0, "liquidity_metrics": {}, "expiry": "2026-08-30",
    }
    strategy_opt = {
        "strategy_id": "strat-opt-1",
        "underlying": "NVDA",
        "score": 0.55,
        "thesis": "Long NVDA call",
        "legs": [{
            "asset_class": "option",
            "contract_symbol": "NVDA260918C00500000",
            "side": "buy",
            "quantity": 1,
            "option_type": "call",
            "strike": 500.0,
            "expiry": "2026-09-18",
            "limit_price": 5.0,
        }],
        "expected_return": 0.3, "probability_profit": 0.6, "confidence": 0.6,
        "max_loss": 500.0, "liquidity_metrics": {}, "expiry": "2026-09-18",
    }
    obs = [{
        "agent_id": "direction_agent",
        "message_type": "DIRECTION_VIEW",
        "timestamp": "2026-08-30T13:30:00Z",
        "confidence": 0.78,
        "signal": {"directional_bias": "bullish", "top_pick": "AAPL"},
        "evidence": [], "risks": [],
    }]
    intent_eq = {
        "intent_id": "i-1", "broker": "ALPACA", "account_mode": "PAPER",
        "strategy_id": "strat-eq-1", "underlying": "AAPL",
        "legs": strategy_eq["legs"], "quantity": 5,
        "limit_price": 180.0, "time_in_force": "day",
    }
    intent_opt = {
        "intent_id": "i-2", "broker": "ALPACA", "account_mode": "PAPER",
        "strategy_id": "strat-opt-1", "underlying": "NVDA",
        "legs": strategy_opt["legs"], "quantity": 1,
        "limit_price": 5.0, "time_in_force": "day",
    }
    er_filled = {
        "status": "filled", "filled_qty": 5, "filled_avg_price": 180.0,
        "broker_order_id": "abc-1",
    }
    er_filled2 = {
        "status": "filled", "filled_qty": 1, "filled_avg_price": 5.0,
        "broker_order_id": "abc-2",
    }
    conn.execute(
        "INSERT INTO decision_journal "
        "(decision_id, timestamp, final_action, underlying_focus, "
        " realized_pnl, outcome_label, selected_strategy_json, "
        " order_intent_json, execution_result_json, agent_observations_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("d-1", "2026-08-29T13:30:00Z", "PROCEED", "AAPL", 75.0, "win",
         json.dumps(strategy_eq), json.dumps(intent_eq), json.dumps(er_filled),
         json.dumps(obs)),
    )
    conn.execute(
        "INSERT INTO decision_journal "
        "(decision_id, timestamp, final_action, underlying_focus, "
        " realized_pnl, outcome_label, selected_strategy_json, "
        " order_intent_json, execution_result_json, agent_observations_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("d-2", "2026-08-30T13:30:00Z", "PROCEED", "NVDA", -120.0, "loss",
         json.dumps(strategy_opt), json.dumps(intent_opt),
         json.dumps(er_filled2), json.dumps(obs)),
    )
    conn.execute(
        "INSERT INTO decision_journal "
        "(decision_id, timestamp, final_action, underlying_focus, "
        " selected_strategy_json, agent_observations_json) "
        "VALUES (?,?,?,?,?,?)",
        ("d-3", "2026-08-30T18:00:00Z", "NO_TRADE", "MSFT", "null",
         json.dumps([])),
    )
    conn.commit()
    conn.close()
    return db


def test_missing_db_writes_placeholder(tmp_path, monkeypatch, capsys):
    """When the DB doesn't exist, the script writes a placeholder and exits 0."""
    missing = tmp_path / "nope.db"
    out = tmp_path / "out" / "index.html"
    rc = dash_main(["--db", str(missing), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    content = out.read_text()
    assert "<!doctype html>" in content.lower()
    assert "No data yet" in content
    assert str(missing) in content


def test_renders_all_sections(tmp_db, tmp_path):
    """The populated dashboard must contain every named section."""
    out = tmp_path / "index.html"
    rc = dash_main(["--db", str(tmp_db), "--out", str(out)])
    assert rc == 0
    content = out.read_text()
    assert content.startswith("<!doctype html>")
    # Required sections (text content)
    for needle in (
        "Aizen Trading",
        "Total realized P&amp;L",
        "Today",
        "Last 7 days",
        "Trades",
        "Win rate",
        "Max drawdown",
        "Sharpe",
        "Next-day plan",
        "Equity curve",
        "Recent cycles",
        "Recent agent signals",
    ):
        assert needle in content, f"missing section: {needle!r}"


def test_kpi_numbers_from_realized_pnl(tmp_db, tmp_path):
    """Total realized = +75 - 120 = -45 across the two PROCEED cycles."""
    out = tmp_path / "index.html"
    dash_main(["--db", str(tmp_db), "--out", str(out)])
    content = out.read_text()
    # Total realized P&L should be -$45.00. The cell uses $.+ format
    # (format_money), so -$45.00 is the rendered form.
    assert "$-45.00" in content
    # Win / Loss count
    assert ">1<" not in content  # sanity: we are not checking exact cell counts
    # 1 win, 1 loss
    assert "1 / 1" in content


def test_next_day_plan_is_most_recent_proceed(tmp_db, tmp_path):
    """The most recent PROCEED cycle's strategy is the next-day plan."""
    out = tmp_path / "index.html"
    dash_main(["--db", str(tmp_db), "--out", str(out)])
    content = out.read_text()
    # The 2nd PROCEED cycle (NVDA) is the most recent.
    assert "NVDA" in content
    assert "Long NVDA call" in content


def test_trade_history_includes_all_proceed_rows(tmp_db, tmp_path):
    """The trade-history table shows both PROCEED cycles."""
    out = tmp_path / "index.html"
    dash_main(["--db", str(tmp_db), "--out", str(out)])
    content = out.read_text()
    assert "AAPL" in content
    assert "NVDA" in content
    # Asset class badges
    assert "equity" in content
    assert "option" in content


def test_signals_section_lists_agent_views(tmp_db, tmp_path):
    """The signals section surfaces the direction_agent observation."""
    out = tmp_path / "index.html"
    dash_main(["--db", str(tmp_db), "--out", str(out)])
    content = out.read_text()
    assert "direction_agent" in content
    assert "DIRECTION_VIEW" in content


def test_empty_db_renders_gracefully(tmp_path):
    """A DB with the schema but zero rows still produces a valid dashboard."""
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE decision_journal (
            decision_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            final_action TEXT NOT NULL DEFAULT 'NO_TRADE',
            underlying_focus TEXT,
            realized_pnl REAL,
            outcome_label TEXT,
            selected_strategy_json TEXT NOT NULL DEFAULT 'null',
            order_intent_json TEXT NOT NULL DEFAULT 'null',
            execution_result_json TEXT NOT NULL DEFAULT 'null',
            agent_observations_json TEXT NOT NULL DEFAULT '[]',
            market_snapshot_json TEXT NOT NULL DEFAULT '{}',
            ml_prediction_json TEXT NOT NULL DEFAULT '[]',
            gnn_output_json TEXT NOT NULL DEFAULT '{}',
            agent_messages_json TEXT NOT NULL DEFAULT '[]',
            strategy_proposal_json TEXT NOT NULL DEFAULT 'null',
            risk_decision_json TEXT NOT NULL DEFAULT 'null',
            model_versions TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT '',
            completed_at TEXT,
            market_state_hash TEXT NOT NULL DEFAULT '',
            schema_version TEXT NOT NULL DEFAULT '1.0',
            run_mode TEXT NOT NULL DEFAULT 'paper',
            topology_version TEXT
        );
    """)
    conn.commit()
    conn.close()
    out = tmp_path / "out.html"
    rc = dash_main(["--db", str(db), "--out", str(out)])
    assert rc == 0
    content = out.read_text()
    assert "<!doctype html>" in content.lower()
    assert "No cycles yet" in content
    assert "No PROCEED cycle yet" in content
