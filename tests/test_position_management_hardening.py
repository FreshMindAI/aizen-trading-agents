"""Tests for the 2026-09-03 position-management hardening.

Four independent failure-mode fixes, each covered by a focused test:

  1. Fail-CLOSED on broker fetch error
     test_orchestrator_fail_closed_on_broker_error
     The orchestrator must short-circuit the cycle to NO_TRADE/BLOCKED
     when Alpaca's list_positions call raises. Previous behavior
     returned [] which silently disabled every pre-flight check.

  2. -25% early warning
     test_early_warning_returns_positions_in_warning_zone
     test_early_warning_excludes_stop_loss_zone
     Positions between -25% and -50% loss are returned; positions
     past -50% (which would be auto-closed) are NOT also returned as
     warnings (the louder signal wins).

  3. 404 from close_position is success
     test_auto_close_stop_loss_404_is_already_closed
     When the broker returns 404 (position was closed by another
     path between our snapshot and the call), the result is
     status="already_closed", not "error".

  4. Kill-switch latch table
     test_kill_switch_latch_persists_for_today
     test_kill_switch_latch_clears_on_new_day
     test_orchestrator_short_circuits_on_latched_kill_switch
     A kill-switch trip is written to kill_switch_latch; subsequent
     ticks on the same UTC day return NO_TRADE without re-running
     the math. The next UTC day has a clean slate.

The orchestrator-level tests use a stubbed AlpacaTradingClient and
the public ``Orchestrator.run_cycle`` API so we exercise the wiring
end-to-end, not just the helper functions in isolation.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents import position_management as pm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pos(symbol: str, qty: float, entry: float, mark: float) -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry_price": entry,
        "current_price": mark,
        "unrealized_pl": (mark - entry) * qty,
    }


def _fresh_db() -> sqlite3.Connection:
    """Build an in-memory DB with the kill_switch_latch table present.

    Mirrors what ``init_db`` does on a normal DB. The 71_* migration
    adds the latch table; we execute it explicitly here so tests
    don't depend on running the full migration set.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    sql_path = REPO / "sql" / "71_kill_switch_latch.sql"
    conn.executescript(sql_path.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def _full_db(tmp_path: Path) -> sqlite3.Connection:
    """Build a file-backed DB with the full schema applied. Used by
    orchestrator-level tests that need the decision_journal table to
    exist for the journal upsert path."""
    from src.db import init_db
    db_path = tmp_path / "trading.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# 2. -25% early warning
# ---------------------------------------------------------------------------
def test_early_warning_returns_positions_in_warning_zone():
    """A position at -30% loss is in the early-warning zone (between
    -25% and -50%) and must be returned with a warning entry."""
    pos = _make_pos("AAPL", 10, 100, 70)  # -30%
    out = pm.early_warning_positions([pos], pct=-0.25)
    assert len(out) == 1
    assert out[0]["symbol"] == "AAPL"
    assert out[0]["loss_pct"] == pytest.approx(-0.30)


def test_early_warning_excludes_stop_loss_zone():
    """A position at -60% loss is past the -50% stop-loss; it should
    NOT also appear in the warning list (the auto-close is the
    louder signal — the warning would be redundant noise)."""
    pos = _make_pos("NVDA", 1, 200, 80)  # -60%
    out = pm.early_warning_positions([pos], pct=-0.25)
    assert out == []


def test_early_warning_excludes_safe_positions():
    """A position at -5% loss is well below the warning threshold;
    no warning, no entry."""
    pos = _make_pos("MSFT", 5, 400, 380)  # -5%
    out = pm.early_warning_positions([pos], pct=-0.25)
    assert out == []


# ---------------------------------------------------------------------------
# 3. 404 from close_position is success
# ---------------------------------------------------------------------------
def test_auto_close_stop_loss_404_is_already_closed():
    """When close_position raises an error containing '404' the
    function returns status='already_closed', not 'error'."""
    fake_pos = _make_pos("AAPL", 1, 100, 30)  # -70% past -50%
    client = MagicMock()
    client.close_position.side_effect = RuntimeError(
        "alpaca trading 404 on DELETE /v2/positions/AAPL: ..."
    )
    results = pm.auto_close_stop_loss(client, [fake_pos], pct=-0.50)
    assert len(results) == 1
    assert results[0]["symbol"] == "AAPL"
    assert results[0]["status"] == "already_closed"
    # No error field on a success path.
    assert "error" not in results[0]


def test_auto_close_stop_loss_real_error_stays_error():
    """A non-404 broker error (e.g. 503) must still surface as
    'error' so the operator sees the failure mode."""
    fake_pos = _make_pos("NVDA", 1, 200, 50)
    client = MagicMock()
    client.close_position.side_effect = RuntimeError(
        "alpaca trading 503 on DELETE /v2/positions/NVDA: ..."
    )
    results = pm.auto_close_stop_loss(client, [fake_pos], pct=-0.50)
    assert results[0]["status"] == "error"
    assert "503" in results[0]["error"]


def test_auto_close_stop_loss_happy_path():
    """The successful close path still works and returns
    status='closed' with the broker_order_id."""
    fake_pos = _make_pos("AAPL", 1, 100, 30)
    client = MagicMock()
    client.close_position.return_value = {"id": "order-123"}
    results = pm.auto_close_stop_loss(client, [fake_pos], pct=-0.50)
    assert results[0]["status"] == "closed"
    assert results[0]["broker_order_id"] == "order-123"


# ---------------------------------------------------------------------------
# 4. Kill-switch latch table
# ---------------------------------------------------------------------------
def test_kill_switch_latch_persists_for_today():
    """After record_kill_switch_latch, is_kill_switch_latched_today
    must return True for the same UTC day and False the day after."""
    conn = _fresh_db()
    now = datetime(2026, 9, 3, 14, 30, 0, tzinfo=timezone.utc)
    assert pm.is_kill_switch_latched_today(conn, now=now) is False
    pm.record_kill_switch_latch(
        conn, total_pnl=-3000.0, threshold_usd=-2000.0, pct=-0.02, now=now,
    )
    assert pm.is_kill_switch_latched_today(conn, now=now) is True
    # Day +1: latch should NOT apply (different UTC date).
    tomorrow = now.replace(day=now.day + 1)
    assert pm.is_kill_switch_latched_today(conn, now=tomorrow) is False


def test_kill_switch_latch_clears_on_new_day():
    """Recording for a different day does not affect a query for
    today. This is the implicit-clear-via-PK behavior we depend on."""
    conn = _fresh_db()
    yesterday = datetime(2026, 9, 2, 11, 0, 0, tzinfo=timezone.utc)
    today = datetime(2026, 9, 3, 11, 0, 0, tzinfo=timezone.utc)
    pm.record_kill_switch_latch(
        conn, total_pnl=-3000.0, threshold_usd=-2000.0, pct=-0.02, now=yesterday,
    )
    assert pm.is_kill_switch_latched_today(conn, now=yesterday) is True
    assert pm.is_kill_switch_latched_today(conn, now=today) is False


def test_kill_switch_latch_no_conn_is_noop():
    """None conn: helper returns False on read, no-op on write. Used
    by tests that don't have a real DB."""
    assert pm.is_kill_switch_latched_today(None) is False
    # Write must not raise.
    pm.record_kill_switch_latch(None, total_pnl=-1.0, threshold_usd=-2.0, pct=-0.02)
    # A DB without the latch table is also a clean no-op.
    bare = sqlite3.connect(":memory:")
    assert pm.is_kill_switch_latched_today(bare) is False
    pm.record_kill_switch_latch(
        bare, total_pnl=-1.0, threshold_usd=-2.0, pct=-0.02,
    )
    # And reading again stays False.
    assert pm.is_kill_switch_latched_today(bare) is False


# ---------------------------------------------------------------------------
# 1. Fail-closed on broker fetch error (orchestrator-level)
# ---------------------------------------------------------------------------
def test_orchestrator_fail_closed_on_broker_error(monkeypatch, tmp_path):
    """If Alpaca list_positions raises, the cycle must return
    NO_TRADE with status='BLOCKED' and reason='broker_unreachable'.
    Previous behavior (return []) would have let the cycle proceed
    to open new positions even with no visibility into existing ones.
    """
    # Force the position-management threshold loader to defaults (no
    # config to read in this minimal test).
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "mock")
    # Make sure no env-var overrides push the kill-switch before the
    # broker error short-circuits the cycle.
    monkeypatch.delenv("AIZEN_DAILY_LOSS_KILL_SWITCH_PCT", raising=False)
    monkeypatch.delenv("AIZEN_EARLY_WARNING_PCT", raising=False)

    # Build a DB with the full schema applied. The orchestrator's
    # journal upsert path needs the decision_journal table to exist.
    conn = _full_db(tmp_path)

    from src.agents.graph import Orchestrator

    # Build the orchestrator with a stubbed broker. We patch the
    # AlpacaTradingClient.list_positions to raise.
    with patch("src.agents.alpaca_trading.AlpacaTradingClient") as MockClient:
        instance = MagicMock()
        instance.list_positions.side_effect = RuntimeError("network unreachable")
        MockClient.return_value = instance

        orch = Orchestrator(conn=conn)
        state = orch.run_cycle()
        # Cycle returned BLOCKED, not PROCEED.
        assert state.final_action == "NO_TRADE"
        assert state.execution_result["status"] == "BLOCKED"
        assert state.execution_result["reason"] == "broker_unreachable"
        # The error is recorded in supplementary for the journal trace.
        assert "network unreachable" in state.execution_result["detail"]["error"]
        # A journal row was written.
        row = orch.journal.get(state.decision_id)
        assert row is not None
        assert row["final_action"] == "NO_TRADE"
    conn.close()


def test_orchestrator_short_circuits_on_latched_kill_switch(monkeypatch, tmp_path):
    """If a previous tick today already tripped the kill-switch, the
    next tick must short-circuit to NO_TRADE without even calling the
    broker. This is the operator-intent persistence the latch exists
    for.
    """
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "mock")
    monkeypatch.delenv("AIZEN_DAILY_LOSS_KILL_SWITCH_PCT", raising=False)
    conn = _full_db(tmp_path)
    # Latch today's date BEFORE the orchestrator runs.
    pm.record_kill_switch_latch(
        conn, total_pnl=-3000.0, threshold_usd=-2000.0, pct=-0.02,
    )

    from src.agents.graph import Orchestrator

    with patch("src.agents.alpaca_trading.AlpacaTradingClient") as MockClient:
        instance = MagicMock()
        # If list_positions is called, return an empty list (it must
        # NOT be called - the latch should short-circuit first).
        instance.list_positions.return_value = []
        MockClient.return_value = instance

        orch = Orchestrator(conn=conn)
        state = orch.run_cycle()
        assert state.final_action == "NO_TRADE"
        assert state.execution_result["status"] == "BLOCKED"
        assert state.execution_result["reason"] == "daily_loss_kill_switch_latched"
        # Broker was never queried (latch short-circuits first).
        instance.list_positions.assert_not_called()
    conn.close()
