"""Tests for the failure knowledge-graph writer (T168 / T169).

Covers:
- classify_error() maps exception class names to (kind, severity).
- record_symbol_failure / record_agent_failure / record_cycle_failure
  insert rows; re-recording the same key increments error_count.
- record_cycle_failure emits ``co_occurs`` edges to per-symbol
  failures for the same decision_id.
- link_agent_to_symbol wires an agent's failure to the symbol it
  processed.
- symbol_failure_counts and symbol_failure_features aggregate
  per-symbol error counts over a rolling window.
- The writer tolerates a missing failure_nodes table (defensive:
  does not crash the calling code path).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.failure_kg import FailureKG, classify_error
from src.db import init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_conn() -> sqlite3.Connection:
    """An isolated in-memory DB with init_db applied (so 70_failure_kg
    creates the failure_nodes / failure_edges tables)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn, sql_dir=Path(REPO) / "sql")
    return conn


# ---------------------------------------------------------------------------
# 1. classify_error
# ---------------------------------------------------------------------------
def test_classify_error_known_llm():
    kind, sev = classify_error(TimeoutError("anthropic 504"))
    assert kind == "request"  # Timeout is in _REQUEST_ERROR_NAMES
    assert sev == "warn"


def test_classify_error_broker():
    kind, sev = classify_error(RuntimeError("Alpaca 403 forbidden"))
    assert kind == "broker"
    assert sev == "error"


def test_classify_error_agent_value_error():
    kind, sev = classify_error(ValueError("ml output was None"))
    assert kind == "agent"
    assert sev == "error"


def test_classify_error_none_returns_info():
    kind, sev = classify_error(None)
    assert kind == "info"
    assert sev == "info"


# ---------------------------------------------------------------------------
# 2. record_symbol_failure
# ---------------------------------------------------------------------------
def test_symbol_failure_persists_row():
    conn = _make_conn()
    kg = FailureKG(conn)
    nid = kg.record_symbol_failure(
        "AAPL", "2026-08-25T13:30:00Z",
        exc=TimeoutError("option chain fetch timed out"),
    )
    assert nid.startswith("symbol_failure:AAPL")
    row = conn.execute(
        "SELECT kind, symbol, severity, error_class, error_count "
        "FROM failure_nodes WHERE node_id = ?", (nid,),
    ).fetchone()
    assert row["kind"] == "symbol_failure"
    assert row["symbol"] == "AAPL"
    assert row["severity"] == "warn"  # Timeout is request-class
    assert row["error_class"] == "TimeoutError"
    assert row["error_count"] == 1


def test_symbol_failure_increments_on_re_record():
    """Re-recording the same (symbol, occurred_at) pair must
    increment error_count, not insert a duplicate row."""
    conn = _make_conn()
    kg = FailureKG(conn)
    for _ in range(3):
        kg.record_symbol_failure(
            "AAPL", "2026-08-25T13:30:00Z",
            exc=TimeoutError("timeout"),
        )
    rows = conn.execute(
        "SELECT error_count FROM failure_nodes WHERE symbol = 'AAPL'",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["error_count"] == 3


# ---------------------------------------------------------------------------
# 3. record_agent_failure
# ---------------------------------------------------------------------------
def test_agent_failure_persists_with_symbol_link():
    conn = _make_conn()
    kg = FailureKG(conn)
    nid = kg.record_agent_failure(
        "direction", "2026-08-25T13:30:00Z", "AAPL",
        exc=ValueError("ml output was None"),
    )
    # Agent-failure node ids are keyed on (agent_id, occurred_at)
    # only — the same agent failing on two symbols at the same
    # minute is the same failure event, so the symbol is NOT in
    # the id (it lives on the row's `symbol` column instead).
    assert nid == "agent_failure:-:direction:-:2026-08-25T13:30:00Z"
    row = conn.execute(
        "SELECT kind, agent_id, symbol, error_class FROM failure_nodes "
        "WHERE node_id = ?", (nid,),
    ).fetchone()
    assert row["kind"] == "agent_failure"
    assert row["agent_id"] == "direction"
    assert row["symbol"] == "AAPL"
    assert row["error_class"] == "ValueError"


# ---------------------------------------------------------------------------
# 4. record_cycle_failure + co_occurs edge
# ---------------------------------------------------------------------------
def test_cycle_failure_emits_co_occurs_edges():
    """A cycle failure must emit co_occurs edges to every per-symbol
    failure row for the same decision_id and day."""
    conn = _make_conn()
    kg = FailureKG(conn)
    # Pre-seed two symbol failures tagged with the same decision_id
    # so record_cycle_failure() can find them via the join.
    kg.record_symbol_failure(
        "AAPL", "2026-08-25T13:30:00Z", exc=RuntimeError("alpaca 503"),
        decision_id="decision-abc",
    )
    kg.record_symbol_failure(
        "MSFT", "2026-08-25T13:30:00Z", exc=RuntimeError("alpaca 503"),
        decision_id="decision-abc",
    )
    # Now record a cycle failure. It should emit 2 co_occurs edges.
    nid = kg.record_cycle_failure(
        "decision-abc", "2026-08-25T13:30:00Z",
        final_action="NO_TRADE", exc=RuntimeError("alpaca 503"),
    )
    edges = conn.execute(
        "SELECT target_node_id, relation, weight FROM failure_edges "
        "WHERE source_node_id = ?", (nid,),
    ).fetchall()
    assert len(edges) == 2
    assert all(e["relation"] == "co_occurs" for e in edges)
    assert all(e["weight"] == 1.0 for e in edges)


# ---------------------------------------------------------------------------
# 5. link_agent_to_symbol
# ---------------------------------------------------------------------------
def test_link_agent_to_symbol_emits_caused_by_edge():
    conn = _make_conn()
    kg = FailureKG(conn)
    kg.record_symbol_failure(
        "AAPL", "2026-08-25T13:30:00Z", exc=RuntimeError("data error"),
    )
    kg.record_agent_failure(
        "direction", "2026-08-25T13:30:00Z", "AAPL",
        exc=ValueError("ml output was None"),
    )
    kg.link_agent_to_symbol("direction", "AAPL", "2026-08-25T13:30:00Z")
    rows = conn.execute(
        "SELECT relation, weight FROM failure_edges "
        "WHERE relation = 'caused_by'",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["weight"] == 1.0


def test_link_agent_to_symbol_no_op_when_missing():
    """If the agent or symbol failure row is missing, the link is a
    silent no-op (the writer is best-effort)."""
    conn = _make_conn()
    kg = FailureKG(conn)
    kg.link_agent_to_symbol("direction", "AAPL", "2026-08-25T13:30:00Z")
    rows = conn.execute("SELECT * FROM failure_edges").fetchall()
    assert rows == []


# ---------------------------------------------------------------------------
# 6. Aggregations
# ---------------------------------------------------------------------------
def test_symbol_failure_counts_groups_by_symbol():
    conn = _make_conn()
    kg = FailureKG(conn)
    kg.record_symbol_failure("AAPL", "2026-08-25T13:30:00Z", exc=TimeoutError("t"))
    kg.record_symbol_failure("AAPL", "2026-08-25T13:35:00Z", exc=TimeoutError("t"))
    kg.record_symbol_failure("MSFT", "2026-08-25T13:30:00Z", exc=TimeoutError("t"))
    counts = kg.symbol_failure_counts("2026-08-25T13:30:00Z")
    assert counts == {"AAPL": 2, "MSFT": 1}


def test_symbol_failure_features_separates_llm_and_broker():
    """The feature dict must distinguish LLM failures from broker
    failures so the GNN can learn each channel separately.

    The split is driven by the *classified* kind (stored in
    ``metadata_json`` via ``classify_error``), NOT by the raw
    exception class name. So a ``RuntimeError("Alpaca 403 ...")``
    is classified as a ``broker`` failure and contributes to the
    broker channel, not the request channel.
    """
    conn = _make_conn()
    kg = FailureKG(conn)
    # TimeoutError -> classify_error -> kind="request" (warn, not in llm/broker)
    kg.record_symbol_failure("AAPL", "2026-08-25T13:30:00Z", exc=TimeoutError("t"))
    # RuntimeError("Alpaca 403 forbidden") -> classify_error ->
    # kind="broker" (heuristic matches "alpaca" in the message)
    kg.record_symbol_failure(
        "AAPL", "2026-08-25T13:35:00Z",
        exc=RuntimeError("Alpaca 403 forbidden"),
    )
    feats = kg.symbol_failure_features("2026-08-25T13:30:00Z")
    aapl = feats["AAPL"]
    assert aapl["failure_count_7d"] == 2.0
    assert aapl["llm_failure_count_7d"] == 0.0
    assert aapl["broker_failure_count_7d"] == 1.0
    # weighted score: 1 warn (0.3) + 1 error (0.7) = 1.0
    assert aapl["weighted_failure_score_7d"] == 1.0


# ---------------------------------------------------------------------------
# 7. Defensive: missing table
# ---------------------------------------------------------------------------
def test_failure_kg_handles_missing_schema():
    """A pre-70-failure-kg DB (no failure_nodes table) must not crash
    the writer; the calling code should fall back to logging."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    kg = FailureKG(conn)
    # All three record methods must raise a clean OperationalError
    # (not crash with a NameError or AttributeError) so the caller
    # can wrap them in try/except and continue.
    with pytest.raises(sqlite3.OperationalError):
        kg.record_symbol_failure("AAPL", "2026-08-25T13:30:00Z",
                                  exc=RuntimeError("x"))
    with pytest.raises(sqlite3.OperationalError):
        kg.record_agent_failure("direction", "2026-08-25T13:30:00Z", "AAPL",
                                 exc=ValueError("x"))
    with pytest.raises(sqlite3.OperationalError):
        kg.record_cycle_failure("decision-x", "2026-08-25T13:30:00Z",
                                 final_action="NO_TRADE", exc=RuntimeError("x"))
