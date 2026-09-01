"""Tests for the hourly self-analysis reflection module.

Covers:
  * build_markdown_report: produces deterministic output and surfaces the
    per-step success rate + NO_TRADE cluster analysis.
  * collect_topology_summary: returns the latest news-driven snapshot
    stats, gracefully handles the "no snapshot yet" case.
  * collect_cycle_traces: reads per-cycle JSONL files from a tmp dir and
    skips lines that fail to parse.
  * collect_decision_summary: aggregates the decision_journal rows
    correctly.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.cli.hourly_reflection import (  # noqa: E402
    PIPELINE_STEPS,
    _failure_reasons_histogram,
    _no_trade_cluster,
    _per_step_success_rate,
    build_json_summary,
    build_markdown_report,
    collect_cycle_traces,
    collect_decision_summary,
    collect_failure_summary,
    collect_topology_summary,
    main,
)
from src.db import connect, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_trace_record(
    cycle_id: str, step: str, success: bool = True,
    reasons: list[str] | None = None, symbol: str = "",
) -> dict:
    return {
        "cycle_id": cycle_id,
        "step": step,
        "symbol": symbol,
        "ts": "2026-08-30T13:30:00Z",
        "success": success,
        "reasons": reasons or [],
        "fields": {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")


# ---------------------------------------------------------------------------
# Cycle-trace collection
# ---------------------------------------------------------------------------
def test_collect_cycle_traces_reads_recent_files(tmp_path: Path):
    now = datetime.now(timezone.utc)
    since_iso = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    trace_dir = tmp_path / "models" / "cycle_traces"
    trace_dir.mkdir(parents=True)

    rec1 = _make_trace_record("c1", "ml", success=True)
    rec2 = _make_trace_record("c1", "final", success=True)
    _write_jsonl(trace_dir / "cycle_c1.jsonl", [rec1, rec2])

    # An old file (mtime well before the since window) — mtime is set
    # on write, so we back-date via os.utime.
    import os
    rec3 = _make_trace_record("c0", "ml", success=True)
    old = trace_dir / "cycle_c0.jsonl"
    _write_jsonl(old, [rec3])
    old_epoch = (now - timedelta(days=30)).timestamp()
    os.utime(old, (old_epoch, old_epoch))

    out = collect_cycle_traces(since_iso, trace_dir)
    ids = sorted({r["cycle_id"] for r in out})
    assert "c1" in ids
    assert "c0" not in ids
    assert len(out) == 2


def test_collect_cycle_traces_skips_bad_jsonl(tmp_path: Path):
    trace_dir = tmp_path / "models" / "cycle_traces"
    trace_dir.mkdir(parents=True)
    p = trace_dir / "cycle_c.jsonl"
    p.write_text("not-json\n", encoding="utf-8")
    p.write_text(json.dumps(_make_trace_record("c", "ml")) + "\n", encoding="utf-8")

    since_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out = collect_cycle_traces(since_iso, trace_dir)
    assert len(out) == 1
    assert out[0]["step"] == "ml"


# ---------------------------------------------------------------------------
# Per-step success rate
# ---------------------------------------------------------------------------
def test_per_step_success_rate_groups_correctly():
    records = [
        _make_trace_record("c1", "ml", success=True),
        _make_trace_record("c1", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c2", "ml", success=True),
        _make_trace_record("c2", "gnn", success=True),
    ]
    out = _per_step_success_rate(records)
    assert out["ml"] == {"n": 3, "n_success": 2, "n_failure": 1}
    assert out["gnn"] == {"n": 1, "n_success": 1, "n_failure": 0}


def test_failure_reasons_histogram_returns_top5():
    records = [
        _make_trace_record("c1", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c2", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c3", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c4", "ml", success=False, reasons=["timeout"]),
        _make_trace_record("c5", "ml", success=True),
    ]
    hist = _failure_reasons_histogram(records)
    assert hist[0] == ("threshold", 3)
    assert ("timeout", 1) in hist


# ---------------------------------------------------------------------------
# NO_TRADE cluster
# ---------------------------------------------------------------------------
def test_no_trade_cluster_buckets_by_first_failing_step():
    # 4 cycles where ml fails first (no symbol crossed threshold),
    # 1 cycle where risk fails first (passed every agent but risk
    # rejected). NO_TRADE = "final" step did not succeed.
    c1 = [
        _make_trace_record("c1", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c1", "gnn", success=True),
        _make_trace_record("c1", "final", success=False),
    ]
    c2 = [
        _make_trace_record("c2", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c2", "final", success=False),
    ]
    c3 = [
        _make_trace_record("c3", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c3", "final", success=False),
    ]
    c4 = [
        _make_trace_record("c4", "ml", success=False, reasons=["threshold"]),
        _make_trace_record("c4", "final", success=False),
    ]
    c5 = [
        _make_trace_record("c5", "ml", success=True),
        _make_trace_record("c5", "gnn", success=True),
        _make_trace_record("c5", "supervisor", success=True),
        _make_trace_record("c5", "risk", success=False, reasons=["max loss"]),
        _make_trace_record("c5", "final", success=False),
    ]
    cluster = _no_trade_cluster(c1 + c2 + c3 + c4 + c5)
    cluster_dict = dict(cluster)
    assert cluster_dict["ml"] == 4
    assert cluster_dict["risk"] == 1


# ---------------------------------------------------------------------------
# Topology + decision summary
# ---------------------------------------------------------------------------
def test_collect_topology_summary_returns_latest(tmp_path: Path):
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    # The create-if-not-exists is part of ensure_schema in option_graph,
    # but init_db() will not have run that yet. Create the table here
    # so the collector can read it.
    conn.executescript(
        """
        CREATE TABLE gnn_option_graph_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            topology_version TEXT NOT NULL,
            node_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE gnn_option_graph_edges (
            snapshot_id TEXT NOT NULL,
            source_symbol TEXT NOT NULL,
            target_symbol TEXT NOT NULL,
            reason TEXT NOT NULL,
            weight REAL NOT NULL,
            edge_features_json TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, source_symbol, target_symbol, reason)
        );
        """
    )
    conn.execute(
        "INSERT INTO gnn_option_graph_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("snap1", "2026-08-30T13:30:00Z", "option-v2-news",
         10, 20, "{}", "2026-08-30T13:30:00Z"),
    )
    conn.execute(
        "INSERT INTO gnn_option_graph_edges VALUES (?, ?, ?, ?, ?, ?)",
        ("snap1", "AAPL", "MSFT", "news_cooccurrence", 0.7, "[]"),
    )
    conn.execute(
        "INSERT INTO gnn_option_graph_edges VALUES (?, ?, ?, ?, ?, ?)",
        ("snap1", "AAPL", "NVDA", "news_sentiment_correlation", 0.9, "[]"),
    )
    conn.commit()
    out = collect_topology_summary(conn)
    assert out["snapshot_id"] == "snap1"
    assert out["node_count"] == 10
    assert out["edge_count"] == 20
    assert out["edge_kinds"]["news_cooccurrence"] == 1
    assert out["edge_kinds"]["news_sentiment_correlation"] == 1


def test_collect_topology_summary_handles_missing_table(tmp_path: Path):
    """When the topology table is absent (fresh DB), the collector
    should return an empty dict instead of raising."""
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    out = collect_topology_summary(conn)
    assert out == {}


def test_collect_decision_summary_aggregates_actions(tmp_path: Path):
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    # decision_journal is created by sql/40_decision_journal.sql via init_db.
    rows = [
        ("d1", "2026-08-30T13:30:00Z", "NVDA", "PROCEED", "[]",
         '{"decision":"APPROVE","checks":[{"name":"max_loss","passed":true}]}'),
        ("d2", "2026-08-30T13:35:00Z", "NVDA", "PROCEED", "[]",
         '{"decision":"APPROVE","checks":[{"name":"max_loss","passed":true}]}'),
        ("d3", "2026-08-30T13:40:00Z", "AAPL", "NO_TRADE", "[]", "[]"),
        ("d4", "2026-08-30T13:45:00Z", "AAPL", "NO_TRADE", "[]",
         '{"decision":"REJECT","checks":[{"name":"position_size","passed":false}]}'),
    ]
    # Build full row tuples matching the decision_journal NOT-NULL contract.
    not_null_blanks = {
        "market_state_hash": "h",
        "schema_version": "1.0",
        "run_mode": "paper",
        "market_snapshot_json": "{}",
        "ml_prediction_json": "[]",
        "gnn_output_json": "{}",
        "agent_messages_json": "[]",
        "agent_observations_json": "[]",
        "strategy_proposal_json": "null",
        "selected_strategy_json": "null",
        "execution_result_json": "{}",
        "model_versions": "[]",
        "created_at": "2026-08-30T13:30:00Z",
    }
    full_rows = [
        (
            r[0], r[1], "2026-08-30T13:30:00Z",  # decision_id, timestamp, completed_at
            not_null_blanks["market_state_hash"],
            not_null_blanks["schema_version"], not_null_blanks["run_mode"],
            r[2], r[3],  # underlying_focus, final_action
            None, None,  # outcome_label, realized_pnl
            not_null_blanks["market_snapshot_json"],
            not_null_blanks["ml_prediction_json"],
            not_null_blanks["gnn_output_json"],
            None,  # topology_version
            not_null_blanks["agent_messages_json"],
            not_null_blanks["agent_observations_json"],
            not_null_blanks["strategy_proposal_json"],
            not_null_blanks["selected_strategy_json"],
            r[5],  # risk_decision_json
            r[4],  # order_intent_json
            not_null_blanks["execution_result_json"],
            not_null_blanks["model_versions"],
            not_null_blanks["created_at"],
        )
        for r in rows
    ]
    conn.executemany(
        "INSERT INTO decision_journal (decision_id, timestamp, completed_at, "
        "market_state_hash, schema_version, run_mode, underlying_focus, final_action, "
        "outcome_label, realized_pnl, market_snapshot_json, ml_prediction_json, "
        "gnn_output_json, topology_version, agent_messages_json, "
        "agent_observations_json, strategy_proposal_json, selected_strategy_json, "
        "risk_decision_json, order_intent_json, execution_result_json, "
        "model_versions, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        full_rows,
    )
    conn.commit()
    out = collect_decision_summary(conn, "2026-08-30T13:00:00Z")
    assert out["total"] == 4
    assert out["by_action"]["PROCEED"] == 2
    assert out["by_action"]["NO_TRADE"] == 2
    # Both AAPL and NVDA have count 2; order between them is not pinned.
    assert dict(out["top_underlyings"]) == {"NVDA": 2, "AAPL": 2}
    assert out["top_risk_reject_reasons"][0] == ("position_size", 1)


# ---------------------------------------------------------------------------
# build_markdown_report
# ---------------------------------------------------------------------------
def test_build_markdown_report_is_deterministic():
    traces = [
        _make_trace_record("c1", "ml", success=True),
        _make_trace_record("c1", "gnn", success=True),
        _make_trace_record("c1", "final", success=False, reasons=["no PROCEED"]),
    ]
    md1 = build_markdown_report(
        traces, {}, {"total": 1, "by_action": {"NO_TRADE": 1},
                     "top_underlyings": [], "top_risk_reject_reasons": []},
        {"total": 0, "by_kind": {}, "by_severity": {}},
        hour="2026-08-30-13", since="2026-08-30T12:00:00Z",
    )
    md2 = build_markdown_report(
        traces, {}, {"total": 1, "by_action": {"NO_TRADE": 1},
                     "top_underlyings": [], "top_risk_reject_reasons": []},
        {"total": 0, "by_kind": {}, "by_severity": {}},
        hour="2026-08-30-13", since="2026-08-30T12:00:00Z",
    )
    assert md1 == md2
    # The markdown should mention the step table and the cluster analysis.
    assert "## Cycle traces (per-step success)" in md1
    assert "## NO_TRADE cluster analysis" in md1
    assert "## Decision journal" in md1
    assert "## Topology" in md1
    assert "## Failure knowledge graph" in md1


def test_build_markdown_report_handles_empty_window():
    md = build_markdown_report(
        [], {}, {"total": 0, "by_action": {}, "top_underlyings": [],
                 "top_risk_reject_reasons": []},
        {"total": 0, "by_kind": {}, "by_severity": {}},
        hour="2026-08-30-13", since="2026-08-30T12:00:00Z",
    )
    # No crash on empty input; the report should still be parseable.
    assert "Hourly Reflection" in md
    assert "Cycles in window: **0**" in md
    assert "_No cycle traces in this window._" in md


def test_build_json_summary_has_schema_version():
    out = build_json_summary(
        [_make_trace_record("c1", "ml", success=True)],
        {"snapshot_id": "s1", "node_count": 10, "edge_count": 20,
         "edge_kinds": {"news_cooccurrence": 1}},
        {"total": 1, "by_action": {"PROCEED": 1}, "top_underlyings": [("NVDA", 1)],
         "top_risk_reject_reasons": []},
        {"total": 0, "by_kind": {}, "by_severity": {}},
        hour="2026-08-30-13", since="2026-08-30T12:00:00Z",
    )
    assert out["schema_version"] == "1.0"
    assert out["hour"] == "2026-08-30-13"
    assert out["per_step"]["ml"]["n"] == 1
    assert out["decisions"]["by_action"]["PROCEED"] == 1


# ---------------------------------------------------------------------------
# main() smoke test
# ---------------------------------------------------------------------------
def test_main_writes_markdown_and_json(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "trading.db"
    out_dir = tmp_path / "reflections"
    trace_dir = tmp_path / "models" / "cycle_traces"
    trace_dir.mkdir(parents=True)
    _write_jsonl(
        trace_dir / "cycle_c.jsonl",
        [_make_trace_record("c", "ml", success=True)],
    )

    monkeypatch.setenv("AIZEN_DB_PATH", str(db_path))
    rc = main([
        "--out", str(out_dir),
        "--db-path", str(db_path),
    ])
    assert rc == 0
    files = sorted(out_dir.glob("*"))
    assert any(f.suffix == ".md" for f in files)
    assert any(f.suffix == ".json" for f in files)


def test_failure_summary_counts_by_kind(tmp_path: Path):
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    # failure_nodes is created by sql/70_failure_kg.sql via init_db.
    conn.execute(
        "INSERT INTO failure_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("n1", "symbol_failure", "AAPL", None, "d1", "2026-08-30T13:30:00Z",
         "2026-08-30", "error", "TimeoutError", "timeout", 3, '{"kind":"request"}',
         "2026-08-30T13:30:00Z"),
    )
    conn.execute(
        "INSERT INTO failure_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("n2", "agent_failure", None, "direction", "d2", "2026-08-30T13:30:00Z",
         "2026-08-30", "error", "ValueError", "bad value", 1, '{"kind":"agent"}',
         "2026-08-30T13:30:00Z"),
    )
    conn.commit()
    out = collect_failure_summary(conn, "2026-08-30T13:00:00Z")
    assert out["total"] == 4
    assert out["by_kind"]["symbol_failure"] == 3
    assert out["by_kind"]["agent_failure"] == 1
