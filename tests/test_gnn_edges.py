"""Edge-reason coverage test (T012 / US1).

All six ``reason`` enum values must appear in a populated snapshot
when the dynamic-1 topology is used (the 6th, ``rolling_corr``, is
added by the dynamic-topology layer on top of the static 5).
The fixed-1 topology covers the first 5 only.

The CHECK constraint on ``gnn_graph_edges.reason`` must reject any
other value.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.db import init_db
from src.gnn.build_snapshot import write_snapshot
from src.gnn.constants import EDGE_REASONS


@pytest.fixture
def tmp_db(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()
    conn = sqlite3.connect("data/trading.db")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def _reasons_at(conn, ts: str) -> set[str]:
    return {
        r["reason"]
        for r in conn.execute(
            "SELECT DISTINCT reason FROM gnn_graph_edges "
            "WHERE snapshot_id IN (SELECT snapshot_id FROM gnn_graph_snapshots "
            "WHERE timestamp=?)", (ts,),
        ).fetchall()
    }


def test_fixed_1_has_five_edge_reasons(tmp_db):
    """The fixed-1 topology emits the 5 static reasons (sector,
    supplier, customer, etf_membership, correlation). The dynamic
    ``rolling_corr`` reason is NOT in a fixed-1 snapshot."""
    ts = "2026-08-25T17:45:00Z"
    write_snapshot(tmp_db, ts, topology_version="fixed-1")
    reasons = _reasons_at(tmp_db, ts)
    assert "rolling_corr" not in reasons
    # The 5 static reasons are all present (the synthetic DB has
    # populated enough data to emit each one).
    static = {r for r in EDGE_REASONS if r != "rolling_corr"}
    missing = static - reasons
    assert not missing, f"missing static reasons in fixed-1 snapshot: {missing}"


def test_dynamic_1_has_six_edge_reasons(tmp_db):
    """The dynamic-1 topology includes the 5 static reasons PLUS
    ``rolling_corr`` (the new dynamic channel)."""
    ts = "2026-08-25T17:45:00Z"
    write_snapshot(tmp_db, ts, topology_version="dynamic-1")
    reasons = _reasons_at(tmp_db, ts)
    missing = set(EDGE_REASONS) - reasons
    # ``rolling_corr`` requires at least one pair of co-moving symbols
    # in underlying_bars; the project DB has this for the standard
    # universe, so the channel should be present.
    assert not missing, f"missing reasons in dynamic-1 snapshot: {missing}"


def test_check_constraint_rejects_bad_reason(tmp_db):
    """The SQLite CHECK on gnn_graph_edges.reason must reject garbage."""
    ts = "2026-08-25T17:45:00Z"
    res = write_snapshot(tmp_db, ts)
    snap_id = res["snapshot_id"]
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.execute(
            "INSERT INTO gnn_graph_edges "
            "(snapshot_id, source_symbol, target_symbol, reason, weight, "
            " topology_version) VALUES (?, ?, ?, ?, ?, ?)",
            (snap_id, "AAPL", "MSFT", "not_a_reason", 1.0, "fixed-1"),
        )
        tmp_db.commit()
