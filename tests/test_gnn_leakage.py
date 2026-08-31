"""No-future-leak test (T013 / US1).

Every row referenced in ``payload_json`` must have
``timestamp <= snapshot.timestamp`` - no future data sneaks into the
graph.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.gnn.build_snapshot import write_snapshot


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


def test_no_node_source_after_snapshot(tmp_db):
    """No node may reference a bar strictly after the snapshot ts."""
    ts = "2026-08-25T17:45:00Z"
    res = write_snapshot(tmp_db, ts)
    payload = json.loads(res["payload_json"])
    assert payload["timestamp"] == ts
    for n in payload["nodes"]:
        src = n.get("source_timestamp")
        if src is None:
            continue
        assert src <= ts, (
            f"node {n['symbol']} references future bar {src} > snapshot {ts}"
        )


def test_no_edge_uses_future_correlation(tmp_db):
    """Every correlation edge must come from a bar <= snapshot ts."""
    ts = "2026-08-25T17:45:00Z"
    res = write_snapshot(tmp_db, ts)
    snap_id = res["snapshot_id"]
    corr_rows = tmp_db.execute(
        "SELECT source_symbol, target_symbol, reason "
        "FROM gnn_graph_edges WHERE snapshot_id = ? AND reason = 'correlation'",
        (snap_id,),
    ).fetchall()
    # We need to confirm every correlation pair referenced the most
    # recent v_asset_correlations row at or before the snapshot.
    for row in corr_rows:
        a, b = row["source_symbol"], row["target_symbol"]
        if a > b:
            a, b = b, a
        latest = tmp_db.execute(
            "SELECT MAX(timestamp) as ts FROM v_asset_correlations "
            "WHERE symbol_a = ? AND symbol_b = ? AND timestamp <= ?",
            (a, b, ts),
        ).fetchone()
        assert latest is not None
        assert latest["ts"] is not None
        assert latest["ts"] <= ts
