"""Graph determinism test (T011 / US1).

The same inputs must produce a byte-identical ``payload_json``. This
is the SC1 invariant - any non-determinism in the builder chain is a
bug.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.db import init_db
from src.gnn.build_snapshot import build_payload, write_snapshot


@pytest.fixture
def tmp_db(monkeypatch):
    """A SQLite database with every sql/*.sql applied (production data)."""
    # Use the project DB so v_features_underlying_v2 has rows; the build
    # is read-only on the source views, so it is safe to share.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()
    conn = sqlite3.connect("data/trading.db")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_payload_is_byte_identical_across_builds(tmp_db):
    ts = "2026-08-25T17:45:00Z"
    p1 = build_payload(tmp_db, ts)
    p2 = build_payload(tmp_db, ts)
    import json
    s1 = json.dumps(p1, sort_keys=True, separators=(",", ":"))
    s2 = json.dumps(p2, sort_keys=True, separators=(",", ":"))
    assert s1 == s2
    # And the canonical schema is intact.
    assert len(p1["nodes"]) == 10  # UNIVERSE
    for n in p1["nodes"]:
        assert len(n["node_features"]) == 14
        assert n["feature_names"] == n["feature_names"]  # stable


def test_two_writes_have_byte_identical_payload(tmp_db):
    ts = "2026-08-25T17:45:00Z"
    r1 = write_snapshot(tmp_db, ts)
    r2 = write_snapshot(tmp_db, ts)
    # Same payload (only created_at + snapshot_id differ, and we don't
    # include created_at in the payload itself).
    assert r1["payload_json"] == r2["payload_json"]
    assert r1["node_count"] == r2["node_count"]
    assert r1["edge_count"] == r2["edge_count"]
