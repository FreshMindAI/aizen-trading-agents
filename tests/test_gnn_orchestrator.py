"""Orchestrator integration tests (T026 / T027 / US3).

Two flows on the same orchestrator wiring:

1. **Real GNN** (T026) - a fresh trained artifact is present in
   ``gnn_model_artifacts``; the orchestrator cycle must populate
   ``decision_journal.gnn_output_json`` with ``model_version != "stub-1"``
   and the JSON validates against ``contracts/gnn_output.schema.json``.

2. **Stub GNN** (T027) - the artifact table is empty; the cycle must
   still complete and write a row with ``model_version == "stub-1"``.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import jsonschema
import pytest

from src.db import init_db


@pytest.fixture
def tmp_db(monkeypatch):
    """An isolated SQLite DB with every sql/*.sql applied."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row
    sql_dir = Path(__file__).resolve().parents[1] / "sql"
    for script in sorted(sql_dir.glob("*.sql")):
        conn.executescript(script.read_text(encoding="utf-8"))
    conn.commit()
    yield conn, Path(tmp.name)
    conn.close()
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def orch_factory(tmp_db, monkeypatch):
    """Build an Orchestrator that points at the tmp DB. Force mock LLM."""
    conn, db_path = tmp_db
    monkeypatch.setenv("RUN_MODE", "dry-run")
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "mock")
    # Configure data to come from the project DB so the agents have
    # actual underlyings to reason about, while the journal writes
    # land in the tmp DB via the orchestrator.
    from src.db import connect
    real_conn = connect()
    # Wipe + re-init decision_journal in the tmp DB so each test gets
    # a clean slate.
    conn.execute("DELETE FROM decision_journal")
    conn.commit()
    from src.agents.graph import Orchestrator
    orch = Orchestrator(conn=conn)
    # Inject a real project DB connection for the inference reads.
    orch.conn = conn
    orch.inference.conn = real_conn
    return orch, conn, real_conn


def _gnn_output_schema():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "specs" /
         "002-phase2-gnn" / "contracts" / "gnn_output.schema.json"
        ).read_text()
    )


def test_stub_path_when_no_artifact(orch_factory):
    """T027: with no GNN artifact, the journal row carries stub-1."""
    orch, tmp_conn, real_conn = orch_factory
    # Make sure no artifact row exists.
    real_conn.execute("DELETE FROM gnn_model_artifacts")
    real_conn.commit()
    state = orch.run_cycle()
    row = orch.journal.get(state.decision_id)
    assert row is not None
    gnn = row["gnn_output_json"]
    assert gnn["model_version"] == "stub-1"
    assert gnn["version"] == "1.0"
    jsonschema.validate(gnn, _gnn_output_schema())


def test_real_path_with_artifact(orch_factory, tmp_path):
    """T026: with a trained artifact, the journal row carries the
    real model_version (not stub-1) and validates against the schema."""
    orch, tmp_conn, real_conn = orch_factory
    # Make sure the gnn_* tables are present on the real DB.
    init_db(real_conn)
    real_conn.execute("DELETE FROM gnn_model_artifacts")
    real_conn.execute("DELETE FROM gnn_graph_snapshots")
    real_conn.execute("DELETE FROM gnn_graph_edges")
    real_conn.commit()

    # Train a tiny artifact.
    from src.gnn.build_snapshot import write_snapshot
    from src.gnn.train import train_model
    timestamps = [
        "2026-08-24T18:45:00Z",
        "2026-08-24T18:30:00Z",
        "2026-08-24T18:00:00Z",
        "2026-08-24T17:30:00Z",
        "2026-08-24T17:00:00Z",
        "2026-08-24T16:45:00Z",
        "2026-08-24T16:30:00Z",
        "2026-08-24T16:15:00Z",
        "2026-08-21T18:45:00Z",
    ]
    res = train_model(
        real_conn,
        timestamps=timestamps,
        epochs=2,
        out_dir=str(tmp_path),
        out_prefix="gnn-orch",
    )
    model_version = res["meta"]["model_version"]
    assert model_version != "stub-1"

    # Build a snapshot the inference service can use.
    write_snapshot(real_conn, "2026-08-24T18:45:00Z")
    # Force a fresh pick of the GNN service inside the inference
    # object now that an artifact is present.
    orch.inference._gnn_loaded = False
    orch.inference._gnn_service = None

    state = orch.run_cycle()
    row = orch.journal.get(state.decision_id)
    assert row is not None
    gnn = row["gnn_output_json"]
    # The journal row is the JSON blob stored verbatim.
    assert gnn["model_version"] == model_version
    assert gnn["model_version"] != "stub-1"
    # node_features present and non-empty.
    assert gnn["node_features"]
    jsonschema.validate(gnn, _gnn_output_schema())
