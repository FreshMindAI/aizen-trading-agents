"""Artifact round-trip test (T020 / US2).

Load a freshly-trained artifact, run inference on a fixture, and
assert the per-symbol outputs are in the contract's allowed ranges
and the ``model_version`` is not the stub sentinel.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import jsonschema
import pytest

from src.gnn.build_snapshot import build_payload, write_snapshot
from src.gnn.service import GNNService
from src.gnn.train import train_model


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


def test_artifact_round_trip_in_range(tmp_db, tmp_path):
    # Train a tiny model.
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
        tmp_db,
        timestamps=timestamps,
        epochs=2,
        out_dir=str(tmp_path),
        out_prefix="gnn-rt",
    )
    # Build a fresh snapshot to predict against.
    write_snapshot(tmp_db, "2026-08-24T18:45:00Z")
    snap_id = tmp_db.execute(
        "SELECT snapshot_id FROM gnn_graph_snapshots "
        "WHERE timestamp='2026-08-24T18:45:00Z' LIMIT 1"
    ).fetchone()["snapshot_id"]

    svc = GNNService.load_latest(tmp_db)
    assert svc is not None
    out = svc.predict(snap_id)
    # Not the stub sentinel.
    assert out.model_version != "stub-1"
    assert out.version == "1.0"
    # Per-symbol range checks.
    for sym, nf in out.node_features.items():
        assert -1.0 <= nf.bias <= 1.0, f"{sym} bias {nf.bias} OOR"
        assert 0.0 <= nf.centrality <= 1.0, f"{sym} centrality {nf.centrality} OOR"
        assert nf.model_version == svc.model_version
    # Output validates against the GNNOutput JSON Schema.
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "specs" /
         "002-phase2-gnn" / "contracts" / "gnn_output.schema.json"
        ).read_text()
    )
    jsonschema.validate(out.model_dump(mode="json"), schema)
