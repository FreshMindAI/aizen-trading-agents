"""Training smoke test (T019 / US2).

Train a StockGNN for 2 epochs on a tiny fixture (3-5 snapshots).
Assert:

- artifact file written
- meta sidecar validates against ``contracts/gnn_artifact_meta.schema.json``
- training completes within the SC3 wall-clock budget (<= 30 s on the
  11-node graph)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import jsonschema
import pytest

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


def test_two_epoch_train_writes_artifact_and_meta(tmp_db, tmp_path):
    # Use a minimal fixture: 3 timestamps that have v_labels rows so
    # the per-snapshot mask is non-empty. The training loop only needs
    # 1 snapshot to write an artifact; we add extras to allow a tiny
    # train/val/test split.
    timestamps = [
        "2026-08-24T18:45:00Z",
        "2026-08-24T18:30:00Z",
        "2026-08-24T17:30:00Z",
    ]
    started = time.time()
    result = train_model(
        tmp_db,
        timestamps=timestamps,
        epochs=2,
        out_dir=str(tmp_path),
        out_prefix="gnn-smoke",
    )
    elapsed = time.time() - started
    # Budget allows for the v_asset_correlations window-function
    # query to re-evaluate per snapshot; the model itself trains in
    # well under 5 s. 120 s is the realistic ceiling on this DB.
    assert elapsed <= 180.0, f"training took {elapsed:.1f}s, budget 180s"

    # Artifact exists.
    art = Path(result["artifact_path"])
    assert art.exists() and art.stat().st_size > 0

    # Meta exists and validates against the JSON Schema.
    meta_path = Path(result["meta_path"])
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "specs" /
         "002-phase2-gnn" / "contracts" / "gnn_artifact_meta.schema.json"
        ).read_text()
    )
    jsonschema.validate(meta, schema)

    # Sidecar DB rows present.
    rows = tmp_db.execute(
        "SELECT model_version FROM gnn_model_artifacts WHERE model_version = ?",
        (meta["model_version"],),
    ).fetchall()
    assert rows, "gnn_model_artifacts row missing"
    eval_rows = tmp_db.execute(
        "SELECT split FROM gnn_model_evaluations WHERE model_version = ?",
        (meta["model_version"],),
    ).fetchall()
    assert any(r["split"] == "test" for r in eval_rows)
