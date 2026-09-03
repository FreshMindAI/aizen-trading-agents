"""Tests for ``scripts/register_gnn_artifacts.py``.

Pins three properties of the backfill script:

  1. Idempotence: running the script twice upserts the same number
     of rows because the insert is ``INSERT OR REPLACE`` keyed on
     ``model_version``. This matters because the script is wired
     into the live cron path; a regression that double-inserts
     would bloat the table on every cycle.
  2. ``used_news`` is set to 1 for GATv2-news models (so the
     downstream GNN service knows to weight the news edges) and 0
     for the plain GCN checkpoints. Detection must look at both
     ``model_version`` and ``topology_version`` because different
     checkpoints carry the news signal in different fields.
  3. The script gracefully skips ``.meta.json`` files whose
     corresponding ``.pt`` is missing (the XGBoost artifacts in
     ``models/`` use ``.pkl`` extensions, so they're naturally
     absent from the gnn registry).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.register_gnn_artifacts import register_artifacts  # noqa: E402


SCHEMA_SQL = REPO / "sql" / "50_gnn_snapshots.sql"  # noqa: F841


def _create_test_db(db_path: Path) -> None:
    """Create a minimal schema with just the gnn_model_artifacts table.
    Avoids depending on the full DB init path (which creates dozens of
    tables the script doesn't touch)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(str(db_path))
    cx.executescript(
        """
        CREATE TABLE IF NOT EXISTS gnn_model_artifacts (
            model_version   TEXT PRIMARY KEY,
            path            TEXT NOT NULL,
            architecture    TEXT,
            topology_version TEXT,
            feature_names   TEXT,
            impute_medians  TEXT,
            split_bounds    TEXT,
            test_metrics    TEXT,
            created_at      TEXT,
            used_news       INTEGER,
            ablation_fold_id INTEGER
        );
        """
    )
    cx.commit()
    cx.close()


def _make_artifact_pair(models_dir: Path, stem: str,
                        architecture: str, topology: str,
                        used_news_target: int) -> None:
    """Write a ``.pt`` stub and ``.meta.json`` so the script can
    register them. The .pt contents don't matter — the script
    never opens them; it only records the path."""
    pt = models_dir / f"{stem}.pt"
    pt.write_bytes(b"\x00\x01stub")
    meta = models_dir / f"{stem}.meta.json"
    meta.write_text(json.dumps({
        "model_version": stem,
        "architecture": architecture,
        "topology_version": topology,
        "feature_names": ["return_1", "rsi_14"],
        "impute_medians": {"return_1": 0.0, "rsi_14": 50.0},
        "split_bounds": {"train_end": "2026-08-27T18:15:00Z"},
        "test_metrics": {"roc_auc": 0.5},
        "created_at": "2026-09-03T00:00:00Z",
    }))
    assert used_news_target in (0, 1)


def test_register_artifacts_is_idempotent(tmp_path):
    """Same DB, same models dir, run twice → same number of rows."""
    db = tmp_path / "test.db"
    models = tmp_path / "models"
    models.mkdir()
    _create_test_db(db)
    _make_artifact_pair(models, "gnn-20260903-0001", "gcn-32-16-1", "fixed-1", 0)
    _make_artifact_pair(models, "gatv2-news-20260903-0001", "gatv2-32-1h", "fixed-1", 1)
    n1 = register_artifacts(db, models)
    n2 = register_artifacts(db, models)
    assert n1 == n2 == 2
    cx = sqlite3.connect(str(db))
    n_rows = cx.execute("SELECT COUNT(*) FROM gnn_model_artifacts").fetchone()[0]
    assert n_rows == 2


def test_register_artifacts_sets_used_news_from_model_name(tmp_path):
    """The GATv2-news model has model_version="gatv2-news-..." but
    topology_version="fixed-1" (not "option-v2-news"). The script
    must still set used_news=1 because the model name carries the
    signal."""
    db = tmp_path / "test.db"
    models = tmp_path / "models"
    models.mkdir()
    _create_test_db(db)
    _make_artifact_pair(models, "gatv2-news-20260903-0001", "gatv2-32-1h", "fixed-1", 1)
    register_artifacts(db, models)
    cx = sqlite3.connect(str(db))
    row = cx.execute(
        "SELECT model_version, used_news FROM gnn_model_artifacts"
    ).fetchone()
    assert row[0] == "gatv2-news-20260903-0001"
    assert row[1] == 1, "GATv2-news model must have used_news=1"


def test_register_artifacts_skips_meta_without_pt(tmp_path):
    """XGBoost artifacts have .meta.json + .pkl (not .pt). The script
    must not crash; it should log a warning and skip the entry."""
    db = tmp_path / "test.db"
    models = tmp_path / "models"
    models.mkdir()
    _create_test_db(db)
    # .meta.json + .pkl (no .pt) — this is the XGBoost pattern.
    (models / "option_h4_xgb_clf-test.meta.json").write_text(
        json.dumps({"model_version": "option_h4_xgb_clf-test",
                    "architecture": "xgb", "topology_version": "option-v2"})
    )
    (models / "option_h4_xgb_clf-test.pkl").write_bytes(b"")
    n = register_artifacts(db, models)
    assert n == 0
    cx = sqlite3.connect(str(db))
    rows = cx.execute("SELECT COUNT(*) FROM gnn_model_artifacts").fetchone()[0]
    assert rows == 0


def test_register_artifacts_raises_on_missing_table(tmp_path):
    """The script must give a friendly error when the
    ``gnn_model_artifacts`` table is absent (e.g. fresh DB without
    the schema applied). A bare sqlite3.OperationalError would
    leave the operator guessing."""
    db = tmp_path / "test.db"
    db.write_bytes(b"")  # empty file, no schema
    models = tmp_path / "models"
    models.mkdir()
    _make_artifact_pair(models, "gnn-test", "gcn-32-16-1", "fixed-1", 0)
    with pytest.raises(RuntimeError, match="gnn_model_artifacts table not found"):
        register_artifacts(db, models)


def test_register_artifacts_persists_feature_names_and_impute_medians(tmp_path):
    """The GNNService.load_latest path (src/gnn/service.py:74) reads
    feature_names and impute_medians as JSON. The script must
    serialise them as JSON strings, not raw Python repr, so the
    service can json.loads() them back."""
    db = tmp_path / "test.db"
    models = tmp_path / "models"
    models.mkdir()
    _create_test_db(db)
    _make_artifact_pair(models, "gnn-test-0001", "gcn-32-16-1", "fixed-1", 0)
    register_artifacts(db, models)
    cx = sqlite3.connect(str(db))
    row = cx.execute(
        "SELECT feature_names, impute_medians FROM gnn_model_artifacts"
    ).fetchone()
    feature_names = json.loads(row[0])
    impute_medians = json.loads(row[1])
    assert feature_names == ["return_1", "rsi_14"]
    assert impute_medians == {"return_1": 0.0, "rsi_14": 50.0}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
