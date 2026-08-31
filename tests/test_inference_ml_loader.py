"""Regression tests for the ML model loader in ``InferenceService``.

Background
----------
The original ``_LATEST_UNDERLYING_SQL`` referenced columns
``direction_probability``, ``predicted_future_realized_vol``, and
``model_version`` on the ``ml_training_dataset`` table that never
existed. The bare ``except Exception`` at line 324 swallowed the
``OperationalError`` and silently stubbed every underlying to
``direction_probability=0.5``, ``model_version="stub"``. This made the
supervisor's ``direction_prob_min=0.55`` gate fire on every cycle
even when the trained XGBoost models had a real prediction.

The fix (Option A in ``docs/ml_inference_root_cause.md``) is to load
the trained XGBoost artifacts from disk at inference time, build the
v2 feature row from ``v_features_underlying_v2``, and call
``predict_proba`` / ``predict`` directly — mirroring how the GNN
service runs the .pt model.

These tests pin the new behavior so the regression cannot return:
  - ``_parse_artifact_stem`` correctly identifies the (task, horizon, kind)
    triple from the filename.
  - ``_latest_artifact`` returns the newest matching .pkl.
  - The inference service produces ``model_version`` values that
    match a real artifact stem (not "stub"), when the system has
    trained artifacts.
  - The as_of cut-off propagates to the underlying feature SQL.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.inference import (  # noqa: E402
    MODEL_DIR,
    InferenceService,
    _latest_artifact,
    _parse_artifact_stem,
)
from src.db import connect, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path):
    p = tmp_path / "test.db"
    c = connect(str(p))
    init_db(c, sql_dir=__import__("pathlib").Path("sql"))
    yield c
    c.close()


def _iso(days_ago: int, hour: int = 12) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).replace(
        hour=hour, minute=0, second=0, microsecond=0,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _insert_underlying(conn, ts: str, sym: str = "NVDA", close: float = 100.0):
    conn.execute(
        "INSERT INTO underlying_bars (symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sym, ts, close, close, close, close, 1000),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stem parser / artifact lookup
# ---------------------------------------------------------------------------
def test_parse_artifact_stem_direction_h4_clf():
    parsed = _parse_artifact_stem("direction_h4_xgb_clf-20260829-152556")
    assert parsed == ("direction", 4, "clf")


def test_parse_artifact_stem_rv_h4_reg():
    parsed = _parse_artifact_stem("rv_h4_xgb_reg-20260829-152611")
    assert parsed == ("rv", 4, "reg")


def test_parse_artifact_stem_option_h16_reg():
    parsed = _parse_artifact_stem("option_h16_xgb_reg-20260829-130039")
    assert parsed == ("option", 16, "reg")


def test_parse_artifact_stem_rejects_unknown_task():
    assert _parse_artifact_stem("garbage-20260829") is None
    assert _parse_artifact_stem("direction_h4_xgb_garbage-20260829") is None
    assert _parse_artifact_stem("just-a-filename.pkl") is None


def test_latest_artifact_returns_newest_match():
    """_latest_artifact picks the newest file matching (task, horizon, kind)."""
    p = _latest_artifact(MODEL_DIR, "direction", 4, "clf")
    if p is None:
        pytest.skip("no direction_h4_xgb_clf artifact in models/")
    # Filename must encode the (task, horizon, kind) triple.
    assert p.stem.startswith("direction_h4_xgb_clf-")
    # Must be a real .pkl that exists on disk.
    assert p.exists()
    # And it must be the newest one (sorted by filename: timestamp suffix).
    candidates = sorted(
        MODEL_DIR.glob("direction_h4_xgb_clf-*.pkl"),
    )
    assert p.name == candidates[-1].name


def test_latest_artifact_returns_none_when_missing():
    """If the model directory has no match, return None (do not raise)."""
    p = _latest_artifact(MODEL_DIR, "direction", 999, "clf")
    assert p is None


# ---------------------------------------------------------------------------
# End-to-end: the inference service uses real XGBoost predictions
# ---------------------------------------------------------------------------
def test_ml_inference_returns_real_model_version_not_stub(conn, monkeypatch):
    """When a trained direction_h4_xgb_clf artifact is on disk, the
    loader must use it and set ``model_version`` to the artifact's
    stem (not "stub"). This is the regression pin: the silent
    stub fallback that produced prob=0.5 + model_version="stub"
    must not return.
    """
    # Skip when no trained artifact is present (CI without the
    # 800k-row dataset cannot train the XGBoost models).
    direction_pkl = _latest_artifact(MODEL_DIR, "direction", 4, "clf")
    rv_pkl = _latest_artifact(MODEL_DIR, "rv", 4, "reg")
    if direction_pkl is None or rv_pkl is None:
        pytest.skip("trained direction / rv artifacts not present in models/")

    # Force the loader to use a fresh, isolated model dir so we can
    # inject a tiny temporary directory if needed; the default
    # model_dir already points at the project models/ folder which
    # has the trained artifacts, so the simple case is enough.
    _insert_underlying(conn, _iso(1), "NVDA", close=100.0)
    svc = InferenceService(
        conn=conn, universe=["NVDA"], as_of=None,
    )
    rows = svc._load_underlying_predictions()
    assert rows, "expected at least one UnderlyingScore"
    u = rows[0]
    # The regression pin: model_version is the real artifact stem,
    # not "stub" and not "n/a".
    assert u.model_version != "stub", (
        f"loader silently fell back to stub; model_version={u.model_version!r}"
    )
    assert u.model_version != "n/a", (
        f"loader returned n/a; model_version={u.model_version!r}"
    )
    # And the version must match a real artifact on disk.
    assert u.model_version == direction_pkl.stem, (
        f"expected model_version to be the trained direction artifact stem, "
        f"got {u.model_version!r} (artifact: {direction_pkl.stem!r})"
    )
    # direction_probability should be a real probability in [0, 1],
    # not the silent 0.5 stub.
    assert 0.0 <= u.direction_probability <= 1.0


def test_ml_inference_respects_as_of_cutoff(conn):
    """The as_of parameter must clamp the underlying feature row to
    timestamp <= as_of. With a future-dated bar present, the loader
    must NOT include it in the latest-row selection.
    """
    direction_pkl = _latest_artifact(MODEL_DIR, "direction", 4, "clf")
    if direction_pkl is None:
        pytest.skip("no trained direction artifact")

    # Insert a PAST bar and a FUTURE bar.
    _insert_underlying(conn, "2026-08-25T13:30:00Z", "NVDA", close=100.0)
    _insert_underlying(conn, "2026-08-29T13:30:00Z", "NVDA", close=999.0)
    svc = InferenceService(
        conn=conn, universe=["NVDA"], as_of="2026-08-26T13:30:00Z",
    )
    rows = svc._load_underlying_predictions()
    assert rows, "expected at least one UnderlyingScore"
    # The returned row's timestamp must be at-or-before the cut-off,
    # never the 2026-08-29 row.
    assert rows[0].timestamp <= "2026-08-26T13:30:00Z", (
        f"future row leaked: timestamp={rows[0].timestamp}"
    )
    assert not rows[0].timestamp.startswith("2026-08-29")


def test_ml_inference_handles_missing_artifact_gracefully(conn, tmp_path):
    """When the model_dir is empty, the loader falls back to the stub
    shape and logs a WARNING. The orchestrator must not crash.
    """
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    _insert_underlying(conn, _iso(1), "NVDA", close=100.0)
    svc = InferenceService(
        conn=conn, universe=["NVDA"], as_of=None, model_dir=empty_dir,
    )
    rows = svc._load_underlying_predictions()
    assert rows, "expected at least one UnderlyingScore even with no model"
    # With no artifact, the loader logs a warning and returns the
    # deterministic stub: prob=0.5, model_version="n/a".
    u = rows[0]
    assert u.model_version == "n/a"
    assert u.direction_probability == pytest.approx(0.5, abs=1e-9)
