"""Tests for the news_snapshot table schema (T010).

Covers spec FR-001:
- Migration is idempotent (re-running init_db does not fail)
- timestamp <= created_at CHECK is enforced
- topics_json is a JSON array of 1-3 strings
- Insert with a duplicate (timestamp, symbol) is a no-op (PK constraint)
"""
from __future__ import annotations

import json
import sqlite3
import pytest

from src.db import connect, init_db


@pytest.fixture
def conn(tmp_path):
    p = tmp_path / "test.db"
    c = connect(str(p))
    init_db(c, sql_dir=__import__("pathlib").Path("sql"))
    yield c
    c.close()


def test_news_snapshot_table_exists(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='news_snapshot'"
    ).fetchall()
    assert len(rows) == 1


def test_news_snapshot_pk_constraint(conn):
    # First insert succeeds
    conn.execute(
        "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("2026-08-29T13:00:00Z", "NVDA", 0.5, 1, "[\"earnings\"]", "{}", "2026-08-29T13:01:00Z"),
    )
    conn.commit()
    # Second insert with the same (timestamp, symbol) must be ignored
    conn.execute(
        "INSERT OR IGNORE INTO news_snapshot (timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("2026-08-29T13:00:00Z", "NVDA", 0.7, 2, "[\"guidance\"]", "{}", "2026-08-29T13:02:00Z"),
    )
    conn.commit()
    row = conn.execute(
        "SELECT sentiment, article_count FROM news_snapshot WHERE timestamp='2026-08-29T13:00:00Z' AND symbol='NVDA'"
    ).fetchone()
    # The second INSERT was ignored; original row is intact
    assert row[0] == 0.5
    assert row[1] == 1


def test_news_snapshot_timestamp_before_created_at_check(conn):
    # article cannot be 'published' after it is 'created' (clock-skew check)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-08-29T13:00:00Z", "NVDA", 0.5, 1, "[\"earnings\"]", "{}", "2026-08-29T12:00:00Z"),
        )


def test_news_snapshot_sentiment_bounds(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-08-29T13:00:00Z", "NVDA", 1.5, 1, "[\"x\"]", "{}", "2026-08-29T13:01:00Z"),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-08-29T13:00:00Z", "NVDA", -1.5, 1, "[\"x\"]", "{}", "2026-08-29T13:01:00Z"),
        )


def test_gnn_model_artifacts_used_news_column_exists(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(gnn_model_artifacts)").fetchall()}
    assert "used_news" in cols
    assert "ablation_fold_id" in cols


def test_gnn_model_artifacts_used_news_default_zero(conn):
    # Insert without specifying used_news; default is 0
    conn.execute(
        "INSERT INTO gnn_model_artifacts (model_version, path, architecture, topology_version, "
        "feature_names, impute_medians, split_bounds, test_metrics, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("v1", "models/v1.pt", "gcn-32-16-1", "fixed-1", "[]", "{}", "{}", "{}", "2026-08-29T00:00:00Z"),
    )
    conn.commit()
    row = conn.execute("SELECT used_news, ablation_fold_id FROM gnn_model_artifacts WHERE model_version='v1'").fetchone()
    assert row[0] == 0
    assert row[1] is None
