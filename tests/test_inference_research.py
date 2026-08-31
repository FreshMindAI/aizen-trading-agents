"""Tests for InferenceService.build_snapshot with research field (T018)."""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.inference import InferenceService  # noqa: E402
from src.db import connect, init_db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    p = tmp_path / "test.db"
    c = connect(str(p))
    init_db(c, sql_dir=__import__("pathlib").Path("sql"))
    yield c
    c.close()


def _insert_news(conn, ts, sym, sent=0.0, count=1, topics=None):
    conn.execute(
        "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, sym, sent, count, json.dumps(topics or []), "{}", ts.replace("Z", "") + "Z"),
    )
    conn.commit()


def test_research_field_is_none_when_flag_off(conn):
    svc = InferenceService(conn=conn, universe=["NVDA"], research_enabled=False)
    snap = svc.build_snapshot()
    assert snap.research is None


def test_research_field_is_none_when_no_rows(conn):
    svc = InferenceService(conn=conn, universe=["NVDA"], research_enabled=True)
    snap = svc.build_snapshot()
    assert snap.research is None  # No news rows -> valid None state


def test_research_field_loaded_when_flag_on(conn):
    _insert_news(conn, "2026-08-29T10:00:00Z", "NVDA", sent=0.5, count=2, topics=["earnings"])
    _insert_news(conn, "2026-08-29T11:00:00Z", "AAPL", sent=-0.3, count=1, topics=["miss"])
    svc = InferenceService(conn=conn, universe=["NVDA", "AAPL"], research_enabled=True)
    snap = svc.build_snapshot()
    assert snap.research is not None
    assert snap.research.feature_flag_state == "news-on"
    assert snap.research.version == "1.0"
    assert "NVDA" in snap.research.per_symbol
    assert "AAPL" in snap.research.per_symbol
    assert snap.research.per_symbol["NVDA"].sentiment == 0.5
    assert snap.research.per_symbol["AAPL"].sentiment == -0.3


def test_research_aggregates_multiple_articles_per_symbol(conn):
    _insert_news(conn, "2026-08-29T08:00:00Z", "NVDA", sent=0.4, count=1, topics=["beat"])
    _insert_news(conn, "2026-08-29T09:00:00Z", "NVDA", sent=0.6, count=2, topics=["rally"])
    svc = InferenceService(conn=conn, universe=["NVDA"], research_enabled=True)
    snap = svc.build_snapshot()
    sr = snap.research.per_symbol["NVDA"]
    # Mean of 0.4 and 0.6
    assert abs(sr.sentiment - 0.5) < 1e-9
    # Volume sums
    assert sr.volume == 3
    # last_article_at is the most recent timestamp
    assert sr.last_article_at == "2026-08-29T09:00:00Z"


def test_research_excludes_symbols_outside_universe(conn):
    _insert_news(conn, "2026-08-29T10:00:00Z", "NVDA", sent=0.5, count=1)
    _insert_news(conn, "2026-08-29T10:00:00Z", "XYZ", sent=0.9, count=1)
    svc = InferenceService(conn=conn, universe=["NVDA"], research_enabled=True)
    snap = svc.build_snapshot()
    assert "NVDA" in snap.research.per_symbol
    assert "XYZ" not in snap.research.per_symbol


def test_research_does_not_load_from_other_db_when_flag_on_but_empty_table(conn):
    svc = InferenceService(conn=conn, universe=["NVDA"], research_enabled=True)
    # Empty news_snapshot -> research is None (valid empty state)
    assert svc.build_snapshot().research is None
