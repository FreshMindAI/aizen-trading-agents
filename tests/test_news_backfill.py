"""Tests for the backfill CLI (spec 003 / T022, US1 acceptance scenarios 1, 4).

Covers:
- _articles_to_cells: aggregates to (timestamp, symbol) rows
- _date_range: inclusive bounds across multi-day spans
- _upsert_rows: inserted/skipped accounting + idempotency
- run(): end-to-end with a fake AlpacaDataClient
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.cli.backfill_news import (  # noqa: E402
    _articles_to_cells,
    _date_range,
    _upsert_rows,
    run,
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


def _article(ts: str, headline: str, symbols: list[str], summary: str = "") -> dict:
    return {"published_at": ts, "headline": headline, "summary": summary, "symbols": symbols}


# ---------------------------------------------------------------------------
# _date_range
# ---------------------------------------------------------------------------
def test_date_range_single_day():
    days = _date_range("2026-08-29", "2026-08-29")
    assert len(days) == 1
    assert days[0][0] == "2026-08-29T00:00:00Z"
    assert days[0][1] == "2026-08-29T23:59:59Z"


def test_date_range_three_days_inclusive():
    days = _date_range("2026-08-28", "2026-08-30")
    assert [d[0][:10] for d in days] == ["2026-08-28", "2026-08-29", "2026-08-30"]


# ---------------------------------------------------------------------------
# _articles_to_cells
# ---------------------------------------------------------------------------
def test_articles_to_cells_filters_by_universe_and_window():
    arts = [
        _article("2026-08-29T10:00:00Z", "NVDA beat", ["NVDA"]),
        _article("2026-08-29T11:00:00Z", "AAPL plunge", ["AAPL"]),
        _article("2026-08-29T12:00:00Z", "TSLA rally", ["TSLA"]),  # not in universe
        _article("2026-08-30T09:00:00Z", "NVDA miss", ["NVDA"]),   # outside window
    ]
    rows = _articles_to_cells(
        arts, universe=["NVDA", "AAPL"],
        day_start="2026-08-29T00:00:00Z", day_end="2026-08-29T23:59:59Z",
    )
    # Two cells: (2026-08-29T10:00, NVDA) and (2026-08-29T11:00, AAPL)
    assert len(rows) == 2
    by_key = {(r[0], r[1]): r for r in rows}
    assert ("2026-08-29T10:00:00Z", "NVDA") in by_key
    assert ("2026-08-29T11:00:00Z", "AAPL") in by_key
    # Sentiment bounds
    for r in rows:
        assert -1.0 <= r[2] <= 1.0
        assert r[3] >= 1
        # topics_json is a JSON array string
        import json
        topics = json.loads(r[4])
        assert isinstance(topics, list)
        assert len(topics) <= 3


def test_articles_to_cells_aggregates_multiple_articles_per_cell():
    arts = [
        _article("2026-08-29T10:00:00Z", "NVDA beat", ["NVDA"], "strong demand"),
        _article("2026-08-29T10:00:00Z", "NVDA rally", ["NVDA"], "long summary"),
    ]
    rows = _articles_to_cells(
        arts, universe=["NVDA"],
        day_start="2026-08-29T00:00:00Z", day_end="2026-08-29T23:59:59Z",
    )
    assert len(rows) == 1
    assert rows[0][3] == 2  # article_count = 2


# ---------------------------------------------------------------------------
# _upsert_rows
# ---------------------------------------------------------------------------
def test_upsert_rows_counts_inserted_and_skipped(conn):
    rows = [
        ("2026-08-29T10:00:00Z", "NVDA", 0.5, 1, "[\"x\"]", "{}", "2026-08-29T10:01:00Z"),
        ("2026-08-29T10:00:00Z", "NVDA", 0.7, 1, "[\"y\"]", "{}", "2026-08-29T10:02:00Z"),
    ]
    inserted, skipped = _upsert_rows(conn, rows)
    assert inserted == 1
    assert skipped == 1


def test_upsert_rows_idempotent_on_rerun(conn):
    rows = [
        ("2026-08-29T10:00:00Z", "NVDA", 0.5, 1, "[\"x\"]", "{}", "2026-08-29T10:01:00Z"),
    ]
    inserted1, skipped1 = _upsert_rows(conn, rows)
    inserted2, skipped2 = _upsert_rows(conn, rows)
    assert inserted1 == 1 and skipped1 == 0
    assert inserted2 == 0 and skipped2 == 1


def test_upsert_rows_empty_input_is_noop(conn):
    inserted, skipped = _upsert_rows(conn, [])
    assert inserted == 0
    assert skipped == 0


# ---------------------------------------------------------------------------
# run() end-to-end
# ---------------------------------------------------------------------------
def test_run_writes_rows_with_fake_client(conn, tmp_path, capsys):
    args = type("Args", (), {
        "start": "2026-08-29",
        "end": "2026-08-29",
        "universe": "NVDA,AAPL",
        "per_day_limit": 50,
        "db_path": str(tmp_path / "test.db"),
        "log_level": "WARNING",
    })()
    fake_client = MagicMock()
    fake_client.fetch_news.return_value = [
        _article("2026-08-29T10:00:00Z", "NVDA beat", ["NVDA"], "strong demand"),
        _article("2026-08-29T11:00:00Z", "AAPL plunge", ["AAPL"], "weak outlook"),
    ]
    with patch("src.agents.cli.backfill_news.AlpacaDataClient", return_value=fake_client):
        rc = run(args)
    assert rc == 0
    captured = capsys.readouterr().out
    assert "done:" in captured
    n = conn.execute("SELECT COUNT(*) FROM news_snapshot").fetchone()[0]
    assert n == 2


def test_run_continues_when_alpaca_raises(conn, tmp_path, capsys):
    from src.agents.alpaca_data import AlpacaDataError

    args = type("Args", (), {
        "start": "2026-08-28",
        "end": "2026-08-29",
        "universe": "NVDA",
        "per_day_limit": 50,
        "db_path": str(tmp_path / "test.db"),
        "log_level": "WARNING",
    })()
    fake_client = MagicMock()
    fake_client.fetch_news.side_effect = AlpacaDataError("401 unauthorized")
    with patch("src.agents.cli.backfill_news.AlpacaDataClient", return_value=fake_client):
        rc = run(args)
    # US1 acceptance scenario 4: single non-fatal warning, exit code 0
    assert rc == 0
    n = conn.execute("SELECT COUNT(*) FROM news_snapshot").fetchone()[0]
    assert n == 0


# ---------------------------------------------------------------------------
# Regression: bug fixed 2026-08-29 — the per-day run() loop previously
# passed `ds[:10], de[:10]` (date-only) to client.fetch_news(). Alpaca's
# /v1beta1/news endpoint returns 0 articles when start == end as dates,
# even when both are on the same day. The fix: pass the full ISO timestamp
# (ds, de), which is what _date_range already produces. This test pins
# the call signature so the regression cannot return.
# ---------------------------------------------------------------------------
def test_run_passes_full_iso_timestamps_to_fetch_news(conn, tmp_path, capsys):
    """Pin the date-format fix: fetch_news must be called with full
    ISO timestamps that include the time-of-day component, not with
    the date-only slice. Without the fix, Alpaca's news endpoint
    returns 0 articles for same-day windows and the backfill is
    silently empty.
    """
    args = type("Args", (), {
        "start": "2026-08-29",
        "end": "2026-08-29",
        "universe": "NVDA",
        "per_day_limit": 50,
        "db_path": str(tmp_path / "test.db"),
        "log_level": "WARNING",
    })()
    fake_client = MagicMock()
    fake_client.fetch_news.return_value = []
    with patch("src.agents.cli.backfill_news.AlpacaDataClient", return_value=fake_client):
        rc = run(args)
    assert rc == 0
    # fetch_news must have been called with at least one full ISO timestamp
    # that includes the time-of-day component (length > 10, the date prefix).
    assert fake_client.fetch_news.call_count >= 1, "fetch_news was never called"
    for call in fake_client.fetch_news.call_args_list:
        # Positional args: (symbols, start, end, limit=...)
        start_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("start")
        end_arg = call.args[2] if len(call.args) > 2 else call.kwargs.get("end")
        assert start_arg is not None and len(start_arg) > 10, (
            f"fetch_news start must be a full ISO timestamp with time component, "
            f"got {start_arg!r} (date-only slice regression)"
        )
        assert end_arg is not None and len(end_arg) > 10, (
            f"fetch_news end must be a full ISO timestamp with time component, "
            f"got {end_arg!r} (date-only slice regression)"
        )
        # The start/end must also include the day boundaries from _date_range.
        assert start_arg.endswith("T00:00:00Z"), f"unexpected start: {start_arg!r}"
        assert end_arg.endswith("T23:59:59Z"), f"unexpected end: {end_arg!r}"


def test_run_writes_rows_when_fake_filters_by_iso_timestamps(conn, tmp_path, capsys):
    """A fake that mimics Alpaca's real behavior: returns 0 articles for
    date-only inputs and the right articles for full ISO timestamps.
    With the bug present, the backfill would insert 0 rows; with the
    fix, it inserts 2 rows.
    """
    iso_window = ("2026-08-29T00:00:00Z", "2026-08-29T23:59:59Z")
    iso_articles = [
        _article("2026-08-29T10:00:00Z", "NVDA beat", ["NVDA"], "strong demand"),
        _article("2026-08-29T11:00:00Z", "NVDA rally", ["NVDA"], "long summary"),
    ]

    def fake_fetch(symbols, start, end, **_kw):
        # If start/end are date-only (the bug), return [] (Alpaca's behavior).
        if len(start) <= 10 and len(end) <= 10 and start == end:
            return []
        # Otherwise, only return articles for the matching full-ISO window.
        if start == iso_window[0] and end == iso_window[1] and tuple(symbols) == ("NVDA",):
            return iso_articles
        return []

    args = type("Args", (), {
        "start": "2026-08-29",
        "end": "2026-08-29",
        "universe": "NVDA",
        "per_day_limit": 50,
        "db_path": str(tmp_path / "test.db"),
        "log_level": "WARNING",
    })()
    fake_client = MagicMock()
    fake_client.fetch_news.side_effect = fake_fetch
    with patch("src.agents.cli.backfill_news.AlpacaDataClient", return_value=fake_client):
        rc = run(args)
    assert rc == 0
    n = conn.execute("SELECT COUNT(*) FROM news_snapshot").fetchone()[0]
    assert n == 2, (
        f"expected 2 rows in news_snapshot, got {n}. "
        f"If 0, the date-only regression has returned: fetch_news was "
        f"called with date-only arguments and Alpaca returned 0 articles."
    )
