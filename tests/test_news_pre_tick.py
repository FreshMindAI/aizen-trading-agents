"""Tests for the news pre-tick refresh CLI (Loop 2 of the autonomous
loops plan).

Covers:
- _articles_to_rows: aggregates Alpaca article dicts to (timestamp,
  symbol) cells with weighted sentiment, top-3 topics, and bounded
  sentiment in [-1, +1].
- refresh_news: idempotent (second call inserts 0 rows), empty input
  doesn't crash, AlpacaDataError is caught and reported as 0.
- rebuild_news_topology: snapshot_id, node_count, edge_count
  (zero-article case still rebuilds a snapshot but with no news edges).
- main() smoke test: empty universe exits 0, normal invocation returns
  0 and prints a summary block.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("AIZEN_TRACE", "0")

from src.agents.cli.news_pre_tick import (  # noqa: E402
    _articles_to_rows,
    _window_iso,
    main,
    rebuild_news_topology,
    refresh_news,
)
from src.db import connect, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path: Path):
    p = tmp_path / "test.db"
    c = connect(str(p))
    init_db(c, sql_dir=REPO / "sql")
    yield c
    c.close()


def _article(ts: str, headline: str, symbols: list[str], summary: str = "") -> dict:
    return {
        "published_at": ts,
        "headline": headline,
        "summary": summary,
        "symbols": symbols,
    }


# ---------------------------------------------------------------------------
# _window_iso
# ---------------------------------------------------------------------------
def test_window_iso_15min_returns_z_suffix():
    start, end = _window_iso(15)
    assert start.endswith("Z") and end.endswith("Z")
    # End is "now"; start is 15 min before. Parse both.
    s = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert e - s == timedelta(minutes=15)


# ---------------------------------------------------------------------------
# _articles_to_rows
# ---------------------------------------------------------------------------
def test_articles_to_rows_aggregates_per_symbol():
    ts = "2026-08-30T13:30:00Z"
    start, end = ts, ts
    arts = [
        _article(ts, "Apple beats earnings", ["AAPL"], "Strong growth and surging demand"),
        _article(ts, "Apple wins contract", ["AAPL"], "Bullish record high"),
        _article(ts, "Nvidia raises guidance", ["NVDA"], "Robust expansion and growth"),
    ]
    rows = _articles_to_rows(arts, ["AAPL", "NVDA"], start, end)
    assert len(rows) == 2
    by_sym = {r[1]: r for r in rows}
    assert by_sym["AAPL"][3] == 2  # article_count
    assert by_sym["NVDA"][3] == 1
    # Sentiment is in [-1, +1]
    for r in rows:
        assert -1.0 <= r[2] <= 1.0
    # Topics JSON parses to a list
    for r in rows:
        topics = json.loads(r[4])
        assert isinstance(topics, list)
        assert len(topics) <= 3


def test_articles_to_rows_filters_out_universe_symbols():
    ts = "2026-08-30T13:30:00Z"
    arts = [
        _article(ts, "Some news", ["TSLA"]),  # not in universe
        _article(ts, "Apple beats", ["AAPL"]),
    ]
    rows = _articles_to_rows(arts, ["AAPL"], ts, ts)
    assert len(rows) == 1
    assert rows[0][1] == "AAPL"


def test_articles_to_rows_drops_out_of_window_articles():
    start = "2026-08-30T13:00:00Z"
    end = "2026-08-30T13:30:00Z"
    arts = [
        _article("2026-08-30T12:00:00Z", "old", ["AAPL"]),   # before window
        _article("2026-08-30T14:00:00Z", "future", ["AAPL"]),  # after window
        _article("2026-08-30T13:15:00Z", "in window", ["AAPL"]),
    ]
    rows = _articles_to_rows(arts, ["AAPL"], start, end)
    assert len(rows) == 1


def test_articles_to_rows_drops_articles_without_published_at():
    arts = [
        {"headline": "no timestamp", "symbols": ["AAPL"]},
        _article("2026-08-30T13:30:00Z", "with ts", ["AAPL"]),
    ]
    rows = _articles_to_rows(arts, ["AAPL"], "2026-08-30T13:00:00Z", "2026-08-30T14:00:00Z")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# refresh_news
# ---------------------------------------------------------------------------
def test_refresh_news_inserts_rows(conn: sqlite3.Connection, monkeypatch):
    # Stub the Alpaca client to return articles timestamped *now* so they
    # fall inside the lookback window.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fake = MagicMock()
    fake.fetch_news.return_value = [
        _article(now, "Apple beats earnings", ["AAPL"], summary="Strong growth"),
        _article(now, "Nvidia upgrades", ["NVDA"], summary="Bullish"),
    ]
    out = refresh_news(
        ["AAPL", "NVDA"], lookback_minutes=15, client=fake,
        conn_factory=lambda: conn,
    )
    assert out["articles_fetched"] == 2
    assert out["rows_inserted"] == 2

    # Verify the rows are in news_snapshot.
    n = conn.execute("SELECT COUNT(*) AS n FROM news_snapshot").fetchone()["n"]
    assert n == 2


def test_refresh_news_idempotent(conn: sqlite3.Connection, monkeypatch):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fake = MagicMock()
    fake.fetch_news.return_value = [
        _article(now, "Apple beats", ["AAPL"]),
    ]
    first = refresh_news(
        ["AAPL"], lookback_minutes=15, client=fake, conn_factory=lambda: conn,
    )
    second = refresh_news(
        ["AAPL"], lookback_minutes=15, client=fake, conn_factory=lambda: conn,
    )
    assert first["rows_inserted"] == 1
    assert second["rows_inserted"] == 0  # PK collision, INSERT OR IGNORE


def test_refresh_news_handles_zero_articles(conn: sqlite3.Connection, monkeypatch):
    fake = MagicMock()
    fake.fetch_news.return_value = []
    out = refresh_news(
        ["AAPL"], lookback_minutes=15, client=fake, conn_factory=lambda: conn,
    )
    assert out["articles_fetched"] == 0
    assert out["rows_inserted"] == 0


def test_refresh_news_handles_alpaca_error(monkeypatch):
    # Construct a client that raises AlpacaDataError on fetch_news.
    from src.agents.alpaca_data import AlpacaDataError
    fake = MagicMock()
    fake.fetch_news.side_effect = AlpacaDataError("rate limit")
    out = refresh_news(["AAPL"], lookback_minutes=15, client=fake)
    assert out == {"articles_fetched": 0, "rows_inserted": 0}


# ---------------------------------------------------------------------------
# rebuild_news_topology
# ---------------------------------------------------------------------------
def test_rebuild_news_topology_writes_snapshot(conn: sqlite3.Connection, monkeypatch):
    """write_option_snapshot must be invoked with news_enabled=True
    and news_cutoff=now. We patch it out so the test doesn't depend
    on the option_contracts table being populated."""
    import src.agents.cli.news_pre_tick as npt

    fake_result = {
        "snapshot_id": "snap-xyz",
        "node_count": 10,
        "edge_count": 4,
    }
    fake_write = MagicMock(return_value=fake_result)
    monkeypatch.setattr(
        "src.agents.cli.news_pre_tick.write_option_snapshot", fake_write,
        raising=False,
    )
    # `rebuild_news_topology` does a `from src.gnn.option_graph import
    # write_option_snapshot` inside the function. Patch that binding
    # instead.
    import src.gnn.option_graph as og
    monkeypatch.setattr(og, "write_option_snapshot", fake_write)

    out = rebuild_news_topology(conn, ["AAPL", "NVDA"], now_iso="2026-08-30T13:35:00Z")
    assert out["snapshot_id"] == "snap-xyz"
    assert out["node_count"] == 10
    assert out["edge_count"] == 4
    assert out["topology_version"] == "option-v2-news"
    # And the call went through with our args.
    args, kwargs = fake_write.call_args
    assert kwargs["news_enabled"] is True
    assert kwargs["news_cutoff"] == "2026-08-30T13:35:00Z"


def test_rebuild_news_topology_handles_no_edges(conn: sqlite3.Connection, monkeypatch):
    import src.gnn.option_graph as og
    fake_write = MagicMock(return_value={
        "snapshot_id": "snap-empty", "node_count": 0, "edge_count": 0,
    })
    monkeypatch.setattr(og, "write_option_snapshot", fake_write)
    out = rebuild_news_topology(conn, ["AAPL"], now_iso="2026-08-30T13:35:00Z")
    assert out["topology_version"] == "option-v2"  # no news edges → fallback label


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def test_main_empty_universe_returns_zero(capsys, tmp_path: Path, monkeypatch):
    # An empty universe is a no-op (logs a warning, exits 0, prints no
    # summary). That's the documented behavior: nothing to refresh, so
    # we skip the topology rebuild too.
    rc = main(["--universe", "", "--db-path", str(tmp_path / "x.db")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "news_pre_tick summary" not in out  # summary not printed


def test_main_invokes_refresh_and_topology(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "trading.db"
    monkeypatch.setenv("AIZEN_DB_PATH", str(db_path))

    # Stub out the network client and the topology writer.
    import src.agents.cli.news_pre_tick as npt

    fake_client = MagicMock()
    fake_client.fetch_news.return_value = [
        _article("2026-08-30T13:30:00Z", "Apple beats", ["AAPL"], summary="growth"),
    ]
    monkeypatch.setattr(npt, "refresh_news",
                        lambda universe, **kw: {"articles_fetched": 1, "rows_inserted": 1})

    fake_topology = {
        "snapshot_id": "snap-1", "node_count": 5, "edge_count": 2,
        "topology_version": "option-v2-news",
    }
    monkeypatch.setattr(npt, "rebuild_news_topology", lambda conn, u, **kw: fake_topology)

    rc = main(["--universe", "AAPL,NVDA", "--db-path", str(db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "articles_fetched   : 1" in out
    assert "rows_inserted      : 1" in out
    assert "topology_snapshot  : snap-1" in out


def test_main_skip_topology_does_not_rebuild(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "trading.db"
    monkeypatch.setenv("AIZEN_DB_PATH", str(db_path))

    import src.agents.cli.news_pre_tick as npt
    monkeypatch.setattr(npt, "refresh_news",
                        lambda universe, **kw: {"articles_fetched": 0, "rows_inserted": 0})
    called = MagicMock()
    monkeypatch.setattr(npt, "rebuild_news_topology", called)

    rc = main(["--universe", "AAPL", "--db-path", str(db_path), "--skip-topology"])
    assert rc == 0
    assert not called.called
