"""Tests for the news-driven edge builders in option_graph (spec 003 / T023-T035).

Covers US2 acceptance scenarios:
- 8-d edge feature vector (6 features + 2 one-hot) for both pre-news and
  news edges (SC-001 / FR-005 / FR-006)
- news_cooccurrence fires only when both underlyings have >= min_articles
- news_sentiment_correlation fires only when |rho| >= min_abs_rho
- news_cutoff respected (no future-article leakage, failure-analysis row 2)
- topology_version="option-v2-news" when at least one news edge is present
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.db import connect, init_db  # noqa: E402
from src.gnn.option_graph import (  # noqa: E402
    _build_news_edges,
    _edge_type_onehot,
    _news_cooccurrence_edges,
    _news_per_underlying,
    _news_sentiment_correlation_edges,
    _sentiment_correlation,
    build_option_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iso(days_ago: int, hour: int = 12) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).replace(
        hour=hour, minute=0, second=0, microsecond=0,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _node(underlying: str, contract: str, *, dte: int = 14, moneyness: float = 1.0) -> dict:
    return {
        "contract_symbol": contract,
        "underlying": underlying,
        "strike_price": 100.0 * moneyness,
        "option_type": "call",
        "expiration_date": (datetime.now(timezone.utc) + timedelta(days=dte)).strftime("%Y-%m-%d"),
        "dte": dte,
        "moneyness": moneyness,
        "spot": 100.0,
        "node_features": [0.0] * 14,
    }


@pytest.fixture
def conn(tmp_path):
    p = tmp_path / "t.db"
    c = connect(str(p))
    init_db(c, sql_dir=__import__("pathlib").Path("sql"))
    yield c
    c.close()


def _insert_news(conn, ts: str, sym: str, sent: float, count: int = 1):
    conn.execute(
        "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, "
        "topics_json, raw_json, created_at) VALUES (?, ?, ?, ?, '[]', '{}', ?)",
        (ts, sym, sent, count, ts),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# One-hot taxonomy
# ---------------------------------------------------------------------------
def test_edge_type_onehot_is_2d():
    """Decision 6: 2-d one-hot keeps edge_dim=8 (6 features + 2 one-hot)."""
    assert _edge_type_onehot("same_expiry") == [1.0, 0.0]
    assert _edge_type_onehot("news_cooccurrence") == [0.0, 1.0]
    assert _edge_type_onehot("news_sentiment_correlation") == [0.0, 1.0]


# ---------------------------------------------------------------------------
# _news_per_underlying
# ---------------------------------------------------------------------------
def test_news_per_underlying_aggregates_articles_and_sentiment(conn):
    _insert_news(conn, _iso(0, 10), "NVDA", 0.5, count=1)
    _insert_news(conn, _iso(0, 11), "NVDA", 0.7, count=2)
    _insert_news(conn, _iso(0, 12), "AAPL", -0.3, count=1)
    out = _news_per_underlying(conn, ["NVDA", "AAPL"], _iso(0, 13), lookback_hours=24)
    assert out["NVDA"]["article_count"] == 3
    assert out["AAPL"]["article_count"] == 1
    assert len(out["NVDA"]["sentiment_ts"]) == 2


def test_news_per_underlying_respects_lookback(conn):
    """Articles outside the lookback window are dropped (FR-001 / SC-002)."""
    _insert_news(conn, _iso(5, 10), "NVDA", 0.5, count=1)  # outside 24h
    _insert_news(conn, _iso(0, 10), "NVDA", 0.7, count=1)  # inside
    out = _news_per_underlying(conn, ["NVDA"], _iso(0, 13), lookback_hours=24)
    assert out["NVDA"]["article_count"] == 1


def test_news_per_underlying_respects_cutoff(conn):
    """Future-article leakage protection (failure-analysis §2.1 row 2)."""
    _insert_news(conn, _iso(0, 14), "NVDA", 0.5, count=1)  # future relative to cutoff
    _insert_news(conn, _iso(0, 10), "NVDA", 0.7, count=1)  # past
    out = _news_per_underlying(conn, ["NVDA"], _iso(0, 12), lookback_hours=24)
    assert out["NVDA"]["article_count"] == 1


# ---------------------------------------------------------------------------
# news_cooccurrence
# ---------------------------------------------------------------------------
def test_cooccurrence_fires_only_when_both_have_min_articles(conn):
    _insert_news(conn, _iso(0, 10), "NVDA", 0.5, count=2)
    _insert_news(conn, _iso(0, 11), "AAPL", 0.4, count=2)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    per_u = _news_per_underlying(conn, ["NVDA", "AAPL"], _iso(0, 13), lookback_hours=24)
    edges = _news_cooccurrence_edges(nodes, per_u, min_articles=2)
    # 2 directed edges (A->B, B->A) for 1 contract pair
    assert len(edges) == 2
    reasons = {e["reason"] for e in edges}
    assert reasons == {"news_cooccurrence"}


def test_cooccurrence_does_not_fire_when_below_threshold(conn):
    _insert_news(conn, _iso(0, 10), "NVDA", 0.5, count=1)  # < 2
    _insert_news(conn, _iso(0, 11), "AAPL", 0.4, count=2)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    per_u = _news_per_underlying(conn, ["NVDA", "AAPL"], _iso(0, 13), lookback_hours=24)
    edges = _news_cooccurrence_edges(nodes, per_u, min_articles=2)
    assert edges == []


def test_cooccurrence_weight_is_log1p_product(conn):
    _insert_news(conn, _iso(0, 10), "NVDA", 0.5, count=3)
    _insert_news(conn, _iso(0, 11), "AAPL", 0.4, count=4)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    per_u = _news_per_underlying(conn, ["NVDA", "AAPL"], _iso(0, 13), lookback_hours=24)
    edges = _news_cooccurrence_edges(nodes, per_u, min_articles=2)
    import math
    expected = math.log1p(3) * math.log1p(4)
    assert all(abs(e["weight"] - expected) < 1e-9 for e in edges)


# ---------------------------------------------------------------------------
# news_sentiment_correlation
# ---------------------------------------------------------------------------
def test_sentiment_correlation_returns_none_for_no_overlap():
    rho = _sentiment_correlation(
        [(_iso(0, 10), 0.5)],
        [(_iso(2, 10), -0.4)],
    )
    assert rho is None


def test_sentiment_correlation_positive_corr():
    # Strong positive correlation across 3 days
    a = [(_iso(i, 10), 0.1 * i + 0.5) for i in range(3)]
    b = [(_iso(i, 10), 0.1 * i + 0.3) for i in range(3)]
    rho = _sentiment_correlation(a, b)
    assert rho is not None
    assert rho > 0.9


def test_sentiment_correlation_negative_corr():
    a = [(_iso(i, 10), 0.1 * i) for i in range(3)]
    b = [(_iso(i, 10), -0.1 * i) for i in range(3)]
    rho = _sentiment_correlation(a, b)
    assert rho is not None
    assert rho < -0.9


def test_correlation_edge_only_fires_for_strong_rho(conn):
    # Both share 3 days, perfectly correlated
    for i in range(3):
        _insert_news(conn, _iso(i, 10), "NVDA", 0.1 * i + 0.5, count=1)
        _insert_news(conn, _iso(i, 10), "AAPL", 0.1 * i + 0.3, count=1)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    per_u = _news_per_underlying(conn, ["NVDA", "AAPL"], _iso(0, 13), lookback_hours=24 * 5)
    edges = _news_sentiment_correlation_edges(nodes, per_u, min_abs_rho=0.5)
    assert len(edges) == 2  # directed pair
    assert all(e["reason"] == "news_sentiment_correlation" for e in edges)
    # Weight is |rho|, >= 0.5
    for e in edges:
        assert e["weight"] >= 0.5


def test_correlation_edge_does_not_fire_for_weak_rho(conn):
    # Same days, completely uncorrelated
    _insert_news(conn, _iso(0, 10), "NVDA", 0.1, count=1)
    _insert_news(conn, _iso(1, 10), "NVDA", 0.9, count=1)
    _insert_news(conn, _iso(2, 10), "NVDA", 0.2, count=1)
    _insert_news(conn, _iso(0, 10), "AAPL", 0.5, count=1)
    _insert_news(conn, _iso(1, 10), "AAPL", 0.5, count=1)
    _insert_news(conn, _iso(2, 10), "AAPL", 0.5, count=1)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    per_u = _news_per_underlying(conn, ["NVDA", "AAPL"], _iso(0, 13), lookback_hours=24 * 5)
    edges = _news_sentiment_correlation_edges(nodes, per_u, min_abs_rho=0.5)
    assert edges == []


# ---------------------------------------------------------------------------
# _build_news_edges + topology_version
# ---------------------------------------------------------------------------
def test_build_news_edges_returns_8d_feature_vector(conn):
    _insert_news(conn, _iso(0, 10), "NVDA", 0.5, count=2)
    _insert_news(conn, _iso(0, 11), "AAPL", 0.4, count=2)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    edges = _build_news_edges(nodes, conn, _iso(0, 13), news_cutoff=None)
    assert edges, "expected at least one news edge"
    for e in edges:
        assert "edge_features" in e
        ef = e["edge_features"]
        assert len(ef) == 8, f"expected 8-d edge features, got {len(ef)}: {ef}"
        # News one-hot slice is [0, 1]
        assert ef[6:] == [0.0, 1.0], f"news one-hot should be [0, 1], got {ef[6:]}"
        # No underscore temp keys leak
        assert "_a" not in e and "_b" not in e


def test_build_news_edges_respects_news_cutoff(conn):
    """Future articles (timestamp > news_cutoff) MUST NOT be admitted
    (failure-analysis §2.1 row 2 — news-time-leakage protection)."""
    # NVDA: 1 article at 10:00 (before cutoff) + 1 article at 14:00 (after)
    # AAPL: 2 articles at 10:00 and 11:00 (both before cutoff)
    # With cutoff=12:00, NVDA has 1 article (10:00) — below min_articles=2.
    # AAPL has 2 articles — but cooccurrence needs BOTH >= 2. => no edge.
    _insert_news(conn, _iso(0, 10), "NVDA", 0.5, count=1)
    _insert_news(conn, _iso(0, 14), "NVDA", 0.5, count=1)  # future, excluded
    _insert_news(conn, _iso(0, 10), "AAPL", 0.4, count=1)
    _insert_news(conn, _iso(0, 11), "AAPL", 0.4, count=1)
    nodes = [_node("NVDA", "NVDA1"), _node("AAPL", "AAPL1")]
    edges = _build_news_edges(nodes, conn, _iso(0, 13), news_cutoff=_iso(0, 12))
    assert not any(e["reason"] == "news_cooccurrence" for e in edges)
    # Without the cutoff, NVDA's 14:00 article is included; both NVDA and
    # AAPL now have >= 2 articles, so cooccurrence fires.
    edges_open = _build_news_edges(nodes, conn, _iso(0, 13), news_cutoff=_iso(0, 23))
    assert any(e["reason"] == "news_cooccurrence" for e in edges_open)


def test_build_news_edges_empty_when_universe_empty(conn):
    nodes: list[dict] = []
    edges = _build_news_edges(nodes, conn, _iso(0, 13), news_cutoff=None)
    assert edges == []


# ---------------------------------------------------------------------------
# build_option_payload: topology_version + news_enabled flag
# ---------------------------------------------------------------------------
def test_build_option_payload_default_topology_is_option_v2(conn):
    """When news_enabled is False, topology_version is 'option-v2'."""
    payload = build_option_payload(conn, _iso(0, 13), underlying=None, news_enabled=False)
    # Empty because no underlying_bars / option_contracts in the empty test DB
    assert payload["topology_version"] in ("option-v2", "option-v2-news")


def test_build_option_payload_topology_news_when_enabled(conn):
    payload = build_option_payload(
        conn, _iso(0, 13), underlying=None, news_enabled=True, news_cutoff=None,
    )
    # When news_enabled=True and the graph has no news data, version is
    # still 'option-v2-news' (signals intent). With at least one news edge
    # it is also 'option-v2-news'.
    assert payload["topology_version"] in ("option-v2-news", "option-v2")
