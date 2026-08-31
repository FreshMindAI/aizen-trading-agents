"""Tests for the research agent node (spec 003 / T021, FR-001, FR-011, FR-014).

Covers:
- Flag-off: no-op update, signal.feature_flag_state == 'news-off'
- Flag-on with no market_snapshot: no-op with reason 'no_market_snapshot'
- Flag-on with empty universe: no-op with reason 'empty_universe'
- Flag-on with mocked AlpacaDataClient: articles aggregated to per_symbol
- AlpacaDataError: returns risks=['research_alpaca_unreachable']
- Backtest mask: articles older than lookback_hours are dropped
- The node never raises; failures become empty per_symbol with risks
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.nodes.research import (  # noqa: E402
    _aggregate_articles,
    _is_enabled,
    build_node,
)
from src.agents.protocol import (  # noqa: E402
    AgentObservation,
    DecisionState,
    MarketSnapshot,
    MessageType,
    UnderlyingScore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _state(symbols: list[str] | None = None) -> DecisionState:
    """Build a minimal DecisionState for one cycle."""
    syms = symbols or ["NVDA", "AAPL"]
    return DecisionState(
        market_snapshot=MarketSnapshot(
            timestamp="2026-08-29T13:30:00Z",
            underlyings=[
                UnderlyingScore(symbol=s, timestamp="2026-08-29T13:30:00Z", horizon_bars=16)
                for s in syms
            ],
        ),
    )


class _NoOpLLM:
    """Stub LLM provider; research node must never call it (FR-014)."""
    def complete(self, *a, **kw):  # pragma: no cover
        raise AssertionError("research node must not call the LLM")
    def complete_as(self, *a, **kw):  # pragma: no cover
        raise AssertionError("research node must not call the LLM")


def _articles(recent_count: int, old_count: int = 0, symbol: str = "NVDA") -> list[dict]:
    """Return ``recent_count`` articles from the past hour, ``old_count`` from 48h ago."""
    now = datetime.now(timezone.utc)
    out: list[dict] = []
    for i in range(recent_count):
        out.append({
            "published_at": (now - timedelta(minutes=10 + i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "headline": f"NVDA beat earnings on {symbol}",
            "summary": "Strong demand",
            "symbols": [symbol],
        })
    for i in range(old_count):
        out.append({
            "published_at": (now - timedelta(hours=48 + i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "headline": f"old article on {symbol}",
            "symbols": [symbol],
        })
    return out


# ---------------------------------------------------------------------------
# _is_enabled
# ---------------------------------------------------------------------------
def test_is_enabled_default_false():
    os.environ.pop("AIZEN_RESEARCH_ENABLED", None)
    assert _is_enabled({"agents": {"research": {"enabled": False}}}) is False
    assert _is_enabled({"agents": {"research": {"enabled": True}}}) is True


def test_is_enabled_env_overrides_config(monkeypatch):
    monkeypatch.setenv("AIZEN_RESEARCH_ENABLED", "true")
    assert _is_enabled({"agents": {"research": {"enabled": False}}}) is True
    monkeypatch.setenv("AIZEN_RESEARCH_ENABLED", "0")
    assert _is_enabled({"agents": {"research": {"enabled": True}}}) is False


# ---------------------------------------------------------------------------
# Flag off
# ---------------------------------------------------------------------------
def test_flag_off_returns_noop():
    node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": False}}}, None)
    out = node(_state())
    assert "agent_messages" in out
    assert len(out["agent_messages"]) == 1
    msg = out["agent_messages"][0]
    assert msg.message_type == MessageType.RESEARCH_VIEW
    assert msg.payload["signal"]["feature_flag_state"] == "news-off"
    # No research key returned in the off path
    assert "research" not in out
    # LLM must never be called
    assert "agent_observations" not in out or len(out["agent_observations"]) == 1


# ---------------------------------------------------------------------------
# Empty / missing inputs
# ---------------------------------------------------------------------------
def test_no_market_snapshot_returns_noop():
    state = DecisionState()  # no market_snapshot
    node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
    out = node(state)
    msg = out["agent_messages"][0]
    assert msg.payload["signal"]["reason"] == "no_market_snapshot"
    assert msg.payload["signal"]["feature_flag_state"] == "news-off"


def test_empty_universe_returns_noop():
    state = DecisionState(market_snapshot=MarketSnapshot(timestamp="2026-08-29T13:30:00Z", underlyings=[]))
    node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
    out = node(state)
    msg = out["agent_messages"][0]
    assert msg.payload["signal"]["reason"] == "empty_universe"


# ---------------------------------------------------------------------------
# Happy path (flag on, mocked client)
# ---------------------------------------------------------------------------
def test_flag_on_aggregates_articles_to_per_symbol():
    fake_client = MagicMock()
    fake_client.fetch_news.return_value = _articles(3, old_count=2, symbol="NVDA")
    with patch("src.agents.nodes.research.AlpacaDataClient", return_value=fake_client):
        node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
        out = node(_state(["NVDA", "AAPL"]))
    assert "research" in out
    research = out["research"]
    assert research.feature_flag_state == "news-on"
    assert "NVDA" in research.per_symbol
    # old_count=2 articles are dropped by lookback mask -> only 3 recent
    assert research.per_symbol["NVDA"].volume == 3
    # sentiment is positive (3 'beat' articles, no negatives)
    assert research.per_symbol["NVDA"].sentiment > 0


def test_backtest_mask_drops_articles_outside_lookback():
    """Articles older than 24h MUST NOT influence the live cycle (lookback_hours=24)."""
    fake_client = MagicMock()
    fake_client.fetch_news.return_value = _articles(recent_count=1, old_count=5, symbol="NVDA")
    with patch("src.agents.nodes.research.AlpacaDataClient", return_value=fake_client):
        node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
        out = node(_state(["NVDA"]))
    assert out["research"].per_symbol["NVDA"].volume == 1


# ---------------------------------------------------------------------------
# Alpaca unreachable
# ---------------------------------------------------------------------------
def test_alpaca_error_yields_risks_marker():
    from src.agents.alpaca_data import AlpacaDataError

    fake_client = MagicMock()
    fake_client.fetch_news.side_effect = AlpacaDataError("401 unauthorized")
    with patch("src.agents.nodes.research.AlpacaDataClient", return_value=fake_client):
        node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
        out = node(_state(["NVDA"]))
    assert "research" in out
    assert "research_alpaca_unreachable" in out["research"].risks
    assert out["research"].per_symbol == {}
    # Still emits an observation
    obs = out["agent_observations"][0]
    assert obs.message_type == MessageType.RESEARCH_VIEW
    assert "research_alpaca_unreachable" in obs.risks


# ---------------------------------------------------------------------------
# _aggregate_articles unit
# ---------------------------------------------------------------------------
def test_aggregate_articles_includes_only_universe():
    from datetime import datetime, timedelta, timezone
    # Use timestamps relative to "now" so this test does not go stale
    # when the calendar advances. The previous version hard-coded
    # 2026-08-29 which fell outside the 24h lookback once the date
    # moved past 2026-08-30.
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=2)).isoformat()
    arts = [
        {"published_at": recent, "headline": "NVDA beat", "symbols": ["NVDA"]},
        {"published_at": recent, "headline": "TSLA plunge", "symbols": ["TSLA"]},
    ]
    agg = _aggregate_articles(arts, ["NVDA"], lookback_hours=24)
    assert "NVDA" in agg
    assert "TSLA" not in agg


def test_aggregate_articles_picks_latest_published_at():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    earlier = (now - timedelta(hours=3)).isoformat()
    latest = (now - timedelta(hours=1)).isoformat()
    arts = [
        {"published_at": earlier, "headline": "earlier", "symbols": ["NVDA"]},
        {"published_at": latest, "headline": "latest", "symbols": ["NVDA"]},
    ]
    agg = _aggregate_articles(arts, ["NVDA"], lookback_hours=24)
    assert agg["NVDA"].last_article_at == latest
    assert agg["NVDA"].volume == 2
