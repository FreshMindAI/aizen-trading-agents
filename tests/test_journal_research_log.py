"""Tests for journal-level research message logging (T019).

The spec requires: "Add agent_messages_json log entry on every research-node
execution." This is satisfied at the orchestrator level via the existing
DecisionJournal.upsert path, which serializes state.agent_messages into
agent_messages_json. These tests prove that:
  1. Flag-off cycles still produce a RESEARCH_VIEW log entry (so debug
     history is complete even when the feature is disabled).
  2. Flag-on cycles with articles produce a populated RESEARCH_VIEW log.
  3. The log entries contain the expected signal fields so post-hoc
     debugging is possible (feature_flag_state, n_articles, etc.).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.graph import Orchestrator  # noqa: E402
from src.agents.journal import DecisionJournal  # noqa: E402
from src.agents.nodes.research import build_node  # noqa: E402
from src.agents.protocol import (  # noqa: E402
    DecisionState,
    MarketSnapshot,
    MessageType,
    UnderlyingScore,
)
from src.db import connect, init_db  # noqa: E402


class _NoOpLLM:
    def complete(self, *a, **kw):  # pragma: no cover
        raise AssertionError("research node must not call the LLM")
    def complete_as(self, *a, **kw):  # pragma: no cover
        raise AssertionError("research node must not call the LLM")


def _state(symbols=("NVDA",)) -> DecisionState:
    return DecisionState(
        market_snapshot=MarketSnapshot(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            underlyings=[
                UnderlyingScore(symbol=s, timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), horizon_bars=16)
                for s in symbols
            ],
        )
    )


def test_flag_off_research_message_is_logged_via_journal(tmp_path):
    """A flag-off cycle MUST still emit a RESEARCH_VIEW message so the
    journal has a complete per-cycle log even when the feature is off."""
    db = tmp_path / "t.db"
    conn = connect(str(db))
    init_db(conn)
    try:
        node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": False}}}, None)
        state = _state()
        out = node(state)
        assert "agent_messages" in out
        msg = out["agent_messages"][0]
        assert msg.message_type == MessageType.RESEARCH_VIEW
        # Persist via the journal like the orchestrator would
        state.agent_messages = [msg]
        journal = DecisionJournal(conn, table="decision_journal", run_mode="paper")
        journal.upsert(state)
        row = journal.get(state.decision_id)
        assert row is not None
        msgs = row["agent_messages_json"]
        assert isinstance(msgs, list)
        assert len(msgs) == 1
        assert msgs[0]["message_type"] == "RESEARCH_VIEW"
        assert msgs[0]["payload"]["signal"]["feature_flag_state"] == "news-off"
    finally:
        conn.close()


def test_flag_on_research_message_is_logged_via_journal(tmp_path):
    """A flag-on cycle with a mocked Alpaca client MUST persist a populated
    RESEARCH_VIEW message with the expected signal fields."""
    db = tmp_path / "t.db"
    conn = connect(str(db))
    init_db(conn)
    try:
        now = datetime.now(timezone.utc)
        arts = [
            {
                "published_at": (now - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "headline": "NVDA beat",
                "summary": "strong demand",
                "symbols": ["NVDA"],
            },
        ]
        fake_client = MagicMock()
        fake_client.fetch_news.return_value = arts
        with patch("src.agents.nodes.research.AlpacaDataClient", return_value=fake_client):
            node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
            state = _state(("NVDA",))
            out = node(state)
        assert "agent_messages" in out
        msg = out["agent_messages"][0]
        state.agent_messages = [msg]
        journal = DecisionJournal(conn, table="decision_journal", run_mode="paper")
        journal.upsert(state)
        row = journal.get(state.decision_id)
        assert row is not None
        msgs = row["agent_messages_json"]
        assert len(msgs) == 1
        m = msgs[0]
        assert m["message_type"] == "RESEARCH_VIEW"
        assert m["payload"]["signal"]["feature_flag_state"] == "news-on"
        assert m["payload"]["signal"]["n_articles"] == 1
        assert m["payload"]["signal"]["n_symbols_with_news"] == 1
    finally:
        conn.close()


def test_alpaca_failure_message_is_logged_with_risk(tmp_path):
    """An Alpaca failure during the research node MUST still be logged so
    debug history is complete (US1 acceptance scenario 4)."""
    db = tmp_path / "t.db"
    conn = connect(str(db))
    init_db(conn)
    try:
        from src.agents.alpaca_data import AlpacaDataError
        fake_client = MagicMock()
        fake_client.fetch_news.side_effect = AlpacaDataError("401 unauthorized")
        with patch("src.agents.nodes.research.AlpacaDataClient", return_value=fake_client):
            node = build_node(_NoOpLLM(), {"agents": {"research": {"enabled": True}}}, None)
            state = _state(("NVDA",))
            out = node(state)
        msg = out["agent_messages"][0]
        state.agent_messages = [msg]
        journal = DecisionJournal(conn, table="decision_journal", run_mode="paper")
        journal.upsert(state)
        row = journal.get(state.decision_id)
        msgs = row["agent_messages_json"]
        assert "research_alpaca_unreachable" in msgs[0]["payload"]["signal"].get("risks", [])
    finally:
        conn.close()
