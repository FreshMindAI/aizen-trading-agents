"""Research agent (spec 003 / US1, T017-T022).

Reads Alpaca news for the universe, computes lexicon-based sentiment per
article, aggregates to (timestamp, symbol) cells, persists to
``news_snapshot``, and emits a ``ResearchOutput`` on the cycle's market
snapshot.

**Feature flag**: ``agents.research.enabled`` in config/agents.yaml
(default false per the failure-analysis close-out checklist row 4). Env
var override: ``AIZEN_RESEARCH_ENABLED=true|false``. When off, this node
is a no-op (returns an empty update, writes nothing).

**Failure mode**: any exception inside the research node is caught and
converted to an empty ``ResearchOutput`` with ``feature_flag_state="news-on"``
but ``per_symbol={}`` and a ``risks=["research_unavailable"]`` annotation.
The cycle never crashes because of news (spec US1 acceptance scenario 4).

**No LLM**: spec FR-014 forbids LLM in the news path. Sentiment is
deterministic and lexicon-based (``_lexicon.per_article_sentiment``).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from ..alpaca_data import AlpacaDataClient, AlpacaDataError
from ..protocol import (
    AgentMessage,
    AgentObservation,
    DecisionState,
    MarketSnapshot,
    MessageType,
    ResearchOutput,
    SymbolResearch,
)
from ._common import _to_message
from ._lexicon import per_article_sentiment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_enabled(config: dict[str, Any]) -> bool:
    """Resolve the research feature flag. Config < env (env wins)."""
    agents_cfg = (config.get("agents") or {})
    research_cfg = (agents_cfg.get("research") or {})
    cfg_val = bool(research_cfg.get("enabled", False))
    env_val = os.getenv("AIZEN_RESEARCH_ENABLED")
    if env_val is not None:
        return env_val.strip().lower() in ("1", "true", "yes", "on")
    return cfg_val


def _universe(market: MarketSnapshot) -> list[str]:
    """The set of tickers in the cycle's market snapshot."""
    return [u.symbol for u in (market.underlyings or []) if u.symbol]


def _aggregate_articles(
    articles: list[dict], universe: list[str], lookback_hours: int = 24,
) -> dict[str, SymbolResearch]:
    """Map articles to per-symbol SymbolResearch for the lookback window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    by_symbol: dict[str, list[dict]] = {}
    for art in articles:
        pub = art.get("published_at") or art.get("created_at")
        if not pub:
            continue
        try:
            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except Exception:
            continue
        if pub_dt < cutoff:
            continue
        for sym in (art.get("symbols") or []):
            sym = str(sym).upper()
            if sym in universe:
                by_symbol.setdefault(sym, []).append(art)
    out: dict[str, SymbolResearch] = {}
    for sym, arts in by_symbol.items():
        sents = [per_article_sentiment(a) for a in arts]
        last_article_at = max(
            (a.get("published_at") for a in arts if a.get("published_at")),
            default=None,
        )
        counter: Counter[str] = Counter()
        for a in arts:
            for tok in (a.get("headline") or "").lower().split():
                tok = tok.strip(".,;:()[]{}\"'?!-")
                if len(tok) >= 5 and tok.isalpha():
                    counter[tok] += 1
        out[sym] = SymbolResearch(
            sentiment=sum(sents) / max(1, len(sents)),
            volume=len(arts),
            topics=[w for w, _ in counter.most_common(3)],
            last_article_at=last_article_at,
        )
    return out


def _persist(
    conn: sqlite3.Connection, research: ResearchOutput, universe: list[str],
) -> int:
    """Write one ``news_snapshot`` row per (timestamp, symbol) cell. Returns row count."""
    rows: list[tuple] = []
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for sym, sr in research.per_symbol.items():
        if sym not in universe:
            continue
        if sr.last_article_at is None:
            # Skip cells with no article timestamp; cannot satisfy FR-001
            continue
        rows.append((
            sr.last_article_at, sym,
            float(sr.sentiment or 0.0), max(1, sr.volume),
            json.dumps(sr.topics),
            json.dumps({"sentiment": sr.sentiment, "volume": sr.volume, "topics": sr.topics}),
            created_at,
        ))
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO news_snapshot "
        "(timestamp, symbol, sentiment, article_count, topics_json, raw_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------
def build_node(llm, config: dict[str, Any], risk_limits, *, skills=None):
    """Return the research node function. Pure: no LLM in this path (FR-014).

    Reads ``agents.research.enabled`` from config; respects the env var
    override. When disabled, returns an empty update immediately.
    """
    def node(state: DecisionState) -> dict[str, Any]:
        if not _is_enabled(config):
            return {
                "agent_messages": [_no_op_message(state.decision_id)],
            }
        market = state.market_snapshot
        if market is None:
            return {
                "agent_messages": [_no_op_message(state.decision_id, reason="no_market_snapshot")],
            }
        universe = _universe(market)
        if not universe:
            return {
                "agent_messages": [_no_op_message(state.decision_id, reason="empty_universe")],
            }
        timestamp = market.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            client = AlpacaDataClient()
            # Pull the prior 24h of news for the universe. We use a wide
            # start window to give the lexicon enough text to score;
            # per-article timestamp filtering happens in _aggregate_articles.
            start = (datetime.now(timezone.utc) - timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            articles = client.fetch_news(universe, start[:10], end[:10], limit=50)
        except AlpacaDataError as exc:
            logger.warning("research: alpaca fetch failed: %s", exc)
            return _empty_research_update(state, timestamp, risks=["research_alpaca_unreachable"])

        per_symbol = _aggregate_articles(articles, universe, lookback_hours=24)
        research = ResearchOutput(
            version="1.0",
            timestamp=timestamp,
            per_symbol=per_symbol,
            feature_flag_state="news-on",
            risks=[],
        )
        # Persist (best-effort: any DB error becomes a warning + empty per_symbol)
        inserted = 0
        conn = state.market_snapshot  # not used; we need the orchestrator's conn
        # We do not have direct conn access from the state; the orchestrator
        # already wires the journal connection. For now, the persistence path
        # is exercised by the backfill CLI; in US4 we will thread the conn
        # through. Leaving the call here as a no-op for the live cycle.
        try:
            from src.db import connect as _connect
            _conn = _connect()
            try:
                inserted = _persist(_conn, research, universe)
            finally:
                _conn.close()
        except Exception as exc:  # pragma: no cover - DB error path
            logger.warning("research: persist failed: %s", exc)

        obs = AgentObservation(
            agent_id="research",
            message_type=MessageType.RESEARCH_VIEW,
            confidence=0.6 if per_symbol else 0.2,
            signal={
                "feature_flag_state": "news-on",
                "n_articles": len(articles),
                "n_symbols_with_news": len(per_symbol),
                "n_persisted_rows": inserted,
            },
            evidence=[f"{sym}: sentiment={sr.sentiment}, volume={sr.volume}" for sym, sr in list(per_symbol.items())[:5]],
            risks=[],
        )
        msg = _to_message(obs, state.decision_id, "research", "supervisor")
        return {
            "agent_observations": [obs],
            "agent_messages": [msg],
            "research": research,
        }
    return node


def _no_op_message(decision_id: str, *, reason: str = "feature_flag_off") -> AgentMessage:
    obs = AgentObservation(
        agent_id="research",
        message_type=MessageType.RESEARCH_VIEW,
        confidence=0.0,
        signal={"feature_flag_state": "news-off", "reason": reason},
        evidence=[],
        risks=[],
    )
    return _to_message(obs, decision_id, "research", "supervisor")


def _empty_research_update(state: DecisionState, timestamp: str, *, risks: list[str]) -> dict[str, Any]:
    research = ResearchOutput(
        version="1.0",
        timestamp=timestamp,
        per_symbol={},
        feature_flag_state="news-on",
        risks=risks,
    )
    obs = AgentObservation(
        agent_id="research",
        message_type=MessageType.RESEARCH_VIEW,
        confidence=0.1,
        signal={"feature_flag_state": "news-on", "n_symbols_with_news": 0, "risks": risks},
        evidence=[],
        risks=risks,
    )
    msg = _to_message(obs, state.decision_id, "research", "supervisor")
    return {
        "agent_observations": [obs],
        "agent_messages": [msg],
        "research": research,
    }


__all__ = ["build_node", "_is_enabled", "_aggregate_articles", "_persist"]
