"""Regression tests for scripts/push_journal_to_hf.py.

The cloud cron's local decision_journal is wiped on every GitHub
Actions invocation - the HF dataset is the only place the trading
signal survives. We must guarantee:

  1. LLM transcript columns (agent_observations_json,
     agent_messages_json) are REDACTED by default. The trading
     signal is meant to be public-friendly; the LLM prompt/response
     envelopes are not.
  2. The opt-in flag --include-llm-transcripts preserves the raw
     LLM content for the user who explicitly wants it.
  3. The signal JSON columns (strategy_proposal_json,
     market_snapshot_json, ml_prediction_json, etc.) are kept in
     BOTH modes - the upload is useless without them.
  4. The HF token is read from env and never echoed, even on
     error.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import push_journal_to_hf as pjhf  # noqa: E402


# ---- fixtures --------------------------------------------------------

@pytest.fixture
def journal_db(tmp_path: Path):
    """A fresh sqlite db with the decision_journal schema and 2 rows.

    Row 1: a real cycle with non-empty LLM transcript blobs.
    Row 2: another cycle.
    """
    db = tmp_path / "trading.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE decision_journal (
            decision_id          TEXT PRIMARY KEY,
            timestamp            TEXT NOT NULL,
            completed_at         TEXT,
            market_state_hash    TEXT NOT NULL DEFAULT '',
            schema_version       TEXT NOT NULL DEFAULT '1.0',
            run_mode             TEXT NOT NULL DEFAULT 'paper',
            underlying_focus     TEXT,
            final_action         TEXT NOT NULL DEFAULT 'NO_TRADE',
            outcome_label        TEXT,
            realized_pnl         REAL,
            market_snapshot_json TEXT NOT NULL,
            ml_prediction_json   TEXT NOT NULL,
            gnn_output_json      TEXT NOT NULL DEFAULT '{}',
            topology_version     TEXT,
            agent_messages_json  TEXT NOT NULL DEFAULT '[]',
            agent_observations_json TEXT NOT NULL DEFAULT '[]',
            strategy_proposal_json TEXT NOT NULL DEFAULT 'null',
            selected_strategy_json TEXT NOT NULL DEFAULT 'null',
            risk_decision_json   TEXT NOT NULL DEFAULT 'null',
            order_intent_json    TEXT NOT NULL DEFAULT 'null',
            execution_result_json TEXT NOT NULL DEFAULT 'null',
            model_versions       TEXT NOT NULL DEFAULT '[]',
            created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
    """)
    rows = [
        (
            "d-001", "2026-09-01T13:30:00Z", "META", "PROCEED",
            json.dumps({"ticker": "META", "price": 575.0}),  # market_snapshot
            json.dumps({"ticker": "META", "pp": 0.71}),      # ml_prediction
            json.dumps([                                    # strategy_proposal
                {"underlying": "META",
                 "legs": [{"action": "buy", "qty": 1, "expiry": "2026-09-02"}]}
            ]),
            json.dumps([                                    # agent_observations
                {"agent": "regime", "note": "trending-up",
                 "_raw_prompt": "what is the regime?",
                 "_raw_response": "trending up because RSI=70"}
            ]),
            json.dumps([                                    # agent_messages
                {"role": "assistant", "content": "buy META call"}
            ]),
        ),
        (
            "d-002", "2026-09-01T14:00:00Z", "AAPL", "NO_TRADE",
            json.dumps({"ticker": "AAPL", "price": 220.0}),
            json.dumps({"ticker": "AAPL", "pp": 0.42}),
            json.dumps([{"underlying": "AAPL", "legs": []}]),
            json.dumps([{"agent": "regime", "note": "sideways"}]),
            json.dumps([{"role": "assistant", "content": "no trade"}]),
        ),
    ]
    for r in rows:
        conn.execute(
            "INSERT INTO decision_journal "
            "(decision_id, timestamp, underlying_focus, final_action, "
            " market_snapshot_json, ml_prediction_json, "
            " strategy_proposal_json, agent_observations_json, "
            " agent_messages_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            r,
        )
    conn.commit()
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ---- 1. Default redaction ---------------------------------------------

def test_default_redacts_llm_transcript_columns(journal_db):
    """By default, the trading-signal columns are kept verbatim and
    the LLM transcript columns are replaced with a redaction marker.
    The prompt/response content MUST NOT survive."""
    rows = pjhf._new_decisions(journal_db, since=None)
    assert len(rows) == 2
    for row in rows:
        obs = json.loads(row["agent_observations_json"])
        msgs = json.loads(row["agent_messages_json"])
        # The original LLM content must be gone.
        for o in obs:
            assert "_raw_prompt" not in o, (
                f"LLM prompt leaked into upload: {o}"
            )
            assert "_raw_response" not in o
        for m in msgs:
            assert m.get("content") != "buy META call"
            assert m.get("content") != "no trade"
        # The redaction marker is present.
        assert all(o.get("_redacted") == "llm_transcript" for o in obs)
        assert all(o.get("column") for o in obs)


def test_default_keeps_signal_columns(journal_db):
    """The trading-signal JSON columns (strategy_proposal,
    market_snapshot, ml_prediction) MUST survive redaction - the
    upload is useless without them."""
    rows = pjhf._new_decisions(journal_db, since=None)
    row = rows[0]
    sp = json.loads(row["strategy_proposal_json"])
    assert sp[0]["underlying"] == "META"
    assert sp[0]["legs"][0]["qty"] == 1
    ms = json.loads(row["market_snapshot_json"])
    assert ms["ticker"] == "META"
    assert ms["price"] == 575.0
    ml = json.loads(row["ml_prediction_json"])
    assert ml["pp"] == 0.71


# ---- 2. Opt-in flag preserves LLM content ----------------------------

def test_include_llm_transcripts_flag_preserves_prompts(journal_db):
    """With --include-llm-transcripts, the LLM prompt/response
    content is preserved verbatim."""
    rows = pjhf._new_decisions(journal_db, since=None,
                               include_llm_transcripts=True)
    assert len(rows) == 2
    obs = json.loads(rows[0]["agent_observations_json"])
    assert any(o.get("_raw_prompt") == "what is the regime?" for o in obs), (
        "with --include-llm-transcripts the prompt should survive"
    )
    msgs = json.loads(rows[0]["agent_messages_json"])
    assert any(m.get("content") == "buy META call" for m in msgs)


# ---- 3. Since-filter still works -------------------------------------

def test_since_filter_excludes_older_rows(journal_db):
    """The state file's last_uploaded_ts filters out rows whose
    timestamp is <= since."""
    rows = pjhf._new_decisions(journal_db, since="2026-09-01T13:45:00Z")
    assert len(rows) == 1, f"expected only d-002, got {[r['decision_id'] for r in rows]}"
    assert rows[0]["decision_id"] == "d-002"


# ---- 4. Token never echoed -------------------------------------------

def test_resolve_token_rejects_garbage(monkeypatch):
    """An obviously-wrong token (does not start with 'hf_') MUST
    raise - the script should never silently use a bogus credential."""
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token-12345")
    with pytest.raises(ValueError, match="does not start with"):
        pjhf._resolve_token()


def test_resolve_token_does_not_echo_token(monkeypatch, caplog):
    """The token string MUST NOT appear in any log output."""
    monkeypatch.setenv("HF_TOKEN", "hf_TESTTOKEN_DO_NOT_LOG_zzzzzzzzz")
    token = pjhf._resolve_token()
    assert token == "hf_TESTTOKEN_DO_NOT_LOG_zzzzzzzzz"
    # caplog.text is the union of all captured records; we never
    # want the token to leak through a logger.
    assert "hf_TESTTOKEN_DO_NOT_LOG_zzzzzzzzz" not in caplog.text
