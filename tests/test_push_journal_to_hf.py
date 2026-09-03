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


# ---- 5. create_repo on first upload (404 fix) ------------------------

def test_main_creates_dataset_repo_before_upload(tmp_path, monkeypatch):
    """Regression: GH Actions cron runs were failing on the first push
    with ``404 Not Found`` because the dataset repo did not exist.
    ``push_journal_to_hf`` must call ``api.create_repo(exist_ok=True)``
    BEFORE ``upload_file`` so the dataset materializes on first run
    and is a no-op on every subsequent run. Mirrors the pattern in
    ``scripts/deploy_to_hf.py:212``."""
    # Fresh DB with a single decision row so the upload branch runs.
    db = tmp_path / "trading.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE decision_journal (
            decision_id TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            final_action TEXT NOT NULL DEFAULT 'NO_TRADE',
            market_snapshot_json TEXT NOT NULL DEFAULT '{}',
            ml_prediction_json   TEXT NOT NULL DEFAULT '{}',
            strategy_proposal_json TEXT NOT NULL DEFAULT 'null',
            agent_observations_json TEXT NOT NULL DEFAULT '[]',
            agent_messages_json     TEXT NOT NULL DEFAULT '[]'
        );
    """)
    conn.execute(
        "INSERT INTO decision_journal "
        "(decision_id, timestamp, final_action) VALUES (?, ?, ?)",
        ("d-100", "2026-09-03T15:00:00Z", "PROCEED"),
    )
    conn.commit()
    conn.close()

    state = tmp_path / "state.json"  # not present -> since=None
    calls: list[tuple] = []

    class _FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))

    class _FakeHfHub:
        @staticmethod
        def hf_hub_download(**kwargs):  # simulate missing remote
            raise FileNotFoundError("snapshot.jsonl")

    monkeypatch.setenv("HF_TOKEN", "hf_FAKE_FOR_TEST_xxxxxxxxxxxxxxx")
    # The script imports HfApi + hf_hub_download at call time inside
    # main() from the huggingface_hub package, so the module-level
    # `push_journal_to_hf` module has no `HfApi` attribute to patch.
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: _FakeApi())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        _FakeHfHub.hf_hub_download)

    rc = pjhf.main([
        "--db", str(db),
        "--state", str(state),
        "--dataset", "TestUser/aizen-journal-test",
    ])
    assert rc == 0
    # create_repo MUST come first, and MUST use exist_ok=True.
    create_calls = [c for c in calls if c[0] == "create_repo"]
    upload_calls = [c for c in calls if c[0] == "upload_file"]
    assert len(create_calls) == 1, (
        f"expected exactly one create_repo call, got {create_calls}"
    )
    assert create_calls[0][1]["repo_id"] == "TestUser/aizen-journal-test"
    assert create_calls[0][1]["repo_type"] == "dataset"
    assert create_calls[0][1]["exist_ok"] is True
    assert len(upload_calls) == 1
    # Order: create_repo runs BEFORE upload_file (the 404 fix).
    create_idx = next(i for i, c in enumerate(calls) if c[0] == "create_repo")
    upload_idx = next(i for i, c in enumerate(calls) if c[0] == "upload_file")
    assert create_idx < upload_idx, (
        f"create_repo must precede upload_file: {calls}"
    )
    assert state.exists()  # state file written so next run is a no-op


def test_main_propagates_upload_file_404(tmp_path, monkeypatch):
    """If the upload itself still 404s (e.g. write-protected existing
    repo), the script must NOT swallow the error. Silent ignore would
    hide a real auth issue from the operator."""
    db = tmp_path / "trading.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE decision_journal (
            decision_id TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            market_snapshot_json TEXT NOT NULL DEFAULT '{}',
            ml_prediction_json   TEXT NOT NULL DEFAULT '{}',
            strategy_proposal_json TEXT NOT NULL DEFAULT 'null',
            agent_observations_json TEXT NOT NULL DEFAULT '[]',
            agent_messages_json     TEXT NOT NULL DEFAULT '[]'
        );
    """)
    conn.execute(
        "INSERT INTO decision_journal "
        "(decision_id, timestamp) VALUES (?, ?)",
        ("d-200", "2026-09-03T15:30:00Z"),
    )
    conn.commit()
    conn.close()

    class _FakeApi:
        def create_repo(self, **kwargs):
            return {"repo_id": kwargs["repo_id"]}

        def upload_file(self, **kwargs):
            raise RuntimeError("404 Client Error: Repository Not Found")

    import huggingface_hub
    monkeypatch.setenv("HF_TOKEN", "hf_FAKE_FOR_TEST_xxxxxxxxxxxxxxx")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: _FakeApi())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **k: (_ for _ in ()).throw(FileNotFoundError()))

    with pytest.raises(RuntimeError, match="404"):
        pjhf.main([
            "--db", str(db),
            "--state", str(tmp_path / "state.json"),
        ])
