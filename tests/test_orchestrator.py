"""End-to-end orchestrator test (dry-run, mock LLM)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.agents.graph import Orchestrator
from src.agents.journal import DecisionJournal
from src.agents.protocol import RiskAction
from src.db import connect


@pytest.fixture
def tmp_db(monkeypatch):
    """A SQLite database with every sql/*.sql applied (views + journal)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row
    sql_dir = Path(__file__).resolve().parents[1] / "sql"
    for script in sorted(sql_dir.glob("*.sql")):
        conn.executescript(script.read_text(encoding="utf-8"))
    conn.commit()
    yield conn
    conn.close()
    Path(tmp.name).unlink(missing_ok=True)


def test_orchestrator_runs_a_cycle_with_dry_run(monkeypatch, tmp_db):
    """The orchestrator must complete a cycle, write a journal row, and
    return a populated DecisionState. ML/GNN may be empty in a fresh test
    DB; agents must still produce observations (even if no candidates)."""
    # Force run_mode=dry-run via env so the execution agent never touches Alpaca.
    monkeypatch.setenv("RUN_MODE", "dry-run")
    # Use a separate project root for config.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()

    orch = Orchestrator(conn=tmp_db)
    state = orch.run_cycle()

    assert state.decision_id
    assert state.cycle_completed_at
    # At minimum the supervisor / execution / risk agents left observations.
    assert len(state.agent_observations) >= 1
    # Cycle was journaled.
    row = orch.journal.get(state.decision_id)
    assert row is not None
    assert row["final_action"] in ("PROCEED", "REJECT", "NO_TRADE", "REDUCE")


def test_orchestrator_falls_back_to_sequential(monkeypatch, tmp_db):
    """Even without langgraph installed, the orchestrator must run end-to-end."""
    monkeypatch.setenv("RUN_MODE", "dry-run")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()
    # Force the LangGraph import to fail by hiding the module so the
    # orchestrator's `try: from langgraph.graph` raises ImportError.
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **kw):
        if name == "langgraph.graph":
            raise ImportError("simulated: langgraph not installed")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    orch = Orchestrator(conn=tmp_db)
    state = orch.run_cycle()
    assert state.cycle_completed_at
