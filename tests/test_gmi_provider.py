"""Smoke test: run a full orchestrator cycle against GMI-serving.

Usage:
    ANTHROPIC_AUTH_TOKEN=... python -m pytest tests/test_gmi_provider.py -v -s
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.agents.llm import LLMMessage, LLMRequest, get_provider
from src.agents.protocol import AgentObservation


pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_AUTH_TOKEN"),
    reason="ANTHROPIC_AUTH_TOKEN not set - skipping GMI integration test",
)


def test_gmi_basic_chat():
    p = get_provider(
        "anthropic",
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.gmi-serving.com"),
        default_model=os.getenv("ANTHROPIC_MODEL", "MiniMaxAI/MiniMax-M3"),
    )
    assert p.api_key or p.auth_token
    req = LLMRequest(
        system="You are the test agent.",
        messages=[LLMMessage(role="user", content="Reply with one short sentence.")],
        max_tokens=32,
    )
    resp = p.complete(req)
    assert resp.text
    assert resp.usage["output_tokens"] > 0
    print(f"  GMI response: {resp.text[:80]!r} (model={resp.model})")


def test_gmi_pydantic_roundtrip():
    p = get_provider(
        "anthropic",
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.gmi-serving.com"),
        default_model=os.getenv("ANTHROPIC_MODEL", "MiniMaxAI/MiniMax-M3"),
    )
    req = LLMRequest(
        system=(
            "You are the test_agent. Return ONLY a JSON object matching the schema. "
            "No prose, no markdown."
        ),
        messages=[LLMMessage(role="user", content=(
            '{"agent_id":"test_agent","message_type":"DIRECTION_VIEW",'
            '"confidence":0.7,"signal":{"direction_edge":0.4},'
            '"evidence":["gmi test"],"risks":[],"data_version":"test-1",'
            '"model_versions":["MiniMax-M3"]}'
        ))],
        max_tokens=128,
        metadata={"_response_model": "AgentObservation"},
    )
    obs = p.complete_as(req, AgentObservation)
    assert obs.confidence > 0
    assert obs.agent_id
    print(f"  parsed observation: confidence={obs.confidence:.2f} agent_id={obs.agent_id}")


def test_gmi_orchestrator_full_cycle():
    """End-to-end: orchestrator runs against GMI and writes a journal row."""
    pytest.importorskip("pandas")
    from src.agents.graph import Orchestrator

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row
    sql_dir = Path(__file__).resolve().parents[1] / "sql"
    for script in sorted(sql_dir.glob("*.sql")):
        conn.executescript(script.read_text(encoding="utf-8"))
    conn.commit()

    os.environ["RUN_MODE"] = "dry-run"
    os.environ["AIZEN_LLM_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_BASE_URL"] = os.getenv(
        "ANTHROPIC_BASE_URL", "https://api.gmi-serving.com"
    )
    os.environ["ANTHROPIC_MODEL"] = os.getenv(
        "ANTHROPIC_MODEL", "MiniMaxAI/MiniMax-M3"
    )

    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()

    orch = Orchestrator(conn=conn)
    state = orch.run_cycle()
    print(f"  decision_id={state.decision_id}")
    print(f"  observations={len(state.agent_observations)}")
    print(f"  final_action={state.final_action}")
    assert state.cycle_completed_at
    assert len(state.agent_observations) >= 1

    # Journal row exists.
    row = orch.journal.get(state.decision_id)
    assert row is not None
    assert row["final_action"] in ("PROCEED", "REJECT", "NO_TRADE", "REDUCE")

    conn.close()
    Path(tmp.name).unlink(missing_ok=True)
