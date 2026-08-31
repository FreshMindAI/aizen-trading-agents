"""Tests for the Alpaca MCP server + per-agent skill registry.

Covers:
  * Tool registration (list_tools shape, JSON-schema presence).
  * Dry-run behaviour for ``submit_order`` / ``cancel_order``.
  * Argument validation (missing fields, wrong types, unknown tools).
  * The per-agent ``SkillRegistry`` enforces the safety contract:
    the execution agent can submit; the research agent cannot.
  * ``AlpacaMCPServer.call`` raises ``ToolError`` for unknown tools.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.mcp import AlpacaMCPServer, ToolError  # noqa: E402
from src.agents.mcp.skills import (  # noqa: E402
    AGENT_SKILLS,
    SkillRegistry,
    build_skill_registry,
)


# ---------------------------------------------------------------------------
# Tool registration + JSON-schema
# ---------------------------------------------------------------------------
def test_list_tools_returns_mcp_shaped_dicts():
    srv = AlpacaMCPServer(run_mode="dry-run")
    tools = srv.list_tools()
    assert tools, "expected at least one tool"
    for t in tools:
        assert {"name", "description", "inputSchema"} <= set(t.keys())
        assert t["inputSchema"]["type"] == "object"
    # All required tools are present.
    names = {t["name"] for t in tools}
    assert {
        "get_account", "get_positions", "get_order", "list_orders",
        "submit_order", "cancel_order", "get_news",
    } <= names


def test_call_unknown_tool_raises():
    srv = AlpacaMCPServer(run_mode="dry-run")
    with pytest.raises(ToolError, match="unknown tool"):
        srv.call("nope")


# ---------------------------------------------------------------------------
# Dry-run: submit_order + cancel_order
# ---------------------------------------------------------------------------
def test_submit_order_dry_run_returns_fake_fill():
    srv = AlpacaMCPServer(run_mode="dry-run")
    out = srv.call("submit_order", {
        "symbol": "NVDA", "qty": 5, "side": "buy", "type": "market",
        "time_in_force": "day",
    })
    assert out["dry_run"] is True
    assert out["symbol"] == "NVDA"
    assert out["qty"] == 5
    assert out["side"] == "buy"
    assert out["status"] == "filled"   # market orders fill immediately
    assert out["filled_qty"] == 5


def test_submit_order_dry_run_limit_accepted_not_filled():
    srv = AlpacaMCPServer(run_mode="dry-run")
    out = srv.call("submit_order", {
        "symbol": "TSLA", "qty": 1, "side": "buy", "type": "limit",
        "limit_price": 300.0,
    })
    assert out["dry_run"] is True
    assert out["status"] == "accepted"  # not yet filled
    assert out["filled_avg_price"] == 300.0


def test_submit_order_missing_required_field_raises():
    srv = AlpacaMCPServer(run_mode="dry-run")
    with pytest.raises(ToolError, match="symbol"):
        srv.call("submit_order", {"qty": 1, "side": "buy"})


def test_submit_order_bad_side_raises():
    srv = AlpacaMCPServer(run_mode="dry-run")
    with pytest.raises(ToolError, match="side"):
        srv.call("submit_order", {"symbol": "NVDA", "qty": 1, "side": "long"})


def test_cancel_order_dry_run():
    srv = AlpacaMCPServer(run_mode="dry-run")
    out = srv.call("cancel_order", {"order_id": "abc-123"})
    assert out["status"] == "canceled"
    assert out["dry_run"] is True


# ---------------------------------------------------------------------------
# Skill registry: per-agent allowed tools
# ---------------------------------------------------------------------------
def test_execution_agent_can_submit_but_research_cannot():
    """The safety contract: only the execution agent can call submit_order."""
    server = AlpacaMCPServer(run_mode="dry-run")
    execution = build_skill_registry("execution", server)
    research = build_skill_registry("research", server)
    risk = build_skill_registry("risk", server)

    # Execution has submit_order.
    assert "submit_order" in execution.allowed_tool_names()
    assert "cancel_order" in execution.allowed_tool_names()

    # Research does NOT have submit_order (read-only).
    assert "submit_order" not in research.allowed_tool_names()
    # Research has market-data tools.
    assert "get_news" in research.allowed_tool_names()
    # Research has no account access.
    assert "get_account" not in research.allowed_tool_names()

    # Risk has only account/positions (no submit, no market data).
    assert "submit_order" not in risk.allowed_tool_names()
    assert "get_news" not in risk.allowed_tool_names()
    assert "get_account" in risk.allowed_tool_names()


def test_skill_registry_blocks_disallowed_tool():
    """A research agent attempting to call submit_order gets a clean refusal."""
    server = AlpacaMCPServer(run_mode="dry-run")
    research = build_skill_registry("research", server)
    with pytest.raises(ToolError, match="not allowed to call 'submit_order'"):
        research.call("submit_order", {"symbol": "NVDA", "qty": 1, "side": "buy"})


def test_skill_registry_allows_listed_tool():
    """The execution agent can submit a dry-run order via its registry."""
    server = AlpacaMCPServer(run_mode="dry-run")
    execution = build_skill_registry("execution", server)
    out = execution.call("submit_order", {
        "symbol": "AAPL", "qty": 2, "side": "buy", "type": "market",
    })
    assert out["symbol"] == "AAPL"
    assert out["dry_run"] is True


def test_skill_registry_lists_correct_tool_schemas():
    """list_tools() on a registry returns only the agent's allowed subset."""
    server = AlpacaMCPServer(run_mode="dry-run")
    execution = build_skill_registry("execution", server)
    tools = execution.list_tools()
    names = {t["name"] for t in tools}
    # Execution sees its 6 allowed tools, not the research-only get_news.
    assert names == {
        "get_account", "get_positions", "get_order", "list_orders",
        "submit_order", "cancel_order",
    }


# ---------------------------------------------------------------------------
# Skill map coverage: every agent in the graph has a skills entry
# ---------------------------------------------------------------------------
def test_all_graph_agents_have_skills():
    """The static skill map must cover every agent the orchestrator wires up.
    A new agent added to graph.py without a skills entry would be a silent
    capability loss — fail loudly here so the omission is caught in CI."""
    expected = {
        "regime", "direction", "volatility", "options_structure",
        "portfolio", "supervisor", "risk", "execution", "research",
    }
    assert set(AGENT_SKILLS.keys()) >= expected


def test_execution_is_the_only_agent_with_submit():
    """Safety invariant: only the execution agent is allowed to call
    submit_order. If another agent ever needs it, the safety contract
    has changed and this test should be updated with a justification."""
    for agent, tools in AGENT_SKILLS.items():
        if agent == "execution":
            assert "submit_order" in tools, "execution must have submit_order"
        else:
            assert "submit_order" not in tools, (
                f"agent {agent!r} is not allowed to submit orders "
                "(safety contract violated)"
            )
