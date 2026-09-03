"""Tests for DecisionState.mcp_preflight field.

The execution node (:mod:`src.agents.nodes.execution`) writes a
``mcp_preflight`` summary onto the per-cycle state so the trace +
journal can confirm the in-process MCP SkillRegistry was exercised.
``DecisionState`` uses ``extra=forbid`` (StrictModel), so the field
must be declared on the model — otherwise every cycle that reaches
the execution node crashes with
``ValidationError: Extra inputs are not permitted``.

These tests pin the schema:
  1. A fresh DecisionState accepts ``mcp_preflight`` as an optional
     dict and the round-trip preserves the value.
  2. The extra="forbid" guarantee still holds for unrelated fields
     (catches a regression that loosens the model too far).
  3. The same shape can be deserialized from a dict that would
     have crashed before the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.protocol import DecisionState  # noqa: E402


def test_decision_state_accepts_mcp_preflight_field():
    """The mcp_preflight field MUST be declared on DecisionState so
    the execution node can write it without raising ValidationError."""
    state = DecisionState()
    state.mcp_preflight = {
        "called": True,
        "n_positions": 3,
    }
    assert state.mcp_preflight == {"called": True, "n_positions": 3}


def test_decision_state_mcp_preflight_optional():
    """Default state must NOT require mcp_preflight — NO_TRADE cycles
    that never reach the execution node should still construct."""
    state = DecisionState()
    assert state.mcp_preflight is None


def test_decision_state_round_trip_mcp_preflight():
    """A state serialized with mcp_preflight and re-parsed keeps the
    value. This is the exact shape the LangGraph state machine uses
    when merging node return dicts into the state."""
    original = DecisionState()
    original.mcp_preflight = {"called": True, "n_positions": 0}
    payload = original.model_dump()
    restored = DecisionState.model_validate(payload)
    assert restored.mcp_preflight == {"called": True, "n_positions": 0}


def test_decision_state_still_rejects_unknown_fields():
    """Regression guard: the mcp_preflight fix must not loosen the
    StrictModel extra=forbid guarantee for unrelated fields."""
    payload = DecisionState().model_dump()
    payload["not_a_real_field"] = "boom"
    with pytest.raises(Exception, match="not_a_real_field|Extra"):
        DecisionState.model_validate(payload)


def test_decision_state_accepts_error_marker_in_mcp_preflight():
    """When the MCP read fails, the execution node writes
    ``{"called": True, "error": "..."}``. The schema must accept this
    shape so the operator still gets the failure signal."""
    state = DecisionState()
    state.mcp_preflight = {
        "called": True,
        "error": "ConnectionError: alpaca unreachable",
    }
    assert "error" in state.mcp_preflight
