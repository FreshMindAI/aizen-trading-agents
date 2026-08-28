"""Directional Agent.

Combines Phase-1 direction probability with GNN directional bias. Returns
an AgentObservation only.
"""

from __future__ import annotations

from typing import Any

from ..protocol import (
    AgentObservation,
    DecisionState,
    MessageType,
)
from ._common import AgentResult, _llm_call, _to_message


def build_node(llm, config: dict[str, Any], risk_limits):
    role = (
        "Combine ML direction_probability and GNN directional_bias into a "
        "single per-underlying bias with confidence."
    )
    thresh = (config.get("thresholds") or {}).get("direction_prob_min", 0.55)

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        payload = {
            "direction_prob_min": thresh,
            "underlyings": [u.model_dump(mode="json") for u in snap.underlyings],
        }
        obs = _llm_call(llm, "direction_agent", role, payload, AgentObservation)
        obs = obs.model_copy(update={"message_type": MessageType.DIRECTION_VIEW})
        msg = _to_message(obs, state.decision_id, "direction_agent", "supervisor")
        return AgentResult(observations=[obs], messages=[msg]).as_update()

    return node
