"""Volatility Agent.

Compares predicted realized vol to historical realized vol; outputs an
expected-vol edge and confidence. No Greeks today (no IV data); fields are
left nullable for forward compatibility.
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
        "Derive a volatility view: predicted-RV vs historical-RV per symbol, "
        "expected edge, and confidence."
    )

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        payload = {
            "thresholds": config.get("thresholds", {}),
            "underlyings": [u.model_dump(mode="json") for u in snap.underlyings],
            "options": [o.model_dump(mode="json") for o in snap.options[:50]],
        }
        obs = _llm_call(llm, "volatility_agent", role, payload, AgentObservation)
        obs = obs.model_copy(update={"message_type": MessageType.VOLATILITY_VIEW})
        msg = _to_message(obs, state.decision_id, "volatility_agent", "supervisor")
        return AgentResult(observations=[obs], messages=[msg]).as_update()

    return node
