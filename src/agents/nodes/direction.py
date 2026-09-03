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
from ._common import AgentResult, _llm_call, _log_agent_observation, _to_message


def build_node(llm, config: dict[str, Any], risk_limits, *, skills=None):
    role = (
        "Combine ML direction_probability and GNN directional_bias into a "
        "single per-underlying bias with confidence."
    )
    thresh = (config.get("thresholds") or {}).get("direction_prob_min", 0.55)
    # The configured 0.55 is unreachable for a model trained on a 30%-up market.
    # Read the artifact's base_rate at module import (see strategy_selector)
    # and report a regime-relative effective threshold so the LLM sees the
    # real gate, not a stale one.
    try:
        from .strategy_selector import _BASE_RATE, _BASE_RATE_CUSHION
        if _BASE_RATE is not None:
            thresh = min(0.95, _BASE_RATE + _BASE_RATE_CUSHION)
    except Exception:
        pass

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        # Surface the GNN bias + centrality alongside the Phase-1
        # direction probability so the LLM (or the deterministic
        # scoring layer) can blend them. The GNN signal lives at
        # state.gnn_output["node_features"][symbol].
        gnn_features: dict[str, dict[str, Any]] = {}
        try:
            gnn_features = (state.gnn_output or {}).get("node_features", {}) or {}
        except Exception:
            gnn_features = {}
        underlyings = []
        for u in snap.underlyings:
            blob = u.model_dump(mode="json")
            gnn_node = gnn_features.get(u.symbol, {}) if gnn_features else {}
            blob["gnn_directional_bias"] = gnn_node.get("bias")
            blob["gnn_centrality"] = gnn_node.get("centrality")
            underlyings.append(blob)
        payload = {
            "direction_prob_min": thresh,
            "gnn_model_version": (state.gnn_output or {}).get("model_version"),
            "underlyings": underlyings,
        }
        obs = _llm_call(llm, "direction_agent", role, payload, AgentObservation)
        obs = obs.model_copy(update={"message_type": MessageType.DIRECTION_VIEW})
        msg = _to_message(obs, state.decision_id, "direction_agent", "supervisor")
        _log_agent_observation("direction_agent", obs)
        return AgentResult(observations=[obs], messages=[msg]).as_update()

    return node
