"""Market Regime Agent.

Detects the current regime (trending / range / high-vol / low-vol / crisis)
from observed realized vol, recent returns, and GNN topology. Returns an
AgentObservation only; never a trade decision.
"""

from __future__ import annotations

from typing import Any

from ..protocol import (
    AgentObservation,
    DecisionState,
    MarketSnapshot,
    MessageType,
    RegimeLabel,
)
from ._common import AgentResult, _llm_call, _to_message


def build_node(llm, config: dict[str, Any], risk_limits):
    role = "Classify the market regime and explain with evidence."

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        payload = _regime_payload(snap)
        obs = _llm_call(llm, "regime_agent", role, payload, AgentObservation)
        # Override message_type to REGIME_VIEW.
        obs = obs.model_copy(update={"message_type": MessageType.REGIME_VIEW})
        msg = _to_message(obs, state.decision_id, "regime_agent", "supervisor")
        return AgentResult(observations=[obs], messages=[msg]).as_update()

    return node


def _regime_payload(snap: MarketSnapshot) -> dict[str, Any]:
    vols: list[float] = []
    rets: list[float] = []
    for u in snap.underlyings:
        if u.predicted_future_realized_vol is not None:
            vols.append(u.predicted_future_realized_vol)
        if u.direction_probability is not None:
            rets.append(u.direction_probability - 0.5)
    avg_vol = sum(vols) / len(vols) if vols else 0.0
    avg_ret = sum(rets) / len(rets) if rets else 0.0
    label = _label_from_metrics(avg_vol, avg_ret)
    return {
        "avg_predicted_vol": avg_vol,
        "avg_directional_bias": avg_ret,
        "n_underlyings": len(snap.underlyings),
        "label_heuristic": label.value,
        "vol_distribution": vols[:25],
    }


def _label_from_metrics(avg_vol: float, avg_ret: float) -> RegimeLabel:
    if avg_vol >= 0.45:
        return RegimeLabel.HIGH_VOLATILITY
    if avg_vol <= 0.10:
        return RegimeLabel.LOW_VOLATILITY
    if avg_ret > 0.05:
        return RegimeLabel.TRENDING_UP
    if avg_ret < -0.05:
        return RegimeLabel.TRENDING_DOWN
    return RegimeLabel.RANGE_BOUND
