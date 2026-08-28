"""Portfolio Agent.

Reads current positions from the snapshot, computes concentration and
exposure, returns an AgentObservation summarizing portfolio risk. Never
issues trades; risk limits are enforced downstream.
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
        "Summarize the current portfolio: positions, concentration, exposure, "
        "and any concentration risks relative to the risk limits."
    )

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        payload = {
            "positions": [p.model_dump(mode="json") for p in snap.portfolio],
            "account_equity": snap.account_equity,
            "account_cash": snap.account_cash,
            "risk_limits": {
                "max_open_positions": risk_limits.max_open_positions,
                "max_gross_exposure_usd": risk_limits.max_gross_exposure_usd,
                "max_concentration_per_underlying": risk_limits.max_concentration_per_underlying,
            },
        }
        obs = _llm_call(llm, "portfolio_agent", role, payload, AgentObservation)
        obs = obs.model_copy(update={"message_type": MessageType.PORTFOLIO_VIEW})
        msg = _to_message(obs, state.decision_id, "portfolio_agent", "supervisor")
        return AgentResult(observations=[obs], messages=[msg]).as_update()

    return node
