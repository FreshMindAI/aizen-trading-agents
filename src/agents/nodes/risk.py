"""Deterministic Risk Agent (LangGraph node adapter).

Wraps `src.agents.risk.evaluate` so the LangGraph state machine treats the
risk check like any other node. If the OrderIntent is missing we emit a
REJECT with a single check explaining why.
"""

from __future__ import annotations

from typing import Any

from ..protocol import (
    DecisionState,
    RiskAction,
    RiskCheck,
    RiskDecision,
)
from ..risk import evaluate


def build_node(llm, config: dict[str, Any], risk_limits, *, skills=None):
    # `llm` is unused - the risk engine is deterministic by design.
    _ = llm

    def node(state: DecisionState) -> dict[str, Any]:
        intent = state.order_intent
        if intent is None:
            decision = RiskDecision(
                decision=RiskAction.REJECT,
                approved_quantity=0,
                max_loss=0.0,
                checks=[RiskCheck(name="no_intent", passed=False, detail="no order intent")],
                reasons=["no order intent to validate"],
            )
            return {"risk_decision": decision}

        decision = evaluate(intent, state.market_snapshot, risk_limits)
        return {"risk_decision": decision}

    return node
