"""Execution Agent.

Validates an OrderIntent one more time (defense in depth), then submits to
Alpaca (or returns a synthetic 'submitted' result in dry-run mode). The
order-validator step ensures the legs, side, quantity, and limit price are
all consistent BEFORE any external call.
"""

from __future__ import annotations

import logging
from typing import Any

from ..alpaca_trading import AlpacaTradingClient
from ..protocol import (
    DecisionState,
    MessageType,
    OrderIntent,
    RiskAction,
    Side,
)
from ._common import _llm_call
from ..protocol import AgentObservation

logger = logging.getLogger(__name__)


def build_node(llm, config: dict[str, Any], risk_limits):
    role = "Validate and submit the OrderIntent through the broker."

    def node(state: DecisionState) -> dict[str, Any]:
        intent = state.order_intent
        if intent is None:
            return _no_intent_result(state)
        if state.risk_decision is None or state.risk_decision.decision == RiskAction.REJECT:
            return _rejected_result(state, intent)

        # Apply REDUCE_SIZE if risk asked for it.
        if state.risk_decision.decision == RiskAction.REDUCE_SIZE:
            intent = intent.model_copy(update={
                "quantity": state.risk_decision.approved_quantity,
            })

        # Defense in depth: validator catches malformed intents even if
        # the supervisor built them.
        if not _validate_intent(intent):
            return _invalid_result(state, intent)

        run_mode = (config.get("run_mode") or "paper").lower()
        if run_mode == "dry-run":
            execution = _dry_run_submission(intent)
        else:
            client = AlpacaTradingClient()
            execution = client.submit_order(intent)

        obs = _llm_call(llm, "execution_agent", role,
                        {"intent": intent.model_dump(mode="json"),
                         "result": execution}, AgentObservation)
        obs = obs.model_copy(update={"message_type": MessageType.EXECUTION_RESULT})

        return {
            "agent_observations": [obs],
            "execution_result": execution,
            "final_action": _final_action_from(execution),
            "cycle_completed_at": _now_iso(),
        }

    return node


def _validate_intent(intent: OrderIntent) -> bool:
    if intent.quantity <= 0 or not intent.legs:
        return False
    for leg in intent.legs:
        if leg.quantity <= 0:
            return False
        if leg.strike <= 0:
            return False
        if leg.side not in (Side.BUY, Side.SELL):
            return False
    return True


def _dry_run_submission(intent: OrderIntent) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "submitted_at": _now_iso(),
        "intent_id": intent.intent_id,
        "strategy_id": intent.strategy_id,
        "broker": intent.broker,
        "account_mode": intent.account_mode,
        "qty": intent.quantity,
    }


def _no_intent_result(state: DecisionState) -> dict[str, Any]:
    return {
        "execution_result": {"status": "skipped", "reason": "no_order_intent"},
        "final_action": "NO_TRADE",
        "cycle_completed_at": _now_iso(),
    }


def _rejected_result(state: DecisionState, intent: OrderIntent) -> dict[str, Any]:
    return {
        "execution_result": {
            "status": "rejected_by_risk",
            "intent_id": intent.intent_id,
            "reasons": state.risk_decision.reasons if state.risk_decision else [],
        },
        "final_action": "REJECT",
        "cycle_completed_at": _now_iso(),
    }


def _invalid_result(state: DecisionState, intent: OrderIntent) -> dict[str, Any]:
    logger.warning("OrderIntent failed validation: %s", intent.model_dump(mode="json"))
    return {
        "execution_result": {"status": "invalid_intent", "intent_id": intent.intent_id},
        "final_action": "REJECT",
        "cycle_completed_at": _now_iso(),
    }


def _final_action_from(execution: dict[str, Any]) -> str:
    status = execution.get("status", "")
    if status in ("submitted", "filled", "dry_run", "partially_filled"):
        return "PROCEED"
    if status in ("rejected_by_risk", "invalid_intent"):
        return "REJECT"
    return "NO_TRADE"


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
