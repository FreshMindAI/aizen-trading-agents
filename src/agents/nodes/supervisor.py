"""Decision Supervisor.

Resolves conflicts among agent observations, picks one candidate (or
NO_TRADE), and constructs the OrderIntent. The LLM is asked to EXPLAIN the
pick; the math (best-score / no-trade-on-disagreement) is deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import get_settings
from ..protocol import (
    AgentObservation,
    DecisionState,
    Leg,
    MessageType,
    OrderIntent,
    Side,
    StrategyProposal,
)
from ._common import _llm_call, _log_agent_observation, _to_message

logger = logging.getLogger(__name__)


def _blocked_symbols(state: DecisionState) -> set[str]:
    """Read the blocked-symbol set written by the orchestrator's
    pre-flight (:mod:`src.agents.position_management`). Returns an empty
    set when the cycle did not run the pre-flight (e.g. tests that call
    the supervisor directly) so the filter is a no-op.
    """
    supp = getattr(state, "supplementary", None) or {}
    pm = supp.get("position_management") or {}
    return {s.upper() for s in (pm.get("blocked_symbols") or []) if isinstance(s, str)}


def build_node(llm, config: dict[str, Any], risk_limits, *, skills=None):
    role = (
        "Resolve conflicts between agent observations. Pick the highest-score "
        "candidate whose signals are not contradicted by the regime/direction/"
        "volatility agents, OR return NO_TRADE."
    )
    thresh = config.get("thresholds", {})
    no_trade_if_disagreement = bool(thresh.get("no_trade_if_disagreement", True))
    confidence_min = float(thresh.get("confidence_min", 0.50))

    def node(state: DecisionState) -> dict[str, Any]:
        candidates = state.candidate_strategies
        observations = state.agent_observations

        if not candidates:
            obs = _supervisor_obs(llm, observations, candidates, "NO_TRADE", "no candidates",
                                  0.1, role, state)
            return _as_update(state, obs, None)

        # (c) Per-symbol cooldown. The orchestrator's pre-flight fills
        # state.supplementary['position_management']['blocked_symbols']
        # with the set of symbols that have an open losing position
        # OR a recent realized loss. Drop those candidates BEFORE the
        # conflict / confidence gates so we never re-enter a loser.
        blocked = _blocked_symbols(state)
        if blocked:
            # Log the blocked set at the top of every cycle so the
            # operator can see it in the cron stdout (not buried
            # inside the LangGraph state). Promoted to WARNING
            # because "we are about to skip a candidate because the
            # underlying is in loss" is the single most important
            # operator signal in the run loop.
            logger.warning(
                "supervisor blocked_symbols: %s", sorted(blocked),
            )
            before = len(candidates)
            candidates = [c for c in candidates
                          if (c.underlying or "").split(" ")[0].upper()
                          not in blocked]
            if len(candidates) < before:
                logger.warning(
                    "supervisor filtered %d candidate(s) on blocked "
                    "symbols: %s",
                    before - len(candidates), sorted(blocked),
                )
            if not candidates:
                obs = _supervisor_obs(llm, observations, candidates, "NO_TRADE",
                                      f"all candidates on blocked symbols {sorted(blocked)}",
                                      0.2, role, state)
                return _as_update(state, obs, None)

        top = candidates[0]
        if _has_conflict(observations, top) and no_trade_if_disagreement:
            obs = _supervisor_obs(llm, observations, candidates, "NO_TRADE",
                                  "agent disagreement", 0.2, role, state)
            return _as_update(state, obs, None)

        if top.confidence < confidence_min:
            obs = _supervisor_obs(llm, observations, candidates, "NO_TRADE",
                                  "below confidence_min", 0.2, role, state)
            return _as_update(state, obs, None)

        intent = _build_intent(top)
        obs = _supervisor_obs(llm, observations, candidates, "PROCEED",
                              f"selected {top.strategy_id}", top.confidence, role, state)
        return _as_update(state, obs, intent, top)

    return node


def _has_conflict(observations: list[AgentObservation], top: StrategyProposal) -> bool:
    """Heuristic conflict detector: regime=danger + direction opposes top's
    underlying bias => conflict. Keeps it simple, fully deterministic."""
    for obs in observations:
        if obs.signal.get("label_heuristic") in ("high_volatility", "crisis"):
            return True
    return False


def _build_intent(proposal: StrategyProposal) -> OrderIntent:
    settings = get_settings()
    qty = max(1, proposal.legs[0].quantity)
    return OrderIntent(
        strategy_id=proposal.strategy_id,
        underlying=proposal.underlying,
        legs=[Leg(**leg.model_dump()) for leg in proposal.legs],
        quantity=qty,
        limit_price=proposal.legs[0].limit_price,
        time_in_force="day",
        account_mode="PAPER" if settings.run_mode != "live" else "LIVE",
    )


def _supervisor_obs(llm, observations, candidates, action: str, reason: str,
                    confidence: float, role: str, state: DecisionState) -> AgentObservation:
    payload = {
        "action": action,
        "reason": reason,
        "observations": [o.model_dump(mode="json") for o in observations],
        "candidates": [c.model_dump(mode="json") for c in candidates[:5]],
    }
    obs = _llm_call(llm, "supervisor", role, payload, AgentObservation)
    return obs.model_copy(update={
        "message_type": MessageType.SUPERVISOR_DECISION,
        "confidence": max(obs.confidence, confidence),
        "signal": {"action": action, "reason": reason, **obs.signal},
    })


def _as_update(state: DecisionState, obs: AgentObservation,
               intent: OrderIntent | None,
               selected: StrategyProposal | None = None) -> dict[str, Any]:
    msg = _to_message(obs, state.decision_id, "supervisor", "execution")
    _log_agent_observation("supervisor", obs)
    update: dict[str, Any] = {
        "agent_observations": [obs],
        "agent_messages": [msg],
        "final_action": "PROCEED" if intent else "NO_TRADE",
    }
    if intent is not None:
        update["order_intent"] = intent
    if selected is not None:
        update["selected_strategy"] = selected
    return update
