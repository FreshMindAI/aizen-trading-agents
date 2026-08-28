"""LangGraph orchestrator.

Wires the Phase-3 nodes into a StateGraph matching doc section 10:

    OBSERVE -> PREDICT -> ANALYZE -> PROPOSE -> CHALLENGE -> SELECT
                                                       -> RISK
                                                       -> EXECUTE/REJECT
                                                       -> RECORD

The graph uses a single `DecisionState` TypedDict (LangGraph wants dicts,
not Pydantic models, at the storage layer; Pydantic validation happens at
the boundary in `run.py`).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from ..config import get_settings, get_yaml
from ..db import connect
from .inference import InferenceService
from .journal import DecisionJournal
from .llm import get_provider
from .protocol import DecisionState
from .risk import RiskLimits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State shape for LangGraph
# ---------------------------------------------------------------------------
class _GraphState(TypedDict, total=False):
    decision_id: str
    schema_version: str
    cycle_started_at: str
    cycle_completed_at: str
    market_snapshot: dict[str, Any]
    ml_predictions: list[dict[str, Any]]
    gnn_output: dict[str, Any]
    topology_version: str | None
    agent_observations: Annotated[list[dict[str, Any]], _reducer_extend]
    agent_messages: Annotated[list[dict[str, Any]], _reducer_extend]
    candidate_strategies: list[dict[str, Any]]
    selected_strategy: dict[str, Any] | None
    risk_decision: dict[str, Any] | None
    order_intent: dict[str, Any] | None
    execution_result: dict[str, Any] | None
    final_action: str
    realized_pnl: float | None
    outcome_label: str | None


def _reducer_extend(left: list, right: list) -> list:
    """Concatenate lists - LangGraph uses this to merge node updates."""
    return list(left or []) + list(right or [])


# ---------------------------------------------------------------------------
# Node adapters - these are the bridges between LangGraph dicts and Pydantic
# ---------------------------------------------------------------------------
def _to_state_dict(state: DecisionState) -> _GraphState:
    return {
        "decision_id": state.decision_id,
        "schema_version": state.schema_version,
        "cycle_started_at": state.cycle_started_at,
        "cycle_completed_at": state.cycle_completed_at,
        "market_snapshot": state.market_snapshot.model_dump(mode="json")
            if state.market_snapshot else None,
        "ml_predictions": [u.model_dump(mode="json") for u in state.ml_predictions],
        "gnn_output": state.gnn_output,
        "topology_version": state.topology_version,
        "agent_observations": [o.model_dump(mode="json") for o in state.agent_observations],
        "agent_messages": [m.model_dump(mode="json") for m in state.agent_messages],
        "candidate_strategies": [s.model_dump(mode="json") for s in state.candidate_strategies],
        "selected_strategy": state.selected_strategy.model_dump(mode="json")
            if state.selected_strategy else None,
        "risk_decision": state.risk_decision.model_dump(mode="json")
            if state.risk_decision else None,
        "order_intent": state.order_intent.model_dump(mode="json")
            if state.order_intent else None,
        "execution_result": state.execution_result,
        "final_action": state.final_action,
        "realized_pnl": state.realized_pnl,
        "outcome_label": state.outcome_label,
    }


def _from_state_dict(d: _GraphState) -> DecisionState:
    return DecisionState.model_validate(d)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """Owns the LangGraph state machine, journal, and inference service.

    The orchestrator is process-local and re-entrant: one orchestrator can
    run many cycles. Each cycle produces exactly one DecisionJournal row.
    """

    def __init__(self, conn: sqlite3.Connection | None = None,
                 config: dict[str, Any] | None = None) -> None:
        self.settings = get_settings()
        self.agents_cfg = get_yaml("agents")
        self.risk_cfg = get_yaml("risk")
        self.config = config or self.agents_cfg
        self.risk_limits = RiskLimits.from_yaml(self.risk_cfg)

        # Inference service reads from SQLite.
        self.conn = conn or connect()
        self.inference = InferenceService(
            self.conn, self.settings.universe, topology_version="stub-1",
        )

        # LLM provider.
        llm_cfg = (self.agents_cfg.get("llm") or {})
        self.llm = get_provider(
            llm_cfg.get("provider", "mock"),
            model=llm_cfg.get("model"),
            timeout_s=int(llm_cfg.get("timeout_s", 30)),
        )

        # Journal.
        self.journal = DecisionJournal(
            self.conn,
            table=self.settings.decision_journal_table,
            run_mode=self.settings.run_mode,
        )

        # Lazy import so the package still works if langgraph isn't installed
        # (the system falls back to a sequential driver in that case).
        self._graph = self._build_graph()
        self._sequential = self._build_sequential_driver()

    # ---- public API ----------------------------------------------------
    def run_cycle(self) -> DecisionState:
        state = self._new_state()
        if self._graph is not None:
            final = self._graph.invoke(_to_state_dict(state))
        else:
            final = self._sequential(_to_state_dict(state))
        out = _from_state_dict(final)
        out.cycle_completed_at = out.cycle_completed_at or _now_iso()
        try:
            self.journal.upsert(out)
        except Exception as exc:  # pragma: no cover - never fail the cycle
            logger.exception("journal upsert failed: %s", exc)
        return out

    # ---- internal ------------------------------------------------------
    def _new_state(self) -> DecisionState:
        snap = self.inference.build_snapshot()
        ml_preds = list(snap.underlyings)
        return DecisionState(
            market_snapshot=snap,
            ml_predictions=ml_preds,
            gnn_output=self.inference.gnn_output(),
            topology_version=self.inference.topology_version,
        )

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.warning("langgraph not installed - falling back to sequential driver")
            return None
        return self._compose(StateGraph, END)

    def _compose(self, sg_lib, END) -> Any:
        from .nodes import (
            direction, execution, options_structure, portfolio,
            regime, risk as risk_node, supervisor, volatility,
        )

        g = sg_lib(_GraphState)

        nodes = {
            "regime": regime.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "direction": direction.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "volatility": volatility.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "options_structure": options_structure.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "portfolio": portfolio.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "supervisor": supervisor.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "risk": risk_node.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "execution": execution.build_node(self.llm,
                                              {**self.agents_cfg, "run_mode": self.settings.run_mode},
                                              self.risk_limits),
        }
        for name, fn in nodes.items():
            g.add_node(name, fn)

        g.set_entry_point("regime")
        g.add_edge("regime", "direction")
        g.add_edge("direction", "volatility")
        g.add_edge("volatility", "options_structure")
        g.add_edge("options_structure", "portfolio")
        g.add_edge("portfolio", "supervisor")

        # Conditional after supervisor:
        #   PROCEED -> risk -> execution -> END
        #   NO_TRADE / REJECT -> END
        def _post_supervisor(state: _GraphState) -> str:
            if state.get("order_intent") is not None:
                return "risk"
            return END

        g.add_conditional_edges("supervisor", _post_supervisor, {
            "risk": "risk", END: END,
        })
        g.add_edge("risk", "execution")
        g.add_edge("execution", END)
        return g.compile()

    # ---- sequential fallback (no langgraph) ---------------------------
    def _build_sequential_driver(self):
        from .nodes import (
            direction, execution, options_structure, portfolio,
            regime, risk as risk_node, supervisor, volatility,
        )
        nodes = {
            "regime": regime.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "direction": direction.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "volatility": volatility.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "options_structure": options_structure.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "portfolio": portfolio.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "supervisor": supervisor.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "risk": risk_node.build_node(self.llm, self.agents_cfg, self.risk_limits),
            "execution": execution.build_node(self.llm,
                                              {**self.agents_cfg, "run_mode": self.settings.run_mode},
                                              self.risk_limits),
        }

        def driver(state: _GraphState) -> _GraphState:
            for name in ("regime", "direction", "volatility", "options_structure",
                         "portfolio", "supervisor"):
                update = nodes[name](_from_state_dict(state))
                state = _merge(state, update)
            if state.get("order_intent") is not None:
                for name in ("risk", "execution"):
                    update = nodes[name](_from_state_dict(state))
                    state = _merge(state, update)
            state["cycle_completed_at"] = _now_iso()
            return state

        return driver


def _merge(state: _GraphState, update: dict[str, Any]) -> _GraphState:
    """Apply a node's partial update to the graph state. Lists are extended,
    scalars overwrite, None values are kept (not used to clear)."""
    out = dict(state)
    for k, v in update.items():
        if isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + v
        else:
            if v is not None:
                out[k] = v
    return out  # type: ignore[return-value]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
