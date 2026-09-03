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
                 config: dict[str, Any] | None = None,
                 *, as_of: str | None = None) -> None:
        self.settings = get_settings()
        self.agents_cfg = get_yaml("agents")
        # get_yaml('risk') returns the wrapped bundle {'risk': {...}}; the
        # risk engine expects the inner dict, so unwrap explicitly. If
        # the bundle is already flat (no 'risk' key), fall back to the
        # whole mapping so old call-sites keep working.
        _risk_bundle = get_yaml("risk")
        self.risk_cfg = _risk_bundle.get("risk", _risk_bundle) if isinstance(_risk_bundle, dict) else _risk_bundle
        self.config = config or self.agents_cfg
        # Two ways to build the risk limits: (a) explicit dollar amounts
        # in risk.yaml (legacy 1-week hackathon defaults), or (b) scaled
        # from the account's capital (default for the multi-week paper
        # account). The flag is opt-out: set ``scale_from_capital: false``
        # in risk.yaml to pin the explicit dollar amounts.
        _scale_from_capital = bool(self.risk_cfg.get("scale_from_capital", True))
        if _scale_from_capital:
            self.risk_limits = RiskLimits.scaled_from_capital(
                float(getattr(self.settings, "capital_usd", 100_000.0))
            )
            logger.info(
                "risk limits scaled from capital: $%.0f → "
                "per-trade $%.0f, per-symbol $%.0f, gross $%.0f",
                self.settings.capital_usd,
                self.risk_limits.max_order_notional_usd,
                self.risk_limits.max_equity_notional_per_symbol,
                self.risk_limits.max_gross_exposure_usd,
            )
        else:
            self.risk_limits = RiskLimits.from_yaml(self.risk_cfg)
        # Point-in-time cut-off (ISO 8601). When set, every loader in
        # the inference path filters on ``timestamp <= as_of`` so the
        # snapshot is consistent with the cycle time. Used by the
        # backtester (spec 003 / T046) and by hand-driven replays.
        self.as_of = as_of

        # Inference service reads from SQLite. The research_enabled flag
        # is resolved from config/agents.yaml (agents.research.enabled) with
        # the env var AIZEN_RESEARCH_ENABLED taking precedence. When True
        # the snapshot is enriched with a ResearchOutput built from the
        # news_snapshot table (spec 003 / T018).
        #
        # Topology version: resolved from env AIZEN_GNN_TOPOLOGY_VERSION
        # (preferred, lets ops flip without a config edit) then from
        # config/agents.yaml (orchestrator.topology_version) then
        # "fixed-1". Values supported today:
        #   "fixed-1"   : static sector/supplier/etf/correlation (default)
        #   "dynamic-1" : fixed-1 PLUS rolling-correlation edges from
        #                 src.gnn.dynamic_topology (recomputed at snapshot
        #                 time from underlying_bars). The new edge
        #                 reason "rolling_corr" is appended to the edge
        #                 list; downstream agents see a topology_version
        #                 of "dynamic-1" on the GNN output.
        self.conn = conn or connect()
        import os as _os
        _research_cfg = ((self.agents_cfg.get("agents") or {}).get("research") or {})
        _env_research = _os.getenv("AIZEN_RESEARCH_ENABLED")
        if _env_research is not None:
            _research_enabled = _env_research.strip().lower() in ("1", "true", "yes", "on")
        else:
            _research_enabled = bool(_research_cfg.get("enabled", False))
        _topo_env = _os.getenv("AIZEN_GNN_TOPOLOGY_VERSION")
        _topo_cfg = (self.agents_cfg.get("orchestrator") or {}).get("topology_version")
        topology_version = _topo_env or _topo_cfg or "fixed-1"
        if topology_version not in ("fixed-1", "dynamic-1"):
            logger.warning(
                "unknown topology_version=%r requested; falling back to 'fixed-1'. "
                "Supported: 'fixed-1', 'dynamic-1'.",
                topology_version,
            )
            topology_version = "fixed-1"
        self.inference = InferenceService(
            self.conn, self.settings.universe, topology_version=topology_version,
            research_enabled=_research_enabled,
            as_of=self.as_of,
        )

        # LLM provider. Resolution order (matches get_provider's contract):
        #   1. env vars AIZEN_LLM_PROVIDER / LLM_PROVIDER (preferred - lets ops
        #      flip providers without editing YAML)
        #   2. config/agents.yaml `llm.provider`
        #   3. 'mock' (default)
        import os as _os
        llm_cfg = (self.agents_cfg.get("llm") or {})
        env_provider = _os.getenv("AIZEN_LLM_PROVIDER") or _os.getenv("LLM_PROVIDER")
        yaml_provider = llm_cfg.get("provider")
        provider_name = (env_provider or yaml_provider or "mock")
        # Defensive guard: a `model:` field written for the MockProvider
        # (e.g. "mock-1") leaks into a real provider when an operator
        # flips the env var. A real provider will then send "mock-1" to
        # the upstream endpoint and 404 (GMI does not know "mock-1").
        # The `model` field is opt-in: if a sentinel-only model is set,
        # drop it and let the provider's class default take over.
        yaml_model = llm_cfg.get("model")
        is_real_provider = provider_name in ("anthropic", "gmi_fallback", "gmi",
                                             "gmi-serving", "openai", "gpt", "oai")
        looks_like_mock_sentinel = isinstance(yaml_model, str) and (
            yaml_model.startswith("mock-") or yaml_model.startswith("stub-")
        )
        if is_real_provider and looks_like_mock_sentinel:
            logger.warning(
                "config/agents.yaml sets llm.model=%r but llm.provider=%r; "
                "the mock-only sentinel would 404 the upstream. "
                "Ignoring the override and using the provider's class default.",
                yaml_model, provider_name,
            )
            yaml_model = None
        self.llm = get_provider(
            provider_name,
            default_model=yaml_model,
            timeout_s=int(llm_cfg.get("timeout_s", 30)),
        )

        # Journal.
        self.journal = DecisionJournal(
            self.conn,
            table=self.settings.decision_journal_table,
            run_mode=self.settings.run_mode,
        )

        # Failure knowledge-graph writer. Persists every caught
        # failure (cycle / agent / symbol) as a typed node with
        # weighted edges so the GNN can treat the failure channel as
        # a first-class signal. The writer is feature-flag-gated: it
        # is a no-op when ``failure_kg_enabled`` is False. The default
        # is True so failures don't get silently dropped again.
        from .failure_kg import FailureKG
        _fk_env = _os.getenv("AIZEN_FAILURE_KG_ENABLED")
        if _fk_env is not None:
            _fk_enabled = _fk_env.strip().lower() in ("1", "true", "yes", "on")
        else:
            _fk_enabled = True
        self.failure_kg_enabled = _fk_enabled
        self.failure_kg = FailureKG(self.conn) if _fk_enabled else None

        # Lazy import so the package still works if langgraph isn't installed
        # (the system falls back to a sequential driver in that case).
        self._graph = self._build_graph()
        self._sequential = self._build_sequential_driver()

    # ---- public API ----------------------------------------------------
    def run_cycle(self) -> DecisionState:
        # Pre-flight position management. Runs BEFORE state construction
        # so the kill-switch can short-circuit and the blocked-symbols
        # set is on the state when the supervisor reads candidates.
        # See :mod:`src.agents.position_management` for the contract.
        from . import position_management as _pm
        _thresholds = _pm._load_thresholds(self.agents_cfg)
        preflight = _pm.PreFlightResult()

        # (a0) Kill-switch latch check FIRST - before any broker call.
        # If a previous tick on the same UTC day already tripped the
        # kill-switch, the operator intent was "no new entries for the
        # rest of today". Short-circuit before touching the broker so
        # we don't waste a list_positions call on a day that's locked.
        if _pm.is_kill_switch_latched_today(self.conn):
            preflight.kill_switch_latched = True
            state = DecisionState()
            state.final_action = "NO_TRADE"
            state.cycle_started_at = _now_iso()
            state.cycle_completed_at = _now_iso()
            state.execution_result = {
                "status": "BLOCKED",
                "reason": "daily_loss_kill_switch_latched",
                "detail": preflight.as_dict(),
            }
            state.supplementary = {"position_management": preflight.as_dict()}
            logger.warning(
                "kill-switch latched earlier today; skipping cycle (no new entries)"
            )
            try:
                self.journal.upsert(state)
            except Exception as exc:  # pragma: no cover
                logger.warning("kill-switch-latch journal upsert failed: %s", exc)
            return state

        # Broker portfolio fetch: fail-CLOSED. The previous behavior
        # (return [] on any exception) silently disabled every
        # position-management check, which would let the orchestrator
        # open new positions even on a day when existing positions
        # were past the stop-loss. We now short-circuit the cycle to
        # NO_TRADE/BLOCKED if we cannot see the live portfolio.
        from .alpaca_trading import AlpacaTradingClient
        _broker_error: str | None = None
        live_positions: list[Any] = []
        try:
            _client = AlpacaTradingClient()
            live_positions = _client.list_positions() or []
        except Exception as exc:  # noqa: BLE001
            _broker_error = f"{type(exc).__name__}: {exc}"
        if _broker_error is not None:
            preflight.broker_unreachable = _broker_error
            state = DecisionState()
            state.final_action = "NO_TRADE"
            state.cycle_started_at = _now_iso()
            state.cycle_completed_at = _now_iso()
            state.execution_result = {
                "status": "BLOCKED",
                "reason": "broker_unreachable",
                "detail": {"error": _broker_error},
            }
            state.supplementary = {"position_management": preflight.as_dict()}
            logger.error(
                "broker portfolio fetch failed; cycle fail-closed: %s",
                _broker_error,
            )
            try:
                self.journal.upsert(state)
            except Exception as exc:  # pragma: no cover
                logger.warning("broker-unreachable journal upsert failed: %s", exc)
            return state

        # (a) Daily loss kill-switch.
        try:
            from .journal import _today_realized_pnl
            realized_today = _today_realized_pnl(self.conn)
        except Exception:
            realized_today = 0.0
        preflight.kill_switch = _pm.check_daily_loss_kill_switch(
            capital_usd=float(getattr(self.settings, "capital_usd", 100_000.0)),
            positions=live_positions,
            realized_pnl_today=realized_today,
            pct=_thresholds["daily_loss_kill_switch_pct"],
        )
        if preflight.kill_switch.breached:
            # Latch the trip so subsequent ticks today cannot re-enter.
            _pm.record_kill_switch_latch(
                self.conn,
                total_pnl=preflight.kill_switch.total_pnl,
                threshold_usd=preflight.kill_switch.threshold_usd,
                pct=preflight.kill_switch.pct,
            )
            preflight.kill_switch_latched = True
            state = DecisionState()
            state.final_action = "NO_TRADE"
            state.cycle_started_at = _now_iso()
            state.cycle_completed_at = _now_iso()
            state.execution_result = {
                "status": "BLOCKED",
                "reason": "daily_loss_kill_switch",
                "detail": preflight.kill_switch.as_dict(),
            }
            state.supplementary = {"position_management": preflight.as_dict()}
            logger.warning(
                "daily-loss kill switch breached: total_pnl=%.2f threshold=%.2f (pct=%.4f) - skipping cycle",
                preflight.kill_switch.total_pnl,
                preflight.kill_switch.threshold_usd,
                preflight.kill_switch.pct,
            )
            # Persist the block as a journal row so the audit trail shows
            # why the cycle did nothing.
            try:
                self.journal.upsert(state)
            except Exception as exc:  # pragma: no cover
                logger.warning("kill-switch journal upsert failed: %s", exc)
            return state
        # (a1) Early warning. -25% (configurable) loss is halfway to
        # the -50% stop-loss. Log a WARNING per position so the
        # operator can see it on the cron trace; record the list in
        # preflight.early_warnings so the journal row also reflects it.
        preflight.early_warnings = _pm.early_warning_positions(
            live_positions, pct=_thresholds["early_warning_pct"],
        )
        for w in preflight.early_warnings:
            logger.warning(
                "early-warning: %s at loss_pct=%.2f%% unrealized_pnl=%.2f "
                "(halfway to %.0f%% stop-loss)",
                w["symbol"], w["loss_pct"] * 100, w["unrealized_pnl"],
                abs(_thresholds["stop_loss_pct"]) * 100,
            )
        # (b) Auto-close per-position stop-losses. Side effect, no state
        # change to the cycle result. Failures are logged but do not
        # abort the cycle (a broker outage is non-fatal).
        try:
            preflight.stop_loss_closes = _pm.auto_close_stop_loss(
                _client, live_positions,
                pct=_thresholds["stop_loss_pct"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_close_stop_loss failed: %s: %s",
                           type(exc).__name__, exc)
        # (c) Per-symbol cooldown. The blocked set is read by the
        # supervisor via state.supplementary['position_management'].
        preflight.blocked_symbols = _pm.get_blocked_symbols(
            positions=live_positions,
            conn=self.conn,
            cooldown_seconds=_thresholds["cooldown_seconds"],
            open_loss_block_pct=_thresholds["open_loss_block_pct"],
        )
        # Build the cycle state with the pre-flight summary attached.
        state = self._new_state()
        state.supplementary = {"position_management": preflight.as_dict()}
        cycle_error: str | None = None
        cycle_exc: BaseException | None = None
        try:
            if self._graph is not None:
                final = self._graph.invoke(_to_state_dict(state))
            else:
                final = self._sequential(_to_state_dict(state))
        except Exception as exc:
            # Cycle blew up (e.g. options market closed during execution).
            # Persist a partial state + a trace so the failure is debuggable
            # rather than a black-box exit. The trace captures what we had
            # *before* the crash; the journal row records the failure mode.
            cycle_error = f"{type(exc).__name__}: {exc}"
            cycle_exc = exc
            logger.exception("cycle failed: %s", cycle_error)
            final = _to_state_dict(state)
            # ``final_action`` is a Literal[PROCEED, NO_TRADE, REJECT, REDUCE];
            # we can't add an ``ERROR`` value without breaking journal/contract
            # invariants. Surface the failure via execution_result instead.
            final["final_action"] = "NO_TRADE"
            final["cycle_completed_at"] = _now_iso()
            existing = final.get("execution_result") or {}
            existing["error"] = cycle_error
            existing["status"] = "ERROR"
            final["execution_result"] = existing
        out = _from_state_dict(final)
        out.cycle_completed_at = out.cycle_completed_at or _now_iso()
        # ---- failure knowledge graph (T169) ---------------------------
        # Record the cycle-level failure as a node + connect to any
        # per-symbol failures for the same decision_id. This block is
        # best-effort: a FailureKG write failure must NOT cascade into
        # a journal write failure (it is the most-recent layer).
        if cycle_exc is not None and self.failure_kg is not None:
            try:
                self.failure_kg.record_cycle_failure(
                    out.decision_id,
                    out.cycle_started_at or _now_iso(),
                    final_action=final.get("final_action"),
                    exc=cycle_exc,
                    metadata={"cycle_completed_at": out.cycle_completed_at},
                )
            except Exception as fk_exc:  # pragma: no cover - defensive
                logger.warning("failure_kg cycle-failure write failed: %s", fk_exc)
        try:
            self.journal.upsert(out)
        except Exception as exc:  # pragma: no cover - never fail the cycle
            logger.exception("journal upsert failed: %s", exc)
        # Per-cycle trace: write JSONL record per step (ML, GNN, topology,
        # research, each agent, final). Lazy import so tests that don't
        # touch trace.py don't pay the import cost. Always runs, even on
        # ERROR cycles, so the on-disk record shows what the model saw
        # before it crashed.
        try:
            from .trace import (
                CycleTraceBuilder,
                StepRecord,
                build_agent_step,
                build_final_step,
                build_gnn_step,
                build_llm_step,
                build_ml_step,
                build_research_step,
                build_topology_step,
            )
            trace = CycleTraceBuilder(out.decision_id, out.cycle_started_at or _now_iso())
            # ML step: re-centered gate = base_rate + 0.10 (mirrors direction node)
            try:
                from .nodes.strategy_selector import _BASE_RATE, _BASE_RATE_CUSHION
                _thresh = min(0.95, _BASE_RATE + _BASE_RATE_CUSHION) if _BASE_RATE is not None else 0.55
            except Exception:
                _thresh = 0.55
            trace.add(build_ml_step(threshold=_thresh, predictions=out.ml_predictions))
            trace.add(build_gnn_step(out.gnn_output))
            trace.add(build_topology_step(out.gnn_output, out.topology_version))
            research_obj = out.market_snapshot.research if out.market_snapshot else None
            trace.add(build_research_step(research_obj))
            for obs in out.agent_observations:
                trace.add(build_agent_step(obs))
            # LLM call step: one record per cycle sourced from the provider's
            # last_call telemetry. Skipped silently when the provider has no
            # call to report (e.g. MockProvider, or a short-circuited cycle).
            try:
                llm_telemetry = getattr(self.llm, "last_call", None)
                llm_step = build_llm_step(llm_telemetry)
                if llm_step is not None:
                    trace.add(llm_step)
            except Exception as exc:  # pragma: no cover - never fail the cycle
                logger.warning("llm_call trace build failed: %s", exc)
            trace.add(build_final_step(out))
            if cycle_error is not None:
                # Append a final "error" step so the trace self-documents
                # the failure mode without grepping the system log.
                trace.add(StepRecord(step="error", success=False, reasons=[cycle_error]))
            trace.write()
        except Exception as exc:  # pragma: no cover - never fail the cycle
            logger.exception("cycle trace write failed: %s", exc)
        return out

    # ---- internal ------------------------------------------------------
    def _new_state(self) -> DecisionState:
        snap = self.inference.build_snapshot()
        ml_preds = list(snap.underlyings)
        state = DecisionState(
            market_snapshot=snap,
            ml_predictions=ml_preds,
            gnn_output=self.inference.gnn_output(),
            topology_version=self.inference.topology_version,
        )
        # When ``as_of`` is set, stamp the cycle start to the cut-off
        # so the journal row reflects the backtest time, not wall-clock.
        if self.as_of is not None:
            state.cycle_started_at = self.as_of
        return state

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
        # Construct one in-process MCP server per orchestrator and hand
        # each agent a per-agent SkillRegistry. The MCP server is
        # constructed here (not in __init__) so the orchestrator
        # already has settings + run_mode resolved.
        from .mcp import AlpacaMCPServer
        from .mcp.skills import build_skill_registry
        mcp_server = AlpacaMCPServer(run_mode=self.settings.run_mode)

        g = sg_lib(_GraphState)

        nodes = {
            "regime": regime.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                        skills=build_skill_registry("regime", mcp_server)),
            "direction": direction.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                              skills=build_skill_registry("direction", mcp_server)),
            "volatility": volatility.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                                skills=build_skill_registry("volatility", mcp_server)),
            "options_structure": options_structure.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                                             skills=build_skill_registry("options_structure", mcp_server)),
            "portfolio": portfolio.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                              skills=build_skill_registry("portfolio", mcp_server)),
            "supervisor": supervisor.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                                skills=build_skill_registry("supervisor", mcp_server)),
            "risk": risk_node.build_node(self.llm, self.agents_cfg, self.risk_limits,
                                          skills=build_skill_registry("risk", mcp_server)),
            "execution": execution.build_node(self.llm,
                                              {**self.agents_cfg, "run_mode": self.settings.run_mode},
                                              self.risk_limits,
                                              skills=build_skill_registry("execution", mcp_server)),
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
