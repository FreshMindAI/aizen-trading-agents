"""Per-cycle structured trace logging.

Why this exists
---------------
The user wants to answer "where is the model failing and where is it
succeeding?" for the 1-week hackathon window. The decision_journal table
stores the *final* state per cycle, but it does not record the
*intermediate* gates: did the dir_prob cross the threshold? Did the GNN
flag agree with the ML signal? Was the news sentiment strongly bearish?
Did the risk engine cap the size?

This module writes a JSONL record per step per cycle so the answer is
"open models/cycle_traces/{decision_id}.jsonl" instead of "grep the
debug log." Each record has:

  cycle_id        stable id for the cycle
  step            "ml" | "gnn" | "topology" | "research" | agent_id | "final"
  symbol          per-symbol step keys it to a ticker; "" for cycle-wide
  ts              ISO-8601 timestamp of the step
  fields          arbitrary JSON-safe dict of the step's inputs
  success         bool: did the gate / model / agent produce a usable signal?
  reasons         list[str]: human-readable explanation when success=False
  duration_ms     wall-clock for the step (optional, for perf triage)

Output paths
------------
Default: ``models/cycle_traces/{decision_id}.jsonl`` (one line per step)
Plus:    ``models/cycle_traces.jsonl`` (one summary line per cycle, the
         last record of each per-cycle file appended here for grep).

Both are gated by the env var ``AIZEN_TRACE=1`` (default on). Set to
``0`` to disable — useful when running the backtester at scale.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Default output location. Repo-relative so it survives a CWD change.
_DEFAULT_TRACE_DIR = Path("models") / "cycle_traces"
_DEFAULT_TRACE_FILE = Path("models") / "cycle_traces.jsonl"


def _trace_enabled() -> bool:
    """Honour AIZEN_TRACE=0 to silence the trace writer."""
    flag = os.getenv("AIZEN_TRACE", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


@dataclass
class StepRecord:
    """One step in a cycle (ML, GNN, an agent, etc.)."""

    step: str
    symbol: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    reasons: list[str] = field(default_factory=list)
    duration_ms: float | None = None

    def to_dict(self, cycle_id: str, ts: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cycle_id": cycle_id,
            "step": self.step,
            "symbol": self.symbol,
            "ts": ts,
            "success": self.success,
        }
        if self.fields:
            d["fields"] = self.fields
        if self.reasons:
            d["reasons"] = list(self.reasons)
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 3)
        return d


class CycleTraceBuilder:
    """Collects step records for one cycle, then writes them as JSONL.

    Usage from ``Orchestrator.run_cycle``::

        trace = CycleTraceBuilder(decision_id, cycle_started_at)
        with trace.step("ml", symbol="AAPL") as s:
            s.fields["direction_probability"] = 0.42
            s.success = s.fields["direction_probability"] >= 0.394
        ...
        trace.write()
    """

    def __init__(
        self,
        decision_id: str,
        cycle_started_at: str,
        *,
        trace_dir: Path | str | None = None,
        summary_path: Path | str | None = None,
    ) -> None:
        self.decision_id = decision_id
        self.cycle_started_at = cycle_started_at
        self.records: list[StepRecord] = []
        self._enabled = _trace_enabled()
        self._trace_dir = Path(trace_dir) if trace_dir is not None else _DEFAULT_TRACE_DIR
        self._summary_path = Path(summary_path) if summary_path is not None else _DEFAULT_TRACE_FILE

    # ---- step context manager -----------------------------------------
    def step(self, name: str, *, symbol: str = "") -> "_ActiveStep":
        """Open a step context. Records timing automatically on __exit__."""
        return _ActiveStep(self, name, symbol)

    # ---- direct record append -----------------------------------------
    def add(self, record: StepRecord) -> None:
        if self._enabled:
            self.records.append(record)

    # ---- persistence --------------------------------------------------
    def write(self) -> Path | None:
        """Write all collected records to JSONL. Returns the per-cycle path."""
        if not self._enabled or not self.records:
            return None
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        per_cycle = self._trace_dir / f"cycle_{self.decision_id}.jsonl"
        with per_cycle.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(self.decision_id, self.cycle_started_at), default=str))
                f.write("\n")
        # Append a single summary line (the last record == "final") to the
        # roll-up file so an operator can `tail -f models/cycle_traces.jsonl`.
        try:
            last = self.records[-1]
            with self._summary_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            **last.to_dict(self.decision_id, self.cycle_started_at),
                            "n_steps": len(self.records),
                        },
                        default=str,
                    )
                )
                f.write("\n")
        except OSError:
            # Roll-up is best-effort; the per-cycle file is the source of truth.
            pass
        return per_cycle


class _ActiveStep:
    """Context manager that times and commits a StepRecord."""

    def __init__(self, builder: CycleTraceBuilder, name: str, symbol: str) -> None:
        self._builder = builder
        self.record = StepRecord(step=name, symbol=symbol)
        self._t0: float = 0.0

    def __enter__(self) -> StepRecord:
        self._t0 = time.perf_counter()
        return self.record

    def __exit__(self, exc_type, exc, tb) -> None:
        self.record.duration_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc is not None:
            self.record.success = False
            self.record.reasons.append(f"exception: {type(exc).__name__}: {exc}")
        self._builder.add(self.record)


# ---------------------------------------------------------------------------
# Convenience builders - the Orchestrator calls these so the per-step
# knowledge (what counts as "success" for ML / GNN / agents) lives next
# to the data shapes, not buried in the graph driver.
# ---------------------------------------------------------------------------
def build_ml_step(
    *,
    threshold: float,
    predictions: Iterable[Any],
) -> StepRecord:
    """One per-cycle 'ml' step summarizing the XGBoost direction predictions.

    ``success`` is True when at least one symbol cleared the gate.
    """
    rows: list[dict[str, Any]] = []
    any_hit = False
    for u in predictions:
        prob = getattr(u, "direction_probability", None)
        sym = getattr(u, "symbol", "")
        model_version = getattr(u, "model_version", None)
        hit = prob is not None and prob >= threshold
        if hit:
            any_hit = True
        rows.append(
            {
                "symbol": sym,
                "direction_probability": prob,
                "predicted_realized_vol": getattr(u, "predicted_future_realized_vol", None),
                "model_version": model_version,
                "threshold": threshold,
                "gate_hit": hit,
            }
        )
    return StepRecord(
        step="ml",
        fields={"predictions": rows, "threshold": threshold, "n_underlyings": len(rows)},
        success=any_hit,
        reasons=[] if any_hit else ["no symbol crossed the direction_probability threshold"],
    )


def build_gnn_step(gnn_output: dict[str, Any] | None) -> StepRecord:
    """One per-cycle 'gnn' step from the GNN payload."""
    if not gnn_output:
        return StepRecord(step="gnn", success=False, reasons=["gnn_output is None"])
    node_features = gnn_output.get("node_features") or {}
    model_version = gnn_output.get("model_version")
    edges = gnn_output.get("edges") or []
    edge_kinds: dict[str, int] = {}
    for e in edges:
        kind = e.get("kind", "unknown") if isinstance(e, dict) else "unknown"
        edge_kinds[kind] = edge_kinds.get(kind, 0) + 1
    rows = []
    any_signal = False
    for sym, feats in node_features.items():
        if not isinstance(feats, dict):
            continue
        bias = feats.get("bias")
        centrality = feats.get("centrality")
        if bias is not None and abs(float(bias)) >= 0.10:
            any_signal = True
        rows.append(
            {
                "symbol": sym,
                "bias": bias,
                "centrality": centrality,
                "bias_strong": (bias is not None and abs(float(bias)) >= 0.10),
            }
        )
    return StepRecord(
        step="gnn",
        fields={
            "model_version": model_version,
            "n_nodes": len(node_features),
            "n_edges": len(edges),
            "edge_kinds": edge_kinds,
            "node_features": rows,
        },
        success=any_signal,
        reasons=[] if any_signal else ["no GNN node had |bias| >= 0.10"],
    )


def build_topology_step(gnn_output: dict[str, Any] | None, topology_version: str | None) -> StepRecord:
    """Graph-level topology stats: density, weight distribution, news-edge count."""
    if not gnn_output:
        return StepRecord(
            step="topology",
            success=False,
            reasons=["gnn_output is None; no topology payload available"],
        )
    edges = gnn_output.get("edges") or []
    weights = [float(e.get("weight", 0.0)) for e in edges if isinstance(e, dict) and e.get("weight") is not None]
    news_edges = sum(1 for e in edges if isinstance(e, dict) and e.get("kind") == "news")
    n_nodes = len(gnn_output.get("node_features") or {})
    n_edges = len(edges)
    density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    fields = {
        "topology_version": topology_version,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "news_edges": news_edges,
        "density": round(density, 6),
        "weight_min": min(weights) if weights else None,
        "weight_max": max(weights) if weights else None,
        "weight_mean": (sum(weights) / len(weights)) if weights else None,
    }
    # Topology is "successful" when we have at least one edge of any kind
    # (a graph with nodes but no edges is degenerate and not useful).
    return StepRecord(
        step="topology",
        fields=fields,
        success=n_edges > 0,
        reasons=[] if n_edges > 0 else ["graph has zero edges"],
    )


def build_research_step(research: Any) -> StepRecord:
    """Research agent: news-derived sentiment per symbol."""
    if research is None:
        return StepRecord(
            step="research",
            success=False,
            reasons=["research disabled (agents.research.enabled=false)"],
        )
    # Feature flag check first: "news-off" is a different failure mode from
    # "news-on but no articles in window". Operators triage these differently.
    flag_state = getattr(research, "feature_flag_state", "news-on")
    if flag_state == "news-off":
        return StepRecord(
            step="research",
            fields={"feature_flag_state": flag_state, "n_symbols_with_news": 0, "n_symbols_total": 0},
            success=False,
            reasons=["research feature flag is off (news-off)"],
        )
    per_symbol: dict[str, Any] = {}
    n_with_news = 0
    sentiment_values: list[float] = []
    try:
        for sym, sr in (research.per_symbol or {}).items():
            sent = getattr(sr, "sentiment", None)
            vol = getattr(sr, "volume", 0)
            per_symbol[sym] = {
                "sentiment": sent,
                "volume": vol,
                "topics": list(getattr(sr, "topics", []) or []),
            }
            if vol and sent is not None:
                n_with_news += 1
                sentiment_values.append(float(sent))
    except AttributeError:
        return StepRecord(step="research", success=False, reasons=["research object missing per_symbol"])
    return StepRecord(
        step="research",
        fields={
            "feature_flag_state": flag_state,
            "n_symbols_with_news": n_with_news,
            "n_symbols_total": len(per_symbol),
            "sentiment_mean": (sum(sentiment_values) / len(sentiment_values)) if sentiment_values else None,
            "per_symbol": per_symbol,
        },
        success=n_with_news > 0,
        reasons=[] if n_with_news > 0 else ["no news articles in the prior window for any symbol"],
    )


def build_agent_step(observation: Any) -> StepRecord:
    """Per-agent observation step. The agent's own ``risks`` become the
    failure reasons; ``success`` is the agent's own confidence >= 0.50."""
    msg_type = getattr(observation, "message_type", None)
    msg_type_str = msg_type.value if hasattr(msg_type, "value") else str(msg_type)
    agent_id = getattr(observation, "agent_id", "")
    confidence = getattr(observation, "confidence", 0.0)
    risks = list(getattr(observation, "risks", []) or [])
    signal = getattr(observation, "signal", {}) or {}
    success = bool(confidence >= 0.50) and not risks
    return StepRecord(
        step=agent_id or "agent",
        fields={
            "message_type": msg_type_str,
            "confidence": confidence,
            "signal": signal,
            "evidence": list(getattr(observation, "evidence", []) or []),
            "data_version": getattr(observation, "data_version", None),
            "model_versions": list(getattr(observation, "model_versions", []) or []),
        },
        success=success,
        reasons=[] if success else [f"confidence={confidence:.2f} below 0.50 or risks present"] + risks,
    )


def build_llm_step(telemetry: Any) -> StepRecord | None:
    """Per-cycle LLM call step. Built from an ``LLMCallTelemetry`` instance
    (src/agents/llm/telemetry.py). Returns ``None`` when no telemetry is
    available (e.g. mock provider, or a cycle that never made an LLM call)
    so the caller can skip the trace append.

    ``success`` is True only when the call succeeded on the FIRST attempt.
    Retries are visible in ``fields.attempts`` and ``fields.per_attempt``;
    a cycle whose LLM call needed a retry is flagged as ``success=False``
    even though the final response was 2xx (operators want to see retries,
    they don't want them buried in a 200 OK).
    """
    if telemetry is None:
        return None
    to_dict = getattr(telemetry, "to_dict", None)
    if not callable(to_dict):
        # Defensive: the orchestrator must pass an LLMCallTelemetry.
        return StepRecord(
            step="llm_call",
            success=False,
            reasons=["telemetry object missing to_dict()"],
        )
    summary = to_dict()
    attempts = int(summary.get("attempts", 0) or 0)
    succeeded = bool(summary.get("succeeded", False))
    final_error = summary.get("final_error")
    final_status = summary.get("final_status")
    # A retry even on success is reported as a soft failure: the operator
    # wants visibility into "we had to retry, even though it ultimately
    # worked". Permanent failure (LLMError raised) keeps success=False.
    success = succeeded and attempts == 1
    reasons: list[str] = []
    if not succeeded:
        if attempts > 0:
            reasons.append(
                f"giving up after {attempts} attempts: last_status={final_status}, "
                f"last_error={final_error!r}"
            )
        else:
            reasons.append("no LLM attempts recorded")
    elif attempts > 1:
        reasons.append(
            f"succeeded on attempt {attempts} after {attempts - 1} retry/retr"
            "ies (last_error={!r})".format(final_error)
        )
    return StepRecord(
        step="llm_call",
        fields={
            "provider": summary.get("provider"),
            "model": summary.get("model"),
            "attempts": attempts,
            "succeeded": succeeded,
            "final_status": final_status,
            "final_error": final_error,
            "total_sleep_s": summary.get("total_sleep_s"),
            "duration_ms": summary.get("duration_ms"),
            "per_attempt": summary.get("per_attempt", []),
        },
        success=success,
        reasons=reasons,
    )


def build_final_step(state: Any) -> StepRecord:
    """Cycle-final summary: action, selected strategy, risk outcome."""
    action = getattr(state, "final_action", None)
    selected = getattr(state, "selected_strategy", None)
    risk_dec = getattr(state, "risk_decision", None)
    order_intent = getattr(state, "order_intent", None)
    fields: dict[str, Any] = {
        "action": action,
        "selected_underlying": getattr(selected, "underlying", None) if selected else None,
        "selected_strategy_id": getattr(selected, "strategy_id", None) if selected else None,
        "risk_action": getattr(risk_dec, "action", None) if risk_dec else None,
        "risk_reasons": list(getattr(risk_dec, "reasons", []) or []) if risk_dec else [],
        "order_legs": len(getattr(order_intent, "legs", []) or []) if order_intent else 0,
    }
    success = action == "PROCEED" and order_intent is not None
    return StepRecord(step="final", fields=fields, success=success, reasons=[] if success else [f"action={action}"])
