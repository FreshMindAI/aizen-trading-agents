"""Tests for per-cycle trace logging (spec 003 / hackathon triage).

The trace is the on-disk answer to "where is the model failing and where
is it succeeding?" Each test pins one of the four shape invariants:

  T1. CycleTraceBuilder writes one JSONL line per step + a roll-up line.
  T2. build_ml_step records per-symbol gate_hit and surfaces failure
      reasons when no symbol crosses the threshold.
  T3. build_gnn_step + build_topology_step surface a sane graph stat
      payload (density, edge kinds, news edge count) and mark degenerate
      graphs as failures.
  T4. build_research_step distinguishes "no news" from "feature off".
  T5. build_agent_step carries the agent's own confidence and risks.
  T6. build_final_step captures the cycle outcome (action, risk_action,
      order_legs) and the failure reason when action != PROCEED.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.trace import (  # noqa: E402
    CycleTraceBuilder,
    StepRecord,
    build_agent_step,
    build_final_step,
    build_gnn_step,
    build_ml_step,
    build_research_step,
    build_topology_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _FakeUnderlying:
    """Minimal stand-in for UnderlyingScore."""

    def __init__(self, symbol: str, direction_probability: float, model_version: str = "v1"):
        self.symbol = symbol
        self.direction_probability = direction_probability
        self.model_version = model_version
        self.predicted_future_realized_vol = 0.20


class _FakeResearch:
    def __init__(self, per_symbol: dict | None, flag: str = "news-on"):
        self.per_symbol = per_symbol or {}
        self.feature_flag_state = flag

    def __bool__(self) -> bool:  # so research is None -> False works
        return bool(self.per_symbol)


class _SymbolR:
    def __init__(self, sentiment: float | None, volume: int = 0, topics: list | None = None):
        self.sentiment = sentiment
        self.volume = volume
        self.topics = topics or []


class _FakeObs:
    def __init__(self, agent_id="regime", message_type="REGIME_VIEW", confidence=0.7,
                 risks=None, signal=None):
        self.agent_id = agent_id
        self.message_type = message_type
        self.confidence = confidence
        self.risks = risks or []
        self.signal = signal or {"bias": "neutral"}
        self.evidence = []
        self.data_version = "v1"
        self.model_versions = []


class _FakeSelected:
    def __init__(self, underlying="AAPL", strategy_id="long_call"):
        self.underlying = underlying
        self.strategy_id = strategy_id


class _FakeRiskDecision:
    def __init__(self, action="APPROVE", reasons=None):
        self.action = action
        self.reasons = reasons or []


class _FakeOrderIntent:
    def __init__(self, legs=None):
        self.legs = legs or []


class _FakeState:
    def __init__(self, *, action="PROCEED", selected=None, risk=None, intent=None):
        self.final_action = action
        self.selected_strategy = selected
        self.risk_decision = risk
        self.order_intent = intent


# ---------------------------------------------------------------------------
# T1: CycleTraceBuilder writes one line per step + a roll-up line.
# ---------------------------------------------------------------------------
def test_builder_writes_per_cycle_file_and_rollup(tmp_path, monkeypatch):
    monkeypatch.setenv("AIZEN_TRACE", "1")
    summary = tmp_path / "rollup.jsonl"
    builder = CycleTraceBuilder(
        decision_id="abc-123",
        cycle_started_at="2026-08-30T13:30:00Z",
        trace_dir=tmp_path / "cycles",
        summary_path=summary,
    )
    with builder.step("ml", symbol="AAPL") as s:
        s.fields["direction_probability"] = 0.42
        s.success = True
    with builder.step("final") as s:
        s.fields["action"] = "NO_TRADE"
        s.success = False
        s.reasons.append("disagreement")
    out = builder.write()
    assert out is not None
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(ln) for ln in lines]
    assert recs[0]["step"] == "ml" and recs[0]["symbol"] == "AAPL" and recs[0]["success"]
    assert recs[1]["step"] == "final" and not recs[1]["success"]
    assert recs[0]["duration_ms"] >= 0.0  # context manager timed it
    # Roll-up file: exactly one line, the last record + n_steps.
    rollup_lines = summary.read_text(encoding="utf-8").strip().splitlines()
    assert len(rollup_lines) == 1
    rollup = json.loads(rollup_lines[0])
    assert rollup["step"] == "final"
    assert rollup["n_steps"] == 2


def test_builder_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIZEN_TRACE", "0")
    builder = CycleTraceBuilder(
        decision_id="x",
        cycle_started_at="2026-08-30T13:30:00Z",
        trace_dir=tmp_path / "cycles",
    )
    builder.add(StepRecord(step="ml", success=True))
    out = builder.write()
    assert out is None
    assert not list((tmp_path / "cycles").glob("*"))


# ---------------------------------------------------------------------------
# T2: build_ml_step
# ---------------------------------------------------------------------------
def test_build_ml_step_marks_success_when_any_symbol_clears_gate():
    preds = [
        _FakeUnderlying("AAPL", 0.20),
        _FakeUnderlying("AMZN", 0.55),  # above default 0.394
        _FakeUnderlying("NVDA", 0.30),
    ]
    step = build_ml_step(threshold=0.394, predictions=preds)
    assert step.step == "ml"
    assert step.success
    assert step.fields["threshold"] == 0.394
    rows = {r["symbol"]: r for r in step.fields["predictions"]}
    assert rows["AMZN"]["gate_hit"] is True
    assert rows["AAPL"]["gate_hit"] is False
    assert rows["NVDA"]["gate_hit"] is False


def test_build_ml_step_failure_when_no_symbol_crosses():
    preds = [_FakeUnderlying("AAPL", 0.20), _FakeUnderlying("AMZN", 0.18)]
    step = build_ml_step(threshold=0.55, predictions=preds)
    assert not step.success
    assert step.reasons == ["no symbol crossed the direction_probability threshold"]


def test_build_ml_step_handles_none_probability():
    preds = [_FakeUnderlying("AAPL", None)]
    step = build_ml_step(threshold=0.55, predictions=preds)
    assert not step.success
    assert step.fields["predictions"][0]["gate_hit"] is False


# ---------------------------------------------------------------------------
# T3: GNN + topology
# ---------------------------------------------------------------------------
def test_build_gnn_step_records_bias_and_centrality():
    gnn = {
        "model_version": "gatv2-news-20260829-0002",
        "node_features": {
            "AAPL": {"bias": 0.42, "centrality": 0.81},
            "MSFT": {"bias": 0.05, "centrality": 0.40},
        },
        "edges": [
            {"src": "AAPL", "dst": "MSFT", "kind": "sector", "weight": 0.30},
            {"src": "AAPL", "dst": "NVDA", "kind": "news", "weight": 0.55},
        ],
    }
    step = build_gnn_step(gnn)
    assert step.success
    assert step.fields["model_version"] == "gatv2-news-20260829-0002"
    assert step.fields["n_nodes"] == 2
    assert step.fields["n_edges"] == 2
    assert step.fields["edge_kinds"] == {"sector": 1, "news": 1}
    aapl = next(r for r in step.fields["node_features"] if r["symbol"] == "AAPL")
    assert aapl["bias_strong"] is True
    msft = next(r for r in step.fields["node_features"] if r["symbol"] == "MSFT")
    assert msft["bias_strong"] is False


def test_build_gnn_step_failure_when_no_payload():
    assert build_gnn_step(None).success is False
    assert build_gnn_step({}).success is False  # no node_features -> no |bias|>=0.10


def test_build_topology_step_computes_density_and_news_edge_count():
    gnn = {
        "model_version": "v1",
        "node_features": {"AAPL": {}, "MSFT": {}, "NVDA": {}},
        "edges": [
            {"src": "AAPL", "dst": "MSFT", "kind": "sector", "weight": 0.5},
            {"src": "AAPL", "dst": "NVDA", "kind": "news", "weight": 0.7},
            {"src": "MSFT", "dst": "NVDA", "kind": "news", "weight": 0.2},
        ],
    }
    step = build_topology_step(gnn, topology_version="v1")
    assert step.step == "topology"
    assert step.success
    assert step.fields["n_nodes"] == 3
    assert step.fields["n_edges"] == 3
    assert step.fields["news_edges"] == 2
    # density of complete graph on 3 nodes = 1.0
    assert step.fields["density"] == pytest.approx(1.0, abs=1e-6)
    assert step.fields["weight_min"] == 0.2
    assert step.fields["weight_max"] == 0.7
    assert step.fields["weight_mean"] == pytest.approx(0.4666, abs=1e-3)


def test_build_topology_step_flags_degenerate_graph():
    gnn = {"node_features": {"AAPL": {}, "MSFT": {}}, "edges": []}
    step = build_topology_step(gnn, topology_version="v1")
    assert not step.success
    assert "zero edges" in step.reasons[0]


# ---------------------------------------------------------------------------
# T4: research
# ---------------------------------------------------------------------------
def test_build_research_step_distinguishes_off_vs_empty():
    # Off -> success=False with feature-off reason
    off = _FakeResearch(per_symbol={}, flag="news-off")
    step = build_research_step(off)
    assert not step.success
    assert "off" in step.reasons[0].lower()
    # On but no news -> success=False with empty-news reason
    empty = _FakeResearch(per_symbol={}, flag="news-on")
    step = build_research_step(empty)
    assert not step.success
    assert "no news" in step.reasons[0]
    # On with news -> success=True
    on = _FakeResearch(per_symbol={"AAPL": _SymbolR(0.4, 3, ["earnings"])}, flag="news-on")
    step = build_research_step(on)
    assert step.success
    assert step.fields["n_symbols_with_news"] == 1
    assert step.fields["sentiment_mean"] == 0.4


def test_build_research_step_handles_none():
    step = build_research_step(None)
    assert not step.success


# ---------------------------------------------------------------------------
# T5: agent observation
# ---------------------------------------------------------------------------
def test_build_agent_step_carries_confidence_and_risks():
    obs = _FakeObs(agent_id="direction", message_type="DIRECTION_VIEW", confidence=0.7)
    step = build_agent_step(obs)
    assert step.step == "direction"
    assert step.success
    assert step.fields["confidence"] == 0.7
    assert step.fields["message_type"] == "DIRECTION_VIEW"


def test_build_agent_step_fails_on_low_confidence_or_risks():
    low = _FakeObs(confidence=0.3)
    step = build_agent_step(low)
    assert not step.success
    risky = _FakeObs(confidence=0.8, risks=["IV/RV gap too narrow"])
    step = build_agent_step(risky)
    assert not step.success
    assert "IV/RV gap too narrow" in step.reasons


# ---------------------------------------------------------------------------
# T6: final step
# ---------------------------------------------------------------------------
def test_build_final_step_success_on_proceed_with_intent():
    state = _FakeState(
        action="PROCEED",
        selected=_FakeSelected(underlying="AAPL", strategy_id="long_call"),
        risk=_FakeRiskDecision(action="APPROVE"),
        intent=_FakeOrderIntent(legs=[1, 2]),
    )
    step = build_final_step(state)
    assert step.step == "final"
    assert step.success
    assert step.fields["action"] == "PROCEED"
    assert step.fields["selected_underlying"] == "AAPL"
    assert step.fields["order_legs"] == 2


def test_build_final_step_records_risk_reject_reason():
    state = _FakeState(action="REJECT", risk=_FakeRiskDecision(action="REJECT", reasons=["DTE 25 > 10"]))
    step = build_final_step(state)
    assert not step.success
    assert step.fields["action"] == "REJECT"
    assert "DTE 25 > 10" in step.fields["risk_reasons"]
    assert step.reasons == ["action=REJECT"]


def test_build_final_step_no_trade_is_failure_with_legs_zero():
    state = _FakeState(action="NO_TRADE", risk=None, intent=None)
    step = build_final_step(state)
    assert not step.success
    assert step.fields["order_legs"] == 0
