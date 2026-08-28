"""Pydantic protocol contract tests."""

from __future__ import annotations

import json
import math

import pytest

from src.agents.protocol import (
    AgentMessage,
    AgentObservation,
    DecisionState,
    Leg,
    MarketSnapshot,
    MessageType,
    OptionScore,
    OptionType,
    OrderIntent,
    RiskAction,
    RiskCheck,
    RiskDecision,
    SCHEMA_VERSION,
    Side,
    StrategyProposal,
    UnderlyingScore,
)


def test_schema_version_constant():
    assert SCHEMA_VERSION == "1.0"


def test_underlying_score_roundtrip():
    s = UnderlyingScore(
        symbol="AAPL", timestamp="2026-08-28T15:30:00Z", horizon_bars=4,
        direction_probability=0.62, predicted_future_realized_vol=0.21,
        gnn_directional_bias=0.15, model_version="dir_h4_xgb_clf-20260826-002633",
    )
    blob = s.model_dump_json()
    parsed = UnderlyingScore.model_validate_json(blob)
    assert parsed.symbol == s.symbol
    assert parsed.direction_probability == 0.62


def test_strategy_proposal_validates_legs():
    p = StrategyProposal(
        underlying="AAPL",
        legs=[Leg(contract_symbol="AAPL260101C00200000",
                  side=Side.BUY, quantity=1,
                  option_type=OptionType.CALL, strike=200.0, expiry="2026-12-31")],
        thesis="long call", expected_return=0.15, probability_profit=0.6,
        confidence=0.55, max_loss=200.0, expiry="2026-12-31",
    )
    assert p.legs[0].option_type == OptionType.CALL
    assert p.score == 0.0  # default


def test_strategy_proposal_rejects_unknown_field():
    with pytest.raises(Exception):
        StrategyProposal.model_validate({
            "underlying": "AAPL", "legs": [], "thesis": "x",
            "expected_return": 0.1, "probability_profit": 0.5,
            "confidence": 0.5, "max_loss": 100.0, "expiry": "x",
            "made_up_field": "should fail",
        })


def test_nan_in_floats_become_none():
    s = UnderlyingScore(
        symbol="AAPL", timestamp="2026-08-28T15:30:00Z", horizon_bars=4,
        direction_probability=float("nan"),
    )
    assert s.direction_probability is None


def test_confidence_bounded():
    with pytest.raises(Exception):
        AgentObservation(
            agent_id="x", message_type=MessageType.SUPERVISOR_DECISION,
            confidence=1.5,
        )


def test_order_intent_serializes_to_broker_safe_json():
    intent = OrderIntent(
        strategy_id="abc", underlying="AAPL",
        legs=[Leg(contract_symbol="AAPL260101C00200000",
                  side=Side.BUY, quantity=2,
                  option_type=OptionType.CALL, strike=200.0, expiry="2026-12-31",
                  limit_price=3.5)],
        quantity=2, limit_price=3.5,
    )
    blob = intent.model_dump_json()
    parsed = OrderIntent.model_validate_json(blob)
    assert parsed.legs[0].limit_price == 3.5
    # JSON-safe: no NaN / no exotic types.
    data = json.loads(blob)
    assert data["broker"] == "ALPACA"
    assert data["account_mode"] == "PAPER"


def test_agent_message_envelope_roundtrip():
    msg = AgentMessage(
        decision_id="d-1", sender="regime", receiver="supervisor",
        message_type=MessageType.REGIME_VIEW,
        payload={"label": "trending_up"},
    )
    blob = msg.model_dump_json()
    parsed = AgentMessage.model_validate_json(blob)
    assert parsed.payload["label"] == "trending_up"
    assert parsed.schema_version == SCHEMA_VERSION


def test_decision_state_market_hash_is_deterministic():
    snap = MarketSnapshot(
        timestamp="2026-08-28T15:30:00Z",
        underlyings=[UnderlyingScore(symbol="AAPL", timestamp="2026-08-28T15:30:00Z",
                                     horizon_bars=4, direction_probability=0.6)],
    )
    s1 = DecisionState(market_snapshot=snap)
    s2 = DecisionState(market_snapshot=snap)
    assert s1.market_state_hash == s2.market_state_hash
    assert len(s1.market_state_hash) == 64  # sha256 hex


def test_risk_decision_minimum_required_fields():
    rd = RiskDecision(
        decision=RiskAction.APPROVE, approved_quantity=1, max_loss=100.0,
    )
    assert rd.checks == []  # default empty
    assert rd.reasons == []
