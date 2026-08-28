"""Versioned Pydantic contracts for the Phase-3 multi-agent protocol.

Every inter-agent message MUST conform to one of the models in this module.
The orchestrator is the only component that owns a DecisionState; agents see
a copy of the state and return a *partial* update dict (LangGraph merge).

Why strict typing here?
- LLM outputs are validated against the same schema, so a hallucinated
  field name or wrong enum never reaches the deterministic risk layer.
- The schemas double as the on-the-wire format and the persisted format in
  the decision_journal table.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class RiskAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REDUCE_SIZE = "REDUCE_SIZE"


class MessageType(str, Enum):
    REGIME_VIEW = "REGIME_VIEW"
    DIRECTION_VIEW = "DIRECTION_VIEW"
    VOLATILITY_VIEW = "VOLATILITY_VIEW"
    STRATEGY_PROPOSAL = "STRATEGY_PROPOSAL"
    PORTFOLIO_VIEW = "PORTFOLIO_VIEW"
    RISK_DECISION = "RISK_DECISION"
    SUPERVISOR_DECISION = "SUPERVISOR_DECISION"
    EXECUTION_RESULT = "EXECUTION_RESULT"


class RegimeLabel(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StrictModel(BaseModel):
    """Base model: reject unknown fields and coerce NaN/inf to safe defaults."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    @field_validator("*", mode="before")
    @classmethod
    def _no_nans(cls, v: Any) -> Any:
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                return None
        return v


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------
class UnderlyingScore(StrictModel):
    """One underlying's ML/GNN scores at a single timestamp."""
    symbol: str
    timestamp: str
    horizon_bars: int
    direction_probability: float | None = None
    predicted_future_realized_vol: float | None = None
    predicted_iv: float | None = None  # not yet populated (no IV data)
    gnn_directional_bias: float | None = None  # [-1, +1]
    gnn_centrality: float | None = None
    model_version: str | None = None


class OptionScore(StrictModel):
    """One option contract's ML scores at a single timestamp."""
    contract_symbol: str
    underlying: str
    timestamp: str
    horizon_bars: int
    probability_profitable: float | None = None
    expected_return: float | None = None
    moneyness: float | None = None
    days_to_expiry: int | None = None
    option_volatility_16: float | None = None
    model_version: str | None = None


class PortfolioPosition(StrictModel):
    """One open position used by the Portfolio Agent."""
    symbol: str                  # OCC option symbol OR underlying ticker
    asset_class: Literal["option", "equity"]
    side: Side
    quantity: int
    entry_price: float
    current_price: float | None = None
    unrealized_pnl: float | None = None
    delta: float | None = None
    vega: float | None = None


class MarketSnapshot(StrictModel):
    """Everything an agent might need to reason about. Built once per cycle."""
    timestamp: str
    underlyings: list[UnderlyingScore] = Field(default_factory=list)
    options: list[OptionScore] = Field(default_factory=list)
    portfolio: list[PortfolioPosition] = Field(default_factory=list)
    account_equity: float | None = None
    account_cash: float | None = None


# ---------------------------------------------------------------------------
# Agent observations (LLM-backed)
# ---------------------------------------------------------------------------
class AgentObservation(StrictModel):
    agent_id: str
    message_type: MessageType
    timestamp: str = Field(default_factory=_utcnow)
    confidence: float = Field(ge=0.0, le=1.0)
    signal: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    data_version: str | None = None
    model_versions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy proposal
# ---------------------------------------------------------------------------
class Leg(StrictModel):
    contract_symbol: str
    side: Side
    quantity: int = Field(gt=0)
    option_type: OptionType
    strike: float
    expiry: str                            # YYYY-MM-DD
    limit_price: float | None = None       # None = market


class StrategyProposal(StrictModel):
    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    underlying: str
    legs: list[Leg]
    thesis: str
    expected_return: float
    probability_profit: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    max_loss: float
    liquidity_metrics: dict[str, float] = Field(default_factory=dict)
    expiry: str                            # YYYY-MM-DD of the longest leg
    score: float = 0.0                     # filled by the scoring module
    created_at: str = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Risk decision
# ---------------------------------------------------------------------------
class RiskCheck(StrictModel):
    name: str
    passed: bool
    detail: str = ""


class RiskDecision(StrictModel):
    decision: RiskAction
    approved_quantity: int
    max_loss: float
    checks: list[RiskCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    decided_at: str = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Order intent
# ---------------------------------------------------------------------------
class OrderIntent(StrictModel):
    """A *validated* trade order. Only the execution layer can submit it."""
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    broker: Literal["ALPACA"] = "ALPACA"
    account_mode: Literal["PAPER", "LIVE"] = "PAPER"
    strategy_id: str
    underlying: str
    legs: list[Leg]
    quantity: int = Field(gt=0)
    limit_price: float | None = None
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    created_at: str = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Inter-agent message envelope
# ---------------------------------------------------------------------------
class AgentMessage(StrictModel):
    """Transport-independent envelope. Phase 3 dispatches in-process; the
    schema is deliberately broker-friendly so Redis/Kafka can be swapped in
    later without touching agent code."""
    schema_version: str = SCHEMA_VERSION
    decision_id: str
    timestamp: str = Field(default_factory=_utcnow)
    sender: str
    receiver: str
    message_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# DecisionState - the only object the orchestrator owns end-to-end
# ---------------------------------------------------------------------------
class DecisionState(StrictModel):
    """Per-cycle state that the LangGraph state machine mutates."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    cycle_started_at: str = Field(default_factory=_utcnow)
    cycle_completed_at: str | None = None

    market_snapshot: MarketSnapshot | None = None
    ml_predictions: list[UnderlyingScore] = Field(default_factory=list)
    gnn_output: dict[str, Any] = Field(default_factory=dict)
    topology_version: str | None = None

    agent_observations: list[AgentObservation] = Field(default_factory=list)
    agent_messages: list[AgentMessage] = Field(default_factory=list)
    candidate_strategies: list[StrategyProposal] = Field(default_factory=list)
    selected_strategy: StrategyProposal | None = None

    risk_decision: RiskDecision | None = None
    order_intent: OrderIntent | None = None
    execution_result: dict[str, Any] | None = None

    final_action: Literal["PROCEED", "REDUCE", "REJECT", "NO_TRADE"] = "NO_TRADE"
    realized_pnl: float | None = None
    outcome_label: str | None = None

    @property
    def market_state_hash(self) -> str:
        """Stable hash for journal rows so identical cycles collapse."""
        import hashlib
        import json
        if self.market_snapshot is None:
            return ""
        payload = self.market_snapshot.model_dump()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
