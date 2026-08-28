"""Phase-3 multi-agent package.

Layout::

    src/agents/
        protocol.py        - Pydantic message contracts (versioned)
        llm/                - LLM provider abstraction (mock, openai, anthropic)
        inference.py        - ML / GNN inference service
        nodes/              - one module per specialized agent
        graph.py            - LangGraph state graph
        journal.py          - SQLite decision journal
        risk.py             - deterministic risk engine
        scoring.py          - candidate strategy scoring
        execution.py        - OrderIntent validator + Alpaca submit
        run.py              - CLI entry point
"""

from .protocol import (  # noqa: F401
    SCHEMA_VERSION,
    AgentMessage,
    AgentObservation,
    DecisionState,
    Leg,
    MarketSnapshot,
    OrderIntent,
    RiskDecision,
    StrategyProposal,
)

__all__ = [
    "SCHEMA_VERSION",
    "AgentMessage",
    "AgentObservation",
    "DecisionState",
    "Leg",
    "MarketSnapshot",
    "OrderIntent",
    "RiskDecision",
    "StrategyProposal",
]
