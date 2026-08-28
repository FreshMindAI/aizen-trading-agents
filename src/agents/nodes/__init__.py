"""LangGraph node adapters for the Phase-3 specialized agents.

Each module exports a single `build_node(llm, config, risk_limits)` factory
that returns a callable `(state: DecisionState) -> dict`. LangGraph merges
the returned dict into the persistent state - we return only the fields the
node mutated.
"""
