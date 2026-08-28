"""Candidate strategy scoring (doc section 11).

The score is a fixed linear combination of normalized signals. The LLM
explains the score; it does not change the formula. Weights live in
config/agents.yaml under `scoring.weights`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .protocol import StrategyProposal


@dataclass(frozen=True)
class ScoringWeights:
    direction_edge: float = 0.30
    volatility_edge: float = 0.20
    option_expected_return: float = 0.25
    probability_profit: float = 0.20
    gnn_confirmation: float = 0.10
    spread_penalty: float = 0.10
    liquidity_penalty: float = 0.05
    portfolio_risk_penalty: float = 0.15

    @classmethod
    def from_mapping(cls, m: Mapping[str, float]) -> "ScoringWeights":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: float(v) for k, v in m.items() if k in known})


def score_candidate(
    proposal: StrategyProposal,
    weights: ScoringWeights,
    *,
    direction_edge: float,
    volatility_edge: float,
    gnn_confirmation: float,
    spread_penalty: float,
    liquidity_penalty: float,
    portfolio_risk_penalty: float,
) -> float:
    """Linear combination. Inputs are expected in [0, 1] (or [-1, 1] for
    directional edges, which we remap to [0, 1])."""
    dir_e = (direction_edge + 1.0) / 2.0
    s = (
        weights.direction_edge * dir_e
        + weights.volatility_edge * volatility_edge
        + weights.option_expected_return * max(0.0, min(1.0, proposal.expected_return))
        + weights.probability_profit * proposal.probability_profit
        + weights.gnn_confirmation * (gnn_confirmation + 1.0) / 2.0
        - weights.spread_penalty * spread_penalty
        - weights.liquidity_penalty * liquidity_penalty
        - weights.portfolio_risk_penalty * portfolio_risk_penalty
    )
    return float(s)


def rank(proposals: list[StrategyProposal]) -> list[StrategyProposal]:
    """Stable descending sort by score; ties broken by confidence."""
    return sorted(proposals, key=lambda p: (p.score, p.confidence), reverse=True)
