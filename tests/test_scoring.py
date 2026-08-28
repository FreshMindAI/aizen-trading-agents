"""Candidate scoring tests."""

from __future__ import annotations

from src.agents.protocol import Leg, OptionType, Side, StrategyProposal
from src.agents.scoring import ScoringWeights, rank, score_candidate


def _proposal(score: float = 0.0, conf: float = 0.5) -> StrategyProposal:
    return StrategyProposal(
        underlying="AAPL",
        legs=[Leg(contract_symbol="AAPL260101C00200000",
                  side=Side.BUY, quantity=1,
                  option_type=OptionType.CALL, strike=200.0, expiry="2026-12-31")],
        thesis="x", expected_return=0.1, probability_profit=0.6,
        confidence=conf, max_loss=100.0, expiry="2026-12-31",
        score=score,
    )


def test_score_increases_with_better_inputs():
    w = ScoringWeights()
    base = score_candidate(_proposal(), w, direction_edge=0.0,
                           volatility_edge=0.0, gnn_confirmation=0.0,
                           spread_penalty=0.0, liquidity_penalty=0.0,
                           portfolio_risk_penalty=0.0)
    better = score_candidate(_proposal(), w, direction_edge=0.8,
                             volatility_edge=0.5, gnn_confirmation=0.4,
                             spread_penalty=0.0, liquidity_penalty=0.0,
                             portfolio_risk_penalty=0.0)
    assert better > base


def test_score_decreases_with_penalties():
    w = ScoringWeights()
    base = score_candidate(_proposal(), w, direction_edge=0.5,
                           volatility_edge=0.0, gnn_confirmation=0.0,
                           spread_penalty=0.0, liquidity_penalty=0.0,
                           portfolio_risk_penalty=0.0)
    penalized = score_candidate(_proposal(), w, direction_edge=0.5,
                                volatility_edge=0.0, gnn_confirmation=0.0,
                                spread_penalty=1.0, liquidity_penalty=1.0,
                                portfolio_risk_penalty=1.0)
    assert penalized < base


def test_rank_sorts_descending():
    a, b, c = _proposal(score=0.1), _proposal(score=0.5), _proposal(score=0.3)
    ranked = rank([a, b, c])
    assert [p.score for p in ranked] == [0.5, 0.3, 0.1]


def test_weights_from_mapping_ignores_unknown_keys():
    w = ScoringWeights.from_mapping({
        "direction_edge": 0.5,
        "made_up_weight": 99.0,
    })
    assert w.direction_edge == 0.5
    # other fields keep their dataclass defaults
    assert w.volatility_edge == ScoringWeights.volatility_edge
