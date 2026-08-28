"""Options Structure Agent.

Ranks candidate option strategies from the universe. Pulls `ml_predictions`,
filters by risk-limits, scores via the linear formula, returns the top N
candidates. The LLM is used to generate the *thesis* and to choose between
near-equal candidates; the math is deterministic.
"""

from __future__ import annotations

from typing import Any

from ..protocol import (
    AgentObservation,
    DecisionState,
    Leg,
    MessageType,
    OptionType,
    Side,
    StrategyProposal,
)
from ..scoring import ScoringWeights, rank, score_candidate
from ._common import AgentResult, _llm_call, _to_message


def build_node(llm, config: dict[str, Any], risk_limits):
    role = (
        "Rank option strategies from ML signals. Choose the top-N candidates, "
        "each with a clear thesis, expected return, probability of profit, "
        "and max loss."
    )
    weights = ScoringWeights.from_mapping(
        (config.get("scoring") or {}).get("weights", {})
    )
    candidate_size = (config.get("scoring") or {}).get("candidate_set_size", 5)
    candidate_min_score = (config.get("thresholds") or {}).get("candidate_min_score", 0.30)

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        candidates = _build_candidates(
            snap, weights, candidate_min_score, candidate_size
        )
        # Have the LLM choose/explain the top candidate, but we keep ALL
        # candidate proposals in the state so the supervisor can see them.
        if candidates:
            obs = _llm_call(llm, "options_structure_agent", role,
                            {"candidates": [c.model_dump(mode="json") for c in candidates]},
                            AgentObservation)
            obs = obs.model_copy(update={
                "message_type": MessageType.STRATEGY_PROPOSAL,
                "signal": {"top_strategy_id": candidates[0].strategy_id,
                           "candidates_returned": len(candidates)},
            })
        else:
            obs = AgentObservation(
                agent_id="options_structure_agent",
                message_type=MessageType.STRATEGY_PROPOSAL,
                confidence=0.2,
                signal={"candidates_returned": 0},
                evidence=["no option chains met the min-score filter"],
                risks=["empty candidate set"],
                data_version="options_structure-1",
                model_versions=["options_structure-1"],
            )
        msg = _to_message(obs, state.decision_id,
                          "options_structure_agent", "supervisor")
        return {
            "agent_observations": [obs],
            "agent_messages": [msg],
            "candidate_strategies": candidates,
        }

    return node


def _build_candidates(
    snap, weights: ScoringWeights, min_score: float, top_n: int
) -> list[StrategyProposal]:
    out: list[StrategyProposal] = []
    # Build one long-call candidate per underlying with a viable option chain.
    for opt in snap.options:
        underlying = opt.underlying
        if underlying not in snap.underlyings:
            continue
        if opt.probability_profitable is None or opt.expected_return is None:
            continue
        if opt.days_to_expiry is None:
            continue
        if not (risk_limits.allowed_expiries_dte_min <= opt.days_to_expiry
                <= risk_limits.allowed_expiries_dte_max):
            continue
        if opt.days_to_expiry <= 0:
            continue
        # Heuristic input features for the scoring formula.
        direction_edge = _direction_edge(snap, underlying)
        volatility_edge = _volatility_edge(snap, underlying)
        gnn_confirmation = _gnn_bias(snap, underlying)
        spread_penalty = 0.10  # TODO: real spread when quotes are available
        liquidity_penalty = 0.20 if opt.option_volatility_16 is None else 0.10
        portfolio_risk_penalty = 0.10

        leg = Leg(
            contract_symbol=opt.contract_symbol,
            side=Side.BUY,
            quantity=1,
            option_type=OptionType.CALL if "C" in opt.contract_symbol else OptionType.PUT,
            strike=_strike_from_symbol(opt.contract_symbol),
            expiry=opt.timestamp[:10] if opt.timestamp else "2099-12-31",
        )
        proposal = StrategyProposal(
            underlying=underlying,
            legs=[leg],
            thesis=(
                f"Long {leg.option_type.value} on {underlying} with "
                f"P(profit)={opt.probability_profit:.2f}, "
                f"expected return={opt.expected_return:.3f}."
            ),
            expected_return=float(opt.expected_return),
            probability_profit=float(opt.probability_profit),
            confidence=float(opt.probability_profit),
            max_loss=float(max(1.0, opt.expected_return * 200.0 + 100.0)),
            liquidity_metrics={"option_volatility_16": float(opt.option_volatility_16 or 0.0)},
            expiry=opt.timestamp[:10] if opt.timestamp else "2099-12-31",
        )
        score = score_candidate(
            proposal, weights,
            direction_edge=direction_edge,
            volatility_edge=volatility_edge,
            gnn_confirmation=gnn_confirmation,
            spread_penalty=spread_penalty,
            liquidity_penalty=liquidity_penalty,
            portfolio_risk_penalty=portfolio_risk_penalty,
        )
        proposal = proposal.model_copy(update={"score": score})
        if score >= min_score:
            out.append(proposal)
    return rank(out)[:top_n]


def _direction_edge(snap, underlying: str) -> float:
    for u in snap.underlyings:
        if u.symbol == underlying and u.direction_probability is not None:
            return (float(u.direction_probability) - 0.5) * 2.0
    return 0.0


def _volatility_edge(snap, underlying: str) -> float:
    for u in snap.underlyings:
        if u.symbol == underlying and u.predicted_future_realized_vol is not None:
            return min(1.0, max(0.0, float(u.predicted_future_realized_vol) * 5.0))
    return 0.0


def _gnn_bias(snap, underlying: str) -> float:
    for u in snap.underlyings:
        if u.symbol == underlying and u.gnn_directional_bias is not None:
            return float(u.gnn_directional_bias)
    return 0.0


def _strike_from_symbol(symbol: str) -> float:
    """OCC symbols end with 8 digits encoding strike*1000."""
    try:
        return float(int(symbol[-8:])) / 1000.0
    except (ValueError, IndexError):
        return 0.0
