"""Options Structure Agent.

Ranks candidate option strategies from the universe. Pulls `ml_predictions`,
filters by risk-limits, scores via the linear formula, returns the top N
candidates. The LLM is used to generate the *thesis* and to choose between
near-equal candidates; the math is deterministic.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

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
from ._common import AgentResult, _llm_call, _log_agent_observation, _to_message


def build_node(llm, config: dict[str, Any], risk_limits, *, skills=None):
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
    # Hackathon mandate: options are MANDATORY for the Alpaca AI Trading
    # Agents Hackathon (Aug 28 - Sept 4, 2026). When a valid option
    # candidate exists (real ML model prediction, in the strict DTE
    # window, clears the min-score gate), we apply a score boost so it
    # reliably outscores the parallel long-equity candidate. The boost
    # is sized to overcome the equity side's score advantage: zero
    # spread/liquidity penalties (the option leg carries 0.10 + 0.10)
    # plus the option model's conservative expected_return.
    option_mandate_boost = float(
        # Env override first so an operator can bump the boost without
        # editing config/agents.yaml (e.g. ``AIZEN_OPTION_MANDATE_BOOST=0.40
        # python -m src.agents.cli.run_loop --once``). Falls back to
        # config/agents.yaml's scoring.option_mandate_boost, then 0.30.
        os.environ.get("AIZEN_OPTION_MANDATE_BOOST")
        or (config.get("scoring") or {}).get("option_mandate_boost", 0.30)
    )
    # Option-side rejection gates (P0-1 fix). The 2026-09-03 AVGO live
    # fill exposed two gaps in the candidate build path:
    #   1) the supervisor's confidence gate was bypassed by inflating
    #      probability_profit with the mandate boost, so a 0.37 P(profit)
    #      option was dressed up to 0.67 and approved;
    #   2) there was no direction-vs-leg match check, so a long put was
    #      built for AVGO despite direction_probability=0.667 (bullish).
    # These two gates are now hard floors inside _build_candidates so a
    # bad candidate dies before scoring. Defaults:
    #   * min_option_probability_profitable = 0.40 (must be profitable
    #     at least 40% of the time, ignoring mandate);
    #   * direction_mismatch_buffer = 0.05 (a long-PUT needs dp<0.45,
    #     a long-CALL needs dp>0.55). Anything in the 0.45-0.55 dead-band
    #     is neutral and both sides are allowed.
    min_option_pp = float(
        os.environ.get("AIZEN_MIN_OPTION_PP")
        or (config.get("thresholds") or {}).get("min_option_probability_profitable", 0.40)
    )
    direction_mismatch_buffer = float(
        os.environ.get("AIZEN_DIRECTION_MISMATCH_BUFFER")
        or (config.get("thresholds") or {}).get("direction_mismatch_buffer", 0.05)
    )

    def node(state: DecisionState) -> dict[str, Any]:
        snap = state.market_snapshot
        if snap is None:
            return {}
        option_candidates, dte_fallback, rejection_summary = _build_candidates(
            snap, weights, candidate_min_score, candidate_size,
            dte_min=risk_limits.allowed_expiries_dte_min,
            dte_max=risk_limits.allowed_expiries_dte_max,
            gnn_output=state.gnn_output or {},
            option_mandate_boost=option_mandate_boost,
            min_option_pp=min_option_pp,
            direction_mismatch_buffer=direction_mismatch_buffer,
        )
        # Parallel options+stocks path: build long-equity candidates
        # from the same ML/GNN signal. The supervisor then chooses the
        # best mix of options and equity based on score + risk limits.
        equity_candidates = _build_equity_candidates(
            snap, weights, candidate_min_score, candidate_size,
            gnn_output=state.gnn_output or {},
            risk_limits=risk_limits,
        )
        # Combine into one ranked list. The supervisor / strategy
        # selector downstream picks one based on the final scoring.
        candidates = rank(option_candidates + equity_candidates)[:candidate_size]
        # Have the LLM choose/explain the top candidate, but we keep ALL
        # candidate proposals in the state so the supervisor can see them.
        if candidates:
            obs = _llm_call(llm, "options_structure_agent", role,
                            {"candidates": [c.model_dump(mode="json") for c in candidates]},
                            AgentObservation)
            n_options = sum(
                1 for c in candidates
                if c.legs and c.legs[0].asset_class == "option"
            )
            n_equity = len(candidates) - n_options
            signal = {
                "top_strategy_id": candidates[0].strategy_id,
                "candidates_returned": len(candidates),
                "candidates_option": n_options,
                "candidates_equity": n_equity,
            }
            if dte_fallback is not None:
                # Surface the widened DTE window in the signal so the
                # trace + supervisor both see why we're not in the strict
                # 5-10 band. Treated as informational, not a risk.
                signal["dte_fallback"] = dte_fallback
            # P0-1: surface option-rejection counts so the operator can
            # see WHY the option side is short. Without this the trace
            # shows "candidates_option=0" with no reason and the next
            # debugging step is always "what was rejected?".
            if rejection_summary and any(rejection_summary.values()):
                signal["option_rejections"] = rejection_summary
            obs = obs.model_copy(update={
                "message_type": MessageType.STRATEGY_PROPOSAL,
                "signal": signal,
            })
        else:
            empty_signal: dict[str, Any] = {"candidates_returned": 0}
            if dte_fallback is not None:
                empty_signal["dte_fallback"] = dte_fallback
            if rejection_summary and any(rejection_summary.values()):
                empty_signal["option_rejections"] = rejection_summary
            obs = AgentObservation(
                agent_id="options_structure_agent",
                message_type=MessageType.STRATEGY_PROPOSAL,
                confidence=0.2,
                signal=empty_signal,
                evidence=["no option chains or equity signals met the min-score filter"],
                risks=["empty candidate set"],
                data_version="options_structure-1",
                model_versions=["options_structure-1"],
            )
        msg = _to_message(obs, state.decision_id,
                          "options_structure_agent", "supervisor")
        _log_agent_observation("options_structure_agent", obs)
        return {
            "agent_observations": [obs],
            "agent_messages": [msg],
            "candidate_strategies": candidates,
        }

    return node


def _build_candidates(
    snap, weights: ScoringWeights, min_score: float, top_n: int,
    dte_min: int, dte_max: int,
    gnn_output: dict | None = None,
    option_mandate_boost: float = 0.0,
    min_option_pp: float = 0.40,
    direction_mismatch_buffer: float = 0.05,
) -> tuple[list[StrategyProposal], str | None, dict[str, list[str]]]:
    """Build option candidate strategies from the option chain.

    Returns ``(proposals, dte_fallback_note, rejection_summary)``.

    The third return value is new in P0-1: a per-reason counter list so
    the operator trace shows WHY options were dropped. Today the
    candidate set is "0" or "5" with no visibility into the
    intermediate rejections, and a 0/5 cycle is the most common
    question in cron debugging. ``rejection_summary`` keys are:

      * ``min_pp`` — contracts whose ``probability_profitable`` is below
        ``min_option_pp`` (default 0.40);
      * ``direction_mismatch`` — contracts whose leg side contradicts
        the direction signal (long-PUT when direction_probability is
        bullish past the buffer, or long-CALL when bearish);
      * ``missing_data`` — contracts with no ``probability_profitable``
        / ``expected_return`` / ``days_to_expiry``.

    Each value is a short list of ``"<symbol>:<strike>:<reason>"`` for
    the first few rejects (capped at 8) so the trace JSON stays small.
    """
    out: list[StrategyProposal] = []
    rejection_summary: dict[str, list[str]] = {
        "min_pp": [],
        "direction_mismatch": [],
        "missing_data": [],
    }
    gnn_features = (gnn_output or {}).get("node_features", {}) or {}
    # Membership is by string symbol; Pydantic UnderlyingScore instances
    # are not equal to plain strings, so build a set up front.
    universe_symbols = {u.symbol for u in snap.underlyings}
    # Pre-pass: find the DTE range that actually has contracts in the
    # snapshot. The hackathon window is 5-10 DTE, but the synthetic
    # option_contracts table only has 18-60 DTE today, so an empty
    # primary window would dead-end the whole pipeline. When the
    # strict window is empty, fall back to "the shortest DTE band that
    # has at least one contract" (capped at 30 to keep the trade
    # short-horizon) and surface a 'dte_fallback' trace field so the
    # operator can see why we widened.
    dte_min_eff, dte_max_eff, dte_fallback = _resolve_dte_window(
        snap.options, dte_min, dte_max,
    )
    # Build one long-call candidate per underlying with a viable option chain.
    for opt in snap.options:
        underlying = opt.underlying
        if underlying not in universe_symbols:
            continue
        if opt.probability_profitable is None or opt.expected_return is None:
            if len(rejection_summary["missing_data"]) < 8:
                rejection_summary["missing_data"].append(
                    f"{opt.contract_symbol}:no_pp_or_er"
                )
            continue
        if opt.days_to_expiry is None:
            if len(rejection_summary["missing_data"]) < 8:
                rejection_summary["missing_data"].append(
                    f"{opt.contract_symbol}:no_dte"
                )
            continue
        dte = opt.days_to_expiry
        if not (dte_min_eff <= dte <= dte_max_eff):
            continue
        if opt.days_to_expiry <= 0:
            continue
        # P0-1 fix: hard floor on probability_profitable. The 2026-09-03
        # AVGO long-put was approved with P(profit)=0.37 because the
        # mandate boost later inflated the supervisor's confidence. We
        # cut the candidate here, BEFORE scoring, so it never sees the
        # boost. The threshold is a probability, not a confidence, so
        # the boost is now only a tie-breaker inside the score itself.
        if float(opt.probability_profitable) < min_option_pp:
            if len(rejection_summary["min_pp"]) < 8:
                rejection_summary["min_pp"].append(
                    f"{opt.contract_symbol}:pp={float(opt.probability_profitable):.2f}"
                )
            continue
        # P0-1 fix: direction-mismatch gate. A long PUT profits from a
        # drop; a long CALL profits from a rise. If the direction
        # signal is strongly opposite the leg's payoff shape, the
        # candidate is dropped with a reason that surfaces in the
        # trace. The buffer keeps 0.45-0.55 neutral symbols (a
        # coin-flip direction) eligible for both sides, so this is
        # not a one-sided restriction.
        leg_option_type = (
            OptionType.CALL if "C" in opt.contract_symbol else OptionType.PUT
        )
        dp = _direction_probability(snap, underlying)
        bull_threshold = 0.50 + direction_mismatch_buffer
        bear_threshold = 0.50 - direction_mismatch_buffer
        if leg_option_type == OptionType.PUT and dp is not None and dp > bull_threshold:
            if len(rejection_summary["direction_mismatch"]) < 8:
                rejection_summary["direction_mismatch"].append(
                    f"{opt.contract_symbol}:PUT vs dp={dp:.2f} (bullish)"
                )
            continue
        if leg_option_type == OptionType.CALL and dp is not None and dp < bear_threshold:
            if len(rejection_summary["direction_mismatch"]) < 8:
                rejection_summary["direction_mismatch"].append(
                    f"{opt.contract_symbol}:CALL vs dp={dp:.2f} (bearish)"
                )
            continue
        # Heuristic input features for the scoring formula.
        direction_edge = _direction_edge(snap, underlying)
        volatility_edge = _volatility_edge(snap, underlying)
        # Prefer the GNN signal from state.gnn_output.node_features;
        # fall back to the UnderlyingScore field for backward compat.
        gnn_confirmation = _gnn_bias_from_features(
            gnn_features, underlying, snap
        )
        spread_penalty = 0.10  # TODO: real spread when quotes are available
        liquidity_penalty = 0.20 if opt.option_volatility_16 is None else 0.10
        portfolio_risk_penalty = 0.10

        leg = Leg(
            contract_symbol=opt.contract_symbol,
            side=Side.BUY,
            quantity=1,
            option_type=OptionType.CALL if "C" in opt.contract_symbol else OptionType.PUT,
            strike=_strike_from_symbol(opt.contract_symbol),
            # Bug A fix: Leg.expiry must be the contract's actual
            # expiration_date, NOT the latest option_bars.timestamp.
            # Using the bar's date (often days older than the live
            # bar refresh cadence) produces a Leg with a past expiry,
            # which the broker rejects. Fall back to the OCC symbol's
            # encoded expiry if the inference layer didn't propagate
            # expiration_date, then to a far-future default.
            expiry=(
                opt.expiration_date
                or _expiry_from_occ_symbol(opt.contract_symbol)
                or "2099-12-31"
            ),
        )
        proposal = StrategyProposal(
            underlying=underlying,
            legs=[leg],
            thesis=(
                f"Long {leg.option_type.value} on {underlying} with "
                f"P(profit)={float(opt.probability_profitable or 0):.2f}, "
                f"expected return={float(opt.expected_return or 0):.3f}, "
                f"gnn_bias={gnn_confirmation:+.2f}."
            ),
            expected_return=float(opt.expected_return or 0.0),
            probability_profit=float(opt.probability_profitable or 0.0),
            confidence=float(opt.probability_profitable or 0.0),
            max_loss=float(max(1.0, (opt.expected_return or 0.0) * 200.0 + 100.0)),
            liquidity_metrics={"option_volatility_16": float(opt.option_volatility_16 or 0.0)},
            # StrategyProposal.expiry is the longest leg's expiry; for
            # a single-leg call/put, that's just this leg's expiry.
            expiry=leg.expiry,
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
        # Hackathon mandate: when this option came from a real ML
        # prediction (model_version starts with the option_h4 artifact
        # name) AND falls inside the strict DTE window we asked for, add
        # a fixed score boost so it reliably outscores the parallel
        # long-equity candidate. The boost is NOT applied to
        # heuristic-fallback rows (model_version = "heuristic-1") —
        # those are placeholder rows the option ML path produced when
        # its features were unavailable, and trading them would not
        # exercise the option_h4 model the way the hackathon rubric
        # expects.
        #
        # P0-1 fix: the boost is applied to the **score** only.
        # Previously the same number was also added to the proposal's
        # confidence so the candidate could clear the supervisor's
        # confidence_min=0.50 gate even when probability_profitable
        # was 0.37. That dressed up a sub-50% model output as a 67%
        # confident trade and is the root cause of the 2026-09-03 AVGO
        # long-put fill. The hard ``min_option_pp`` floor above is the
        # real confidence gate now; the boost is just a tie-breaker
        # between options and equity inside the score formula.
        model_version = str(opt.model_version or "")
        is_real_ml = (
            model_version.startswith("option_h4_")
            and not model_version.startswith("heuristic")
        )
        in_strict_window = dte_min <= dte <= dte_max
        if option_mandate_boost and is_real_ml and in_strict_window:
            score = score + option_mandate_boost
            # Surface the boost on the thesis so the trace shows why
            # this option outranked the equity leg. Confidence is NOT
            # touched here on purpose — see the long comment above.
            proposal = proposal.model_copy(update={
                "thesis": proposal.thesis + f" [mandate+{option_mandate_boost:.2f}]",
            })
        proposal = proposal.model_copy(update={"score": score})
        if score >= min_score:
            out.append(proposal)
    if dte_fallback is not None:
        # Stash the fallback on the last surviving candidate's signal so
        # the supervisor / trace can show the operator *why* the window
        # was widened. The note is informational, not a risk.
        if out:
            out[0] = out[0].model_copy(update={"thesis": out[0].thesis + f" [DTE fallback: {dte_fallback}]"})
    return rank(out)[:top_n], dte_fallback, rejection_summary


def _direction_probability(snap, underlying: str) -> float | None:
    """Read ``direction_probability`` for an underlying from the snapshot.
    Returns ``None`` when missing (the caller treats that as "no signal"
    and skips the direction-mismatch gate)."""
    for u in snap.underlyings:
        if u.symbol == underlying:
            dp = getattr(u, "direction_probability", None)
            if dp is None:
                return None
            try:
                return float(dp)
            except (TypeError, ValueError):
                return None
    return None


def _build_equity_candidates(
    snap, weights: ScoringWeights, min_score: float, top_n: int,
    gnn_output: dict | None,
    risk_limits,
) -> list[StrategyProposal]:
    """Build long-equity candidate strategies from the directional signal.

    Parallel path to :func:`_build_candidates` (options). The thesis is
    the same — a strong directional signal produces a candidate — but
    instead of an option leg we emit a single ``equity`` leg sized to
    the per-symbol notional cap. Long-only by design (puts would
    require a margin account and the Pattern Day Trader rule; we run
    cash in the hackathon).

    Gating rules (all must hold for a symbol to produce a candidate):
      * ``direction_probability >= risk_limits.min_equity_direction_probability``
        (default 0.55; the signal must be more bullish than coin-flip)
      * ``last_price`` is set (no fabricated price; missing price means
        the symbol has no underlying bar at the cut-off and should be
        skipped)
      * quantity * last_price <= ``max_equity_notional_per_symbol``
        (so the proposal is already pre-sized within risk; the
        downstream risk engine will still re-check the notional cap)

    Returns at most ``top_n`` candidates, ranked by score. The function
    never raises — symbols that fail any gate are silently dropped.
    """
    if snap is None:
        return []
    gnn_features = (gnn_output or {}).get("node_features", {}) or {}
    min_dp = float(getattr(risk_limits, "min_equity_direction_probability", 0.55))
    max_per_symbol = float(getattr(risk_limits, "max_equity_notional_per_symbol", 1500.0))
    out: list[StrategyProposal] = []
    for u in snap.underlyings:
        # Gate 1: directional signal must be above threshold.
        if u.direction_probability is None:
            continue
        if u.direction_probability < min_dp:
            continue
        # Gate 2: must have a price (no fabricated last_price).
        if u.last_price is None or u.last_price <= 0:
            continue
        price = float(u.last_price)
        # Gate 3: pre-size within per-symbol notional cap. Floor qty at 1
        # so the proposal is non-empty; the risk engine will REDUCE_SIZE
        # or REJECT if the broader notional total is exceeded.
        raw_qty = int(max_per_symbol // price)
        if raw_qty < 1:
            # Price too high to even buy one share under the cap.
            continue
        qty = raw_qty
        notional = qty * price
        # Build the long-equity leg. asset_class=equity, no option_type /
        # strike / expiry (the Leg schema makes those Optional for
        # equity legs).
        leg = Leg(
            asset_class="equity",
            contract_symbol=u.symbol,
            side=Side.BUY,
            quantity=qty,
            limit_price=round(price, 2),
        )
        # Equity scoring inputs:
        # * direction_edge: same as the option path — bullish model => +1
        # * volatility_edge: use predicted rv, capped at 1
        # * expected_return: simplified to ``max(0, dp - 0.5) * 2``
        #   (no premium to decay, so the proxy is just the directional
        #   edge, doubled so 0.7 dp -> 0.4 er)
        # * probability_profit: the direction_probability itself
        # * gnn_confirmation: from the GNN node features (or the
        #   UnderlyingScore fallback for backward compat)
        direction_edge = (float(u.direction_probability) - 0.5) * 2.0
        volatility_edge = min(1.0, max(0.0, float(u.predicted_future_realized_vol or 0.0) * 5.0))
        er = max(0.0, min(1.0, (float(u.direction_probability) - 0.5) * 2.0))
        pp = max(0.0, min(1.0, float(u.direction_probability)))
        gnn_confirmation = _gnn_bias_from_features(gnn_features, u.symbol, snap)
        # Stocks have negligible spread and high liquidity for our
        # universe (SPY-grade names), so both penalties are zero.
        spread_penalty = 0.0
        liquidity_penalty = 0.0
        # Portfolio risk penalty: equity legs tie up the full notional,
        # so a moderate penalty is fair. Sized so a single equity
        # candidate can still beat the option score when dp is strong.
        portfolio_risk_penalty = 0.10
        proposal = StrategyProposal(
            underlying=u.symbol,
            legs=[leg],
            thesis=(
                f"Long {qty}sh of {u.symbol} @ ~${price:.2f} "
                f"(notional=${notional:.0f}), P(up)={pp:.2f}, "
                f"gnn_bias={gnn_confirmation:+.2f}."
            ),
            expected_return=er,
            probability_profit=pp,
            confidence=pp,
            max_loss=float(notional),  # worst case: stock goes to zero
            liquidity_metrics={"equity_notional": float(notional)},
            expiry=snap.timestamp[:10] if snap.timestamp else "2099-12-31",
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


def _resolve_dte_window(
    options, dte_min: int, dte_max: int,
) -> tuple[int, int, str | None]:
    """Return (min, max, fallback_note).

    If the strict [dte_min, dte_max] band has at least one option, return
    it untouched. Otherwise widen to the shortest available DTE band and
    return a non-None ``fallback_note`` describing the widening.

    Widening rules (in priority order):
      1. Try [shortest, shortest+10] — gives ~2 weekly expirations to
         choose from without stepping too far into theta territory.
      2. If that's empty too, widen to [shortest, min(longest, dte_max+5)]
         so the supervisor still has at least one contract to score.
      3. The intent cap is dte_min+25, but if even the shortest
         available is above that cap, use [shortest, min(longest,
         shortest+10)] — a band centered on the shortest DTE — to
         guarantee a non-empty window.
    """
    available = sorted({int(o.days_to_expiry) for o in options if o.days_to_expiry is not None and o.days_to_expiry > 0})
    if not available:
        return dte_min, dte_max, "no_dte_data"
    in_strict = [d for d in available if dte_min <= d <= dte_max]
    if in_strict:
        return dte_min, dte_max, None
    # Strict window is empty: widen.
    shortest = available[0]
    longest = available[-1]
    # The intent cap is dte_min+25 to keep the trade short-horizon, but
    # if shortest already exceeds that cap, fall back to a window of
    # [shortest, min(longest, shortest+10)] so min <= max is preserved.
    intent_cap = dte_min + 25
    if shortest <= intent_cap:
        widened_min = shortest
        widened_max = min(longest, intent_cap)
    else:
        widened_min = shortest
        widened_max = min(longest, shortest + 10)
    note = f"strict [{dte_min},{dte_max}] empty; widened to [{widened_min},{widened_max}]"
    return widened_min, widened_max, note


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


def _gnn_bias_from_features(
    gnn_features: dict[str, dict[str, Any]], underlying: str, snap
) -> float:
    """Pull bias from state.gnn_output.node_features, falling back to
    the legacy UnderlyingScore field for backward compat."""
    node = gnn_features.get(underlying) if gnn_features else None
    if isinstance(node, dict) and node.get("bias") is not None:
        return float(node["bias"])
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


def _expiry_from_occ_symbol(symbol: str) -> str | None:
    """Decode the 6-digit YYMMDD expiry code from an OCC option symbol.

    OCC symbols have the form ``TICKER + YYMMDD + C/P + 8-digit-strike``,
    e.g. ``META260902C00572500`` -> expiry ``2026-09-02``. Used as a
    defense-in-depth fallback when the inference layer didn't
    propagate ``option_contracts.expiration_date`` to the
    OptionScore. Returns None on any parse failure rather than a
    fabricated date - the Leg builder falls through to its "2099-12-31"
    far-future default in that case.
    """
    if len(symbol) < 15:
        return None
    code = symbol[-15:-9]   # 6 digits: YYMMDD
    if not code.isdigit():
        return None
    try:
        from datetime import date
        yy, mm, dd = int(code[0:2]), int(code[2:4]), int(code[4:6])
        # OCC uses a rolling 50-year window. For YYMMDD < 70 (i.e.
        # 2000-2069) we map to 20YY; for 70-99 (2070-2099) we map
        # to 19YY. Today's hackathon window is 2026 so we use 20YY
        # unconditionally.
        year = 2000 + yy
        return date(year, mm, dd).isoformat()
    except (ValueError, IndexError):
        return None
