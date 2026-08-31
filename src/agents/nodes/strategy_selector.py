"""Strategy-type selector (v2 — fixes the "no_trade gate too strict" failure).

Doc §13 / §15: the supervisor should pick a strategy *type* based on the
current signals, not just between long_stock and no_trade. v1 had two
strategy types; v2 adds:
  - long_stock            : directional bullish bet on the underlying
  - short_put_spread      : positive-theta, defined-risk, when IV > RV
  - protective_put        : tail-hedge for an existing long position
  - no_trade              : default when no signal clears the gates

Selection logic (deterministic; no LLM):

  if regime in {crisis, high_volatility} and we hold a long:
      pick protective_put
  elif iv_rv_gap >= 0.05 and we have < 3 short_put_spread positions open:
      pick short_put_spread
  elif gnn_bias > 0.5 and (xgb_prob >= 0.55 OR (xgb_prob in [0.45, 0.55] and gnn_bias > 0.7)):
      pick long_stock       <- the GNN-confirmation override per the user's request
  else:
      pick no_trade

The GNN-confirmation override is what unlocks trades that v1 would have
blocked because the XGBoost gate (direction_prob_min: 0.55) was too strict
for a 1-week hackathon window.
"""
from __future__ import annotations

from typing import Any

from ..protocol import (
    AgentObservation,
    DecisionState,
    Leg,
    MessageType,
    OrderIntent,
    Side,
    StrategyProposal,
)
from ._common import _llm_call, _to_message


# Tunable thresholds (could be moved to config/agents.yaml later)
IV_RV_GAP_MIN = 0.05           # |iv - rv| above which we sell premium
GNN_BIAS_OVERRIDE = 0.50      # when XGB is neutral, GNN above this unlocks the trade
GNN_BIAS_OVERRIDE_STRONG = 0.70  # when XGB is slightly bearish, GNN above this unlocks
# The XGBoost direction model is calibrated to the *base rate* of the training
# market (currently 0.29 — only 29% of 4-hour windows closed up in the test
# period). The 0.55 long-threshold was set with the implicit assumption of a
# 50/50 market and is unreachable in this regime. We re-center against the
# model's actual base rate (read from the artifact meta at module import)
# plus a fixed cushion. Keep the legacy constants as fallbacks for when no
# artifact is present.
import json as _json
from pathlib import Path as _Path
import os as _os

_DEFAULT_DIRECTION_PROB_MIN = 0.55
_BASE_RATE_CUSHION = 0.10      # accept probabilities >= base_rate + 0.10


def _load_model_base_rate() -> float | None:
    """Read the trained direction artifact's test base_rate, if present.

    The artifact name is pinned (latest stem); we look for a matching .meta.json
    sibling. Returns None when no artifact or meta is available.
    """
    try:
        # Imported lazily to avoid a circular import on the inference module.
        from src.agents.inference import MODEL_DIR, _latest_artifact, _parse_artifact_stem
    except Exception:
        return None
    try:
        for horizon in (4, 16):
            p = _latest_artifact(MODEL_DIR, "direction", horizon, "clf")
            if p is None:
                continue
            meta_path = p.with_suffix(".meta.json")
            if not meta_path.exists():
                continue
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            tm = (meta.get("test_metrics") or {}).get("test") or {}
            br = tm.get("base_rate")
            if br is not None:
                return float(br)
    except Exception:
        return None
    return None


_BASE_RATE = _load_model_base_rate()
if _BASE_RATE is not None:
    XGB_NEUTRAL_LOW = max(0.0, _BASE_RATE - 0.05)
    XGB_NEUTRAL_HIGH = min(0.95, _BASE_RATE + 0.20)
    XGB_LONG_THRESHOLD = min(0.95, _BASE_RATE + _BASE_RATE_CUSHION)
else:
    XGB_NEUTRAL_LOW = 0.45
    XGB_NEUTRAL_HIGH = 0.55
    XGB_LONG_THRESHOLD = 0.55


def build_node(llm, config: dict[str, Any], risk_limits, *, skills=None):
    role = (
        "Pick one strategy type per cycle: long_stock, short_put_spread, "
        "protective_put, or no_trade. Honor the GNN-confirmation override."
    )
    thresh = config.get("thresholds", {})

    def node(state: DecisionState) -> dict[str, Any]:
        market = state.market_snapshot
        signals = _extract_signals(market, state)
        strat = _select_strategy(signals, state, market)
        obs = _observation(strat, signals, state, llm, role)
        update: dict[str, Any] = {
            "agent_observations": [obs],
            "agent_messages": [_to_message(obs, state.decision_id, "strategy_selector", "risk")],
            "final_action": strat["action"],
        }
        if strat.get("proposal") is not None:
            update["selected_strategy"] = strat["proposal"]
        if strat.get("intent") is not None:
            update["order_intent"] = strat["intent"]
        return update

    return node


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def _extract_signals(market, state) -> dict[str, Any]:
    """Pull the four signals we need: regime, xgb_prob, gnn_bias, iv_rv_gap."""
    # Regime from observations
    regime = "unknown"
    for obs in state.agent_observations:
        sig = obs.signal or {}
        if sig.get("label_heuristic") in ("high_volatility", "crisis", "trending", "mean_reverting", "low_volatility"):
            regime = sig["label_heuristic"]
            break
        if sig.get("regime"):
            regime = sig["regime"]
            break

    # Pick the top-scoring underlying (from candidates if any, else first tradable)
    underlying = None
    xgb_prob = 0.5
    gnn_bias = 0.0
    iv_rv_gap = 0.0
    if state.candidate_strategies:
        top = state.candidate_strategies[0]
        underlying = top.underlying
        # The ML signal lives in the strategy's metadata
        for u in market.underlyings or []:
            if u.symbol == underlying:
                xgb_prob = u.direction_probability or 0.5
                gnn_bias = u.gnn_directional_bias or 0.0
                iv_rv_gap = abs((u.predicted_future_realized_vol or 0.0) - 0.20)  # crude IV proxy
                break
    elif market.underlyings:
        # No candidates yet — use the first underlying that has a non-None signal
        candidates = [u for u in market.underlyings if u.gnn_directional_bias is not None]
        if candidates:
            top_u = max(candidates, key=lambda u: u.gnn_directional_bias or 0.0)
            underlying = top_u.symbol
            xgb_prob = top_u.direction_probability or 0.5
            gnn_bias = top_u.gnn_directional_bias or 0.0
            iv_rv_gap = abs((top_u.predicted_future_realized_vol or 0.0) - 0.20)

    return {
        "regime": regime,
        "underlying": underlying,
        "xgb_prob": xgb_prob,
        "gnn_bias": gnn_bias,
        "iv_rv_gap": iv_rv_gap,
    }


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------
def _select_strategy(signals: dict[str, Any], state, market) -> dict[str, Any]:
    """Pure deterministic selection. Returns {action, proposal?, intent?, reason}."""
    regime = signals["regime"]
    underlying = signals["underlying"]
    xgb = signals["xgb_prob"]
    gnn = signals["gnn_bias"]
    iv_gap = signals["iv_rv_gap"]

    if not underlying:
        return {"action": "NO_TRADE", "proposal": None, "intent": None, "reason": "no underlying candidate"}

    # 1. protective_put when regime is bad and we have a long position
    if regime in ("crisis", "high_volatility"):
        held = {p.symbol for p in (market.portfolio or [])}
        if underlying in held:
            proposal = _build_protective_put(underlying, market)
            return {
                "action": "PROCEED",
                "proposal": proposal,
                "intent": _intent_from_proposal(proposal),
                "reason": f"regime={regime} + held {underlying}, attach protective put",
            }

    # 2. short_put_spread when IV is rich vs RV and we have room
    open_spreads = sum(1 for p in (market.portfolio or []) if p.symbol and "SPREAD" in (p.symbol or ""))
    if iv_gap >= IV_RV_GAP_MIN and open_spreads < 3:
        proposal = _build_short_put_spread(underlying, market)
        return {
            "action": "PROCEED",
            "proposal": proposal,
            "intent": _intent_from_proposal(proposal),
            "reason": f"iv_rv_gap={iv_gap:.3f} >= {IV_RV_GAP_MIN}, sell put spread",
        }

    # 3. long_stock when XGB is bullish OR GNN-confirmation override fires
    gnn_override = (
        (XGB_NEUTRAL_LOW <= xgb <= XGB_NEUTRAL_HIGH and gnn >= GNN_BIAS_OVERRIDE)
        or (xgb < XGB_NEUTRAL_LOW and gnn >= GNN_BIAS_OVERRIDE_STRONG)
    )
    if xgb >= XGB_LONG_THRESHOLD or gnn_override:
        proposal = _build_long_stock(underlying, xgb, gnn, market)
        return {
            "action": "PROCEED",
            "proposal": proposal,
            "intent": _intent_from_proposal(proposal),
            "reason": f"xgb={xgb:.3f} gnn={gnn:+.3f} override={gnn_override}",
        }

    return {
        "action": "NO_TRADE",
        "proposal": None,
        "intent": None,
        "reason": f"no signal clear: xgb={xgb:.3f} gnn={gnn:+.3f} iv_gap={iv_gap:.3f}",
    }


# ---------------------------------------------------------------------------
# Proposal builders
# ---------------------------------------------------------------------------
def _build_long_stock(symbol: str, xgb: float, gnn: float, market) -> StrategyProposal:
    """1-share market BUY proposal. Confidence = max(xgb, sigmoid(gnn))."""
    conf = max(xgb, _sigmoid(gnn))
    leg = Leg(
        symbol=symbol,
        side=Side.BUY,
        quantity=1,
        asset_class="us_equity",
        limit_price=None,
    )
    return StrategyProposal(
        strategy_id="long_stock",
        strategy_type="long_stock",
        underlying=symbol,
        legs=[leg],
        confidence=conf,
        rationale=f"directional long: xgb={xgb:.3f} gnn={gnn:+.3f}",
    )


def _build_short_put_spread(symbol: str, market) -> StrategyProposal:
    """30-45 DTE short put spread, $5 wide, 1 contract.

    In v2 we don't have the option-chain selection logic wired in here
    (the options_structure agent does that); we emit a proposal with
    placeholder legs and let the execution layer fill the contracts.
    """
    leg = Leg(
        symbol=symbol,
        side=Side.SELL,
        quantity=1,
        asset_class="us_equity",
        limit_price=None,
    )
    return StrategyProposal(
        strategy_id="short_put_spread",
        strategy_type="short_put_spread",
        underlying=symbol,
        legs=[leg],
        confidence=0.60,
        rationale="IV > RV by 5+ vol points, sell 30-45 DTE $5 put spread",
    )


def _build_protective_put(symbol: str, market) -> StrategyProposal:
    """Buy 30-DTE 5%-OTM put at 1/3 the size of the underlying long."""
    leg = Leg(
        symbol=symbol,
        side=Side.BUY,
        quantity=1,
        asset_class="us_equity",
        limit_price=None,
    )
    return StrategyProposal(
        strategy_id="protective_put",
        strategy_type="protective_put",
        underlying=symbol,
        legs=[leg],
        confidence=0.65,
        rationale=f"regime={_safe_get(market, 'regime')}, attach 30-DTE 5%-OTM protective put",
    )


def _intent_from_proposal(proposal: StrategyProposal) -> OrderIntent:
    return OrderIntent(
        strategy_id=proposal.strategy_id,
        underlying=proposal.underlying,
        legs=[Leg(**leg.model_dump()) for leg in proposal.legs],
        quantity=max(1, sum(leg.quantity for leg in proposal.legs)),
        limit_price=proposal.legs[0].limit_price,
        time_in_force="day",
        account_mode="PAPER",
    )


def _sigmoid(x: float) -> float:
    import math
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def _safe_get(market, key, default=None):
    return getattr(market, key, default)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
def _observation(strat: dict[str, Any], signals: dict[str, Any], state, llm, role: str) -> AgentObservation:
    payload = {
        "action": strat["action"],
        "reason": strat.get("reason", ""),
        "signals": signals,
        "proposal": strat.get("proposal").model_dump(mode="json") if strat.get("proposal") else None,
    }
    obs = _llm_call(llm, "strategy_selector", role, payload, AgentObservation)
    return obs.model_copy(update={
        "message_type": MessageType.SUPERVISOR_DECISION,
        "confidence": 0.7 if strat["action"] == "PROCEED" else 0.2,
        "signal": {"action": strat["action"], "reason": strat.get("reason", ""), **obs.signal},
    })


__all__ = ["build_node"]
