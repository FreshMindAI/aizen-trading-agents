"""Tests for the option_mandate_boost in options_structure.

The Alpaca AI Trading Agents Hackathon (Aug 28 - Sept 4, 2026) requires
options to be the trade instrument. To make this guarantee structural
(so a single walk-forward re-tune of the scoring weights can't quietly
flip the system to equity-only), ``_build_candidates`` adds a fixed
``option_mandate_boost`` to every option candidate that:

  * came from a real ``option_h4`` ML model prediction (not the
    ``heuristic-1`` placeholder), AND
  * falls inside the strict DTE window we asked for.

The boost is sized to overcome (a) the option leg's spread + liquidity
penalties vs. the equity leg's zero penalties, and (b) the option
model's conservative expected_return vs. the equity path's
``(dp - 0.5) * 2`` proxy. These tests pin the three gates and the
thesis tag so the boost can never silently stop firing.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.nodes.options_structure import _build_candidates  # noqa: E402
from src.agents.protocol import (  # noqa: E402
    MarketSnapshot, OptionScore, ResearchOutput, UnderlyingScore,
)
from src.agents.scoring import ScoringWeights  # noqa: E402


# ---- helpers ----

def _u(sym: str, dp: float, price: float = 100.0) -> UnderlyingScore:
    return UnderlyingScore(
        symbol=sym, timestamp="2026-08-30T13:30:00Z", horizon_bars=4,
        direction_probability=dp, last_price=price, model_version="test",
    )


def _o(
    contract: str, underlying: str,
    dte: int = 8, pp: float = 0.6, er: float = 0.05,
    model_version: str = "option_h4_xgb_clf-20260829-152605",
) -> OptionScore:
    return OptionScore(
        contract_symbol=contract, underlying=underlying,
        timestamp="2026-08-30T13:30:00Z", horizon_bars=4,
        probability_profitable=pp, expected_return=er,
        moneyness=100.0, days_to_expiry=dte, model_version=model_version,
    )


def _snap(underlyings, options) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-08-30T13:30:00Z",
        underlyings=underlyings, options=options,
        portfolio=[], account_equity=100_000.0, account_cash=100_000.0,
        research=ResearchOutput(
            version="1.0", timestamp="2026-08-30T13:30:00Z",
            per_symbol={}, feature_flag_state="news-off",
        ),
    )


# ---- boost applied ----

def test_real_ml_option_in_strict_dte_window_gets_mandate_boost():
    """An option from the option_h4 model in the 5-10 DTE window must
    receive the full boost. We compare the candidate's score WITH boost
    against an equivalent run WITHOUT boost and assert the delta equals
    the configured boost value."""
    snap = _snap(
        underlyings=[_u("AAPL", dp=0.65, price=180.0)],
        options=[_o("AAPL260908C00200000", "AAPL", dte=8)],
    )
    weights = ScoringWeights()

    no_boost, _ = _build_candidates(
        snap, weights, 0.0, 5,           # min_score=0 so the candidate is kept
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.0,
    )
    with_boost, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.20,
    )

    assert len(no_boost) == 1
    assert len(with_boost) == 1
    delta = with_boost[0].score - no_boost[0].score
    assert delta == pytest.approx(0.20, abs=1e-9), (
        f"expected a +0.20 mandate boost, got delta={delta:.4f}"
    )
    # The thesis must show the boost so the trace tells the operator
    # why the option outranked equity.
    assert "[mandate+0.20]" in with_boost[0].thesis
    assert "[mandate" not in no_boost[0].thesis


def test_heuristic_option_does_not_get_mandate_boost():
    """Heuristic-fallback options (model_version = "heuristic-1") are
    placeholders. Boosting them would mask the missing ML model."""
    snap = _snap(
        underlyings=[_u("AAPL", dp=0.65, price=180.0)],
        options=[_o(
            "AAPL260908C00200000", "AAPL", dte=8,
            model_version="heuristic-1",
        )],
    )
    weights = ScoringWeights()

    no_boost, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.0,
    )
    with_boost, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.20,
    )
    assert len(no_boost) == 1
    assert len(with_boost) == 1
    delta = with_boost[0].score - no_boost[0].score
    assert delta == pytest.approx(0.0, abs=1e-9), (
        f"heuristic-fallback option must NOT get the boost; got delta={delta:.4f}"
    )
    assert "[mandate" not in with_boost[0].thesis


def test_option_outside_strict_dte_window_does_not_get_mandate_boost():
    """A real-ML option in the widened (DTE fallback) window is not in
    the strict [5, 10] band. The boost only applies to options in the
    strict window. Falling back to the heuristic still happens for
    these, but the boost stays off so the supervisor's score stays a
    pure reflection of model quality."""
    snap = _snap(
        underlyings=[_u("AAPL", dp=0.65, price=180.0)],
        options=[_o("AAPL260918C00200000", "AAPL", dte=18)],  # widened band
    )
    weights = ScoringWeights()

    with_boost, dte_note = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.20,
    )
    # The strict window is empty, so the DTE fallback fires.
    assert dte_note is not None
    # Candidate must still be present (the band was widened), but it
    # must NOT carry the mandate boost.
    assert len(with_boost) == 1
    assert "[mandate" not in with_boost[0].thesis


def test_boost_makes_option_outrank_equivalent_equity_candidate():
    """End-to-end: with a directional signal that's just over the
    equity threshold and an option in the strict window, the option
    must outrank the equity candidate when the boost is on. Without
    the boost, the equity leg's lack of spread/liquidity penalties
    lets it outrank the option leg."""
    from src.agents.nodes.options_structure import _build_equity_candidates
    from src.agents.risk import RiskLimits

    snap = _snap(
        underlyings=[_u("AAPL", dp=0.62, price=180.0)],   # above 0.55
        options=[_o("AAPL260908C00200000", "AAPL", dte=8,
                    pp=0.55, er=0.05)],                   # real ML
    )
    weights = ScoringWeights()
    limits = RiskLimits(
        allowed_expiries_dte_min=5, allowed_expiries_dte_max=10,
        max_equity_notional_per_symbol=1500.0,
        max_equity_notional_total=5000.0,
        min_equity_direction_probability=0.55,
    )

    options, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.20,
    )
    equities = _build_equity_candidates(
        snap, weights, 0.0, 5, gnn_output={}, risk_limits=limits,
    )
    assert len(options) == 1
    assert len(equities) == 1
    # With the boost on, the option must now outrank equity.
    assert options[0].score > equities[0].score, (
        f"option score ({options[0].score:.3f}) must beat equity "
        f"({equities[0].score:.3f}) when mandate boost is on"
    )
    # And the option must carry the mandate tag.
    assert "[mandate+0.20]" in options[0].thesis
