"""Regression tests for the 2026-09-03 P0-1 fixes.

Three related changes landed together to address the AVGO live-fill
incident and the 0.00 GNN bias symptom:

  1. ``options_structure._build_candidates`` now applies two hard
     floors BEFORE scoring: a probability_profitable floor and a
     direction-mismatch gate. The previous behaviour approved a
     long put on AVGO with direction_probability=0.667 (bullish
     signal) and probability_profitable=0.37 (below confidence_min).

  2. The hackathon mandate boost no longer inflates
     ``StrategyProposal.confidence`` past the supervisor gate. The
     boost is now applied to the **score** only, so a 0.37 P(profit)
     candidate stays at 0.37 confidence and is correctly rejected by
     ``supervisor.top.confidence < confidence_min``.

  3. The options_structure signal now carries an
     ``option_rejections`` summary so the operator trace shows WHY
     options were dropped (the previous "0 candidates" with no
     reason was the most common cron-debugging question).

These tests pin the new contract at the ``_build_candidates`` level
so the regression is caught even when the supervisor's policy or
the option-mandate boost are re-tuned.
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


# ---------- helpers ----------

def _u(sym: str, dp: float, price: float = 200.0,
       rv: float = 0.2) -> UnderlyingScore:
    return UnderlyingScore(
        symbol=sym, timestamp="2026-09-03T16:00:00Z", horizon_bars=4,
        direction_probability=dp, predicted_future_realized_vol=rv,
        gnn_directional_bias=None, last_price=price, model_version="test-1",
    )


def _opt(contract: str, underlying: str, dte: int, pp: float, er: float,
         model_version: str = "option_h4_xgb_clf-test",
         expiration_date: str = "2026-09-09") -> OptionScore:
    return OptionScore(
        contract_symbol=contract, underlying=underlying,
        timestamp="2026-09-03T16:00:00Z", horizon_bars=4,
        probability_profitable=pp, expected_return=er,
        days_to_expiry=dte, option_volatility_16=0.20,
        model_version=model_version, expiration_date=expiration_date,
    )


def _snap(underlyings, options) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-09-03T16:00:00Z",
        underlyings=underlyings, options=options, portfolio=[],
        account_equity=100_000.0, account_cash=100_000.0,
        research=ResearchOutput(
            version="1.0", timestamp="2026-09-03T16:00:00Z",
            per_symbol={}, feature_flag_state="news-on",
        ),
    )


def _weights() -> ScoringWeights:
    return ScoringWeights()


# ============================================================
# Fix #1: direction-mismatch filter
# ============================================================

def test_long_put_on_bullish_signal_is_rejected():
    """The 2026-09-03 AVGO long-put bug: direction_probability=0.667
    (bullish) but the system built a long put. With the new
    direction-mismatch gate, this contract is dropped before scoring
    and the rejection is surfaced in the summary."""
    snap = _snap(
        underlyings=[_u("AVGO", dp=0.667)],
        options=[
            _opt("AVGO260909P00352500", "AVGO", dte=6, pp=0.45, er=0.012),
            # A control: a call on the same bullish signal SHOULD survive.
            _opt("AVGO260909C00352500", "AVGO", dte=6, pp=0.55, er=0.020),
        ],
    )
    proposals, dte_fallback, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0,
    )
    # The put is gone, the call survives.
    survivors = [p.legs[0].contract_symbol for p in proposals]
    assert "AVGO260909P00352500" not in survivors, (
        f"long PUT on a bullish (dp=0.667) signal should be rejected; "
        f"survivors: {survivors}"
    )
    assert "AVGO260909C00352500" in survivors
    # The rejection summary mentions the put + reason.
    assert "AVGO260909P00352500:PUT vs dp=0.67 (bullish)" in rejections["direction_mismatch"]


def test_long_call_on_bearish_signal_is_rejected():
    """Mirror of the long-put case. dp<0.45 means the signal is
    bearish, so a long call (which profits from a rise) is
    direction-mismatched and must be dropped."""
    snap = _snap(
        underlyings=[_u("META", dp=0.30)],
        options=[
            _opt("META260909C00615000", "META", dte=6, pp=0.50, er=0.015),
            _opt("META260909P00615000", "META", dte=6, pp=0.55, er=0.020),
        ],
    )
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0,
    )
    survivors = [p.legs[0].contract_symbol for p in proposals]
    assert "META260909C00615000" not in survivors
    assert "META260909P00615000" in survivors
    assert any("META260909C00615000:CALL vs dp=0.30 (bearish)" in r
               for r in rejections["direction_mismatch"])


def test_neutral_signal_allows_both_calls_and_puts():
    """The 0.45-0.55 dead-band is the "no signal" range. A long call
    and a long put on a coin-flip direction must both be eligible
    (the gate is direction-mismatch, not direction-preference)."""
    snap = _snap(
        underlyings=[_u("NFLX", dp=0.50)],
        options=[
            _opt("NFLX260909C00750000", "NFLX", dte=6, pp=0.45, er=0.010),
            _opt("NFLX260909P00750000", "NFLX", dte=6, pp=0.45, er=0.010),
        ],
    )
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0,
    )
    survivors = [p.legs[0].contract_symbol for p in proposals]
    assert "NFLX260909C00750000" in survivors
    assert "NFLX260909P00750000" in survivors
    # Neutral direction → no direction_mismatch rejections.
    assert rejections["direction_mismatch"] == []


def test_missing_direction_probability_does_not_trigger_mismatch_gate():
    """If direction_probability is None (no signal at all), the
    direction-mismatch gate must be skipped, NOT auto-reject. A
    missing signal is not the same as an opposing one."""
    snap = _snap(
        underlyings=[_u("AMZN", dp=None)],
        options=[
            _opt("AMZN260909C00370000", "AMZN", dte=6, pp=0.45, er=0.010),
            _opt("AMZN260909P00370000", "AMZN", dte=6, pp=0.45, er=0.010),
        ],
    )
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0,
    )
    survivors = [p.legs[0].contract_symbol for p in proposals]
    assert "AMZN260909C00370000" in survivors
    assert "AMZN260909P00370000" in survivors
    assert rejections["direction_mismatch"] == []


# ============================================================
# Fix #2: probability_profitable floor
# ============================================================

def test_low_pp_is_rejected_even_with_mandate_boost():
    """The 2026-09-03 AVGO long-put had P(profit)=0.37 and was
    approved because the mandate boost inflated its confidence to
    0.67. The fix: P(profit) < 0.40 is rejected BEFORE the boost
    runs, so the candidate never reaches the score or the
    supervisor's confidence gate."""
    snap = _snap(
        underlyings=[_u("AVGO", dp=0.40)],  # neutral
        options=[
            # This is the bad candidate from the live incident.
            _opt("AVGO260909P00352500", "AVGO", dte=6, pp=0.37, er=0.012),
        ],
    )
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.30,  # the old behaviour would have inflated this
    )
    assert proposals == [], (
        f"low-PP candidate should be rejected by the min_pp floor; "
        f"got {[p.legs[0].contract_symbol for p in proposals]}"
    )
    assert any("AVGO260909P00352500:pp=0.37" in r for r in rejections["min_pp"])


def test_mandate_boost_no_longer_inflates_confidence():
    """The mandate boost used to add to BOTH score and confidence.
    The fix is to add it to score only. Verify by building a real
    option_h4 candidate inside the strict DTE window, with the
    boost, and reading the proposal's confidence: it must equal
    probability_profitable (NOT probability_profitable + 0.30)."""
    snap = _snap(
        underlyings=[_u("SPY", dp=0.55)],
        options=[
            _opt("SPY260909C00550000", "SPY", dte=7,
                 pp=0.42, er=0.010,
                 model_version="option_h4_xgb_clf-real-20260903"),
        ],
    )
    proposals, _, _ = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.30,
    )
    assert len(proposals) == 1
    p = proposals[0]
    # The thesis shows the boost was applied (score side).
    assert "mandate+0.30" in p.thesis
    # But the confidence is NOT inflated: it must equal
    # probability_profitable (0.42), NOT 0.42 + 0.30 = 0.72.
    assert abs(float(p.confidence) - 0.42) < 1e-6, (
        f"confidence inflated past probability_profitable: got {p.confidence}, "
        f"want 0.42 (the candidate's P(profit))"
    )


def test_min_pp_is_configurable():
    """AIZEN_MIN_OPTION_PP / config overrides the default floor."""
    snap = _snap(
        underlyings=[_u("AAPL", dp=0.55)],
        options=[
            _opt("AAPL260909C00330000", "AAPL", dte=6, pp=0.35, er=0.010),
        ],
    )
    # Floor 0.30 → candidate survives.
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0, min_option_pp=0.30,
    )
    assert len(proposals) == 1
    assert rejections["min_pp"] == []
    # Floor 0.50 → same candidate is dropped.
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0, min_option_pp=0.50,
    )
    assert proposals == []
    assert any("AAPL260909C00330000:pp=0.35" in r for r in rejections["min_pp"])


# ============================================================
# Fix #3: rejection summary in the signal
# ============================================================

def test_rejection_summary_lists_each_reason_bucket():
    """All three rejection buckets should be populated by the same
    synthetic snapshot, and the counts must match the data."""
    snap = _snap(
        underlyings=[_u("AVGO", dp=0.70), _u("SPY", dp=0.55)],
        options=[
            # missing pp
            _opt("AVGO260909C00350000", "AVGO", dte=6,
                 pp=None, er=0.010),
            # low pp
            _opt("AVGO260909C00400000", "AVGO", dte=6,
                 pp=0.30, er=0.010),
            # direction mismatch (long call on bullish signal? no,
            # but a long PUT on dp=0.70 is mismatched)
            _opt("AVGO260909P00400000", "AVGO", dte=6,
                 pp=0.50, er=0.010),
            # the good one
            _opt("SPY260909C00550000", "SPY", dte=6,
                 pp=0.55, er=0.020),
        ],
    )
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0,
    )
    assert len(rejections["missing_data"]) == 1
    assert "AVGO260909C00350000:no_pp_or_er" in rejections["missing_data"]
    assert len(rejections["min_pp"]) == 1
    assert any("AVGO260909C00400000" in r for r in rejections["min_pp"])
    assert len(rejections["direction_mismatch"]) == 1
    assert any("AVGO260909P00400000" in r and "bullish" in r
               for r in rejections["direction_mismatch"])
    # The good candidate survives.
    assert len(proposals) == 1
    assert proposals[0].legs[0].contract_symbol == "SPY260909C00550000"


def test_rejection_summary_caps_at_eight_entries():
    """Cap the rejection list at 8 entries per bucket so a 200-
    contract snapshot doesn't blow up the trace JSON. Operators
    can still see the breakdown; the count is captured separately."""
    n_options = 20
    options = [
        _opt(f"AVGO2609{'09' if i < 10 else '16'}P{str(350+i*5).zfill(8)}",
             "AVGO", dte=6, pp=0.30, er=0.010)
        for i in range(n_options)
    ]
    snap = _snap(
        underlyings=[_u("AVGO", dp=0.55)],
        options=options,
    )
    _, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.0,
    )
    assert len(rejections["min_pp"]) == 8, (
        f"rejection list should cap at 8; got {len(rejections['min_pp'])}"
    )


# ============================================================
# Integration: the AVGO live incident is now blocked
# ============================================================

def test_avgo_live_incident_reproduction():
    """The exact cycle that produced the 2026-09-03 16:23:44Z AVGO
    long-put fill: direction_probability=0.667, P(profit)=0.37,
    long PUT leg. With the new gates, this contract is rejected by
    BOTH the min_pp floor AND the direction-mismatch filter, so it
    cannot reach the score, the supervisor, or the broker."""
    snap = _snap(
        underlyings=[_u("AVGO", dp=0.667)],
        options=[
            # The exact contract from the live incident.
            _opt("AVGO260909P00352500", "AVGO", dte=6, pp=0.37, er=0.012),
        ],
    )
    proposals, _, rejections = _build_candidates(
        snap, _weights(), min_score=0.0, top_n=10,
        dte_min=5, dte_max=10, gnn_output={"node_features": {}},
        option_mandate_boost=0.30,  # was the bug: it used to inflate conf past 0.50
    )
    assert proposals == [], (
        "the AVGO live incident (long put, dp=0.667, P(profit)=0.37) "
        "must NOT produce a candidate after the P0-1 fix"
    )
    # The pp=0.37 contract is rejected by min_pp (catches it first).
    assert any("AVGO260909P00352500:pp=0.37" in r for r in rejections["min_pp"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
