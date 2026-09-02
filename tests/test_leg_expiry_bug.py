"""Regression tests for the Bug A fix: Leg.expiry must use the
contract's actual expiration_date, NOT the latest bar's timestamp.

Before the fix, ``options_structure._build_candidates`` set
``Leg.expiry = opt.timestamp[:10]`` - the bar's date. When the
option_bars table is stale (which is the common case: the live-tick
refresh may not have written today yet), this made the Leg's expiry
date be in the past relative to the decision timestamp. The OCC
symbol's encoded 6-digit expiry (e.g. ``260902`` = 2026-09-02) was
correct, but the broker received a Leg with a conflicting past
date and rejected the order.

The fix: the inference layer now populates ``OptionScore.expiration_date``
from ``option_contracts.expiration_date`` and ``_build_candidates``
uses that field (with a defense-in-depth fallback to the OCC
symbol's encoded date via ``_expiry_from_occ_symbol``).
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.nodes.options_structure import (  # noqa: E402
    _build_candidates, _expiry_from_occ_symbol,
)
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
    dte: int = 5, pp: float = 0.6, er: float = 0.05,
    model_version: str = "option_h4_xgb_clf-20260829-152605",
    bar_ts: str = "2026-08-28T19:45:00Z",
    expiration_date: str = "2026-09-02",
) -> OptionScore:
    """Build an OptionScore that simulates a stale bar (bar_ts is
    4 days before the contract's actual expiration). The bar's date
    MUST NOT be used as Leg.expiry."""
    return OptionScore(
        contract_symbol=contract, underlying=underlying,
        timestamp=bar_ts, horizon_bars=4,
        probability_profitable=pp, expected_return=er,
        moneyness=100.0, days_to_expiry=dte, model_version=model_version,
        expiration_date=expiration_date,
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


# ---- OCC helper ----

def test_expiry_from_occ_symbol_decodes_yymmdd():
    """Sanity check: the defense-in-depth helper returns the right
    YYYY-MM-DD for known OCC symbol patterns."""
    assert _expiry_from_occ_symbol("META260902C00572500") == "2026-09-02"
    assert _expiry_from_occ_symbol("AAPL260918P00150000") == "2026-09-18"
    assert _expiry_from_occ_symbol("SPY251231C00450000") == "2025-12-31"


def test_expiry_from_occ_symbol_returns_none_on_garbage():
    """Garbage input must not return a fabricated date - the Leg
    builder falls through to its far-future default in that case."""
    assert _expiry_from_occ_symbol("X") is None
    assert _expiry_from_occ_symbol("") is None
    assert _expiry_from_occ_symbol("NOTANOPTIONSYMBOL") is None


# ---- Bug A core: Leg.expiry comes from contract, not bar ----

def test_leg_expiry_uses_contract_expiration_not_bar_timestamp():
    """Core regression: when the OptionScore carries the contract's
    actual expiration_date, the Leg.expiry MUST equal that field -
    NOT the bar's timestamp (which is 4 days stale in this test)."""
    snap = _snap(
        underlyings=[_u("META", dp=0.65, price=180.0)],
        options=[_o(
            "META260902C00572500", "META", dte=5,
            bar_ts="2026-08-28T19:45:00Z",      # stale bar
            expiration_date="2026-09-02",       # contract's real expiry
        )],
    )
    weights = ScoringWeights()
    options, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.0,
    )
    assert len(options) == 1
    leg = options[0].legs[0]
    # Leg.expiry MUST be the contract's expiration_date, not the
    # bar's timestamp. The bar's date is 2026-08-28, the contract
    # expires 2026-09-02.
    assert leg.expiry == "2026-09-02", (
        f"Leg.expiry should be the contract's expiration_date "
        f"('2026-09-02'), got {leg.expiry!r} (bar's date '2026-08-28'?)"
    )
    assert leg.expiry != "2026-08-28", (
        "REGRESSION: Leg.expiry is using the bar's stale timestamp "
        "instead of the contract's expiration_date."
    )


def test_leg_expiry_falls_back_to_occ_symbol_when_expiration_date_missing():
    """When the inference layer didn't propagate expiration_date
    (e.g. a synthetic OptionScore in a test), the Leg.expiry MUST
    still be the contract's actual expiry, via the OCC symbol."""
    snap = _snap(
        underlyings=[_u("META", dp=0.65, price=180.0)],
        options=[_o(
            "META260902C00572500", "META", dte=5,
            bar_ts="2026-08-28T19:45:00Z",
            expiration_date=None,                # missing!
        )],
    )
    weights = ScoringWeights()
    options, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.0,
    )
    assert len(options) == 1
    leg = options[0].legs[0]
    # The OCC symbol encodes 2026-09-02; that MUST be the fallback
    # value, not the bar's date.
    assert leg.expiry == "2026-09-02", (
        f"Leg.expiry should fall back to OCC symbol's encoded date "
        f"('2026-09-02'), got {leg.expiry!r}"
    )


def test_strategy_proposal_expiry_uses_leg_expiry():
    """StrategyProposal.expiry is the longest leg's expiry. For a
    single-leg proposal that's just the leg's expiry - and MUST be
    the contract's expiry, not the bar's."""
    snap = _snap(
        underlyings=[_u("META", dp=0.65, price=180.0)],
        options=[_o(
            "META260902C00572500", "META", dte=5,
            bar_ts="2026-08-28T19:45:00Z",
            expiration_date="2026-09-02",
        )],
    )
    weights = ScoringWeights()
    options, _ = _build_candidates(
        snap, weights, 0.0, 5,
        dte_min=5, dte_max=10, gnn_output={},
        option_mandate_boost=0.0,
    )
    proposal = options[0]
    assert proposal.expiry == "2026-09-02"
    assert proposal.expiry == proposal.legs[0].expiry
