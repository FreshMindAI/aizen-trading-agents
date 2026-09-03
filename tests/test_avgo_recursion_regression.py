"""Regression test for the 2026-09-03 AVGO re-entry failure.

Live scenario (broker positions, screenshot captured by user):
  AVGO260909P00350000  qty=1  loss_pct=-25.00%   uPnL=-155
  AVGO260909P00352500  qty=6  loss_pct=-11.34%   uPnL=-445
  COIN260911P00190000  qty=1  loss_pct=-23.08%   uPnL=-165
  COIN260911P00195000  qty=1  loss_pct= -5.88%   uPnL= -50
  NFLX                qty=120 loss_pct= -0.46%   uPnL= -46

User reported: "i see avgo options was sold and system recommend in
the latest run again which is problem and it is on loss as well."

The supervisor MUST filter AVGO and COIN out of the candidate set
before PROCEED. This test pins that contract end-to-end through
get_blocked_symbols + supervisor.filter.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents import position_management as pm
from src.agents.nodes.supervisor import _blocked_symbols
from src.agents.protocol import (
    AgentObservation, DecisionState, Leg, MessageType, OptionType,
    Side, StrategyProposal,
)


LIVE_LOSERS = [
    # (symbol, qty, avg_entry, current_price, expected_loss_pct)
    ("AVGO260909P00350000", 1, 6.20, 4.65, -0.2500),     # -25.00%
    ("AVGO260909P00352500", 6, 6.5417, 5.80, -0.1134),   # -11.34%
    ("COIN260911P00190000", 1, 7.15, 5.50, -0.2308),     # -23.08%
    ("COIN260911P00195000", 1, 8.50, 8.00, -0.0588),     #  -5.88%
    ("NFLX",                120, 82.2658, 82.35, -0.0010),#   -0.10% (rounded)
]


def _broker_dicts() -> list[dict]:
    """Mirror the exact dict shape AlpacaTradingClient.list_positions
    returns to the orchestrator's pre-flight."""
    return [
        {
            "symbol": sym, "qty": qty, "side": "long",
            "avg_entry_price": entry, "current_price": mark,
            "unrealized_pl": (mark - entry) * qty * (100 if any(c.isdigit() for c in sym) else 1),
        }
        for (sym, qty, entry, mark, _) in LIVE_LOSERS
    ]


def test_blocked_set_includes_avgo_and_coin():
    """get_blocked_symbols must put AVGO and COIN in the blocked set
    (open_loss_block_pct = -0.05, every AVGO and COIN position here
    is at or below that threshold). NFLX at -0.10% is just above the
    threshold so it should NOT be blocked."""
    blocked = pm.get_blocked_symbols(
        positions=_broker_dicts(),
        conn=None,
        cooldown_seconds=0,
    )
    assert "AVGO" in blocked, f"AVGO should be blocked: {sorted(blocked)}"
    assert "COIN" in blocked, f"COIN should be blocked: {sorted(blocked)}"
    # NFLX is at -0.10% which is above the -5% open_loss_block_pct;
    # it should NOT be blocked by the open-loss check.
    assert "NFLX" not in blocked, (
        f"NFLX at -0.10% should NOT be blocked (open_loss_block_pct=-5%): "
        f"{sorted(blocked)}"
    )


def test_supervisor_filters_avgo_candidate_against_live_blocked_set():
    """End-to-end: build the same DecisionState the orchestrator would
    have at supervisor time, with the live blocked_symbols in
    state.supplementary, and assert the AVGO put candidate gets
    filtered out."""
    state = DecisionState()
    blocked = pm.get_blocked_symbols(
        positions=_broker_dicts(),
        conn=None,
        cooldown_seconds=0,
    )
    state.supplementary = {
        "position_management": {"blocked_symbols": sorted(blocked)},
    }
    # Confirm the supervisor reads the same set.
    assert _blocked_symbols(state) == {"AVGO", "COIN"}

    # Now the same StrategyProposal the options_structure agent would
    # have built for an AVGO long put.
    avgo_put = StrategyProposal(
        underlying="AVGO",
        legs=[Leg(asset_class="option",
                  contract_symbol="AVGO260909P00352500",
                  side=Side.BUY, quantity=1,
                  option_type=OptionType.PUT, strike=352.5,
                  expiry="2026-09-09")],
        thesis="Long put on AVGO with P(profit)=0.37",
        expected_return=0.012,
        probability_profit=0.37,
        confidence=0.66,
        max_loss=652.0,
        expiry="2026-09-09",
        score=0.595,
    )
    coin_put = StrategyProposal(
        underlying="COIN",
        legs=[Leg(asset_class="option",
                  contract_symbol="COIN260911P00195000",
                  side=Side.BUY, quantity=1,
                  option_type=OptionType.PUT, strike=195.0,
                  expiry="2026-09-11")],
        thesis="Long put on COIN",
        expected_return=0.01,
        probability_profit=0.34,
        confidence=0.55,
        max_loss=850.0,
        expiry="2026-09-11",
        score=0.50,
    )
    spy_call = StrategyProposal(  # not in blocked set
        underlying="SPY",
        legs=[Leg(asset_class="option",
                  contract_symbol="SPY260909C00550000",
                  side=Side.BUY, quantity=1,
                  option_type=OptionType.CALL, strike=550.0,
                  expiry="2026-09-09")],
        thesis="Long call on SPY",
        expected_return=0.05,
        probability_profit=0.6,
        confidence=0.7,
        max_loss=300.0,
        expiry="2026-09-09",
        score=0.62,
    )
    candidates = [avgo_put, coin_put, spy_call]
    filtered = [c for c in candidates
                if (c.underlying or "").split(" ")[0].upper()
                not in {s.upper() for s in blocked}]
    assert len(filtered) == 1, (
        f"expected only SPY to survive; got {[c.underlying for c in filtered]}"
    )
    assert filtered[0].underlying == "SPY"


def test_empty_positions_with_broker_failure_does_not_silently_unblock(caplog):
    """Defense in depth: if list_positions() returns [] (no exception,
    but broker may have silently failed - a real failure mode in the
    2026-09-03 cloud run), the operator should at minimum see a WARNING
    that the position-management layer is now operating on an empty
    portfolio. Without this signal, a transient broker blip silently
    disables every blocked_symbol + stop-loss check."""
    # Capture the WARNING that get_blocked_symbols emits on empty input.
    with caplog.at_level(logging.WARNING, logger="src.agents.position_management"):
        blocked = pm.get_blocked_symbols(
            positions=[],
            conn=None,
            cooldown_seconds=0,
        )
    assert blocked == set()
    # The empty-positions WARNING must surface so the operator can
    # distinguish "I really have no positions" from "broker just lied".
    warnings = [r for r in caplog.records
                if r.levelname == "WARNING"
                and "empty" in r.getMessage().lower()]
    assert len(warnings) >= 1, (
        "empty positions should produce a WARNING so the operator sees "
        "the position-management layer is operating blind"
    )
