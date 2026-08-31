"""Tests for the options_structure DTE-window fallback (hackathon sizing).

Background
----------
The hackathon window is 5-10 DTE, but the synthetic option_contracts
table was generated with 18-60 DTE contracts. Without a fallback, the
strict 5-10 filter empties the candidate set on every cycle and the
pipeline dead-ends at the options_structure_agent with the
"empty candidate set" risk.

The fallback widens the window to the shortest DTE band that has
contracts, capped at dte_min+25 to keep the trade short-horizon
(matches the 1h XGBoost signal-decay window). The fallback note is
surfaced in the obs signal so the trace + supervisor can see why
we widened.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.nodes.options_structure import _resolve_dte_window  # noqa: E402


def _opt(dte: int):
    """Minimal OptionScore stub - we only need days_to_expiry."""
    from src.agents.protocol import OptionScore
    return OptionScore(
        contract_symbol="X", underlying="X", timestamp="2026-08-17T00:00:00Z",
        horizon_bars=4, days_to_expiry=dte,
    )


def test_strict_window_hits_returns_unchanged():
    options = [_opt(d) for d in (5, 7, 9, 12, 18, 30)]
    mn, mx, note = _resolve_dte_window(options, dte_min=5, dte_max=10)
    assert (mn, mx) == (5, 10)
    assert note is None


def test_strict_window_empty_widens_to_shortest_band():
    options = [_opt(d) for d in (18, 20, 25, 30, 40, 60)]
    mn, mx, note = _resolve_dte_window(options, dte_min=5, dte_max=10)
    assert note is not None
    assert mn == 18  # shortest available
    assert mx >= mn
    # When shortest (18) <= intent_cap (30), the upper bound is the intent cap
    assert mx == 5 + 25


def test_no_dte_data_returns_note_and_keeps_strict_window():
    options = [_opt(0), _opt(-1)]
    mn, mx, note = _resolve_dte_window(options, dte_min=5, dte_max=10)
    assert note == "no_dte_data"
    assert (mn, mx) == (5, 10)


def test_shortest_above_intent_cap_widens_to_shortest_plus_ten():
    """When the shortest available DTE is already above dte_min+25, the
    intent cap would produce min > max. Fall back to a 10-DTE-wide
    band centered on the shortest available."""
    options = [_opt(d) for d in (32, 40, 50, 58)]
    mn, mx, note = _resolve_dte_window(options, dte_min=5, dte_max=10)
    assert note is not None
    assert mn == 32  # shortest available
    assert mx == min(58, 32 + 10)  # = 42
    assert mn <= mx  # critical: non-empty window
