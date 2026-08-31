"""Tests for the long-equity candidate generator (parallel options+stocks path).

The generator lives in :mod:`src.agents.nodes.options_structure` and is
called ``_build_equity_candidates``. It is the parallel of
``_build_candidates`` (options) — same scoring formula, different
instrument.

Why a separate function instead of a flag on ``_build_candidates``?
  * The two paths have different gating rules (DTE vs. price, OI/volume
    vs. none) and different scoring inputs (option premium proxy vs.
    pure directional edge).
  * A single combined function would carry a per-iteration branch on
    instrument type, which is harder to test in isolation.

The tests below pin the contract:
  * Only symbols with ``direction_probability >= min_equity_direction_probability``
    (default 0.55) become candidates.
  * Symbols without a ``last_price`` are silently dropped (no fabricated
    price).
  * The leg carries ``asset_class='equity'`` and a BUY side; the per-leg
    quantity is sized to the per-symbol notional cap.
  * High-confidence signals out-score weak ones, and the function
    returns the top-N in descending score order.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.nodes.options_structure import _build_equity_candidates  # noqa: E402
from src.agents.protocol import (  # noqa: E402
    Leg, MarketSnapshot, OptionScore, ResearchOutput, SymbolResearch,
    UnderlyingScore,
)
from src.agents.risk import RiskLimits  # noqa: E402
from src.agents.scoring import ScoringWeights  # noqa: E402


def _u(sym: str, dp: float | None, price: float | None = 200.0,
       rv: float | None = 0.2, gnn_bias: float | None = None) -> UnderlyingScore:
    return UnderlyingScore(
        symbol=sym, timestamp="2026-08-30T13:30:00Z", horizon_bars=4,
        direction_probability=dp, predicted_future_realized_vol=rv,
        gnn_directional_bias=gnn_bias, last_price=price,
        model_version="test-1",
    )


def _snap(underlyings: list[UnderlyingScore]) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-08-30T13:30:00Z",
        underlyings=underlyings,
        options=[],
        portfolio=[],
        account_equity=100_000.0,
        account_cash=100_000.0,
        research=ResearchOutput(
            version="1.0", timestamp="2026-08-30T13:30:00Z",
            per_symbol={}, feature_flag_state="news-off",
        ),
    )


def _limits() -> RiskLimits:
    return RiskLimits(
        max_equity_notional_per_symbol=1500.0,
        max_equity_notional_total=5000.0,
        min_equity_direction_probability=0.55,
    )


def _weights() -> ScoringWeights:
    return ScoringWeights()


def test_emits_long_equity_leg_with_correct_asset_class():
    """Strong BULL signal -> single equity leg, asset_class=equity, side=BUY."""
    snap = _snap([_u("AAPL", dp=0.72, price=180.0)])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert len(out) == 1
    p = out[0]
    assert p.underlying == "AAPL"
    assert len(p.legs) == 1
    leg = p.legs[0]
    assert isinstance(leg, Leg)
    assert leg.asset_class == "equity"
    assert leg.contract_symbol == "AAPL"
    assert leg.side.value == "buy"
    # Quantity: 1500 // 180 = 8 shares. Notional = 8 * 180 = $1440.
    assert leg.quantity == 8
    assert 0 < leg.limit_price <= 200.0
    # No option-only fields on equity leg.
    assert leg.option_type is None
    assert leg.strike is None
    assert leg.expiry is None


def test_skips_when_direction_probability_below_threshold():
    """Default 0.55 threshold; a 0.50 (coin-flip) signal must NOT generate a candidate."""
    snap = _snap([_u("AAPL", dp=0.50, price=180.0)])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert out == []


def test_skips_when_last_price_is_missing():
    """No fabricated price: a symbol without a price is dropped."""
    snap = _snap([_u("AAPL", dp=0.70, price=None)])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert out == []


def test_skips_when_price_too_high_to_buy_one_share():
    """Stock > per-symbol cap means even 1 share is over the limit."""
    # cap is $1500; price $3000 -> 0 shares allowed -> drop.
    snap = _snap([_u("BRK.A", dp=0.70, price=3000.0)])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert out == []


def test_top_n_truncation():
    """Only the top-N (by score) survive."""
    snap = _snap([
        _u(f"SYM{i}", dp=0.55 + i * 0.02, price=100.0) for i in range(10)
    ])
    out = _build_equity_candidates(snap, _weights(), 0.30, top_n=3, gnn_output={}, risk_limits=_limits())
    assert len(out) == 3
    # Descending by score.
    for a, b in zip(out, out[1:]):
        assert a.score >= b.score


def test_high_signal_outsorts_low_signal():
    """Higher direction_probability -> higher score."""
    snap = _snap([
        _u("WEAK", dp=0.56, price=100.0),
        _u("STRONG", dp=0.85, price=100.0),
    ])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert [p.underlying for p in out] == ["STRONG", "WEAK"]


def test_gnn_bias_pushed_through_to_thesis():
    """The thesis surfaces the GNN bias (operator-visible signal chain)."""
    snap = _snap([_u("AAPL", dp=0.70, price=100.0, gnn_bias=0.42)])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert len(out) == 1
    assert "gnn_bias=+0.42" in out[0].thesis


def test_min_score_filter_drops_weak_candidates():
    """A high min_score threshold drops weak signals."""
    snap = _snap([_u("AAPL", dp=0.56, price=100.0)])
    out = _build_equity_candidates(snap, _weights(), min_score=0.99, top_n=5, gnn_output={}, risk_limits=_limits())
    assert out == []


def test_empty_snapshot_returns_empty_list():
    """A snapshot with no underlyings -> no candidates, no exceptions."""
    snap = _snap([])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert out == []


def test_handles_none_snapshot_gracefully():
    """Defensive: a None snapshot is treated as empty."""
    out = _build_equity_candidates(None, _weights(), 0.30, 5, {}, _limits())  # type: ignore[arg-type]
    assert out == []


def test_uses_gnn_node_features_when_present():
    """When gnn_output.node_features has a per-symbol bias, it is preferred."""
    snap = _snap([_u("AAPL", dp=0.70, price=100.0, gnn_bias=None)])
    gnn = {"node_features": {"AAPL": {"bias": 0.8, "centrality": 0.5}}}
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, gnn, _limits())
    assert len(out) == 1
    assert "gnn_bias=+0.80" in out[0].thesis


def test_max_loss_equals_full_notional():
    """For a long equity leg, the worst case is the stock going to zero;
    max_loss should equal qty * limit_price (the full notional)."""
    snap = _snap([_u("AAPL", dp=0.70, price=150.0)])
    out = _build_equity_candidates(snap, _weights(), 0.30, 5, {}, _limits())
    assert len(out) == 1
    leg = out[0].legs[0]
    assert out[0].max_loss == pytest.approx(leg.quantity * leg.limit_price, rel=1e-3)
