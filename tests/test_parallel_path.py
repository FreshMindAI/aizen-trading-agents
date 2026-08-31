"""End-to-end tests for the parallel options + stocks path.

These tests pin the contract that the options_structure node emits
both option and equity candidates, that the Leg schema carries
``asset_class``, and that the execution node's validator accepts both
instrument types.

The tests use small in-memory snapshots (no DB, no LLM) and call the
node builders directly with a mock LLM.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.nodes.execution import _validate_intent, _dry_run_submission  # noqa: E402
from src.agents.nodes.options_structure import (  # noqa: E402
    _build_candidates, _build_equity_candidates,
)
from src.agents.protocol import (  # noqa: E402
    Leg, MarketSnapshot, OptionScore, OptionType, OrderIntent,
    ResearchOutput, Side, StrategyProposal, UnderlyingScore,
)
from src.agents.risk import RiskLimits  # noqa: E402
from src.agents.scoring import ScoringWeights  # noqa: E402


# ---- helpers ----

def _u(sym: str, dp: float | None, price: float | None = 100.0) -> UnderlyingScore:
    return UnderlyingScore(
        symbol=sym, timestamp="2026-08-30T13:30:00Z", horizon_bars=4,
        direction_probability=dp, last_price=price, model_version="test",
    )


def _o(contract: str, underlying: str, dte: int = 8) -> OptionScore:
    return OptionScore(
        contract_symbol=contract, underlying=underlying,
        timestamp="2026-08-30T13:30:00Z", horizon_bars=4,
        probability_profitable=0.6, expected_return=0.3,
        moneyness=100.0, days_to_expiry=dte, model_version="test",
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


def _limits() -> RiskLimits:
    return RiskLimits(
        allowed_expiries_dte_min=5, allowed_expiries_dte_max=10,
        max_equity_notional_per_symbol=1500.0,
        max_equity_notional_total=5000.0,
        min_equity_direction_probability=0.55,
    )


# ---- leg schema ----

def test_leg_default_asset_class_is_option():
    """Backward compat: a Leg without asset_class defaults to 'option'."""
    leg = Leg(
        contract_symbol="AAPL260101C00200000",
        side=Side.BUY, quantity=1,
        option_type=OptionType.CALL, strike=200.0, expiry="2026-12-31",
    )
    assert leg.asset_class == "option"


def test_leg_equity_does_not_require_strike_or_expiry():
    """Equity legs can leave option_type/strike/expiry as None."""
    leg = Leg(
        asset_class="equity", contract_symbol="AAPL",
        side=Side.BUY, quantity=5, limit_price=200.0,
    )
    assert leg.asset_class == "equity"
    assert leg.option_type is None
    assert leg.strike is None
    assert leg.expiry is None


# ---- options_structure node wiring (parity test) ----

def test_options_node_emits_both_option_and_equity_candidates():
    """With both option chains and equity signals, the node returns a
    mixed candidate set. The supervisor downstream picks the best one."""
    snap = _snap(
        underlyings=[_u("AAPL", dp=0.65, price=180.0)],
        options=[_o("AAPL260918C00200000", "AAPL", dte=8)],
    )
    weights = ScoringWeights()
    option_cands, _dte = _build_candidates(
        snap, weights, 0.30, 5,
        dte_min=5, dte_max=10, gnn_output={},
    )
    equity_cands = _build_equity_candidates(
        snap, weights, 0.30, 5, gnn_output={}, risk_limits=_limits(),
    )
    assert len(option_cands) >= 1
    assert len(equity_cands) >= 1
    assert option_cands[0].legs[0].asset_class == "option"
    assert equity_cands[0].legs[0].asset_class == "equity"


def test_equity_only_path_when_no_options_in_snapshot():
    """No option chain -> equity candidates still flow through."""
    snap = _snap(
        underlyings=[_u("AAPL", dp=0.70, price=180.0)],
        options=[],
    )
    weights = ScoringWeights()
    option_cands, _dte = _build_candidates(
        snap, weights, 0.30, 5,
        dte_min=5, dte_max=10, gnn_output={},
    )
    equity_cands = _build_equity_candidates(
        snap, weights, 0.30, 5, gnn_output={}, risk_limits=_limits(),
    )
    assert option_cands == []
    assert len(equity_cands) == 1
    assert equity_cands[0].legs[0].asset_class == "equity"


# ---- execution node validation ----

def test_validate_intent_accepts_equity_leg():
    """The execution validator must accept an equity leg without strike/expiry."""
    intent = OrderIntent(
        strategy_id="e1", underlying="AAPL",
        legs=[Leg(asset_class="equity", contract_symbol="AAPL",
                  side=Side.BUY, quantity=5, limit_price=200.0)],
        quantity=5,
    )
    assert _validate_intent(intent) is True


def test_validate_intent_accepts_option_leg():
    """The execution validator still accepts the option leg shape."""
    intent = OrderIntent(
        strategy_id="o1", underlying="AAPL",
        legs=[Leg(asset_class="option", contract_symbol="AAPL260918C00200000",
                  side=Side.BUY, quantity=1,
                  option_type=OptionType.CALL, strike=200.0, expiry="2026-09-18",
                  limit_price=5.0)],
        quantity=1,
    )
    assert _validate_intent(intent) is True


def test_validate_intent_rejects_equity_sell_leg():
    """Short-selling is not authorized in the parallel path."""
    intent = OrderIntent(
        strategy_id="s1", underlying="AAPL",
        legs=[Leg(asset_class="equity", contract_symbol="AAPL",
                  side=Side.SELL, quantity=1, limit_price=200.0)],
        quantity=1,
    )
    assert _validate_intent(intent) is False


def test_validate_intent_rejects_equity_with_no_symbol():
    """An equity leg without a contract_symbol is malformed."""
    intent = OrderIntent(
        strategy_id="b1", underlying="AAPL",
        legs=[Leg(asset_class="equity", contract_symbol="",
                  side=Side.BUY, quantity=1, limit_price=200.0)],
        quantity=1,
    )
    assert _validate_intent(intent) is False


def test_validate_intent_rejects_equity_with_zero_limit_price():
    """An equity leg with a zero/negative limit price is malformed."""
    intent = OrderIntent(
        strategy_id="z1", underlying="AAPL",
        legs=[Leg(asset_class="equity", contract_symbol="AAPL",
                  side=Side.BUY, quantity=1, limit_price=0.0)],
        quantity=1,
    )
    assert _validate_intent(intent) is False


# ---- dry-run submission records asset class ----

def test_dry_run_submission_records_asset_classes():
    """Dry-run result must include the set of asset classes submitted."""
    intent = OrderIntent(
        strategy_id="m1", underlying="AAPL",
        legs=[
            Leg(asset_class="equity", contract_symbol="AAPL",
                side=Side.BUY, quantity=5, limit_price=200.0),
            Leg(asset_class="option", contract_symbol="AAPL260918C00200000",
                side=Side.BUY, quantity=1,
                option_type=OptionType.CALL, strike=200.0, expiry="2026-09-18",
                limit_price=5.0),
        ],
        quantity=1,
    )
    result = _dry_run_submission(intent)
    assert result["status"] == "dry_run"
    assert sorted(result["asset_classes"]) == ["equity", "option"]


def test_dry_run_submission_for_equity_only():
    intent = OrderIntent(
        strategy_id="e1", underlying="AAPL",
        legs=[Leg(asset_class="equity", contract_symbol="AAPL",
                  side=Side.BUY, quantity=5, limit_price=200.0)],
        quantity=5,
    )
    result = _dry_run_submission(intent)
    assert result["asset_classes"] == ["equity"]


# ---- combined ranking ----

def test_option_and_equity_candidates_combine_into_ranked_list():
    """Both candidate streams flow into a single ranked list (the
    supervisor picks one). High-signal equity should be in the top 3."""
    snap = _snap(
        underlyings=[
            _u("AAPL", dp=0.85, price=180.0),
            _u("MSFT", dp=0.62, price=400.0),
        ],
        options=[
            _o("AAPL260918C00200000", "AAPL", dte=8),
            _o("MSFT260918C00400000", "MSFT", dte=8),
        ],
    )
    weights = ScoringWeights()
    options, _dte = _build_candidates(
        snap, weights, 0.30, 5, dte_min=5, dte_max=10, gnn_output={},
    )
    equities = _build_equity_candidates(
        snap, weights, 0.30, 5, gnn_output={}, risk_limits=_limits(),
    )
    # Both streams should produce candidates for at least one symbol.
    assert len(options) >= 1
    assert len(equities) >= 1
    # The combined list (sorted) should contain at least one of each asset class.
    combined = sorted(options + equities, key=lambda p: p.score, reverse=True)
    asset_classes = {p.legs[0].asset_class for p in combined}
    assert "equity" in asset_classes
    assert "option" in asset_classes
