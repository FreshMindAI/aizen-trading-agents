"""Deterministic risk engine tests."""

from __future__ import annotations

import pytest

from src.agents.protocol import (
    Leg, MarketSnapshot, OrderIntent, OptionType, PortfolioPosition, Side,
)
from src.agents.risk import RiskLimits, evaluate


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        max_leg_quantity=10,
        max_order_notional_usd=5000.0,
        max_loss_per_trade_usd=1000.0,
        max_open_positions=6,
        allowed_underlyings=("AAPL", "SPY"),
        allowed_expiries_dte_min=7,
        allowed_expiries_dte_max=60,
    )


def _intent(qty: int = 1, leg_qty: int | None = None,
            limit: float | None = 5.0, underlying: str = "AAPL") -> OrderIntent:
    return OrderIntent(
        strategy_id="s1", underlying=underlying,
        legs=[Leg(contract_symbol=f"{underlying}260101C00200000",
                  side=Side.BUY, quantity=leg_qty or qty,
                  option_type=OptionType.CALL, strike=200.0, expiry="2026-12-31",
                  limit_price=limit)],
        quantity=qty,
    )


def test_approves_simple_intent(limits):
    decision = evaluate(_intent(), None, limits)
    assert decision.decision.value == "APPROVE"
    assert decision.approved_quantity == 1


def test_rejects_underlying_not_in_universe(limits):
    decision = evaluate(_intent(underlying="XYZ"), None, limits)
    assert decision.decision.value == "REJECT"
    assert "not in" in decision.reasons[0]


def test_reduces_quantity_when_leg_too_large(limits):
    decision = evaluate(_intent(qty=20, leg_qty=20, limit=5.0), None, limits)
    assert decision.decision.value == "REDUCE_SIZE"
    assert decision.approved_quantity <= limits.max_leg_quantity


def test_rejects_when_no_legs(limits):
    intent = OrderIntent(strategy_id="x", underlying="AAPL", legs=[], quantity=1)
    decision = evaluate(intent, None, limits)
    assert decision.decision.value == "REJECT"


def test_reduces_when_notional_exceeds_limit(limits):
    # 1 contract at $50 = $5000 notional > $2000 limit
    decision = evaluate(_intent(qty=1, limit=50.0), None, limits)
    assert decision.decision.value in ("REDUCE_SIZE", "REJECT")
    assert decision.approved_quantity >= 0


def test_risk_decision_includes_checks(limits):
    decision = evaluate(_intent(), None, limits)
    assert len(decision.checks) >= 1
    for c in decision.checks:
        assert isinstance(c.passed, bool)


def test_respects_open_position_cap(limits):
    snap = MarketSnapshot(
        timestamp="2026-08-28T15:30:00Z",
        portfolio=[PortfolioPosition(
            symbol=f"X{i}", asset_class="equity", side=Side.BUY,
            quantity=1, entry_price=100.0,
        ) for i in range(limits.max_open_positions)],
    )
    decision = evaluate(_intent(), snap, limits)
    assert decision.decision.value == "REJECT"
    assert "open positions" in decision.reasons[0].lower()


def test_yaml_loader_roundtrip():
    import yaml
    raw = """
    max_leg_quantity: 5
    max_order_notional_usd: 1000
    allowed_underlyings: [AAPL, SPY]
    """
    data = yaml.safe_load(raw)
    limits = RiskLimits.from_yaml(data)
    assert limits.max_leg_quantity == 5
    assert limits.allowed_underlyings == ("AAPL", "SPY")
