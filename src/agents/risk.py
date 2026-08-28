"""Deterministic risk engine.

The risk engine is the only authority on whether an OrderIntent goes to the
broker. It MUST be a pure function of (config, portfolio, OrderIntent) and
MUST NOT call an LLM. The graph routes around it on REJECT/REDUCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .protocol import (
    Leg,
    MarketSnapshot,
    OrderIntent,
    PortfolioPosition,
    RiskAction,
    RiskCheck,
    RiskDecision,
    Side,
)


@dataclass(frozen=True)
class RiskLimits:
    max_leg_quantity: int = 10
    max_order_notional_usd: float = 2000.0
    max_loss_per_trade_usd: float = 300.0
    max_open_positions: int = 6
    max_concentration_per_underlying: float = 0.40
    max_gross_exposure_usd: float = 8000.0
    max_net_delta_per_underlying: float = 50.0
    max_net_vega_per_underlying: float = 100.0
    max_bid_ask_spread_pct: float = 0.15
    min_open_interest: int = 50
    min_option_volume: int = 10
    allowed_underlyings: tuple[str, ...] = ()
    allowed_expiries_dte_min: int = 7
    allowed_expiries_dte_max: int = 60
    require_liquid_underlying: bool = True

    @classmethod
    def from_yaml(cls, raw: Mapping[str, Any]) -> "RiskLimits":
        allowed = tuple(raw.get("allowed_underlyings") or ())
        return cls(
            max_leg_quantity=int(raw.get("max_leg_quantity", cls.max_leg_quantity)),
            max_order_notional_usd=float(raw.get("max_order_notional_usd", cls.max_order_notional_usd)),
            max_loss_per_trade_usd=float(raw.get("max_loss_per_trade_usd", cls.max_loss_per_trade_usd)),
            max_open_positions=int(raw.get("max_open_positions", cls.max_open_positions)),
            max_concentration_per_underlying=float(
                raw.get("max_concentration_per_underlying", cls.max_concentration_per_underlying)
            ),
            max_gross_exposure_usd=float(raw.get("max_gross_exposure_usd", cls.max_gross_exposure_usd)),
            max_net_delta_per_underlying=float(
                raw.get("max_net_delta_per_underlying", cls.max_net_delta_per_underlying)
            ),
            max_net_vega_per_underlying=float(
                raw.get("max_net_vega_per_underlying", cls.max_net_vega_per_underlying)
            ),
            max_bid_ask_spread_pct=float(
                raw.get("max_bid_ask_spread_pct", cls.max_bid_ask_spread_pct)
            ),
            min_open_interest=int(raw.get("min_open_interest", cls.min_open_interest)),
            min_option_volume=int(raw.get("min_option_volume", cls.min_option_volume)),
            allowed_underlyings=allowed,
            allowed_expiries_dte_min=int(
                raw.get("allowed_expiries_dte_min", cls.allowed_expiries_dte_min)
            ),
            allowed_expiries_dte_max=int(
                raw.get("allowed_expiries_dte_max", cls.allowed_expiries_dte_max)
            ),
            require_liquid_underlying=bool(
                raw.get("require_liquid_underlying", cls.require_liquid_underlying)
            ),
        )


def _notional_for_intent(intent: OrderIntent) -> float:
    """Conservative notional estimate: sum of quantity * (limit_price or 0)."""
    total = 0.0
    for leg in intent.legs:
        price = leg.limit_price or 0.0
        total += leg.quantity * price * 100.0  # option multiplier
    return total


def evaluate(intent: OrderIntent, snapshot: MarketSnapshot | None,
             limits: RiskLimits) -> RiskDecision:
    """Return a deterministic APPROVE / REDUCE_SIZE / REJECT decision.

    The decision is a list of named checks plus an aggregate. LLM agents
    never see a partial approval - they see the final decision.
    """
    checks: list[RiskCheck] = []
    reasons: list[str] = []

    # --- universe + structure ---
    if not intent.legs:
        return RiskDecision(
            decision=RiskAction.REJECT,
            approved_quantity=0,
            max_loss=0.0,
            checks=[RiskCheck(name="non_empty_legs", passed=False, detail="no legs")],
            reasons=["OrderIntent has no legs"],
        )

    if limits.allowed_underlyings and intent.underlying not in limits.allowed_underlyings:
        return RiskDecision(
            decision=RiskAction.REJECT,
            approved_quantity=0,
            max_loss=0.0,
            checks=[RiskCheck(name="universe", passed=False,
                              detail=f"{intent.underlying} not in allowed list")],
            reasons=[f"{intent.underlying} not in risk.allowed_underlyings"],
        )

    approved_qty = intent.quantity
    for leg in intent.legs:
        if leg.quantity > limits.max_leg_quantity:
            approved_qty = min(approved_qty, limits.max_leg_quantity)
            reasons.append(
                f"leg {leg.contract_symbol} quantity {leg.quantity} "
                f"capped to {limits.max_leg_quantity}"
            )
            checks.append(RiskCheck(
                name="leg_qty", passed=False,
                detail=f"{leg.contract_symbol} {leg.quantity}>{limits.max_leg_quantity}",
            ))

    # --- notional ---
    notional = _notional_for_intent(intent)
    if notional > limits.max_order_notional_usd:
        scale = limits.max_order_notional_usd / max(notional, 1.0)
        new_qty = max(1, int(intent.quantity * scale))
        if new_qty < intent.quantity:
            approved_qty = min(approved_qty, new_qty)
            reasons.append(
                f"notional ${notional:.0f} exceeds ${limits.max_order_notional_usd:.0f}, "
                f"scaled qty to {approved_qty}"
            )
        checks.append(RiskCheck(
            name="notional", passed=False,
            detail=f"${notional:.0f} > ${limits.max_order_notional_usd:.0f}",
        ))

    # --- per-trade max loss (synthetic, no Greeks yet) ---
    leg_loss = sum(
        (leg.limit_price or 5.0) * 100.0 * leg.quantity
        for leg in intent.legs if leg.side == Side.BUY
    )
    if leg_loss > limits.max_loss_per_trade_usd:
        scale = limits.max_loss_per_trade_usd / max(leg_loss, 1.0)
        new_qty = max(1, int(intent.quantity * scale))
        approved_qty = min(approved_qty, new_qty)
        reasons.append(
            f"max-loss ${leg_loss:.0f} > ${limits.max_loss_per_trade_usd:.0f}, "
            f"scaled qty to {approved_qty}"
        )
        checks.append(RiskCheck(
            name="max_loss", passed=False,
            detail=f"${leg_loss:.0f} > ${limits.max_loss_per_trade_usd:.0f}",
        ))

    # --- portfolio concentration ---
    open_positions = len(snapshot.portfolio) if snapshot else 0
    if open_positions >= limits.max_open_positions:
        return RiskDecision(
            decision=RiskAction.REJECT,
            approved_quantity=0,
            max_loss=0.0,
            checks=[RiskCheck(
                name="open_positions", passed=False,
                detail=f"{open_positions} >= {limits.max_open_positions}",
            )],
            reasons=["max open positions reached"],
        )

    # --- decision ---
    if approved_qty <= 0:
        return RiskDecision(
            decision=RiskAction.REJECT,
            approved_quantity=0,
            max_loss=0.0,
            checks=checks or [RiskCheck(name="general", passed=False, detail="rejected")],
            reasons=reasons or ["rejected by risk"],
        )
    # If any check failed, we are not at APPROVE - the intent was modified.
    if checks and any(not c.passed for c in checks):
        return RiskDecision(
            decision=RiskAction.REDUCE_SIZE,
            approved_quantity=approved_qty,
            max_loss=leg_loss,
            checks=checks,
            reasons=reasons,
        )
    if approved_qty < intent.quantity:
        return RiskDecision(
            decision=RiskAction.REDUCE_SIZE,
            approved_quantity=approved_qty,
            max_loss=leg_loss,
            checks=checks,
            reasons=reasons,
        )
    if not checks:
        checks.append(RiskCheck(name="all", passed=True, detail="all checks passed"))
    return RiskDecision(
        decision=RiskAction.APPROVE,
        approved_quantity=approved_qty,
        max_loss=leg_loss,
        checks=checks,
        reasons=["all risk checks passed"],
    )
