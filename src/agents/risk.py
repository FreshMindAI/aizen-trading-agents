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
    # Equity-specific limits (hackathon parallel path). Equity legs
    # bypass option premium/theta concerns but introduce Pattern Day
    # Trader risk and full notional exposure. Defaults sized for a
    # 1-week paper account with cash (no leverage).
    max_equity_notional_per_symbol: float = 1500.0
    max_equity_notional_total: float = 5000.0
    min_equity_direction_probability: float = 0.55
    # When both option and equity candidates clear the score gate, the
    # equity one is preferred by this score-margin. Negative = prefer
    # options. Set to 0 to make it a tie (option wins as default).
    equity_preferred_when_score_margin: float = 0.0

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
            max_equity_notional_per_symbol=float(
                raw.get("max_equity_notional_per_symbol", cls.max_equity_notional_per_symbol)
            ),
            max_equity_notional_total=float(
                raw.get("max_equity_notional_total", cls.max_equity_notional_total)
            ),
            min_equity_direction_probability=float(
                raw.get("min_equity_direction_probability", cls.min_equity_direction_probability)
            ),
            equity_preferred_when_score_margin=float(
                raw.get("equity_preferred_when_score_margin", cls.equity_preferred_when_score_margin)
            ),
        )

    @classmethod
    def scaled_from_capital(cls, capital_usd: float) -> "RiskLimits":
        """Build a :class:`RiskLimits` whose dollar caps are a fixed % of
        the account's capital.

        Allocation policy (round-trip, cash account, no leverage):
          * 2 % per trade  → ``max_order_notional_usd``
          * 5 % per symbol → ``max_equity_notional_per_symbol`` and per-leg
            max-notional proxy
          * 10 % gross equity (no leverage, no margin)
          * 40 % gross (mix of equity + options) → ``max_gross_exposure_usd``
          * 5 % per-trade max loss → ``max_loss_per_trade_usd``
          * 50 % max concentration per underlying (within gross)
          * 6 concurrent open positions (1-week hackathon, not a 6-month fund)
          * 200-share max leg (caps illiquid mega-caps to ~$200k notional
            — well above the per-symbol cap, so the per-symbol cap binds
            first)
          * 50-share max net delta per underlying (50 * $200 = $10k delta
            exposure max — ~10% of the $100k account)
          * 200 vega per underlying (options side)

        The percentages are hard-coded on purpose: they encode the
        "1-week hackathon paper account with $100k cash" policy. A real
        prop shop would re-derive them from a Kelly / vol-target model
        in a separate ``RiskPolicy`` class. The function is deterministic
        and pure so the same capital value produces the same limits
        across cycles.
        """
        if capital_usd <= 0:
            raise ValueError(f"capital_usd must be > 0; got {capital_usd}")
        return cls(
            max_leg_quantity=200,
            max_order_notional_usd=round(capital_usd * 0.02, 2),
            max_loss_per_trade_usd=round(capital_usd * 0.05, 2),
            max_open_positions=6,
            max_concentration_per_underlying=0.50,
            max_gross_exposure_usd=round(capital_usd * 0.40, 2),
            max_net_delta_per_underlying=50.0,
            max_net_vega_per_underlying=200.0,
            max_bid_ask_spread_pct=0.05,         # tighter on a bigger book
            min_open_interest=100,                # need deeper OI for size
            min_option_volume=50,
            allowed_underlyings=(),
            allowed_expiries_dte_min=5,
            allowed_expiries_dte_max=10,
            require_liquid_underlying=True,
            # Equity-specific (5% of capital per name, 10% of capital gross)
            max_equity_notional_per_symbol=round(capital_usd * 0.05, 2),
            max_equity_notional_total=round(capital_usd * 0.10, 2),
            min_equity_direction_probability=0.55,
            equity_preferred_when_score_margin=0.0,
        )


def _notional_for_intent(intent: OrderIntent) -> float:
    """Conservative notional estimate: sum of quantity * price.

    Option contracts carry a 100x multiplier (one contract = 100 shares);
    equity legs do not. The notional for an equity leg is therefore
    ``quantity * price`` while an option leg is ``quantity * price * 100``.
    """
    total = 0.0
    for leg in intent.legs:
        price = leg.limit_price or 0.0
        if leg.asset_class == "equity":
            total += leg.quantity * price
        else:
            total += leg.quantity * price * 100.0  # option multiplier
    return total


def _notional_for_intent_by_asset_class(intent: OrderIntent) -> tuple[float, float]:
    """Split the notional into (equity_notional, option_notional).

    Used by the equity-specific risk checks (per-symbol and total caps).
    """
    equity_notional = 0.0
    option_notional = 0.0
    for leg in intent.legs:
        price = leg.limit_price or 0.0
        if leg.asset_class == "equity":
            equity_notional += leg.quantity * price
        else:
            option_notional += leg.quantity * price * 100.0
    return equity_notional, option_notional


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

    # --- equity-specific notional caps (parallel options+stocks path) ---
    equity_notional, _opt_notional = _notional_for_intent_by_asset_class(intent)
    if equity_notional > 0:
        # Per-symbol cap: a single equity leg cannot exceed
        # max_equity_notional_per_symbol (default $1.5k).
        for leg in intent.legs:
            if leg.asset_class != "equity":
                continue
            leg_notional = leg.quantity * (leg.limit_price or 0.0)
            if leg_notional > limits.max_equity_notional_per_symbol:
                # Cap the leg quantity to the per-symbol notional limit.
                max_qty = max(1, int(limits.max_equity_notional_per_symbol / max(leg.limit_price or 1.0, 0.01)))
                if max_qty < leg.quantity:
                    approved_qty = min(approved_qty, max_qty)
                    reasons.append(
                        f"equity leg {leg.contract_symbol} notional "
                        f"${leg_notional:.0f} exceeds per-symbol cap "
                        f"${limits.max_equity_notional_per_symbol:.0f}, "
                        f"scaled qty to {max_qty}"
                    )
                    checks.append(RiskCheck(
                        name="equity_per_symbol_notional", passed=False,
                        detail=f"{leg.contract_symbol} ${leg_notional:.0f} > "
                               f"${limits.max_equity_notional_per_symbol:.0f}",
                    ))
        # Total equity cap: across all symbols in the intent, equity
        # notional cannot exceed max_equity_notional_total (default $5k).
        if equity_notional > limits.max_equity_notional_total:
            scale = limits.max_equity_notional_total / max(equity_notional, 1.0)
            new_qty = max(1, int(intent.quantity * scale))
            if new_qty < intent.quantity:
                approved_qty = min(approved_qty, new_qty)
                reasons.append(
                    f"total equity notional ${equity_notional:.0f} exceeds "
                    f"${limits.max_equity_notional_total:.0f}, scaled qty to {new_qty}"
                )
            checks.append(RiskCheck(
                name="equity_total_notional", passed=False,
                detail=f"${equity_notional:.0f} > ${limits.max_equity_notional_total:.0f}",
            ))

    # --- per-trade max loss (synthetic, no Greeks yet) ---
    def _leg_max_loss(leg) -> float:
        if leg.asset_class == "equity":
            # Equity: max loss bounded by full notional (price -> 0).
            return float(leg.limit_price or 0.0) * float(leg.quantity)
        # Option: premium paid * 100 multiplier * qty (the price paid).
        return float(leg.limit_price or 5.0) * 100.0 * float(leg.quantity)
    leg_loss = sum(
        _leg_max_loss(leg)
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
