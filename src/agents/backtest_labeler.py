"""Forward-outcome labeler for the point-in-time backtest (spec 003 / T046).

Given a :class:`DecisionState` produced by the orchestrator at cycle
timestamp T, computes three classes of forward labels:

  1. **Short-horizon return** (``forward_return_h1`` / ``forward_return_h4``):
     the underlying's ``future_return`` from the existing leak-safe
     ``v_labels`` view (1h / 4h on the 15Min bar grid). Re-uses the
     forward-return column the model itself is trained against.

  2. **3-class target** (``target_class``): the band label
     (``down``/``flat``/``up``) the model was trained to predict, from
     ``v_labels.target_class`` at the 4h horizon.

  3. **Option-structure payoff** (``option_payoff``): the realized P&L
     of the proposed multi-leg structure if it had been held to expiry.
     For each leg, the entry price is the option bar at-or-before T
     and the exit price is the option bar at-or-before ``expiration_date``.
     Available only when every leg has both an entry and exit bar;
     otherwise ``None`` (the cycle is reported but the payoff column
     is left NULL).

The labeler NEVER mutates the orchestrator's state and NEVER writes to
``decision_journal``. It reads from SQLite, computes, and returns a
``CycleLabels`` dataclass. The caller (BacktestRunner) is responsible
for persisting the result to ``backtest_cycles``.

Why a separate module?
  - The orchestrator's inference path is cutoff-aware but does not know
    about forward outcomes; the labeler is the other half of the
    point-in-time contract.
  - Putting the forward-return math here means the backtester has a
    single, testable surface for "what actually happened" — re-usable
    later for any other replay path (live-paper-eval, walk-forward
    train/eval, etc.).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .protocol import DecisionState, Leg, OrderIntent, Side


@dataclass
class CycleLabels:
    """All forward outcomes for one replayed cycle."""
    forward_return_h1: float | None = None
    forward_return_h4: float | None = None
    target_class: str | None = None
    option_payoff: float | None = None
    coverage_h1: bool = False
    coverage_h4: bool = False
    coverage_payoff: bool = False
    # Equity-path labels (parallel options+stocks). Only populated when
    # the order intent contains an asset_class='equity' leg. For a long
    # equity leg, the payoff at horizon H is the simple stock return
    # scaled by the leg quantity (no 100x multiplier).
    equity_payoff_h1: float | None = None
    equity_payoff_h4: float | None = None
    coverage_equity_h1: bool = False
    coverage_equity_h4: bool = False

    def to_row_dict(self, *, decision_id: str, cycle_as_of: str,
                    final_action: str, predicted_underlying: str | None,
                    predicted_strategy_id: str | None,
                    predicted_legs_json: str | None,
                    run_at: str, model_version: str | None,
                    feature_flag_state: str) -> dict[str, Any]:
        """Build a row dict matching the ``backtest_cycles`` schema."""
        return {
            "decision_id": decision_id,
            "cycle_as_of": cycle_as_of,
            "final_action": final_action,
            "predicted_underlying": predicted_underlying,
            "predicted_strategy_id": predicted_strategy_id,
            "predicted_legs_json": predicted_legs_json,
            "forward_return_h1": self.forward_return_h1,
            "forward_return_h4": self.forward_return_h4,
            "target_class": self.target_class,
            "option_payoff": self.option_payoff,
            "hit_h4": _hit(self.forward_return_h4, predicted_underlying),
            "hit_h1": _hit(self.forward_return_h1, predicted_underlying),
            "coverage_h1": 1 if self.coverage_h1 else 0,
            "coverage_h4": 1 if self.coverage_h4 else 0,
            "coverage_payoff": 1 if self.coverage_payoff else 0,
            "equity_payoff_h1": self.equity_payoff_h1,
            "equity_payoff_h4": self.equity_payoff_h4,
            "equity_hit_h4": _hit(self.equity_payoff_h4, predicted_underlying),
            "equity_hit_h1": _hit(self.equity_payoff_h1, predicted_underlying),
            "coverage_equity_h1": 1 if self.coverage_equity_h1 else 0,
            "coverage_equity_h4": 1 if self.coverage_equity_h4 else 0,
            "run_at": run_at,
            "model_version": model_version,
            "feature_flag_state": feature_flag_state,
            "notes": None,
        }


class Labeler:
    """Compute forward labels for a DecisionState at cycle time T.

    Parameters
    ----------
    conn : sqlite3.Connection
        Read-only access to the system-of-record DB. The labeler uses
        ``v_labels`` (leak-safe) and ``option_bars`` / ``option_contracts``
        / ``underlying_bars`` for option payoff.
    horizon_bars_h1, horizon_bars_h4 : int
        The 1h / 4h horizons in 15Min bar units (4 / 16). Defaults match
        ``sql/20_view_labels.sql``.
    """

    def __init__(self, conn: sqlite3.Connection, *,
                 horizon_bars_h1: int = 4,
                 horizon_bars_h4: int = 16) -> None:
        self.conn = conn
        self.horizon_bars_h1 = horizon_bars_h1
        self.horizon_bars_h4 = horizon_bars_h4

    # ---- public ---------------------------------------------------------
    def label(self, state: DecisionState, as_of: str) -> CycleLabels:
        """Compute all labels for a single cycle."""
        underlying = _first_underlying(state)
        labels = CycleLabels()

        if underlying is not None:
            h1, cls1 = self._short_horizon(underlying, as_of, self.horizon_bars_h1)
            h4, cls4 = self._short_horizon(underlying, as_of, self.horizon_bars_h4)
            labels.forward_return_h1 = h1
            labels.forward_return_h4 = h4
            labels.target_class = cls4 or cls1
            labels.coverage_h1 = h1 is not None
            labels.coverage_h4 = h4 is not None

        # Option payoff only for PROCEED cycles with an order intent.
        if state.order_intent is not None and state.order_intent.legs:
            payoff = self._option_payoff(state.order_intent, as_of)
            labels.option_payoff = payoff
            labels.coverage_payoff = payoff is not None
            # Equity payoff: only when the order intent has at least one
            # asset_class='equity' leg. Long-only by design (the
            # execution validator already rejects SELL equity legs).
            if any(leg.asset_class == "equity" for leg in state.order_intent.legs):
                eq_h1 = self._equity_payoff(state.order_intent, as_of, self.horizon_bars_h1)
                eq_h4 = self._equity_payoff(state.order_intent, as_of, self.horizon_bars_h4)
                labels.equity_payoff_h1 = eq_h1
                labels.equity_payoff_h4 = eq_h4
                labels.coverage_equity_h1 = eq_h1 is not None
                labels.coverage_equity_h4 = eq_h4 is not None

        return labels

    # ---- short-horizon via v_labels ------------------------------------
    def _short_horizon(self, symbol: str, as_of: str, h: int) -> tuple[float | None, str | None]:
        """Return ``(future_return, target_class)`` for ``symbol`` at the
        most-recent v_labels row at-or-before ``as_of`` with ``horizon_bars = h``.

        Returns ``(None, None)`` when no row matches — e.g. T is at the
        end of the dataset and no LEAD(h) bar exists.
        """
        prev_factory = self.conn.row_factory
        self.conn.row_factory = sqlite3.Row
        try:
            row = self.conn.execute(
                """
                SELECT future_return, target_class
                FROM   v_labels
                WHERE  symbol = ? AND horizon_bars = ?
                  AND  timestamp <= ?
                ORDER  BY timestamp DESC
                LIMIT  1
                """,
                (symbol, h, as_of),
            ).fetchone()
        finally:
            self.conn.row_factory = prev_factory
        if row is None:
            return None, None
        ret = row["future_return"]
        cls = row["target_class"]
        return (
            float(ret) if ret is not None else None,
            str(cls) if cls is not None else None,
        )

    # ---- option-structure payoff ---------------------------------------
    def _option_payoff(self, intent: OrderIntent, as_of: str) -> float | None:
        """Realized P&L of the proposed structure if held to expiry.

        For each leg:
          - entry = option_bars[leg].close at-or-before ``as_of``
          - exit  = option_bars[leg].close at-or-before ``expiration_date``
                    AND strictly after ``as_of`` (so we don't re-use
                    the entry bar when no bar exists near expiry)
          - leg_pnl = (exit - entry) * 100 * quantity * (1 if buy else -1)

        Multiplier 100 is the standard OCC equity-option contract size.
        Returns ``None`` if any leg is missing an entry OR exit bar.
        """
        total = 0.0
        for leg in intent.legs:
            entry = _option_close_at_or_before(self.conn, leg.contract_symbol, as_of)
            if entry is None:
                return None
            exit_ = _option_close_at_or_before(
                self.conn, leg.contract_symbol, leg.expiry,
                strictly_after=as_of,
            )
            if exit_ is None:
                return None
            sign = 1 if leg.side == Side.BUY else -1
            total += (exit_ - entry) * 100.0 * leg.quantity * sign
        return round(total, 4)

    # ---- equity payoff (parallel options+stocks path) ------------------
    def _equity_payoff(self, intent: OrderIntent, as_of: str,
                       horizon_bars: int) -> float | None:
        """Realized P&L of the equity leg(s) at a short horizon (1h / 4h).

        For each equity leg:
          - entry = underlying_bars[symbol].close at-or-before ``as_of``
          - exit  = underlying_bars[symbol].close at-or-before
                    ``as_of + horizon_bars * 15min`` (one bar per 15 min
                    matches the dataset's 15Min grid; the actual exit
                    bar is the latest one at-or-before the target).
          - leg_pnl = (exit - entry) * quantity * (1 if buy else -1)

        No 100x multiplier (one share = one share). Returns ``None`` if
        any equity leg is missing an entry or exit bar.
        """
        # Convert horizon (15Min bars) to a target ISO timestamp.
        # The dataset is 15Min grid so 1h = 4 bars, 4h = 16 bars.
        try:
            from datetime import datetime, timedelta, timezone
            t0 = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            target = t0 + timedelta(minutes=15 * horizon_bars)
            target_iso = target.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None
        total = 0.0
        for leg in intent.legs:
            if leg.asset_class != "equity":
                continue
            entry = _equity_close_at_or_before(
                self.conn, leg.contract_symbol, as_of
            )
            if entry is None:
                return None
            exit_ = _equity_close_at_or_before(
                self.conn, leg.contract_symbol, target_iso,
                strictly_after=as_of,
            )
            if exit_ is None:
                return None
            sign = 1 if leg.side == Side.BUY else -1
            total += (exit_ - entry) * float(leg.quantity) * sign
        return round(total, 4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _first_underlying(state: DecisionState) -> str | None:
    """Return the primary underlying for the cycle.

    Prefers the order intent's underlying (PROCEED cycles); falls back to
    the market snapshot's first underlying; returns None if neither
    exists (e.g. empty universe).
    """
    if state.order_intent is not None and state.order_intent.underlying:
        return state.order_intent.underlying
    if state.selected_strategy is not None and state.selected_strategy.underlying:
        return state.selected_strategy.underlying
    if state.market_snapshot is not None and state.market_snapshot.underlyings:
        return state.market_snapshot.underlyings[0].symbol
    return None


def _option_close_at_or_before(conn: sqlite3.Connection,
                               contract: str, as_of: str,
                               *, strictly_after: str | None = None) -> float | None:
    """Return the option bar close for ``contract`` at-or-before ``as_of``.

    The ``as_of`` may be a full ISO timestamp (e.g. ``2026-08-25T13:30:00Z``)
    or just a date (``2026-08-29``); the WHERE clause filters by string
    comparison which is correct for the canonical storage format.

    When ``strictly_after`` is provided, the bar must be at-or-before
    ``as_of`` AND strictly after ``strictly_after`` (used for the
    exit bar so we don't re-use the entry bar when there's no bar
    near the expiry date). Returns ``None`` when no bar satisfies.
    """
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        if strictly_after is None:
            row = conn.execute(
                """
                SELECT close FROM option_bars
                WHERE  contract_symbol = ? AND timestamp <= ?
                ORDER  BY timestamp DESC
                LIMIT  1
                """,
                (contract, as_of),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT close FROM option_bars
                WHERE  contract_symbol = ?
                  AND  timestamp <= ?
                  AND  timestamp >  ?
                ORDER  BY timestamp DESC
                LIMIT  1
                """,
                (contract, as_of, strictly_after),
            ).fetchone()
    finally:
        conn.row_factory = prev_factory
    if row is None or row["close"] is None:
        return None
    return float(row["close"])


def _equity_close_at_or_before(conn: sqlite3.Connection,
                               symbol: str, as_of: str,
                               *, strictly_after: str | None = None) -> float | None:
    """Return the underlying_bars close for ``symbol`` at-or-before ``as_of``.

    Same shape as :func:`_option_close_at_or_before` but reading from
    ``underlying_bars`` and keyed on the underlying ticker instead of an
    OCC contract symbol. Used by the equity-payoff labeler to compute
    entry / exit prices for a long-equity leg.
    """
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        if strictly_after is None:
            row = conn.execute(
                """
                SELECT close FROM underlying_bars
                WHERE  symbol = ? AND timestamp <= ?
                ORDER  BY timestamp DESC
                LIMIT  1
                """,
                (symbol, as_of),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT close FROM underlying_bars
                WHERE  symbol = ?
                  AND  timestamp <= ?
                  AND  timestamp >  ?
                ORDER  BY timestamp DESC
                LIMIT  1
                """,
                (symbol, as_of, strictly_after),
            ).fetchone()
    finally:
        conn.row_factory = prev_factory
    if row is None or row["close"] is None:
        return None
    return float(row["close"])


def _hit(forward_return: float | None, predicted_underlying: str | None) -> int | None:
    """Hit detection: 1 if sign(forward_return) matches the model's bias,
    0 if it doesn't, None if either side is undeterminable.

    The model's bias is not explicitly stored on the DecisionState today;
    we use a simple proxy: PROCEED cycles are treated as "long" (sign=+1)
    and any positive forward return is a hit. A more sophisticated impl
    would inspect ``state.selected_strategy.underlying_bias``. Spec
    003 / T047 — kept simple here to avoid pulling more state out.
    """
    if forward_return is None or predicted_underlying is None:
        return None
    # Heuristic: any non-NaN forward return is the "target" direction;
    # predicted_underlying is the universe ticker the model picked. We
    # treat a positive forward_return as a "long" and the model's
    # selection of *any* underlying as a "long" proxy (PROCEED only).
    if forward_return > 0:
        return 1
    if forward_return < 0:
        return 0
    return None  # exact zero is non-informative
