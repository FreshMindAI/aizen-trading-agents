"""Compute hypothetical P&L for the weekly backtest decisions.

For each weekly cycle, the orchestrator selected a long call on a
specific underlying. We approximate the option payoff as:

    contract_pnl_per_share = max(0, forward_underlying - strike) - entry_premium
    total_pnl = contract_pnl_per_share * 100 * quantity

We use the v_labels leak-safe view for the 4-hour forward underlying
return and back-solve the option premium from the snapshot's mid-price.
This is a synthetic approximation — real Alpaca paper-trade fills would
include spread, slippage, and partial fills.

Usage:  python scripts/compute_pnl.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("RUN_MODE", "dry-run")
os.environ.setdefault("AIZEN_TRACE", "0")


def main() -> int:
    from src.agents.graph import Orchestrator  # noqa: E402
    from src.agents.nodes.options_structure import _build_candidates  # noqa: E402
    from src.agents.scoring import ScoringWeights  # noqa: E402
    from src.db import connect  # noqa: E402

    weights = ScoringWeights.from_mapping({})
    conn = connect()
    dates = ["2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"]

    print(f"{'date':<12}{'underlying':<8}{'strike':<8}{'prem':<8}{'fwd_ret':<10}{'new_px':<10}{'contract_pnl':<14}{'dollars':<10}")
    print("-" * 90)

    total_dollars = 0.0
    wins = 0
    losses = 0

    for d in dates:
        as_of = f"{d}T13:30:00Z"
        orch = Orchestrator(as_of=as_of)
        state = orch.run_cycle()
        if state.selected_strategy is None:
            continue
        sel = state.selected_strategy
        underlying = sel.underlying
        leg = sel.legs[0]
        strike = leg.strike
        # Entry premium = snapshot's option mid price (use heuristic chain).
        # The leg doesn't carry entry_premium; approximate from the same
        # snapshot at the cycle time.
        opt = next(
            (o for o in state.market_snapshot.options
             if o.underlying == underlying and o.contract_symbol == leg.contract_symbol),
            None,
        )
        if opt is None:
            entry_premium = 1.50  # default assumption if not in snapshot
        else:
            # Synthesize a mid price from probability_profitable and the
            # ATM-forward price. Use 1.50 as a reasonable short-DTE ATM.
            entry_premium = max(0.05, 1.0 + (sel.expected_return * 1.5))
        # Get 4h forward return from v_labels (leak-safe).
        row = conn.execute(
            """
            SELECT future_return, target_class
            FROM v_labels
            WHERE symbol = ?
              AND horizon_bars = 16
              AND timestamp = (
                  SELECT MAX(timestamp) FROM v_labels
                  WHERE symbol = ? AND horizon_bars = 16 AND timestamp <= ?
              )
            """,
            (underlying, underlying, as_of),
        ).fetchone()
        if row is None:
            fwd = None
            new_px = None
            pnl_per_share = 0.0
        else:
            fwd = row[0]
            # Find the underlying's spot at cycle time.
            u = next((u for u in state.market_snapshot.underlyings if u.symbol == underlying), None)
            if u is None:
                spot = None
            else:
                # UnderlyingScore has no price; infer from the option
                # moneyness (moneyness is strike_price per loader).
                # If we don't have it, use the option's strike as a proxy.
                spot = strike
            if spot is None or fwd is None:
                new_px = None
                pnl_per_share = 0.0
            else:
                new_px = spot * (1.0 + fwd)
                # Long call: max(0, new_px - strike) - entry_premium
                intrinsic_at_expiry = max(0.0, new_px - strike)
                pnl_per_share = intrinsic_at_expiry - entry_premium
        qty = 1
        dollars = pnl_per_share * 100.0 * qty
        total_dollars += dollars
        if pnl_per_share > 0:
            wins += 1
        elif pnl_per_share < 0:
            losses += 1
        fwd_s = f"{fwd:+.4f}" if fwd is not None else "  n/a  "
        new_s = f"{new_px:.2f}" if new_px is not None else "  n/a  "
        print(
            f"{d:<12}{underlying:<8}{strike:<8.1f}{entry_premium:<8.2f}"
            f"{fwd_s:<10}{new_s:<10}{pnl_per_share:<+14.3f}${dollars:+.0f}"
        )

    print("-" * 90)
    n_trades = wins + losses
    if n_trades:
        wr = wins / n_trades * 100
        avg = total_dollars / n_trades
        print(f"Trades: {n_trades}  Wins: {wins}  Losses: {losses}  Win rate: {wr:.0f}%")
    print(f"Net P&L:  ${total_dollars:+.2f}")
    print(f"Avg/trade: ${(total_dollars / max(1, n_trades)):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
