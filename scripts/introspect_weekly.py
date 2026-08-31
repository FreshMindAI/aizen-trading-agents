"""Introspect the multi-agent pipeline on the user's three target dates.

Prints a one-line summary per agent per cycle so the user can see which
gate is failing on Aug 3, Aug 10, Aug 17 (and any other date in the
argument list).

Usage:
    python scripts/introspect_weekly.py
    python scripts/introspect_weekly.py --dates 2026-08-03,2026-08-10,2026-08-17
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Force the same env the user runs in.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("RUN_MODE", "dry-run")
os.environ.setdefault("AIZEN_TRACE", "1")


def _fmt_obs(o) -> str:
    msg = o.message_type.value if hasattr(o.message_type, "value") else str(o.message_type)
    sig = o.signal or {}
    sig_summary = ""
    if msg == "DIRECTION_VIEW":
        sig_summary = f" bias={sig.get('bias','?')}"
    elif msg == "VOLATILITY_VIEW":
        sig_summary = f" iv_rv={sig.get('iv_rv_gap', '?')}"
    elif msg == "STRATEGY_PROPOSAL":
        sig_summary = f" score={sig.get('score','?')}"
    risk = ""
    if o.risks:
        risk = f"  risk[0]={o.risks[0][:60]}"
    return f"  {o.agent_id:<14} conf={o.confidence:.2f}  msg={msg:<18}{sig_summary}{risk}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dates",
        default="2026-08-03,2026-08-10,2026-08-17",
        help="Comma-separated YYYY-MM-DD list (each at 13:30Z = 09:30 ET).",
    )
    args = ap.parse_args()
    from src.agents.graph import Orchestrator  # noqa: E402

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    for d in dates:
        as_of = f"{d}T13:30:00Z"
        print(f"=== {d}  ({as_of}) ===")
        orch = Orchestrator(as_of=as_of)
        state = orch.run_cycle()
        # ML: list which symbols crossed the gate
        try:
            from src.agents.nodes.strategy_selector import (
                _BASE_RATE, _BASE_RATE_CUSHION,
            )
            thr = min(0.95, _BASE_RATE + _BASE_RATE_CUSHION) if _BASE_RATE is not None else 0.55
        except Exception:
            thr = 0.55
        for u in state.ml_predictions:
            p = u.direction_probability
            hit = "HIT" if (p is not None and p >= thr) else "   "
            print(f"  ml  {u.symbol:<6}  dir_prob={p if p is not None else 'n/a':.4f}  [{hit}]  (thr={thr:.4f})")
        # Per-agent observations
        for o in state.agent_observations:
            print(_fmt_obs(o))
        # Final
        sel = state.selected_strategy
        legs = len(state.order_intent.legs) if state.order_intent else 0
        print(
            f"  FINAL: action={state.final_action}  selected={sel.underlying if sel else None}  "
            f"strategy={sel.strategy_id if sel else None}  legs={legs}"
        )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
