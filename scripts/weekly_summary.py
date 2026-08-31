"""Run each date in --dates as a fresh Orchestrator and print a one-line
summary. Faster than the full BacktestRunner because we skip the
labeler and persistence — the user wants the per-cycle decisions, not
the held-to-expiry payoff (which needs future data we don't have in
the synthetic DB anyway).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("RUN_MODE", "dry-run")
os.environ.setdefault("AIZEN_TRACE", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2026-07-20,2026-07-27,2026-08-03,2026-08-10,2026-08-17,2026-08-24")
    ap.add_argument("--market-open", default="13:30")
    args = ap.parse_args()
    from src.agents.graph import Orchestrator  # noqa: E402
    from src.agents.nodes.options_structure import _build_candidates  # noqa: E402
    from src.agents.scoring import ScoringWeights  # noqa: E402

    weights = ScoringWeights.from_mapping({})
    print(f"{'date':<12}{'action':<10}{'underlying':<10}{'strategy_id':<10}{'score':<8}{'cands':<6}{'fallback'}")
    print("-" * 100)
    for d in [s.strip() for s in args.dates.split(",") if s.strip()]:
        as_of = f"{d}T{args.market_open}:00Z"
        orch = Orchestrator(as_of=as_of)
        state = orch.run_cycle()
        sel = state.selected_strategy
        cands = state.candidate_strategies
        fb = ""
        if state.market_snapshot:
            # Re-run the candidate builder to extract the fallback note
            try:
                _, fb = _build_candidates(
                    state.market_snapshot, weights, min_score=0.30, top_n=5,
                    dte_min=5, dte_max=10, gnn_output=state.gnn_output or {},
                )
            except Exception:
                fb = ""
        if sel is not None:
            print(
                f"{d:<12}{state.final_action:<10}{sel.underlying:<10}"
                f"{sel.strategy_id[:8]:<10}{sel.score:<8.3f}{len(cands):<6}{fb or '-'}"
            )
        else:
            print(f"{d:<12}{state.final_action:<10}{'-':<10}{'-':<10}{'-':<8}{len(cands):<6}{fb or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
