"""Run N cycles with trace on and print the JSONL files for inspection.

The user wants to see exactly which gate is failing for each of the
target weeks. This script:
  1. Runs the orchestrator at each --date
  2. Reads the per-cycle trace JSONL the orchestrator wrote
  3. Prints it as a structured breakdown so the user can grep / diff

Usage:
    python scripts/dump_cycle_traces.py --dates 2026-08-03,2026-08-10,2026-08-17
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Force the same env the user runs in.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("RUN_MODE", "dry-run")
os.environ.setdefault("AIZEN_TRACE", "1")


def _print_step(rec: dict, indent: int = 2) -> None:
    pad = " " * indent
    marker = "[OK]  " if rec.get("success") else "[FAIL]"
    sym = f" {rec['symbol']}" if rec.get("symbol") else ""
    f = rec.get("fields", {}) or {}
    step = rec["step"]
    if step == "ml":
        thr = f.get("threshold", "?")
        n = f.get("n_underlyings", "?")
        hits = sum(1 for p in f.get("predictions", []) if p.get("gate_hit"))
        print(f"{pad}{marker} ml  n={n} threshold={thr} hits={hits}")
        for p in f.get("predictions", []):
            hit = "HIT" if p.get("gate_hit") else "   "
            print(f"{pad}      [{hit}] {p.get('symbol','?'):<6} dir_prob={p.get('direction_probability')}")
    elif step == "gnn":
        print(f"{pad}{marker} gnn  n_nodes={f.get('n_nodes')} n_edges={f.get('n_edges')} kinds={f.get('edge_kinds')} mv={f.get('model_version')}")
    elif step == "topology":
        print(f"{pad}{marker} topology  news_edges={f.get('news_edges')}/{f.get('n_edges')} density={f.get('density')}")
    elif step == "research":
        print(f"{pad}{marker} research  flag={f.get('feature_flag_state')} symbols_with_news={f.get('n_symbols_with_news')}/{f.get('n_symbols_total')}")
    elif step == "final":
        print(f"{pad}{marker} final  action={f.get('action')} risk={f.get('risk_action')} legs={f.get('order_legs')} underlying={f.get('selected_underlying')}")
        for r in f.get("risk_reasons", []):
            print(f"{pad}      risk_reason: {r}")
    elif step == "error":
        print(f"{pad}{marker} error")
    else:
        msg = f.get("message_type", "")
        conf = f.get("confidence")
        print(f"{pad}{marker} {step}  msg={msg} conf={conf} mv={f.get('model_versions')}")
    if rec.get("reasons"):
        for r in rec["reasons"]:
            print(f"{pad}      reason: {r[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2026-08-03,2026-08-10,2026-08-17")
    ap.add_argument("--market-open", default="13:30",
                    help="UTC time of cycle, e.g. 13:30 = 09:30 ET pre-DST")
    args = ap.parse_args()
    from src.agents.graph import Orchestrator  # noqa: E402

    trace_dir = REPO / "models" / "cycle_traces"
    for d in [s.strip() for s in args.dates.split(",") if s.strip()]:
        as_of = f"{d}T{args.market_open}:00Z"
        print(f"\n========== {d} ({as_of}) ==========")
        orch = Orchestrator(as_of=as_of)
        state = orch.run_cycle()
        # Locate the trace file the orchestrator just wrote.
        cands = sorted(trace_dir.glob(f"cycle_{state.decision_id}.jsonl"))
        if not cands:
            print(f"  (no trace file found for {state.decision_id})")
            continue
        for line in cands[0].read_text(encoding="utf-8").splitlines():
            _print_step(json.loads(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
