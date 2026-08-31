"""Pretty-print the per-cycle trace for a given decision_id.

Usage:
    python scripts/print_cycle_trace.py
    python scripts/print_cycle_trace.py cycle_2026-08-17T12-30-00Z-abc123.jsonl
    python scripts/print_cycle_trace.py --latest 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TRACE_DIR = REPO / "models" / "cycle_traces"


def _print_rec(rec: dict) -> None:
    marker = "[OK]  " if rec.get("success") else "[FAIL]"
    sym = f" {rec['symbol']}" if rec.get("symbol") else ""
    f = rec.get("fields", {}) or {}
    step = rec["step"]
    if step == "ml":
        print(f"  {marker} ml{sym}  n={f.get('n_underlyings', '?')} threshold={f.get('threshold', '?')}")
    elif step == "gnn":
        print(
            f"  {marker} gnn  n_nodes={f.get('n_nodes')} n_edges={f.get('n_edges')} "
            f"kinds={f.get('edge_kinds')} mv={f.get('model_version')}"
        )
    elif step == "topology":
        print(
            f"  {marker} topology  news_edges={f.get('news_edges')}/{f.get('n_edges')} "
            f"density={f.get('density')} weight_mean={f.get('weight_mean')}"
        )
    elif step == "research":
        print(
            f"  {marker} research  flag={f.get('feature_flag_state')} "
            f"symbols_with_news={f.get('n_symbols_with_news')}/{f.get('n_symbols_total')} "
            f"sentiment_mean={f.get('sentiment_mean')}"
        )
    elif step == "final":
        print(
            f"  {marker} final  action={f.get('action')} risk={f.get('risk_action')} "
            f"legs={f.get('order_legs')} underlying={f.get('selected_underlying')} "
            f"strategy={f.get('selected_strategy_id')}"
        )
        if f.get("risk_reasons"):
            for r in f["risk_reasons"]:
                print(f"          risk_reason: {r}")
    else:
        # Per-agent observation
        print(
            f"  {marker} {step}  msg={f.get('message_type')} "
            f"conf={f.get('confidence')} data_v={f.get('data_version')}"
        )
    if rec.get("reasons"):
        for r in rec["reasons"]:
            print(f"          reason: {r}")
    if rec.get("duration_ms") is not None:
        print(f"          duration_ms: {rec['duration_ms']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", help="Trace JSONL file (in models/cycle_traces).")
    p.add_argument("--latest", type=int, default=0, help="Show the latest N trace files.")
    args = p.parse_args(argv)

    if not TRACE_DIR.exists():
        print(f"No trace dir at {TRACE_DIR}")
        return 1

    if args.file:
        files = [TRACE_DIR / args.file]
    elif args.latest:
        files = sorted(TRACE_DIR.glob("cycle_*.jsonl"), key=lambda p: p.stat().st_mtime)[-args.latest :]
    else:
        files = sorted(TRACE_DIR.glob("cycle_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            print(f"No trace files in {TRACE_DIR}")
            return 1
        files = files[-1:]

    for f in files:
        if not f.exists():
            print(f"missing: {f}")
            continue
        print(f"--- {f.name} ---")
        for line in f.read_text(encoding="utf-8").splitlines():
            _print_rec(json.loads(line))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
