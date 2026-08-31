"""Quick pretty-printer for backtest JSON reports."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def main(path: str) -> None:
    p = REPO / path
    r = json.loads(p.read_text(encoding="utf-8"))
    print(f"== {p.name} ==")
    print(f"created_at: {r.get('created_at', '-')}")
    cfg = r.get("config", {})
    print(f"interval:   {cfg.get('interval')}  start: {cfg.get('start_date')}  end: {cfg.get('end_date')}")
    print(f"universe:   {cfg.get('universe')}")
    print()
    print("== Per-cycle ==")
    for c in r["cycles"]:
        action = c["final_action"]
        underlying = c.get("predicted_underlying") or "-"
        strat = c.get("predicted_strategy_id") or "-"
        h4 = c.get("forward_return_h4")
        h4s = f"{h4:+.4f}" if h4 is not None else "  n/a "
        payoff = c.get("option_payoff")
        pays = f"  \${payoff:+.0f}" if payoff is not None else "    n/a"
        hit = c.get("hit_h4")
        hits = " hit" if hit == 1 else "miss" if hit == 0 else " n/a"
        notes = (c.get("notes") or "")[:60]
        print(
            f"  {c['cycle_as_of'][:10]}  {action:<8}  {underlying:<6} {strat:<14} "
            f"h4={h4s}  payoff={pays}  {hits}  {notes}"
        )
    print()
    print("== Aggregate ==")
    for k, v in r["aggregate"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/backtest_weekly_6w.json")
