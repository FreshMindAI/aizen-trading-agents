"""Single-cycle entry point for Render Cron Job deployment.

Render Cron Jobs invoke a command on a schedule, exit, and start fresh
on the next tick. This is the right primitive for a 15-min trading
loop because:

  * Each tick is independently debuggable (one decision_id per invocation).
  * Render's ephemeral disk model is fine because we use a persistent
    disk mounted at /var/data/aizen.
  * Cold-start latency (~5s) is well under our 15-min cadence.

What this script does (per tick):
  1. Resolve the DB path from AIZEN_DB_PATH or fall back to /var/data/aizen/trading.db
     (Render's persistent disk mount).
  2. Initialize the schema (idempotent).
  3. Pull a 1-bar refresh from Alpaca for the universe.
  4. Pull the latest news snapshot (when the research flag is on).
  5. Construct an Orchestrator (mock LLM by default; auto-loads ML artifacts
     and the latest GNN snapshot).
  6. Run one cycle: fetch snapshot -> run agents -> journal upsert -> write
     per-cycle trace.
  7. On PROCEED, the execution node submits the order to Alpaca (paper
     by default; dry-run if RUN_MODE=dry-run).
  8. Print a 4-line summary and exit.

Usage (local):
    python -m src.agents.cli.run_loop --once

Usage (Render):
    render.yaml declares the schedule and the env vars; Render calls
    `python -m src.agents.cli.run_loop --once` every 15 minutes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Pin env BEFORE importing anything that reads it.
# Default to mock LLM so the cron job never tries to reach GMI / MiniMaxAI /
# MiniMax-M3 (returning HTTP 402) on a cold start. Set AIZEN_LLM_PROVIDER=
# in render.yaml to override.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("AIZEN_TRACE", "1")
os.environ.setdefault("RUN_MODE", "paper")  # Alpaca paper account
os.environ.setdefault("AIZEN_DB_PATH", "/var/data/aizen/trading.db")

logger = logging.getLogger("run_loop")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_db_path() -> Path:
    raw = os.getenv("AIZEN_DB_PATH", "/var/data/aizen/trading.db")
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _refresh_data(orch) -> dict[str, int]:
    """One-bar refresh of the underlying data + option chain.

    Returns a small dict of {symbol: n_bars_written} so the operator
    can see at a glance whether the tick actually received new data.
    Skipped silently if Alpaca is unreachable (we don't want a broker
    outage to kill the cron job).
    """
    import json as _json
    from src.agents.alpaca_trading import AlpacaTradingError
    from src.agents.data_refresh import refresh_one_bar
    written: dict[str, int] = {}
    try:
        written = refresh_one_bar(orch.settings.universe, db_path=_resolve_db_path())
    except AlpacaTradingError as exc:
        logger.warning("Alpaca refresh failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Data refresh failed: %s: %s", type(exc).__name__, exc)
    return written


def _print_summary(state, n_refreshed: int, started_at: str, started_at_epoch: float) -> None:
    sel = state.selected_strategy
    action = state.final_action
    n_cands = len(state.candidate_strategies)
    n_legs = len(state.order_intent.legs) if state.order_intent else 0
    exec_status = (state.execution_result or {}).get("status") or "ok"
    print("== run_loop summary ==")
    print(f"  started_at    : {started_at}")
    print(f"  duration_s    : {round(time.time() - started_at_epoch, 2)}")
    print(f"  refreshed     : {sum(n_refreshed.values())} bars across {len(n_refreshed)} symbols")
    print(f"  decision_id   : {state.decision_id}")
    print(f"  final_action  : {action}")
    print(f"  underlying    : {sel.underlying if sel else '-'}")
    print(f"  strategy_id   : {sel.strategy_id if sel else '-'}")
    print(f"  score         : {round(sel.score, 3) if sel else '-'}")
    print(f"  candidates    : {n_cands}")
    print(f"  order_legs    : {n_legs}")
    print(f"  exec_status   : {exec_status}")
    print(f"  cycle_started : {state.cycle_started_at}")
    # On PROCEED, this is the "next-day plan" the operator reads before
    # the next market open. On NO_TRADE, print the highest-scoring
    # candidate so the operator can see what *almost* traded.
    if action == "PROCEED" and sel is not None:
        print(f"  next_plan     : PROCEED {sel.underlying} | score={round(sel.score, 3)} | thesis='{(sel.thesis or '')[:80]}'")
    elif sel is not None:
        print(f"  next_plan     : NO_TRADE (top candidate was {sel.underlying} @ {round(sel.score, 3)})")


def run_once() -> int:
    """Run a single cycle. Returns process exit code."""
    started = _now_iso()
    started_epoch = time.time()
    from src.agents.graph import Orchestrator
    from src.db import connect, init_db

    db_path = _resolve_db_path()
    conn = connect(db_path)
    init_db(conn)  # idempotent
    try:
        orch = Orchestrator(conn=conn)
        n_refreshed = _refresh_data(orch)
        state = orch.run_cycle()
        _print_summary(state, n_refreshed, started, started_epoch)
        # Rebuild the static dashboard after every cycle. Cheap (~100ms
        # for an empty DB, ~1s for 1000s of cycles). Render pushes the
        # resulting `dashboard/index.html` to gh-pages via the workflow
        # in .github/workflows/dashboard.yml so judges can browse P&L
        # + trade history without auth.
        try:
            _rebuild_dashboard()
        except Exception as exc:  # noqa: BLE001
            logger.warning("dashboard rebuild failed: %s: %s", type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("cycle failed: %s\n%s", exc, traceback.format_exc())
        return 1
    finally:
        conn.close()
    return 0


def _rebuild_dashboard() -> None:
    """Re-render dashboard/index.html from the live DB.

    Imported lazily so the cron tick still runs even if the dashboard
    module path changes (e.g. dashboard/ removed during surgery).
    """
    import subprocess
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "build_dashboard.py"
    if not script.exists():
        logger.warning("dashboard script not found: %s", script)
        return
    env = os.environ.copy()
    env.setdefault("AIZEN_DB_PATH", str(_resolve_db_path()))
    subprocess.run(
        [sys.executable, "-m", "scripts.build_dashboard"],
        cwd=str(repo),
        env=env,
        check=False,
        capture_output=True,
        timeout=30,
    )


def run_forever(interval_seconds: int) -> int:
    """Local long-lived loop. NOT used by Render (Render uses --once
    in a Cron Job). Kept for local debugging."""
    stop = {"flag": False}

    def _on_signal(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    logger.info("run_forever starting; interval=%ds", interval_seconds)
    while not stop["flag"]:
        rc = run_once()
        if rc != 0:
            logger.warning("cycle returned %d; sleeping anyway", rc)
        for _ in range(interval_seconds):
            if stop["flag"]:
                break
            time.sleep(1)
    logger.info("run_forever stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true",
                   help="Run exactly one cycle and exit (Render Cron Job mode).")
    p.add_argument("--interval", type=int, default=900,
                   help="Seconds between cycles when --once is not set (default 900 = 15 min).")
    args = p.parse_args(argv)
    if args.once:
        return run_once()
    return run_forever(args.interval)


if __name__ == "__main__":
    sys.exit(main())
