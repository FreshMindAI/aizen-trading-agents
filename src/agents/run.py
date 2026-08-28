"""CLI entry point for the Phase-3 multi-agent system.

Usage:
    python -m src.agents.run --once                 # single decision cycle
    python -m src.agents.run --loop --interval 60   # poll every N seconds
    python -m src.agents.run --once --mode dry-run  # no broker calls

The orchestrator decides the actual mode from config/agents.yaml; CLI flags
override it for the current invocation only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ..config import get_settings, get_yaml
from ..db import connect
from .graph import Orchestrator

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase-3 multi-agent orchestrator")
    ap.add_argument("--once", action="store_true", help="run a single decision cycle and exit")
    ap.add_argument("--loop", action="store_true", help="loop on --interval seconds")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between cycles in --loop mode (default 300)")
    ap.add_argument("--mode", choices=["paper", "dry-run", "live"], default=None,
                    help="override run_mode from config")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="stop after N cycles (--loop mode)")
    ap.add_argument("--config", type=Path, default=None,
                    help="override config/agents.yaml path")
    ap.add_argument("--log-level", default=None,
                    help="DEBUG/INFO/WARNING/ERROR (overrides YAML)")
    args = ap.parse_args(argv)

    settings = get_settings()
    log_level = (args.log_level
                 or get_yaml("settings").get("logging", {}).get("level", "INFO"))
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format=get_yaml("settings").get("logging", {}).get(
            "format", "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"),
    )

    # Optional mode override (only for this process).
    if args.mode:
        import os
        os.environ["RUN_MODE"] = args.mode
        object.__setattr__(settings, "run_mode", args.mode)

    if not (args.once or args.loop):
        ap.print_help()
        return 1

    conn = connect()
    orchestrator = Orchestrator(conn=conn)
    cycles = 0
    try:
        if args.once:
            cycles = 1
            _run_one(orchestrator)
        else:
            max_cycles = args.max_cycles or 10_000
            while cycles < max_cycles:
                _run_one(orchestrator)
                cycles += 1
                logger.info("cycle %d done; sleeping %ds", cycles, args.interval)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("interrupted after %d cycles", cycles)
    return 0


def _run_one(orchestrator: Orchestrator) -> None:
    state = orchestrator.run_cycle()
    summary = {
        "decision_id": state.decision_id,
        "final_action": state.final_action,
        "n_observations": len(state.agent_observations),
        "n_candidates": len(state.candidate_strategies),
        "risk_decision": state.risk_decision.decision.value if state.risk_decision else None,
        "execution_status": (state.execution_result or {}).get("status"),
    }
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
