"""Point-in-time backtest CLI (spec 003 / T049).

Replays past cycle timestamps against the SQLite system of record and
writes a :class:`BacktestReport` (JSON) with per-cycle forward outcomes
and aggregate metrics. Mirrors the structure of ``backfill_news.py`` so
the surface is uniform across the agent-CLI suite.

Why this exists
  - The orchestrator today is wall-clock driven; there is no way to ask
    "what would have happened if I had run this on Monday at 09:30 ET
    with last week's news?" This CLI is the answer.
  - Each cycle instantiates a fresh ``Orchestrator(as_of=T)`` so the
    inference layer's ``timestamp <= as_of`` filter is enforced per-cycle
    and the forward-leak guard (spec 003 / T046) is automatically
    applied. The labeler reads v_labels (leak-safe) and option_bars
    at-or-before the cut-off to compute forward outcomes.

Usage:
    python -m src.agents.cli.backtest \
        --start 2026-08-04 --end 2026-08-29 \
        --interval weekly --universe NVDA \
        --out models/backtest_report_nvda.json

    # Daily run, news flag off (legacy orchestrator path):
    python -m src.agents.cli.backtest \
        --start 2026-08-18 --end 2026-08-25 \
        --interval daily --universe NVDA,AAPL \
        --news-disabled \
        --out models/backtest_daily_no_news.json

Notes
  - LLM provider is forced to ``mock`` regardless of env vars. The
    backtester does not perform live calls; any LLM-backed reasoning
    in the orchestrator is short-circuited to the deterministic stub.
  - The CLI never mutates ``decision_journal``; per-cycle rows go to
    the dedicated ``backtest_cycles`` table so the live journal is
    preserved untouched.
  - Cycle timestamps for which ``underlying_bars`` has no row at-or-
    before the cycle time are silently skipped and reported in
    ``skipped_no_data`` (no exception is raised).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# IMPORTANT: the security constraint set by the user requires that the
# backtester never call live LLM endpoints (GMI / MiniMaxAI / MiniMax-M3
# returning HTTP 402, etc.). Pin the provider to ``mock`` BEFORE any
# import of the LLM factory can read the env var.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
# Force dry-run mode so the execution node short-circuits the Alpaca
# submit_order call. A 60-cycle backtest against the live broker would
# either exhaust the rate limit or return 422 "options market orders
# are only allowed during market hours" outside 09:30-16:00 ET. The
# backtest's contract is "what would the system have decided?"; not
# "did the broker accept the order?" - dry-run is the right mode.
os.environ["RUN_MODE"] = "dry-run"
# The trace writer is per-cycle, so a 60-cycle backtest would emit
# 60 JSONL files. Keep it off during the multi-week runs and re-enable
# for individual cycle diagnostics.
os.environ.setdefault("AIZEN_TRACE", "0")

from src.agents.backtest import (  # noqa: E402
    BacktestReport,
    BacktestRunner,
    generate_cycle_timestamps,
)
from src.agents.llm import get_provider  # noqa: E402
from src.db import connect, init_db  # noqa: E402

logger = logging.getLogger(__name__)


def _build_runner(
    *,
    db_path: str | None,
    news_enabled: bool,
    config_overrides: dict[str, Any] | None = None,
) -> BacktestRunner:
    """Construct a BacktestRunner with the mock LLM provider pinned.

    The LLM is shared across cycles so the test-fixture pattern (a single
    provider that records every call) is preserved. With the mock
    provider, ``complete`` returns deterministic text and no network
    call is attempted.
    """
    llm = get_provider("mock")
    # The orchestrator's feature flag is read from env / yaml at
    # construction time. Setting the env var here before the first
    # Orchestrator instantiation in the runner ensures every cycle
    # sees the same flag value.
    os.environ["AIZEN_RESEARCH_ENABLED"] = "1" if news_enabled else "0"
    return BacktestRunner(
        db_path=db_path,
        llm=llm,
        news_enabled=news_enabled,
        config_overrides=config_overrides,
    )


def _print_summary(report: BacktestReport, out_path: Path) -> None:
    """Print a 6-line summary to stdout."""
    a = report.aggregate
    print(f"[backtest] n_cycles       : {a.get('n_cycles', 0)}")
    print(f"[backtest] n_proceed      : {a.get('n_proceed', 0)}")
    print(f"[backtest] hit_rate_h4    : {_fmt_pct(a.get('hit_rate_h4'))}")
    print(f"[backtest] mean_pnl       : {_fmt_float(a.get('mean_pnl'))}")
    print(f"[backtest] sharpe_approx  : {_fmt_float(a.get('sharpe_approx'))}")
    print(f"[backtest] out            : {out_path}")
    if report.skipped_no_data:
        print(
            f"[backtest] skipped_no_data: {len(report.skipped_no_data)} cycles "
            f"(e.g. {report.skipped_no_data[:3]})"
        )
    if report.errors:
        print(f"[backtest] errors         : {len(report.errors)} cycles failed")


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{float(v) * 100:.1f}%"


def _fmt_float(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.4f}"


def run(args: argparse.Namespace) -> int:
    if args.start > args.end:
        print(
            f"[backtest] error: --start ({args.start}) is after --end ({args.end})",
            file=sys.stderr,
        )
        return 2

    cycles = generate_cycle_timestamps(
        args.start, args.end,
        interval=args.interval,
        market_open_utc=args.market_open,
    )
    if not cycles:
        print(
            f"[backtest] warning: no cycles in [{args.start}, {args.end}] "
            f"with interval={args.interval}",
            file=sys.stderr,
        )

    # Ensure the schema is present (the backtest_cycles table may not
    # exist on a fresh DB). init_db is idempotent.
    bootstrap_conn = connect(args.db_path) if args.db_path else connect()
    try:
        init_db(bootstrap_conn)
    finally:
        bootstrap_conn.close()

    config_overrides: dict[str, Any] = {}
    if args.universe:
        config_overrides["universe"] = [
            s.strip().upper() for s in args.universe.split(",") if s.strip()
        ]

    runner = _build_runner(
        db_path=args.db_path,
        news_enabled=args.news_enabled,
        config_overrides=config_overrides,
    )
    print(
        f"[backtest] cycles={len(cycles)} interval={args.interval} "
        f"universe={args.universe} news={'on' if args.news_enabled else 'off'}"
    )
    report = runner.run(cycles)
    out_path = Path(args.out)
    runner.write_report(report, out_path)
    _print_summary(report, out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Point-in-time backtest of the multi-agent pipeline. "
            "Replays each cycle timestamp with the inference layer "
            "clamped to timestamp <= T."
        ),
    )
    p.add_argument("--start", required=True,
                   help="Start date YYYY-MM-DD (UTC).")
    p.add_argument("--end", required=True,
                   help="End date YYYY-MM-DD (UTC), inclusive.")
    p.add_argument("--interval", choices=("daily", "weekly"), default="daily",
                   help="Cycle interval: 'daily' (every weekday) or 'weekly' (Mondays only).")
    p.add_argument("--market-open", default="13:30:00Z",
                   help="Cycle time-of-day in UTC (default 13:30 = 09:30 ET pre-DST).")
    p.add_argument("--universe", default="",
                   help="Comma-separated tickers (overrides config/agents.yaml).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--news-enabled", dest="news_enabled",
                     action="store_true", default=True,
                     help="Run with the research (news) node enabled (default).")
    grp.add_argument("--news-disabled", dest="news_enabled",
                     action="store_false",
                     help="Run with the research node disabled (legacy path).")
    p.add_argument("--out", default="models/backtest_report.json",
                   help="Output JSON path (default models/backtest_report.json).")
    p.add_argument("--db-path", default=None,
                   help="SQLite DB path (default: project's default DB).")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
