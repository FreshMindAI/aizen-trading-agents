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
# Important: do NOT ``setdefault("AIZEN_LLM_PROVIDER", "mock")`` here.
# cron-loop.yml (GitHub Actions) and render.yaml (Render) both set
# ``AIZEN_LLM_PROVIDER=gmi_fallback`` at the workflow level; if we
# setdefault() in-process, that env var never reaches the LLM
# provider resolver and every cycle runs against the mock stub —
# which is exactly the silent-fallback we hit on 2026-09-02/03.
# We DO set RUN_MODE/AIZEN_DB_PATH/AIZEN_TRACE because those are
# infra defaults that are safe to leave alone if the env didn't
# override them.
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
    from src.agents.data_refresh import (
        refresh_one_bar,
        refresh_option_chains,
        populate_option_contracts,
    )
    written: dict[str, int] = {}
    # TIMEFRAME / LOOKBACK_MINUTES / FEED let operators switch the
    # data feed from the cron env without touching code. Default to
    # 15Min + IEX (works on the free paper account; matches the
    # 15-min cron cadence so each tick sees 16 fresh 15-min bars
    # = 4 hours of intraday history). Set FEED=sip if the account
    # is upgraded.
    timeframe = os.getenv("TIMEFRAME", "15Min")
    lookback = int(os.getenv("LOOKBACK_MINUTES", "240"))
    feed = os.getenv("ALPACA_FEED", "iex")
    try:
        written = refresh_one_bar(
            orch.settings.universe,
            db_path=_resolve_db_path(),
            timeframe=timeframe,
            lookback_minutes=lookback,
            feed=feed,
        )
    except AlpacaTradingError as exc:
        logger.warning("Alpaca refresh failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Data refresh failed: %s: %s", type(exc).__name__, exc)
    # Populate option_contracts (the static list of tradeable option
    # symbols for each underlying). MUST run BEFORE refresh_option_chains,
    # which only READS from this table to pick the ATM subset. The cloud
    # runner wipes the DB on every invocation, so this step is the
    # difference between the option path producing candidates and
    # silently falling through to the heuristic. Same try/except +
    # silent-on-error pattern as the other refresh calls.
    try:
        n_contracts = populate_option_contracts(
            orch.settings.universe,
            db_path=_resolve_db_path(),
            min_dte=1,
            max_dte=30,
            band_pct=0.10,
        )
        if n_contracts:
            logger.info(
                "option contracts populated: %d symbols, %d contracts",
                len(n_contracts), sum(n_contracts.values()),
            )
        else:
            # populate_option_contracts returns {} on broker errors. The
            # function is silent on failure by design (a broker outage
            # must not fail the cycle), but the operator needs a clear
            # signal that the option path is going to fall through to
            # the no_dte_data heuristic. This WARNING is the at-a-glance
            # "options won't trade this cycle" marker.
            logger.warning(
                "option contracts NOT populated (broker error or no chain); "
                "options_structure_agent will return candidates_returned=0"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Option contract populate failed: %s: %s",
                       type(exc).__name__, exc)
    # Option chain refresh: pulls the last 4h of option bars for the
    # ATM subset of ``option_contracts`` and upserts into ``option_bars``.
    # Without this, the option_h4 XGBoost model has no per-contract bar
    # history to score and the option path falls through to the heuristic
    # (which then fails the 0.30 candidate_min_score gate). The function
    # is no-op on broker errors so a broker outage still leaves the
    # underlying refresh intact.
    try:
        opt_written = refresh_option_chains(
            orch.settings.universe,
            db_path=_resolve_db_path(),
            lookback_minutes=lookback,
            max_contracts_per_symbol=int(os.getenv("OPTION_CHAIN_CAP", "6")),
            timeframe=timeframe,
            feed="indicative",  # free tier: 15-min delayed bars are sufficient
        )
        if opt_written:
            logger.info("option bars refreshed: %d contracts, %d rows",
                        len(opt_written), sum(opt_written.values()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Option chain refresh failed: %s: %s",
                       type(exc).__name__, exc)
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
    # Surface the blocked_symbols set the supervisor used to filter
    # candidates. Operators want this visible in the cron stdout so
    # they can see at a glance "we filtered AVGO because it's already
    # in loss" rather than having to dig into the cycle trace.
    _pm = (state.supplementary or {}).get("position_management") or {}
    _blocked = _pm.get("blocked_symbols") or []
    if _blocked:
        print(f"  blocked       : {sorted(_blocked)}")
    _early = _pm.get("early_warnings") or []
    if _early:
        syms = sorted({w.get("symbol", "?") for w in _early})
        worst = min((w.get("loss_pct", 0.0) for w in _early), default=0.0)
        print(f"  early_warn    : {syms} (worst loss_pct={worst*100:.1f}%)")
    _stops = _pm.get("stop_loss_closes") or []
    if _stops:
        syms = sorted({s.get("symbol", "?") for s in _stops})
        print(f"  stopped       : {syms} (auto-closed by stop-loss)")
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

    # Market-hours gate. The cron job ticks 24/7 but Alpaca's data
    # feed + GMI are wasted outside US equity hours (and 403s the
    # bar pull on some paper accounts). Skip the cycle cleanly with
    # exit 0 so Render / GitHub Actions record a successful no-op.
    from src.agents.market_hours import evaluate as _market_hours_evaluate
    gate = _market_hours_evaluate()
    if not gate.should_run:
        print(f"== run_loop summary ==")
        print(f"  started_at    : {started}")
        print(f"  duration_s    : {round(time.time() - started_epoch, 2)}")
        print(f"  final_action  : SKIP (market closed)")
        print(f"  reason        : {gate.reason}")
        print(f"  now_et        : {gate.now_et.isoformat()}")
        return 0

    from src.agents.graph import Orchestrator
    from src.db import connect, init_db

    db_path = _resolve_db_path()
    conn = connect(db_path)
    init_db(conn)  # idempotent
    # Startup data-state log. Emits one INFO line so the operator
    # can see, at a glance, what the cron tick had to work with:
    # which LLM provider is active, how many underlying / option /
    # news rows the DB holds, and when the news snapshot was last
    # refreshed (the news-pre-tick workflow is separate from this
    # one, so a stale news timestamp often explains a low GNN bias
    # in the cycle).
    try:
        _log_startup_data_state(conn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup data-state log failed: %s: %s",
                       type(exc).__name__, exc)
    try:
        orch = Orchestrator(conn=conn)
        n_refreshed = _refresh_data(orch)
        # Post-refresh data-state log. The startup line above was the
        # BEFORE state; this line shows what the refresh actually
        # delivered. If option_contracts is still 0 here, the WARNING
        # above is the only signal the operator has that no options
        # will trade this cycle.
        try:
            _log_startup_data_state(conn, label="data_state_after_refresh")
        except Exception as exc:  # noqa: BLE001
            logger.warning("post-refresh data-state log failed: %s: %s",
                           type(exc).__name__, exc)
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


def _log_startup_data_state(conn, label: str = "data_state") -> None:
    """Emit one INFO line summarizing the live data state at cycle start.

    Why this exists: the operator's most common debugging question after
    a NO_TRADE cycle is "what did the agent see?" Previously, the only
    way to answer was to dig into the per-cycle JSONL trace under
    models/cycle_traces/. This log line gives the high-level answer in
    the cron stdout (visible in the GH Actions log) so the operator can
    tell at a glance whether the cycle had:

      * a live LLM provider (``mock`` = stub signals, ``gmi_fallback`` /
        ``anthropic`` / ``openai`` = real reasoning)
      * enough option_contracts to build a candidate set
      * enough option_bars to score the option_h4 model
      * a fresh news snapshot (the news-pre-tick workflow runs every
        15 min; a stale timestamp here means news is missing or the
        workflow is broken)

    ``label`` is the leading key on the log line (default ``data_state``
    for the BEFORE line, ``data_state_after_refresh`` for the AFTER
    line emitted after _refresh_data() finishes). The two-line form
    gives the operator a clear before/after diff of what the cron
    tick actually accomplished.

    Format (single line, fixed-width labels for grep-ability)::

        data_state llm=mock underlying_bars=254939 option_contracts=15290 option_bars=195466 news_snapshot=907 news_last=2026-09-02T18:50:00Z news_age_min=12 option_dte_max=10
    """
    def _count(table: str) -> int:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001
            return -1

    underlying_bars = _count("underlying_bars")
    option_contracts = _count("option_contracts")
    option_bars = _count("option_bars")
    news_snapshot = _count("news_snapshot")
    # Most-recent news timestamp + age in minutes. Used to flag stale
    # news before the agent runs.
    news_last_iso: str | None = None
    news_age_min: int | None = None
    try:
        row = conn.execute(
            "SELECT timestamp FROM news_snapshot ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            news_last_iso = str(row[0])
            from datetime import datetime, timezone
            try:
                last_dt = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
                age_s = (datetime.now(timezone.utc) - last_dt).total_seconds()
                news_age_min = int(age_s // 60)
            except Exception:  # noqa: BLE001
                news_age_min = None
    except Exception:  # noqa: BLE001
        pass
    # Max days_to_expiry currently in option_contracts. A 0 here means
    # all contracts are already expired (which is why
    # options_structure_agent returned candidates_returned=0 on
    # 2026-09-03: the latest expiry was 2026-09-02, yesterday).
    option_dte_max: int | None = None
    try:
        row = conn.execute(
            "SELECT MAX(julianday(expiration_date) - julianday('now')) "
            "FROM option_contracts WHERE expiration_date IS NOT NULL"
        ).fetchone()
        if row and row[0] is not None:
            option_dte_max = int(float(row[0]))
    except Exception:  # noqa: BLE001
        pass
    llm_provider = os.environ.get("AIZEN_LLM_PROVIDER", "(unset)")
    news_age_str = (
        f"{news_age_min}m" if news_age_min is not None else "?"
    )
    dte_str = str(option_dte_max) if option_dte_max is not None else "?"
    logger.info(
        "%s llm=%s underlying_bars=%d option_contracts=%d "
        "option_bars=%d news_snapshot=%d news_last=%s news_age_min=%s "
        "option_dte_max=%s",
        label, llm_provider, underlying_bars, option_contracts,
        option_bars, news_snapshot, news_last_iso or "(none)",
        news_age_str, dte_str,
    )
    # High-visibility WARNING when the option path is guaranteed to
    # fall through to no_dte_data. The cron-loop.yml operator can then
    # tell at a glance that no option trade is possible this cycle,
    # even without grepping for the data_state line. Two triggers:
    #   (a) option_contracts == 0 -- the populate step never wrote rows
    #       (broker error / no chain for today).
    #   (b) option_dte_max <= 0 -- rows exist but every contract is
    #       already expired (data was populated for a past session).
    # Both surface the same NO_TRADE outcome so the operator gets a
    # single clear signal.
    if option_contracts == 0 or (option_dte_max is not None and option_dte_max <= 0):
        reason = (
            f"no option_contracts in DB" if option_contracts == 0
            else f"all option_contracts expired (max DTE={option_dte_max})"
        )
        logger.warning(
            "no tradable options: %s. options_structure_agent will return "
            "candidates_returned=0 and dte_fallback=no_dte_data. This usually "
            "means populate_option_contracts() couldn't reach Alpaca OR the "
            "chain is empty for today (weekend / holiday / after-hours).",
            reason,
        )
    # Stale news is the other common cause of a low GNN bias and
    # NO_TRADE. Flag when the news-pre-tick workflow hasn't refreshed
    # in over 30 min (two cron cycles).
    if news_age_min is not None and news_age_min > 30:
        logger.warning(
            "stale news: last news_snapshot was %d min ago "
            "(news-pre-tick workflow may be broken or throttled). "
            "GNN bias is being computed against outdated sentiment.",
            news_age_min,
        )


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
