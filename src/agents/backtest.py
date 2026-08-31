"""Point-in-time backtest runner (spec 003 / T046 / T048 / T049).

Takes a list of cycle timestamps and runs the full multi-agent pipeline
at each one, with the inference layer clamped to ``timestamp <= T`` so
no row written after the cycle is visible. Compares the predicted
action to forward outcomes measured by the :class:`Labeler`.

The runner writes per-cycle rows to ``backtest_cycles`` (a dedicated
table so the live ``decision_journal`` is not polluted) and returns a
``BacktestReport`` with aggregate metrics. A typical 3-week daily
replay produces ~15 cycles and writes one row per cycle.

Why a separate runner (not just a CLI flag on the orchestrator)?
  - The runner owns the cycle-timestamp generation, the per-cycle
    fresh ``Orchestrator`` instantiation, the labeler dispatch, and
    the aggregate metrics. Splitting it out keeps the orchestrator
    single-purpose and lets the runner be tested in isolation.

Usage:
    runner = BacktestRunner(db_path="data/trading.db",
                            llm=get_provider("mock"))
    report = runner.run([
        "2026-08-04T13:30:00Z",
        "2026-08-11T13:30:00Z",
        "2026-08-18T13:30:00Z",
    ])
    runner.write_report(report, "models/backtest_report.json")
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..db import connect, utc_now_iso
from .backtest_labeler import Labeler
from .graph import Orchestrator
from .journal import DecisionJournal
from .protocol import DecisionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------
@dataclass
class BacktestCycleResult:
    """Per-cycle result of a backtest."""
    cycle_as_of: str
    decision_id: str
    final_action: str
    predicted_underlying: str | None
    predicted_strategy_id: str | None
    predicted_legs_json: str | None
    forward_return_h1: float | None
    forward_return_h4: float | None
    option_payoff: float | None
    target_class: str | None
    hit_h4: int | None
    hit_h1: int | None
    coverage_h1: int
    coverage_h4: int
    coverage_payoff: int
    run_at: str
    model_version: str | None
    feature_flag_state: str
    # Equity-path fields (parallel options+stocks). All NULL for
    # option-only cycles; populated when the order intent carries an
    # asset_class='equity' leg. The labeler computes entry/exit prices
    # from underlying_bars and reports the simple long-stock return.
    equity_payoff_h1: float | None = None
    equity_payoff_h4: float | None = None
    equity_hit_h1: int | None = None
    equity_hit_h4: int | None = None
    coverage_equity_h1: int = 0
    coverage_equity_h4: int = 0
    notes: str | None = None


@dataclass
class BacktestReport:
    """Aggregate backtest report."""
    cycles: list[BacktestCycleResult]
    aggregate: dict[str, Any]
    config: dict[str, Any]
    created_at: str
    skipped_no_data: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "config": self.config,
            "aggregate": self.aggregate,
            "skipped_no_data": self.skipped_no_data,
            "errors": self.errors,
            "cycles": [asdict(c) for c in self.cycles],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class BacktestRunner:
    """Run a sequence of past cycles and persist results to ``backtest_cycles``.

    Parameters
    ----------
    db_path : str | Path | None
        SQLite database path. Defaults to the project's default DB.
    llm : object
        An LLM provider (any object with a ``.complete`` method).
        ``get_provider("mock")`` is the canonical choice for tests.
    news_enabled : bool
        Whether the research feature flag is on for this run. Routes
        through the existing ``AIZEN_RESEARCH_ENABLED`` resolution
        path inside ``Orchestrator.__init__``; passed here only as
        documentation of intent (the orchestrator reads the env / yaml).
    config_overrides : dict | None
        Optional override of the ``config/agents.yaml`` payload.
    """

    def __init__(self, *, db_path: str | Path | None = None,
                 llm: Any = None,
                 news_enabled: bool = True,
                 config_overrides: dict[str, Any] | None = None) -> None:
        self.db_path = db_path
        self.llm = llm
        self.news_enabled = news_enabled
        self.config_overrides = config_overrides or {}

    # ---- public ---------------------------------------------------------
    def run(self, cycle_timestamps: list[str]) -> BacktestReport:
        """Run a backtest over the given list of cycle timestamps.

        Each timestamp is replayed against the database as if it were
        "now" — the inference layer's cutoff enforces no-future-data.
        Per-cycle results are written to ``backtest_cycles`` and an
        aggregate report is returned.
        """
        conn = connect(self.db_path) if self.db_path else connect()
        try:
            return self._run_impl(conn, cycle_timestamps)
        finally:
            conn.close()

    def write_report(self, report: BacktestReport, path: str | Path) -> Path:
        """Write the report as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return p

    # ---- internal -------------------------------------------------------
    def _run_impl(self, conn: sqlite3.Connection,
                  cycle_timestamps: list[str]) -> BacktestReport:
        # Drop a fresh Orchestrator for each cycle so the as_of cut-off
        # is correct. We still need a per-cycle DB cursor / row_factory
        # because the orchestrator's conn is shared.
        journal = DecisionJournal(conn, table="decision_journal", run_mode="paper")
        labeler = Labeler(conn)

        cycles: list[BacktestCycleResult] = []
        skipped: list[str] = []
        errors: dict[str, str] = {}
        run_at = utc_now_iso()

        for ts in cycle_timestamps:
            try:
                if not _has_underlying_data(conn, ts):
                    skipped.append(ts)
                    continue
                # We pass a NEW Orchestrator per cycle so each call gets
                # a fresh InferenceService with the correct as_of. The
                # LLM is shared (mock) to keep tests deterministic.
                orch = Orchestrator(
                    conn=conn,
                    config=self.config_overrides or None,
                    as_of=ts,
                )
                state = orch.run_cycle()
                result = self._build_result(
                    state, ts, labeler, run_at=run_at,
                )
                _persist_result(conn, result)
                cycles.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("backtest cycle %s failed: %s", ts, exc)
                errors[ts] = f"{type(exc).__name__}: {exc}"[:500]

        aggregate = _aggregate(cycles)
        return BacktestReport(
            cycles=cycles,
            aggregate=aggregate,
            config={
                "news_enabled": self.news_enabled,
                "n_requested": len(cycle_timestamps),
            },
            created_at=utc_now_iso(),
            skipped_no_data=skipped,
            errors=errors,
        )

    def _build_result(self, state: DecisionState, cycle_as_of: str,
                      labeler: Labeler, *, run_at: str) -> BacktestCycleResult:
        labels = labeler.label(state, cycle_as_of)
        predicted_underlying = (
            state.order_intent.underlying
            if state.order_intent is not None
            else (
                state.selected_strategy.underlying
                if state.selected_strategy is not None
                else None
            )
        )
        predicted_strategy_id = (
            state.selected_strategy.strategy_id
            if state.selected_strategy is not None
            else None
        )
        legs_json = (
            json.dumps([leg.model_dump(mode="json") for leg in state.order_intent.legs])
            if state.order_intent is not None
            else None
        )
        # The model version is best-effort from the gnn_output dict;
        # inference.topology_version is also carried on the state.
        model_version = (
            state.gnn_output.get("model_version")
            if isinstance(state.gnn_output, dict)
            else None
        ) or state.topology_version
        feature_flag_state = (
            "news-on" if (state.market_snapshot and state.market_snapshot.research)
            else "news-off"
        )
        decision_id = (
            f"backtest-{cycle_as_of.replace(':', '').replace('-', '')}-"
            f"{predicted_underlying or 'unknown'}"
        )

        return BacktestCycleResult(
            cycle_as_of=cycle_as_of,
            decision_id=decision_id,
            final_action=state.final_action or "NO_TRADE",
            predicted_underlying=predicted_underlying,
            predicted_strategy_id=predicted_strategy_id,
            predicted_legs_json=legs_json,
            forward_return_h1=labels.forward_return_h1,
            forward_return_h4=labels.forward_return_h4,
            option_payoff=labels.option_payoff,
            target_class=labels.target_class,
            hit_h4=_hit_int(labels.forward_return_h4),
            hit_h1=_hit_int(labels.forward_return_h1),
            coverage_h1=1 if labels.coverage_h1 else 0,
            coverage_h4=1 if labels.coverage_h4 else 0,
            coverage_payoff=1 if labels.coverage_payoff else 0,
            equity_payoff_h1=labels.equity_payoff_h1,
            equity_payoff_h4=labels.equity_payoff_h4,
            equity_hit_h1=_hit_int(labels.equity_payoff_h1),
            equity_hit_h4=_hit_int(labels.equity_payoff_h4),
            coverage_equity_h1=1 if labels.coverage_equity_h1 else 0,
            coverage_equity_h4=1 if labels.coverage_equity_h4 else 0,
            run_at=run_at,
            model_version=model_version,
            feature_flag_state=feature_flag_state,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_underlying_data(conn: sqlite3.Connection, ts: str) -> bool:
    """Return True if at least one underlying_bars row exists at-or-before
    ``ts`` (so a cycle replay can produce non-empty data)."""
    row = conn.execute(
        "SELECT 1 FROM underlying_bars WHERE timestamp <= ? LIMIT 1",
        (ts,),
    ).fetchone()
    return row is not None


def _persist_result(conn: sqlite3.Connection, result: BacktestCycleResult) -> None:
    """Insert-or-replace one row in ``backtest_cycles``.

    The full column list (including the equity-path fields added by
    ``sql/61_backtest_cycles_equity.sql``) is preferred. If the live DB
    hasn't run the migration yet (e.g. a stale Render disk), fall back
    to a legacy insert that omits the equity columns so the cycle row
    is still written — the option-side metrics remain usable.
    """
    row = asdict(result)
    cols = ",".join(row.keys())
    placeholders = ",".join("?" for _ in row)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO backtest_cycles ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
    except sqlite3.OperationalError as exc:
        if "no column named" not in str(exc):
            raise
        # Legacy DB: drop the equity-only fields and retry once.
        legacy_keys = [k for k in row
                       if not k.startswith(("equity_", "coverage_equity_"))]
        legacy_row = [row[k] for k in legacy_keys]
        legacy_cols = ",".join(legacy_keys)
        legacy_ph = ",".join("?" for _ in legacy_keys)
        logger.warning(
            "backtest_cycles missing equity columns; writing legacy row "
            "(run sql/61_backtest_cycles_equity.sql to upgrade)"
        )
        conn.execute(
            f"INSERT OR REPLACE INTO backtest_cycles ({legacy_cols}) "
            f"VALUES ({legacy_ph})",
            legacy_row,
        )
    conn.commit()


def _hit_int(forward_return: float | None) -> int | None:
    """Map a forward return to a hit value (1=positive, 0=negative, None=zero/unknown)."""
    if forward_return is None:
        return None
    if forward_return > 0:
        return 1
    if forward_return < 0:
        return 0
    return None


def _aggregate(cycles: list[BacktestCycleResult]) -> dict[str, Any]:
    """Compute the headline aggregate metrics from a list of cycle results."""
    n = len(cycles)
    n_proceed = sum(1 for c in cycles if c.final_action == "PROCEED")
    n_no_trade = sum(1 for c in cycles if c.final_action == "NO_TRADE")
    other = n - n_proceed - n_no_trade

    # Hit rate over PROCEED cycles that had a computable h4 label.
    proceed_with_h4 = [c for c in cycles
                       if c.final_action == "PROCEED" and c.coverage_h4]
    hits = [c.hit_h4 for c in proceed_with_h4 if c.hit_h4 is not None]
    hit_rate_h4 = (sum(hits) / len(hits)) if hits else None

    # PnL over cycles that had a computable option payoff.
    pnls = [c.option_payoff for c in cycles if c.option_payoff is not None]
    mean_pnl = (sum(pnls) / len(pnls)) if pnls else None
    sharpe_approx = _compute_sharpe(pnls) if pnls else None

    # Sanity: mean forward return at h4 should be near zero (efficient market).
    h4_returns = [c.forward_return_h4 for c in cycles
                  if c.forward_return_h4 is not None]
    mean_h4 = (sum(h4_returns) / len(h4_returns)) if h4_returns else None

    # ---- equity path aggregates (parallel options+stocks) ---------------
    # Only the cycles whose order intent carried an equity leg contribute.
    n_proceed_equity = sum(1 for c in cycles
                           if c.final_action == "PROCEED" and c.coverage_equity_h4)
    equity_hits = [c.equity_hit_h4 for c in cycles
                   if c.equity_hit_h4 is not None]
    hit_rate_equity_h4 = (
        sum(equity_hits) / len(equity_hits) if equity_hits else None
    )
    equity_pnls = [c.equity_payoff_h4 for c in cycles
                   if c.equity_payoff_h4 is not None]
    mean_equity_pnl_h4 = (
        sum(equity_pnls) / len(equity_pnls) if equity_pnls else None
    )
    equity_coverage = (
        sum(1 for c in cycles if c.coverage_equity_h4) / n if n else None
    )

    return {
        "n_cycles": n,
        "n_proceed": n_proceed,
        "n_no_trade": n_no_trade,
        "n_other": other,
        "n_proceed_with_h4": len(proceed_with_h4),
        "hit_rate_h4": hit_rate_h4,
        "mean_pnl": mean_pnl,
        "sharpe_approx": sharpe_approx,
        "mean_forward_return_h4": mean_h4,
        "coverage_payoff": sum(1 for c in cycles if c.coverage_payoff),
        # ---- equity path -------------------------------------------
        "n_proceed_equity": n_proceed_equity,
        "hit_rate_equity_h4": hit_rate_equity_h4,
        "mean_equity_pnl_h4": mean_equity_pnl_h4,
        "equity_coverage": equity_coverage,
    }


def _compute_sharpe(pnl: list[float]) -> float:
    """Approximate Sharpe on a PnL stream. Re-uses the same formula as
    :func:`src.gnn.walk_forward_ablation._compute_sharpe` so the two
    report paths agree on degenerate handling (returns 0.0 on
    insufficient or constant input)."""
    if not pnl or len(pnl) <= 1:
        return 0.0
    mean = sum(pnl) / len(pnl)
    var = sum((x - mean) ** 2 for x in pnl) / max(1, len(pnl) - 1)
    std = float(var) ** 0.5
    if std < 1e-12:
        return 0.0
    import math
    return mean / std * math.sqrt(len(pnl))


# ---------------------------------------------------------------------------
# Cycle-timestamp generator (used by the CLI)
# ---------------------------------------------------------------------------
def generate_cycle_timestamps(
    start_date: str,
    end_date: str,
    *,
    interval: str = "daily",
    market_open_utc: str = "13:30:00Z",
) -> list[str]:
    """Generate cycle timestamps in ``[start_date, end_date]``.

    ``interval`` is ``"daily"`` (every weekday) or ``"weekly"`` (Mondays
    only). ``market_open_utc`` is the cycle time-of-day appended to
    each calendar date.

    Raises ``ValueError`` for an unknown interval.
    """
    if interval not in ("daily", "weekly"):
        raise ValueError(f"interval must be 'daily' or 'weekly'; got {interval!r}")
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    out: list[str] = []
    d = start.date() if hasattr(start, "date") else start
    end_d = end.date() if hasattr(end, "date") else end
    cur = d
    while cur <= end_d:
        wd = cur.weekday()  # Mon=0..Sun=6
        if interval == "daily" and wd < 5:
            out.append(f"{cur.isoformat()}T{market_open_utc}")
        elif interval == "weekly" and wd == 0:
            out.append(f"{cur.isoformat()}T{market_open_utc}")
        cur = cur + timedelta(days=1)
    return out
