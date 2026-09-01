"""Hourly self-analysis: deterministic reflection over the last hour of cycles.

What this module does
---------------------
The 15-min trade tick writes a per-cycle JSONL to
``models/cycle_traces/cycle_{decision_id}.jsonl`` and a one-line summary
to ``models/cycle_traces.jsonl``. The hourly cron reads everything that
landed in the last hour and produces a single human-readable report
that an operator can browse without touching the Actions tab:

  * Decision-journal stats: PROCEED / NO_TRADE / REJECT / REDUCE counts,
    top-3 selected underlyings, top-3 risk-reject reasons.
  * Cycle-trace per-step success rate (ml, gnn, topology, research, each
    agent). Top-5 failure reasons histogram.
  * NO_TRADE cluster analysis: for cycles that ended NO_TRADE, what was
    the *first* failing step? This is the most actionable signal
    ("always NO_TRADE because ml never crossed threshold" vs "always
    NO_TRADE because risk rejected the order").
  * Topology summary: latest news-driven GNN snapshot stats.
  * Failure KG summary: counts by kind in the last hour.

The reflection is **deterministic** in v1 — no LLM call. This is
intentional: the report is a debug artifact, and a non-deterministic
report would be hard to diff between runs.

Public surface
--------------
::

    from src.agents.cli.hourly_reflection import (
        collect_cycle_traces, collect_topology_summary,
        collect_decision_summary, collect_failure_summary,
        build_markdown_report, build_json_summary, main,
    )

CLI::

    python -m src.agents.cli.hourly_reflection --out models/reflections/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Pin env BEFORE importing anything that reads it.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")
os.environ.setdefault("AIZEN_TRACE", "0")  # the reflection does not write traces

from src.db import connect, init_db, utc_now_iso  # noqa: E402

logger = logging.getLogger("hourly_reflection")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Default location of the per-cycle JSONL files. Matches the cron-loop
# workflow output (CycleTraceBuilder writes to models/cycle_traces/).
DEFAULT_TRACE_DIR = Path("models") / "cycle_traces"

# Steps that count as "first failing step" candidates (in the order the
# orchestrator runs them). The reflection uses this order to bucket
# NO_TRADE cycles by which gate was the first to fail.
PIPELINE_STEPS: tuple[str, ...] = (
    "ml",
    "gnn",
    "topology",
    "research",
    "regime",
    "direction",
    "volatility",
    "options_structure",
    "portfolio",
    "supervisor",
    "risk",
    "execution",
)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------
def _hour_key(now_utc: datetime | None = None) -> str:
    """Return the canonical hour key ``YYYY-MM-DD-HH`` for a UTC instant."""
    n = now_utc or datetime.now(timezone.utc)
    return n.strftime("%Y-%m-%d-%H")


def collect_cycle_traces(
    since: str,
    trace_dir: Path | str = DEFAULT_TRACE_DIR,
) -> list[dict[str, Any]]:
    """Read every cycle_*.jsonl with mtime >= ``since`` (ISO 8601 UTC).

    Returns a flat list of step records (one dict per line per cycle).
    Lines that fail to parse are skipped with a warning — the file is
    best-effort input; one bad line must not kill the reflection.
    """
    out: list[dict[str, Any]] = []
    base = Path(trace_dir)
    if not base.exists():
        return out
    since_epoch = _parse_iso_to_epoch(since)
    for jsonl in sorted(base.glob("cycle_*.jsonl")):
        try:
            mtime_epoch = jsonl.stat().st_mtime
        except OSError:
            continue
        if mtime_epoch < since_epoch:
            continue
        try:
            with jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("bad jsonl in %s: %s", jsonl, exc)
        except OSError as exc:
            logger.warning("read failed for %s: %s", jsonl, exc)
    return out


def collect_topology_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Read the most recent news-driven GNN topology snapshot.

    Returns an empty dict when the table is missing or no row exists —
    the reflection handles the "no snapshot yet" case gracefully.
    """
    try:
        row = conn.execute(
            """
            SELECT snapshot_id, timestamp, topology_version,
                   node_count, edge_count, created_at
            FROM   gnn_option_graph_snapshots
            WHERE  topology_version LIKE '%news%'
               OR  topology_version = 'option-v2'
            ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        return {}
    if row is None:
        return {}
    snap = dict(row)
    # Edge-kind histogram from the most recent snapshot.
    edge_kinds: dict[str, int] = {}
    try:
        for er in conn.execute(
            "SELECT reason, COUNT(*) AS n FROM gnn_option_graph_edges "
            "WHERE snapshot_id = ? GROUP BY reason",
            (snap["snapshot_id"],),
        ).fetchall():
            edge_kinds[str(er["reason"])] = int(er["n"])
    except sqlite3.OperationalError:
        pass
    snap["edge_kinds"] = edge_kinds
    return snap


def collect_decision_summary(
    conn: sqlite3.Connection, since: str,
) -> dict[str, Any]:
    """Aggregate decision_journal rows whose ``timestamp >= since``."""
    out: dict[str, Any] = {
        "total": 0,
        "by_action": {"PROCEED": 0, "NO_TRADE": 0, "REJECT": 0, "REDUCE": 0},
        "top_underlyings": [],
        "top_risk_reject_reasons": [],
        "since": since,
    }
    try:
        rows = conn.execute(
            "SELECT final_action, underlying_focus, risk_decision_json, "
            "       order_intent_json, completed_at "
            "FROM   decision_journal WHERE timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT 500",
            (since,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("decision_journal read failed: %s", exc)
        return out
    out["total"] = len(rows)
    underlyings: Counter[str] = Counter()
    risk_reasons: Counter[str] = Counter()
    for r in rows:
        action = r["final_action"] or "NO_TRADE"
        out["by_action"][action] = out["by_action"].get(action, 0) + 1
        if r["underlying_focus"]:
            underlyings[r["underlying_focus"]] += 1
        # risk_decision_json is the JSON-encoded RiskDecision envelope
        # ({"decision", "checks", "reasons", ...}); pull the checks list
        # out of it. Older rows may store a bare list of RiskCheck dicts;
        # tolerate both shapes.
        try:
            rd = json.loads(r["risk_decision_json"] or "[]")
        except (TypeError, ValueError):
            rd = []
        if isinstance(rd, dict):
            checks = rd.get("checks") or []
        else:
            checks = rd or []
        for check in checks:
            if not isinstance(check, dict):
                continue
            if not check.get("passed", True):
                name = check.get("name") or "?"
                risk_reasons[name] += 1
    out["top_underlyings"] = underlyings.most_common(3)
    out["top_risk_reject_reasons"] = risk_reasons.most_common(3)
    return out


def collect_failure_summary(
    conn: sqlite3.Connection, since: str,
) -> dict[str, Any]:
    """Failure-KG counts by kind over the last hour."""
    out: dict[str, Any] = {
        "total": 0,
        "by_kind": {"symbol_failure": 0, "agent_failure": 0, "cycle_failure": 0},
        "by_severity": {"info": 0, "warn": 0, "error": 0, "critical": 0},
        "since": since,
    }
    try:
        rows = conn.execute(
            "SELECT kind, severity, error_count FROM failure_nodes "
            "WHERE created_at >= ?",
            (since,),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        out["total"] += int(r["error_count"] or 0)
        k = r["kind"] or "?"
        out["by_kind"][k] = out["by_kind"].get(k, 0) + int(r["error_count"] or 0)
        s = r["severity"] or "info"
        out["by_severity"][s] = out["by_severity"].get(s, 0) + int(r["error_count"] or 0)
    return out


# ---------------------------------------------------------------------------
# Per-cycle step analysis
# ---------------------------------------------------------------------------
def _per_step_success_rate(records: list[dict]) -> dict[str, dict[str, int]]:
    """Group records by step and compute {n, n_success, n_failure}."""
    out: dict[str, dict[str, int]] = {}
    for r in records:
        step = r.get("step", "?")
        slot = out.setdefault(step, {"n": 0, "n_success": 0, "n_failure": 0})
        slot["n"] += 1
        if r.get("success"):
            slot["n_success"] += 1
        else:
            slot["n_failure"] += 1
    return out


def _failure_reasons_histogram(records: list[dict]) -> list[tuple[str, int]]:
    """Top failure reasons across all steps. Reasons are free-text strings
    the orchestrator / trace builder emitted; we count exact occurrences."""
    c: Counter[str] = Counter()
    for r in records:
        if r.get("success"):
            continue
        for reason in r.get("reasons", []) or []:
            if isinstance(reason, str) and reason.strip():
                c[reason.strip()] += 1
    return c.most_common(5)


def _no_trade_cluster(records: list[dict]) -> list[tuple[str, int]]:
    """For cycles that ended NO_TRADE, group by the first-failing step.

    A cycle "ends NO_TRADE" when its 'final' step recorded success=False
    (final_action != 'PROCEED' or order_intent absent). For those cycles,
    we walk the per-step records in PIPELINE_STEPS order and return the
    first step where success=False. The most-common first-failure step
    tells the operator where the pipeline is stuck.
    """
    by_cycle: dict[str, list[dict]] = {}
    for r in records:
        cid = r.get("cycle_id", "")
        if cid:
            by_cycle.setdefault(cid, []).append(r)
    first_fail: Counter[str] = Counter()
    for cid, steps in by_cycle.items():
        # Index by step name for O(1) lookup. Multiple records for the
        # same step (e.g. one per symbol) collapse via "any failure".
        any_fail_per_step: dict[str, bool] = {}
        for s in steps:
            name = s.get("step", "?")
            if s.get("success"):
                # successful step resets the "first fail" only if we
                # haven't seen a failure for this step yet. We treat
                # any failure in this step as the marker.
                any_fail_per_step.setdefault(name, False)
            else:
                any_fail_per_step[name] = True
        # Walk PIPELINE_STEPS in order; the first step with a failure wins.
        chosen: str | None = None
        for name in PIPELINE_STEPS:
            if any_fail_per_step.get(name):
                chosen = name
                break
        if chosen is None:
            # No step failed in the standard list; bucket as 'other'.
            chosen = "other"
        first_fail[chosen] += 1
    return first_fail.most_common()


# ---------------------------------------------------------------------------
# Report builders (deterministic, no LLM)
# ---------------------------------------------------------------------------
def _step_table(per_step: dict[str, dict[str, int]]) -> list[str]:
    """Markdown table: per-step success rate."""
    lines = ["| step | n | success | failure | success_rate |",
             "|------|---|---------|---------|--------------|"]
    for step in sorted(per_step):
        s = per_step[step]
        rate = (s["n_success"] / s["n"] * 100.0) if s["n"] else 0.0
        lines.append(
            f"| `{step}` | {s['n']} | {s['n_success']} | {s['n_failure']} | {rate:.1f}% |"
        )
    return lines


def build_markdown_report(
    cycle_traces: list[dict],
    topology: dict,
    decisions: dict,
    failures: dict,
    *,
    hour: str,
    since: str,
) -> str:
    """Deterministic markdown report."""
    lines: list[str] = []
    lines.append(f"# Hourly Reflection — {hour}")
    lines.append("")
    lines.append(f"- Window start (UTC): `{since}`")
    lines.append(f"- Cycles in window: **{len({r.get('cycle_id', '') for r in cycle_traces if r.get('cycle_id')})}**")
    lines.append(f"- Step records read: **{len(cycle_traces)}**")
    lines.append("")

    # ---- Decision journal --------------------------------------------------
    lines.append("## Decision journal")
    lines.append("")
    lines.append(f"- Total cycles: **{decisions.get('total', 0)}**")
    by_action = decisions.get("by_action", {})
    if by_action:
        parts = ", ".join(f"`{k}`={v}" for k, v in sorted(by_action.items()) if v)
        lines.append(f"- By action: {parts}")
    top_u = decisions.get("top_underlyings") or []
    if top_u:
        syms = ", ".join(f"`{s}`×{n}" for s, n in top_u)
        lines.append(f"- Top underlyings: {syms}")
    top_r = decisions.get("top_risk_reject_reasons") or []
    if top_r:
        rs = ", ".join(f"`{name}`×{n}" for name, n in top_r)
        lines.append(f"- Top risk-reject reasons: {rs}")
    lines.append("")

    # ---- Cycle traces ------------------------------------------------------
    lines.append("## Cycle traces (per-step success)")
    lines.append("")
    per_step = _per_step_success_rate(cycle_traces)
    if per_step:
        lines.extend(_step_table(per_step))
    else:
        lines.append("_No cycle traces in this window._")
    lines.append("")

    # ---- Failure reasons histogram ----------------------------------------
    lines.append("## Top failure reasons")
    lines.append("")
    hist = _failure_reasons_histogram(cycle_traces)
    if hist:
        for reason, n in hist:
            # Truncate long reasons for readability.
            short = reason if len(reason) <= 100 else reason[:97] + "..."
            lines.append(f"- `{n}×` {short}")
    else:
        lines.append("_No failure reasons in this window._")
    lines.append("")

    # ---- NO_TRADE cluster --------------------------------------------------
    lines.append("## NO_TRADE cluster analysis")
    lines.append("")
    cluster = _no_trade_cluster(cycle_traces)
    if cluster:
        total = sum(n for _, n in cluster) or 1
        lines.append("| first-failing step | n | share |")
        lines.append("|--------------------|---|-------|")
        for step, n in cluster:
            lines.append(f"| `{step}` | {n} | {n / total * 100:.1f}% |")
    else:
        lines.append("_No NO_TRADE cycles in this window._")
    lines.append("")

    # ---- Topology ----------------------------------------------------------
    lines.append("## Topology (latest GNN snapshot)")
    lines.append("")
    if topology:
        lines.append(
            f"- snapshot_id: `{topology.get('snapshot_id', '?')}`"
        )
        lines.append(
            f"- timestamp: `{topology.get('timestamp', '?')}`"
        )
        lines.append(
            f"- topology_version: `{topology.get('topology_version', '?')}`"
        )
        lines.append(
            f"- nodes: **{topology.get('node_count', 0)}**  "
            f"edges: **{topology.get('edge_count', 0)}**"
        )
        ek = topology.get("edge_kinds") or {}
        if ek:
            kinds = ", ".join(f"`{k}`={v}" for k, v in sorted(ek.items()))
            lines.append(f"- edge kinds: {kinds}")
    else:
        lines.append("_No GNN snapshot found._")
    lines.append("")

    # ---- Failure KG --------------------------------------------------------
    lines.append("## Failure knowledge graph (last hour)")
    lines.append("")
    lines.append(f"- Total events: **{failures.get('total', 0)}**")
    bk = failures.get("by_kind") or {}
    if bk:
        parts = ", ".join(f"`{k}`={v}" for k, v in sorted(bk.items()) if v)
        lines.append(f"- By kind: {parts}")
    bs = failures.get("by_severity") or {}
    if bs:
        parts = ", ".join(f"`{k}`={v}" for k, v in sorted(bs.items()) if v)
        lines.append(f"- By severity: {parts}")
    lines.append("")
    return "\n".join(lines)


def build_json_summary(
    cycle_traces: list[dict],
    topology: dict,
    decisions: dict,
    failures: dict,
    *,
    hour: str,
    since: str,
) -> dict[str, Any]:
    """Machine-readable mirror of build_markdown_report."""
    cluster = _no_trade_cluster(cycle_traces)
    return {
        "schema_version": "1.0",
        "hour": hour,
        "since": since,
        "created_at": utc_now_iso(),
        "n_step_records": len(cycle_traces),
        "per_step": _per_step_success_rate(cycle_traces),
        "top_failure_reasons": _failure_reasons_histogram(cycle_traces),
        "no_trade_cluster": [{"first_failing_step": s, "n": n} for s, n in cluster],
        "decisions": decisions,
        "topology": topology,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?$"
)


def _parse_iso_to_epoch(iso: str) -> float:
    """Parse an ISO 8601 string to a POSIX epoch (seconds)."""
    from datetime import datetime
    if not _ISO_RE.match(iso.strip()):
        # Fall back to a conservative "0" so we don't skip files on a
        # malformed input — the caller is responsible for valid ISO.
        return 0.0
    s = iso.strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", default="models/reflections/",
                   help="Output directory (default models/reflections/)")
    p.add_argument("--since", default=None,
                   help="Window start (ISO 8601 UTC). Default: now - 1h.")
    p.add_argument("--db-path", default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    hour = _hour_key(now)
    since = args.since or (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = connect(args.db_path)
    try:
        init_db(conn)  # idempotent
        traces = collect_cycle_traces(since)
        topology = collect_topology_summary(conn)
        decisions = collect_decision_summary(conn, since)
        failures = collect_failure_summary(conn, since)
    finally:
        conn.close()

    md = build_markdown_report(
        traces, topology, decisions, failures,
        hour=hour, since=since,
    )
    js = build_json_summary(
        traces, topology, decisions, failures,
        hour=hour, since=since,
    )

    md_path = out_dir / f"{hour}.md"
    js_path = out_dir / f"{hour}.json"
    md_path.write_text(md, encoding="utf-8")
    js_path.write_text(json.dumps(js, indent=2, sort_keys=True, default=str),
                       encoding="utf-8")

    print("== hourly_reflection summary ==")
    print(f"  hour        : {hour}")
    print(f"  since       : {since}")
    print(f"  n_records   : {len(traces)}")
    print(f"  cycles      : {decisions.get('total', 0)}")
    print(f"  by_action   : {decisions.get('by_action', {})}")
    print(f"  topology    : "
          f"{'snapshot_id=' + str(topology.get('snapshot_id')) if topology else 'none'}")
    print(f"  failures    : {failures.get('total', 0)} events")
    print(f"  md_path     : {md_path}")
    print(f"  json_path   : {js_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
