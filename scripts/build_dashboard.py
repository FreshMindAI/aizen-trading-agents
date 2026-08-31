"""Build a self-contained static HTML dashboard from the live SQLite DB.

Output: ``dashboard/index.html`` — drop-in for GitHub Pages (push to
``gh-pages`` branch on every Render tick).

The dashboard is intentionally framework-free (no React/Vue/Tailwind
runtime). A single HTML file with inline CSS + a tiny embedded JS for
the "next-day plan" toggle. The judges can open it on any browser; no
server, no build step, no external CDN.

What it shows
-------------
1. **Header** — system name, last cycle time, broker mode (paper/live).
2. **KPI strip** — total P&L, today's P&L, week P&L, trade count,
   win rate, max drawdown. Numbers come from ``daily_pnl`` (when
   populated) and ``decision_journal.outcome_label`` (when
   reconciled). Empty values render as ``—``.
3. **Next-day plan** — the most recent PROCEED cycle's
   ``selected_strategy``: symbol, side, score, thesis, legs, score.
   This is what the operator reads before the next market open.
4. **Trade history** — last 25 cycles: timestamp, underlying, action,
   asset_class, score, fill status, realized P&L, outcome label.
5. **ML / GNN signals** — most-recent agent_observations' confidence
   per agent; the operator can see the regime / direction / volatility
   view that drove the last decision.
6. **Footer** — DB row counts, build timestamp, link to the journal.

Usage
-----
::

    python scripts/build_dashboard.py            # default DB
    python scripts/build_dashboard.py --db /var/data/aizen/trading.db
    python scripts/build_dashboard.py --out dashboard/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.db import connect as _connect  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_field(d: dict | None, key: str, default=None):
    if not d:
        return default
    v = d.get(key)
    if v is None:
        return default
    return v


def _row_factory(conn: sqlite3.Connection):
    """Set row factory to sqlite3.Row locally and return the prev value."""
    prev = conn.row_factory
    conn.row_factory = sqlite3.Row
    return prev


def _restore_factory(conn: sqlite3.Connection, prev):
    conn.row_factory = prev


# ---- data loaders ----

def load_kpis(conn: sqlite3.Connection) -> dict:
    """Compute the headline P&L numbers shown in the KPI strip."""
    out: dict = {
        "total_realized": 0.0,
        "today_realized": 0.0,
        "week_realized": 0.0,
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "max_drawdown": 0.0,
        "sharpe_approx": 0.0,
        "last_cycle": None,
        "last_action": None,
        "next_plan": None,
        "cycle_count": 0,
    }
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Last 7 days inclusive
    from datetime import timedelta
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    prev = _row_factory(conn)
    try:
        # Last cycle
        last = conn.execute(
            "SELECT decision_id, timestamp, final_action, underlying_focus, "
            "       realized_pnl, outcome_label "
            "FROM decision_journal ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if last:
            out["last_cycle"] = last["timestamp"]
            out["last_action"] = last["final_action"]
        # Total + win/loss
        agg = conn.execute(
            "SELECT COUNT(*) AS n, "
            "       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            "       SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS losses, "
            "       SUM(realized_pnl) AS total "
            "FROM decision_journal "
            "WHERE final_action IN ('PROCEED','REDUCE') "
            "  AND realized_pnl IS NOT NULL"
        ).fetchone()
        if agg:
            out["trade_count"] = int(agg["n"] or 0)
            out["win_count"] = int(agg["wins"] or 0)
            out["loss_count"] = int(agg["losses"] or 0)
            out["total_realized"] = float(agg["total"] or 0.0)
        # Today
        today_row = conn.execute(
            "SELECT SUM(realized_pnl) AS pnl FROM decision_journal "
            "WHERE substr(timestamp, 1, 10) = ? AND realized_pnl IS NOT NULL",
            (today,),
        ).fetchone()
        if today_row:
            out["today_realized"] = float(today_row["pnl"] or 0.0)
        # Last 7 days
        week_row = conn.execute(
            "SELECT SUM(realized_pnl) AS pnl FROM decision_journal "
            "WHERE substr(timestamp, 1, 10) >= ? AND realized_pnl IS NOT NULL",
            (week_ago,),
        ).fetchone()
        if week_row:
            out["week_realized"] = float(week_row["pnl"] or 0.0)
        # Total cycles
        n_cycles = conn.execute("SELECT COUNT(*) AS n FROM decision_journal").fetchone()
        out["cycle_count"] = int(n_cycles["n"] or 0)
    finally:
        _restore_factory(conn, prev)

    # Max drawdown + sharpe approx from the realized_pnl stream
    prev = _row_factory(conn)
    try:
        rows = conn.execute(
            "SELECT realized_pnl FROM decision_journal "
            "WHERE realized_pnl IS NOT NULL ORDER BY timestamp ASC"
        ).fetchall()
        eq: list[float] = []
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in rows:
            running += float(r["realized_pnl"] or 0.0)
            peak = max(peak, running)
            max_dd = min(max_dd, running - peak)
            eq.append(float(r["realized_pnl"] or 0.0))
        out["max_drawdown"] = max_dd
        if len(eq) > 1:
            mean = sum(eq) / len(eq)
            var = sum((p - mean) ** 2 for p in eq) / max(1, len(eq) - 1)
            std = math.sqrt(var) if var > 0 else 1e-9
            out["sharpe_approx"] = (mean / std) * math.sqrt(len(eq))
    finally:
        _restore_factory(conn, prev)
    return out


def load_next_day_plan(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent PROCEED cycle's selected_strategy.

    Operator reads this to know what the system plans to do next.
    """
    prev = _row_factory(conn)
    try:
        row = conn.execute(
            "SELECT decision_id, timestamp, underlying_focus, "
            "       selected_strategy_json, order_intent_json, "
            "       execution_result_json "
            "FROM decision_journal "
            "WHERE final_action = 'PROCEED' "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    finally:
        _restore_factory(conn, prev)
    if row is None:
        return None
    try:
        sel = json.loads(row["selected_strategy_json"] or "{}") or {}
    except Exception:
        sel = {}
    if not isinstance(sel, dict):
        sel = {}
    try:
        intent = json.loads(row["order_intent_json"] or "{}") or {}
    except Exception:
        intent = {}
    if not isinstance(intent, dict):
        intent = {}
    return {
        "decision_id": row["decision_id"],
        "timestamp": row["timestamp"],
        "underlying": row["underlying_focus"],
        "score": sel.get("score"),
        "thesis": sel.get("thesis"),
        "legs": sel.get("legs", []),
        "intent": intent,
    }


def load_trade_history(conn: sqlite3.Connection, limit: int = 25) -> list[dict]:
    """Return the most recent N cycles, oldest-last display order (newest first)."""
    prev = _row_factory(conn)
    try:
        rows = conn.execute(
            "SELECT decision_id, timestamp, underlying_focus, final_action, "
            "       realized_pnl, outcome_label, "
            "       selected_strategy_json, order_intent_json, "
            "       execution_result_json "
            "FROM decision_journal "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        _restore_factory(conn, prev)
    out: list[dict] = []
    for r in rows:
        try:
            sel = json.loads(r["selected_strategy_json"] or "{}") or {}
        except Exception:
            sel = {}
        if not isinstance(sel, dict):
            sel = {}
        try:
            intent = json.loads(r["order_intent_json"] or "{}") or {}
        except Exception:
            intent = {}
        if not isinstance(intent, dict):
            intent = {}
        try:
            er = json.loads(r["execution_result_json"] or "{}") or {}
        except Exception:
            er = {}
        legs = sel.get("legs", [])
        asset_class = "—"
        if legs:
            asset_class = legs[0].get("asset_class", "option")
        fill_status = er.get("status", "—")
        if fill_status == "filled" and er.get("filled_qty"):
            fill_status = f"filled {er['filled_qty']} @ {er.get('filled_avg_price','-')}"
        out.append({
            "decision_id": r["decision_id"],
            "timestamp": r["timestamp"],
            "underlying": r["underlying_focus"] or "—",
            "action": r["final_action"],
            "asset_class": asset_class,
            "score": sel.get("score"),
            "fill": fill_status,
            "realized_pnl": r["realized_pnl"],
            "outcome": r["outcome_label"],
        })
    return out


def load_recent_signals(conn: sqlite3.Connection, limit: int = 8) -> list[dict]:
    """Pull the most recent N agent_observations across recent cycles.

    Surface the agent_id, message_type, confidence, and a one-line
    signal — the operator can see what the regime / direction /
    volatility view was for the most recent decision.
    """
    prev = _row_factory(conn)
    try:
        rows = conn.execute(
            "SELECT timestamp, agent_observations_json "
            "FROM decision_journal "
            "WHERE agent_observations_json != '[]' "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        _restore_factory(conn, prev)
    out: list[dict] = []
    for r in rows:
        try:
            obs_list = json.loads(r["agent_observations_json"] or "[]") or []
        except Exception:
            continue
        if not isinstance(obs_list, list):
            continue
        for o in obs_list:
            signal = o.get("signal", {}) or {}
            # Pick the first informative key
            sig_keys = (
                "directional_bias", "regime", "volatility_view", "top_strategy_id",
                "candidates_returned", "dte_fallback", "top_pick",
            )
            sig_str = "—"
            for k in sig_keys:
                if k in signal:
                    sig_str = f"{k}={signal[k]}"
                    break
            out.append({
                "timestamp": r["timestamp"],
                "agent_id": o.get("agent_id", "?"),
                "message_type": o.get("message_type", "?"),
                "confidence": o.get("confidence"),
                "signal": sig_str,
                "risks": o.get("risks", []),
            })
    return out[:limit]


def load_equity_curve(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Cumulative realized P&L over the last N reconciled trades.

    Used for the inline SVG sparkline. Each point = (timestamp,
    cumulative_pnl). Returns at most ``limit`` points.
    """
    prev = _row_factory(conn)
    try:
        rows = conn.execute(
            "SELECT timestamp, realized_pnl FROM decision_journal "
            "WHERE realized_pnl IS NOT NULL "
            "ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        _restore_factory(conn, prev)
    out: list[dict] = []
    cum = 0.0
    for r in rows:
        cum += float(r["realized_pnl"] or 0.0)
        out.append({
            "timestamp": r["timestamp"],
            "cumulative": round(cum, 4),
        })
    return out


# ---- rendering ----

_CSS = """
:root {
  --bg: #fafafa;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #2962ff;
  --good: #1b8e3a;
  --bad: #c62828;
  --neutral: #555;
  --line: #e6e6e6;
  --card: #ffffff;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --radius: 8px;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115;
    --fg: #e8e8e8;
    --muted: #9aa0a6;
    --accent: #82b1ff;
    --good: #4caf50;
    --bad: #ef5350;
    --line: #2a2d33;
    --card: #161a1f;
    --shadow: 0 1px 3px rgba(0,0,0,0.4);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 14px; line-height: 1.5; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 20px 80px; }
header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--line); }
h1 { font-size: 22px; font-weight: 600; margin: 0; }
h2 { font-size: 16px; font-weight: 600; margin: 0 0 12px; color: var(--fg); }
.sub { color: var(--muted); font-size: 13px; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
.kpi { background: var(--card); border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--shadow); }
.kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.kpi .value { font-size: 22px; font-weight: 600; font-family: var(--mono); margin-top: 4px; }
.kpi.good .value { color: var(--good); }
.kpi.bad .value { color: var(--bad); }
.card { background: var(--card); border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow); margin-bottom: 20px; }
.empty { color: var(--muted); font-style: italic; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
tr:last-child td { border-bottom: none; }
td.num { font-family: var(--mono); text-align: right; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
.badge.proceed { background: rgba(27,142,58,0.12); color: var(--good); }
.badge.reduce { background: rgba(255,152,0,0.14); color: #ef6c00; }
.badge.reject { background: rgba(198,40,40,0.12); color: var(--bad); }
.badge.no_trade { background: rgba(0,0,0,0.06); color: var(--neutral); }
.badge.equity { background: rgba(41,98,255,0.12); color: var(--accent); }
.badge.option { background: rgba(0,150,136,0.12); color: #00796b; }
.kv { display: grid; grid-template-columns: 140px 1fr; gap: 6px 16px; font-size: 13px; }
.kv .k { color: var(--muted); }
.kv .v { font-family: var(--mono); word-break: break-word; }
.signal-row { display: grid; grid-template-columns: 200px 120px 1fr 80px; gap: 12px; padding: 6px 0; border-bottom: 1px solid var(--line); font-size: 12px; }
.signal-row:last-child { border-bottom: none; }
.signal-row .conf { font-family: var(--mono); text-align: right; }
.sparkline { display: block; width: 100%; height: 60px; margin-top: 12px; }
footer { color: var(--muted); font-size: 11px; margin-top: 32px; text-align: center; }
code { font-family: var(--mono); background: rgba(0,0,0,0.04); padding: 1px 5px; border-radius: 3px; }
@media (max-width: 600px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } .kv { grid-template-columns: 100px 1fr; } }
"""


def _fmt_money(v) -> str:
    if v is None or v == 0:
        return "—"
    return f"${v:,.2f}"


def _fmt_pct(num, denom) -> str:
    if not denom:
        return "—"
    return f"{(num / denom) * 100:.0f}%"


def _badge(action: str) -> str:
    cls = {
        "PROCEED": "proceed",
        "REDUCE": "reduce",
        "REJECT": "reject",
        "NO_TRADE": "no_trade",
    }.get((action or "").upper(), "no_trade")
    return f'<span class="badge {cls}">{html.escape(action or "—")}</span>'


def _class_asset(ac: str) -> str:
    return f'<span class="badge {ac if ac in ("equity", "option") else "no_trade"}">{html.escape(ac or "—")}</span>'


def _class_pnl(v) -> str:
    if v is None:
        return "—"
    cls = "good" if v > 0 else ("bad" if v < 0 else "")
    return f'<span class="{cls}">{_fmt_money(v)}</span>'


def _render_sparkline(points: list[dict]) -> str:
    if len(points) < 2:
        return '<div class="empty">No reconciled trades yet.</div>'
    cum = [p["cumulative"] for p in points]
    lo, hi = min(cum), max(cum)
    span = (hi - lo) or 1.0
    w, h = 800, 60
    step = w / max(1, len(cum) - 1)
    pts = []
    for i, v in enumerate(cum):
        x = i * step
        y = h - ((v - lo) / span) * h
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    last_x = (len(cum) - 1) * step
    last_y = h - ((cum[-1] - lo) / span) * h
    return (
        f'<svg class="sparkline" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="var(--accent)" stroke-width="1.5" points="{polyline}"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="var(--accent)"/>'
        f'</svg>'
        f'<div class="sub" style="margin-top:6px;">'
        f'  range ${lo:,.2f} → ${hi:,.2f}  ·  {len(points)} reconciled trades'
        f'</div>'
    )


def _render_plan(plan: dict | None) -> str:
    if plan is None:
        return '<div class="empty">No PROCEED cycle yet. The next cycle\'s plan will appear here.</div>'
    legs_html = ""
    if plan.get("legs"):
        for leg in plan["legs"]:
            ac = leg.get("asset_class", "?")
            sym = leg.get("contract_symbol", "?")
            side = leg.get("side", "?")
            qty = leg.get("quantity", "?")
            line = f"{side} {qty}× {sym} ({ac})"
            if ac == "option":
                line += f" {leg.get('option_type','?')} @ {leg.get('strike','?')} exp {leg.get('expiry','?')}"
            elif ac == "equity" and leg.get("limit_price") is not None:
                line += f" @ ${leg['limit_price']}"
            legs_html += f"<li>{html.escape(line)}</li>"
    return f"""
    <div class="kv">
      <div class="k">Decision</div><div class="v"><code>{html.escape(plan.get('decision_id','—'))}</code></div>
      <div class="k">Cycle at</div><div class="v">{html.escape(plan.get('timestamp','—'))}</div>
      <div class="k">Underlying</div><div class="v"><strong>{html.escape(plan.get('underlying','—'))}</strong></div>
      <div class="k">Score</div><div class="v">{plan.get('score') if plan.get('score') is not None else '—'}</div>
      <div class="k">Thesis</div><div class="v">{html.escape(plan.get('thesis') or '—')}</div>
      <div class="k">Legs</div><div class="v"><ul style="margin:0; padding-left:16px;">{legs_html or '<li>—</li>'}</ul></div>
    </div>
    """


def _render_trades(trades: list[dict]) -> str:
    if not trades:
        return '<div class="empty">No cycles yet.</div>'
    rows = []
    for t in trades:
        score = t.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        rows.append(f"""
        <tr>
          <td><code>{html.escape(t['decision_id'][:8])}…</code></td>
          <td>{html.escape(t['timestamp'][:19])}</td>
          <td><strong>{html.escape(t['underlying'])}</strong></td>
          <td>{_badge(t['action'])}</td>
          <td>{_class_asset(t.get('asset_class', '—'))}</td>
          <td class="num">{score_str}</td>
          <td>{html.escape(str(t.get('fill', '—')))}</td>
          <td class="num">{_class_pnl(t.get('realized_pnl'))}</td>
          <td>{html.escape(t.get('outcome') or '—')}</td>
        </tr>
        """)
    return f"""
    <table>
      <thead>
        <tr>
          <th>decision</th><th>timestamp (UTC)</th><th>underlying</th>
          <th>action</th><th>class</th><th>score</th>
          <th>fill</th><th>realized PnL</th><th>outcome</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _render_signals(signals: list[dict]) -> str:
    if not signals:
        return '<div class="empty">No agent signals yet.</div>'
    rows = []
    for s in signals:
        risk_str = ", ".join(s.get("risks") or []) or "—"
        conf = s.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        rows.append(f"""
        <div class="signal-row">
          <div>{html.escape(str(s.get('timestamp','—'))[:19])} · <strong>{html.escape(s.get('agent_id','?'))}</strong></div>
          <div>{html.escape(s.get('message_type','?'))}</div>
          <div>{html.escape(str(s.get('signal','—')))}<br><span class="sub">risks: {html.escape(risk_str)}</span></div>
          <div class="conf">{conf_str}</div>
        </div>
        """)
    return "".join(rows)


def render(conn: sqlite3.Connection, build_ts: str) -> str:
    kpis = load_kpis(conn)
    plan = load_next_day_plan(conn)
    trades = load_trade_history(conn, limit=25)
    signals = load_recent_signals(conn, limit=8)
    equity = load_equity_curve(conn, limit=50)

    win_rate = _fmt_pct(kpis["win_count"], max(1, kpis["trade_count"]))
    pnl_class = "good" if kpis["total_realized"] > 0 else (
        "bad" if kpis["total_realized"] < 0 else "")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Aizen Trading Dashboard</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Aizen Trading — Live Dashboard</h1>
      <div class="sub">Built {html.escape(build_ts)} · {kpis['cycle_count']} cycles on record</div>
    </div>
    <div class="sub">
      Last cycle: <code>{html.escape(kpis.get('last_cycle') or '—')}</code><br>
      Last action: {_badge(kpis.get('last_action'))}
    </div>
  </header>

  <div class="kpi-row">
    <div class="kpi {pnl_class}">
      <div class="label">Total realized P&amp;L</div>
      <div class="value">{_fmt_money(kpis['total_realized'])}</div>
    </div>
    <div class="kpi {('good' if kpis['today_realized']>0 else 'bad') if kpis['today_realized'] else ''}">
      <div class="label">Today</div>
      <div class="value">{_fmt_money(kpis['today_realized'])}</div>
    </div>
    <div class="kpi {('good' if kpis['week_realized']>0 else 'bad') if kpis['week_realized'] else ''}">
      <div class="label">Last 7 days</div>
      <div class="value">{_fmt_money(kpis['week_realized'])}</div>
    </div>
    <div class="kpi">
      <div class="label">Trades</div>
      <div class="value">{kpis['trade_count']}</div>
    </div>
    <div class="kpi">
      <div class="label">Win rate</div>
      <div class="value">{win_rate}</div>
    </div>
    <div class="kpi">
      <div class="label">W / L</div>
      <div class="value">{kpis['win_count']} / {kpis['loss_count']}</div>
    </div>
    <div class="kpi bad">
      <div class="label">Max drawdown</div>
      <div class="value">{_fmt_money(kpis['max_drawdown'])}</div>
    </div>
    <div class="kpi">
      <div class="label">Sharpe (approx)</div>
      <div class="value">{(f"{kpis['sharpe_approx']:.2f}" if kpis['sharpe_approx'] else '—')}</div>
    </div>
  </div>

  <section class="card">
    <h2>Next-day plan</h2>
    {_render_plan(plan)}
  </section>

  <section class="card">
    <h2>Equity curve (cumulative realized P&amp;L)</h2>
    {_render_sparkline(equity)}
  </section>

  <section class="card">
    <h2>Recent cycles</h2>
    {_render_trades(trades)}
  </section>

  <section class="card">
    <h2>Recent agent signals</h2>
    {_render_signals(signals)}
  </section>

  <footer>
    Auto-built by <code>scripts/build_dashboard.py</code> on every Render cron tick ·
    data source: <code>decision_journal</code> ·
    view the raw cycle traces in <code>models/cycle_traces/</code>
  </footer>
</div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.getenv("AIZEN_DB_PATH", str(REPO / "data" / "trading.db")))
    p.add_argument("--out", default=str(REPO / "dashboard" / "index.html"))
    args = p.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        # Render the dashboard with a 'no data' notice rather than failing.
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Aizen Trading Dashboard</title></head>"
            f"<body style='font-family:system-ui;padding:40px;'>"
            f"<h1>Aizen Trading Dashboard</h1>"
            f"<p>No data yet. Database not found at "
            f"<code>{html.escape(str(db_path))}</code>.</p>"
            f"<p>This page is auto-generated by "
            f"<code>scripts/build_dashboard.py</code> on every Render cron tick.</p>"
            f"</body></html>"
        )
        print(f"[build_dashboard] db not found; wrote placeholder to {out}")
        return 0

    conn = _connect(str(db_path))
    try:
        out_path = REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html_doc = render(conn, _now_iso())
        out_path.write_text(html_doc, encoding="utf-8")
        print(f"[build_dashboard] wrote {out_path} ({len(html_doc):,} bytes)")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
