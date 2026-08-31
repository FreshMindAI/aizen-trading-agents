"""Option-graph snapshot builder (GATv2 doc §3-§7).

Doc §4: Node types are UNDERLYING, OPTION, EXPIRY, STRIKE, MARKET_INDEX, SECTOR.
Doc §4.2: Edge types are option->underlying, option->option_same_expiry, etc.

We don't build the full heterogeneous graph in v2 — we build a
*homogeneous option graph* where nodes are option contracts and edges
capture the four "option->option" relations from the doc. The underlying
graph is still built by `build_snapshot.build_payload` and consumed by
the XGBoost side; the option graph is consumed by GATv2.

This file is the v2 option-graph builder. It writes to:
- gnn_option_graph_snapshots  (new table, see sql/51_option_graph.sql)
- gnn_option_graph_edges      (new table)
- gnn_option_graph_nodes      (new table)

For the hackathon, we focus on ATM-ish contracts (the 14,892 we already
have) and only edges that have computable features in our DB. We'll add
EXPIRY and STRIKE node types in v3.

Edge features (doc §5, normalized):
- strike_distance    : |strike_i - strike_j| / spot
- expiry_distance    : |DTE_i - DTE_j| / max_dte
- moneyness_distance : |(K/S)_i - (K/S)_j|
- iv_difference      : |iv_i - iv_j|  (placeholder: 0 if missing)
- return_correlation : rolling 64-bar correlation of underlying return
- liquidity_similarity : 1 - |vol_i - vol_j| / max(vol_i, vol_j)
- edge_type_onehot   : 4-dim (same_expiry, nearby_strike, neighboring_expiry, call_put_pair)
- timestamp          : 1-dim (normalized snapshot age)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..db import connect, utc_now_iso


# ---------------------------------------------------------------------------
# Edge-type one-hot encoding
# ---------------------------------------------------------------------------
# Structural edge types (pre-news).
EDGE_TYPES_STRUCTURAL: tuple[str, ...] = (
    "same_expiry",          # same expiration date
    "nearby_strike",        # |strike_i - strike_j| / spot <= 0.05
    "neighboring_expiry",   # |DTE_i - DTE_j| <= 7
    "call_put_pair",        # same strike, same expiry, opposite type
)
# News-driven edge types (spec 003 / FR-005, FR-006).
EDGE_TYPES_NEWS: tuple[str, ...] = (
    "news_cooccurrence",            # two underlyings both had >=2 articles in lookback
    "news_sentiment_correlation",  # |rho| of per-symbol sentiment over rolling window
)
# Public alias for back-compat with tests / external callers.
EDGE_TYPES: tuple[str, ...] = EDGE_TYPES_STRUCTURAL + EDGE_TYPES_NEWS
STRUCTURAL_REASONS: frozenset[str] = frozenset(EDGE_TYPES_STRUCTURAL)
NEWS_REASONS: frozenset[str] = frozenset(EDGE_TYPES_NEWS)

NEARBY_STRIKE_BAND = 0.05    # 5% of spot
NEIGHBORING_EXPIRY_DAYS = 7

# News-edge configuration (spec 003 / config/gnn.yaml).
NEWS_DEFAULT_LOOKBACK_HOURS = 24
NEWS_DEFAULT_MIN_ARTICLES = 2
NEWS_DEFAULT_MIN_ABS_RHO = 0.5
NEWS_DEFAULT_WINDOW_DAYS = 5


def _edge_type_onehot(reason: str) -> list[float]:
    """Return a 2-d [is_structural, is_news] one-hot.

    The full 6-d one-hot would inflate the GATv2 edge_attr to 12 dim and
    force a model re-architecture; spec decision 6 keeps edge_dim=8 total
    (6 feature dims + 2 one-hot) so existing checkpoints remain valid.
    Distinguishing structural from news at the GNN level is enough for the
    ablation: news edges get a different one-hot slice, so attention
    weights can learn them as a separate channel.
    """
    if reason in NEWS_REASONS:
        return [0.0, 1.0]
    # default: structural
    return [1.0, 0.0]


# ---------------------------------------------------------------------------
# Schema bootstrap (idempotent)
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gnn_option_graph_snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    topology_version TEXT NOT NULL,
    node_count       INTEGER NOT NULL,
    edge_count       INTEGER NOT NULL,
    payload_json     TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_opt_snapshot_ts
    ON gnn_option_graph_snapshots(timestamp DESC);

CREATE TABLE IF NOT EXISTS gnn_option_graph_nodes (
    snapshot_id     TEXT NOT NULL,
    contract_symbol TEXT NOT NULL,
    underlying      TEXT NOT NULL,
    strike_price    REAL NOT NULL,
    option_type     TEXT NOT NULL,
    expiration_date TEXT NOT NULL,
    dte             INTEGER NOT NULL,
    moneyness       REAL NOT NULL,
    node_features_json TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, contract_symbol)
);

CREATE TABLE IF NOT EXISTS gnn_option_graph_edges (
    snapshot_id    TEXT NOT NULL,
    source_symbol  TEXT NOT NULL,
    target_symbol  TEXT NOT NULL,
    reason         TEXT NOT NULL,
    weight         REAL NOT NULL,
    edge_features_json TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, source_symbol, target_symbol, reason)
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Build the option graph
# ---------------------------------------------------------------------------
def _select_contracts_at(
    conn: sqlite3.Connection, timestamp: str, underlying: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Pick the ATM-ish contracts that have option bars <= timestamp.

    We require the contract to have at least one bar in the last 7 days
    so the node features are non-empty.
    """
    sql = (
        "SELECT oc.contract_symbol, oc.underlying_symbol AS underlying, "
        "       oc.strike_price, oc.option_type, oc.expiration_date, "
        "       CAST(julianday(oc.expiration_date) - julianday(substr(?, 1, 10)) AS INTEGER) AS dte, "
        "       oc.strike_price / ub.close AS moneyness, "
        "       ub.close AS spot "
        "FROM   option_contracts oc "
        "JOIN   option_bars ob ON ob.contract_symbol = oc.contract_symbol "
        "JOIN   (SELECT symbol, close FROM underlying_bars "
        "        WHERE (symbol, timestamp) IN ( "
        "            SELECT symbol, MAX(timestamp) FROM underlying_bars "
        "            WHERE timestamp <= ? GROUP BY symbol)) ub "
        "       ON ub.symbol = oc.underlying_symbol "
        "WHERE  ob.timestamp <= ? "
        "  AND  ob.timestamp >= datetime(?, '-7 days') "
        "  AND  CAST(julianday(oc.expiration_date) - julianday(substr(?, 1, 10)) AS INTEGER) BETWEEN 7 AND 60 "
        "  AND  oc.strike_price / ub.close BETWEEN 0.90 AND 1.10 "
    )
    params: list[Any] = [timestamp, timestamp, timestamp, timestamp, timestamp]
    if underlying:
        placeholders = ",".join("?" * len(underlying))
        sql += f" AND oc.underlying_symbol IN ({placeholders})"
        params += list(underlying)
    sql += " GROUP BY oc.contract_symbol ORDER BY oc.underlying_symbol, oc.expiration_date, oc.strike_price, oc.option_type"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _node_features(contract: dict[str, Any], conn: sqlite3.Connection) -> list[float]:
    """Per-contract node features: a 14-dim vector mirroring the underlying
    features, but option-specific where it makes sense."""
    sym = contract["contract_symbol"]
    # Last option bar <= snapshot timestamp
    row = conn.execute(
        "SELECT close, volume FROM option_bars "
        "WHERE contract_symbol = ? AND timestamp <= ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (sym, contract.get("__snapshot_ts", "")),
    ).fetchone()
    last_close = float(row["close"]) if row and row["close"] is not None else 0.0
    last_vol = float(row["volume"]) if row and row["volume"] is not None else 0.0

    # 5-bar return, 16-bar return, 16-bar vol
    rets = conn.execute(
        "SELECT close, timestamp FROM option_bars "
        "WHERE contract_symbol = ? AND timestamp <= ? "
        "ORDER BY timestamp DESC LIMIT 32",
        (sym, contract.get("__snapshot_ts", "")),
    ).fetchall()
    closes = [float(r["close"]) for r in rets if r["close"] is not None]
    if len(closes) < 2:
        r1 = r4 = r16 = v16 = 0.0
    else:
        # closes[0] is the most recent
        r1 = (closes[0] - closes[1]) / max(closes[1], 1e-9) if len(closes) > 1 else 0.0
        r4 = (closes[0] - closes[min(4, len(closes)-1)]) / max(closes[min(4, len(closes)-1)], 1e-9) if len(closes) > 4 else r1
        r16 = (closes[0] - closes[min(16, len(closes)-1)]) / max(closes[min(16, len(closes)-1)], 1e-9) if len(closes) > 16 else r4
        v16 = (
            sum((closes[i] - closes[i+1]) ** 2 for i in range(min(16, len(closes)-1)))
            / max(1, min(16, len(closes)-1))
        ) ** 0.5 if len(closes) > 1 else 0.0

    dte = float(contract.get("dte", 0))
    moneyness = float(contract.get("moneyness", 1.0))
    spot = float(contract.get("spot", 0.0))
    return [
        r1, r4, r16, v16,
        1.0 if contract["option_type"] == "call" else 0.0,
        moneyness,
        dte / 60.0,            # normalized DTE
        last_close,
        last_vol,
        spot,
        float(contract["strike_price"]),
        contract["underlying"] in ("SPY", "QQQ") and 1.0 or 0.0,  # is_index
        1.0 if moneyness > 1.0 else 0.0,    # is_call_itm
        1.0 if moneyness < 1.0 else 0.0,    # is_put_itm
    ]


def _edge_features(
    a: dict[str, Any], b: dict[str, Any], reason: str,
    conn: sqlite3.Connection | None = None, snapshot_ts: str | None = None,
) -> list[float]:
    """Compute the 8-d edge feature vector for an (a, b, reason) edge.

    Layout: [strike_dist, dte_dist, money_dist, iv_diff, rho, liq_sim,
             is_structural, is_news] (6-d features + 2-d one-hot).

    The correlation query is only attempted when both ``conn`` and
    ``snapshot_ts`` are provided AND a/b share the same underlying
    (cross-underlying pairs use rho=0 by default).
    """
    spot_a = float(a["spot"]); spot_b = float(b["spot"])
    spot_avg = max((spot_a + spot_b) / 2.0, 1e-9)
    strike_dist = abs(a["strike_price"] - b["strike_price"]) / spot_avg
    dte_dist = abs(a["dte"] - b["dte"]) / 60.0
    money_dist = abs(a["moneyness"] - b["moneyness"])
    rho = 0.0
    if conn is not None and snapshot_ts is not None and a["underlying"] == b["underlying"]:
        row = conn.execute(
            "SELECT correlation FROM v_asset_correlations "
            "WHERE symbol_a = ? AND symbol_b = ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (a["underlying"], b["underlying"], snapshot_ts),
        ).fetchone()
        if row and row["correlation"] is not None:
            rho = float(row["correlation"])
    va = float(a.get("__last_vol", 0.0)); vb = float(b.get("__last_vol", 0.0))
    denom = max(va, vb, 1.0)
    liq_sim = 1.0 - abs(va - vb) / denom
    onehot = _edge_type_onehot(reason)
    return [
        strike_dist, dte_dist, money_dist, 0.0,  # iv_difference placeholder
        rho, liq_sim, *onehot,
    ]


def _build_edges(
    nodes: list[dict[str, Any]],
    conn: sqlite3.Connection,
    snapshot_ts: str,
    news_edges: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build all directed edges with the 8-dim edge feature vector.

    The output vector layout is:
        [strike_dist, dte_dist, money_dist, iv_diff, rho, liq_sim, is_structural, is_news]
    (6-d features + 2-d one-hot = 8 dim total, matching GATv2's edge_dim=8).
    """
    edges: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, float], dict[str, Any]] = {}
    for n in nodes:
        by_key[(n["underlying"], n["expiration_date"], n["strike_price"])] = n
    spot_by_underlying: dict[str, float] = {n["underlying"]: float(n["spot"]) for n in nodes}

    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j:
                continue
            weight = 0.0
            reason = None
            # 1. same_expiry
            if a["expiration_date"] == b["expiration_date"] and a["underlying"] == b["underlying"]:
                reason = "same_expiry"
                weight = 1.0
            # 2. nearby_strike (same expiry, different strike, within 5% of spot)
            elif (
                a["underlying"] == b["underlying"]
                and a["expiration_date"] == b["expiration_date"]
                and abs(a["strike_price"] - b["strike_price"]) / max(spot_by_underlying[a["underlying"]], 1e-9) <= NEARBY_STRIKE_BAND
            ):
                reason = "nearby_strike"
                weight = 0.8
            # 3. neighboring_expiry (same underlying, same strike, expiry within 7 days)
            elif (
                a["underlying"] == b["underlying"]
                and a["strike_price"] == b["strike_price"]
                and abs(a["dte"] - b["dte"]) <= NEIGHBORING_EXPIRY_DAYS
            ):
                reason = "neighboring_expiry"
                weight = 0.7
            # 4. call_put_pair (same strike, same expiry, opposite type)
            elif (
                a["underlying"] == b["underlying"]
                and a["strike_price"] == b["strike_price"]
                and a["expiration_date"] == b["expiration_date"]
                and a["option_type"] != b["option_type"]
            ):
                reason = "call_put_pair"
                weight = 0.9
            if reason is None:
                continue
            ef = _edge_features(a, b, reason, conn=conn, snapshot_ts=snapshot_ts)
            edges.append({
                "source": a["contract_symbol"],
                "target": b["contract_symbol"],
                "reason": reason,
                "weight": float(weight),
                "edge_features": ef,
            })
    if news_edges:
        edges.extend(news_edges)
    return edges


# ---------------------------------------------------------------------------
# News edge builder (spec 003 / FR-005, FR-006)
# ---------------------------------------------------------------------------
def _news_per_underlying(
    conn: sqlite3.Connection,
    universe: list[str],
    snapshot_ts: str,
    lookback_hours: int = NEWS_DEFAULT_LOOKBACK_HOURS,
    cutoff: str | None = None,
) -> dict[str, dict[str, Any]]:
    """For each underlying in the universe, return
        {"article_count": int, "sentiment_ts": [(ts, sentiment), ...]}.

    ``cutoff`` is the upper bound on article timestamps (failure-analysis
    §2.1 row 2 — no future-article leakage). When ``None``, defaults to
    ``snapshot_ts``.
    """
    cut = cutoff or snapshot_ts
    placeholders = ",".join("?" for _ in universe)
    rows = conn.execute(
        f"""
        SELECT symbol, timestamp, sentiment, article_count
        FROM   news_snapshot
        WHERE  symbol IN ({placeholders})
          AND  timestamp <= ?
          AND  timestamp >= datetime(?, '-' || ? || ' hours')
        ORDER  BY symbol, timestamp DESC
        """,
        [*universe, cut, cut, int(lookback_hours)],
    ).fetchall()
    out: dict[str, dict[str, Any]] = {u: {"article_count": 0, "sentiment_ts": []} for u in universe}
    for r in rows:
        out[r["symbol"]]["article_count"] += int(r["article_count"] or 0)
        out[r["symbol"]]["sentiment_ts"].append((str(r["timestamp"]), float(r["sentiment"] or 0.0)))
    return out


def _news_cooccurrence_edges(
    nodes: list[dict[str, Any]],
    per_u: dict[str, dict[str, Any]],
    *,
    min_articles: int = NEWS_DEFAULT_MIN_ARTICLES,
) -> list[dict[str, Any]]:
    """For each pair of underlyings (A, B) where BOTH had >= min_articles
    articles in the lookback window, emit a directed edge for every
    (contract_A, contract_B) pair. Weight = log(1 + count_A) * log(1 + count_B)
    (geometric mean of the two news volumes, deterministic and bounded).
    """
    import math
    eligible = {u for u, p in per_u.items() if p["article_count"] >= min_articles}
    by_underlying: dict[str, list[dict]] = {}
    for n in nodes:
        by_underlying.setdefault(n["underlying"], []).append(n)
    edges: list[dict[str, Any]] = []
    eligible_sorted = sorted(eligible)
    for i, ua in enumerate(eligible_sorted):
        for ub in eligible_sorted[i + 1:]:
            ca = math.log1p(per_u[ua]["article_count"])
            cb = math.log1p(per_u[ub]["article_count"])
            w = ca * cb
            for a in by_underlying.get(ua, []):
                for b in by_underlying.get(ub, []):
                    edges.append({
                        "source": a["contract_symbol"],
                        "target": b["contract_symbol"],
                        "reason": "news_cooccurrence",
                        "weight": float(w),
                        "_a": a, "_b": b,
                    })
                    edges.append({
                        "source": b["contract_symbol"],
                        "target": a["contract_symbol"],
                        "reason": "news_cooccurrence",
                        "weight": float(w),
                        "_a": b, "_b": a,
                    })
    return edges


def _sentiment_correlation(
    sent_ts_a: list[tuple[str, float]], sent_ts_b: list[tuple[str, float]],
) -> float | None:
    """Pearson correlation on the timestamps the two series share
    (approximate; anchored to whole-day buckets to keep it fast). Returns
    None when fewer than 2 shared points."""
    if not sent_ts_a or not sent_ts_b:
        return None
    # Bucket to day granularity for the join
    a_map: dict[str, float] = {}
    for ts, s in sent_ts_a:
        day = ts[:10]
        # average if multiple articles on the same day
        a_map[day] = a_map.get(day, 0.0) + s
    b_map: dict[str, float] = {}
    for ts, s in sent_ts_b:
        day = ts[:10]
        b_map[day] = b_map.get(day, 0.0) + s
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 2:
        return None
    xs = [a_map[d] for d in common]
    ys = [b_map[d] for d in common]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = (dx2 * dy2) ** 0.5
    if denom < 1e-12:
        return None
    return num / denom


def _news_sentiment_correlation_edges(
    nodes: list[dict[str, Any]],
    per_u: dict[str, dict[str, Any]],
    *,
    min_abs_rho: float = NEWS_DEFAULT_MIN_ABS_RHO,
) -> list[dict[str, Any]]:
    """For each pair (A, B) whose sentiment time series have |rho| >=
    min_abs_rho, emit a directed edge for every (contract_A, contract_B)
    pair. Weight = |rho| (in [min_abs_rho, 1])."""
    by_underlying: dict[str, list[dict]] = {}
    for n in nodes:
        by_underlying.setdefault(n["underlying"], []).append(n)
    underlyings = sorted(per_u.keys())
    edges: list[dict[str, Any]] = []
    for i, ua in enumerate(underlyings):
        for ub in underlyings[i + 1:]:
            rho = _sentiment_correlation(per_u[ua]["sentiment_ts"], per_u[ub]["sentiment_ts"])
            if rho is None or abs(rho) < min_abs_rho:
                continue
            w = abs(rho)
            for a in by_underlying.get(ua, []):
                for b in by_underlying.get(ub, []):
                    edges.append({
                        "source": a["contract_symbol"],
                        "target": b["contract_symbol"],
                        "reason": "news_sentiment_correlation",
                        "weight": float(w),
                        "_a": a, "_b": b,
                    })
                    edges.append({
                        "source": b["contract_symbol"],
                        "target": a["contract_symbol"],
                        "reason": "news_sentiment_correlation",
                        "weight": float(w),
                        "_a": b, "_b": a,
                    })
    return edges


def _build_news_edges(
    nodes: list[dict[str, Any]],
    conn: sqlite3.Connection,
    snapshot_ts: str,
    news_cutoff: str | None = None,
    *,
    lookback_hours: int = NEWS_DEFAULT_LOOKBACK_HOURS,
    min_articles: int = NEWS_DEFAULT_MIN_ARTICLES,
    min_abs_rho: float = NEWS_DEFAULT_MIN_ABS_RHO,
) -> list[dict[str, Any]]:
    """Build the two news-driven edge types for an option graph snapshot.

    For each news edge, the 8-d edge feature vector is the standard
    [strike_dist, dte_dist, money_dist, iv_diff, rho, liq_sim, is_structural, is_news]
    layout, with the one-hot set to [0, 1] (news). The two `a`/`b` contract
    dicts are attached temporarily on the edge (`_a`, `_b`) so the caller
    can build the feature vector without re-walking nodes; they are
    stripped before persistence.
    """
    universe = sorted({n["underlying"] for n in nodes})
    if not universe:
        return []
    per_u = _news_per_underlying(
        conn, universe, snapshot_ts, lookback_hours=lookback_hours, cutoff=news_cutoff,
    )
    raw = (
        _news_cooccurrence_edges(nodes, per_u, min_articles=min_articles)
        + _news_sentiment_correlation_edges(nodes, per_u, min_abs_rho=min_abs_rho)
    )
    out: list[dict[str, Any]] = []
    for e in raw:
        a = e.pop("_a")
        b = e.pop("_b")
        ef = _edge_features(a, b, e["reason"], conn=conn, snapshot_ts=snapshot_ts)
        e["edge_features"] = ef
        out.append(e)
    return out


def build_option_payload(
    conn: sqlite3.Connection,
    timestamp: str,
    underlying: tuple[str, ...] | None = None,
    *,
    news_enabled: bool = False,
    news_cutoff: str | None = None,
) -> dict[str, Any]:
    """Build the option-graph payload in memory (no DB write).

    When ``news_enabled`` is True, the option graph re-wires daily based on
    the most recent ``news_snapshot`` rows (<= ``news_cutoff`` or
    ``timestamp``). The two new edge types are ``news_cooccurrence`` and
    ``news_sentiment_correlation`` (spec 003 / US2, FR-005/FR-006).

    The returned ``topology_version`` is ``"option-v2-news"`` when at least
    one news edge is present, else ``"option-v2"`` (auditability per
    spec FR-007).
    """
    contracts = _select_contracts_at(conn, timestamp, underlying=underlying)
    if not contracts:
        return {
            "timestamp": timestamp,
            "topology_version": "option-v2-news" if news_enabled else "option-v2",
            "nodes": [],
            "edges": [],
        }
    nodes: list[dict[str, Any]] = []
    for c in contracts:
        c["__snapshot_ts"] = timestamp
        nf = _node_features(c, conn)
        # Cache last_vol for edge building
        row = conn.execute(
            "SELECT volume FROM option_bars WHERE contract_symbol = ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (c["contract_symbol"], timestamp),
        ).fetchone()
        c["__last_vol"] = float(row["volume"]) if row and row["volume"] is not None else 0.0
        nodes.append({
            "contract_symbol": c["contract_symbol"],
            "underlying": c["underlying"],
            "strike_price": float(c["strike_price"]),
            "option_type": c["option_type"],
            "expiration_date": c["expiration_date"],
            "dte": int(c["dte"]),
            "moneyness": float(c["moneyness"]),
            "spot": float(c["spot"]),
            "node_features": nf,
        })
    news_edges: list[dict[str, Any]] = []
    if news_enabled:
        news_edges = _build_news_edges(nodes, conn, timestamp, news_cutoff=news_cutoff)
    edges = _build_edges(nodes, conn, timestamp, news_edges=news_edges)
    topology_version = "option-v2-news" if news_edges else (
        "option-v2-news" if news_enabled else "option-v2"
    )
    return {
        "timestamp": timestamp,
        "topology_version": topology_version,
        "nodes": nodes,
        "edges": edges,
    }


def write_option_snapshot(
    conn: sqlite3.Connection,
    timestamp: str,
    underlying: tuple[str, ...] | None = None,
    *,
    news_enabled: bool = False,
    news_cutoff: str | None = None,
) -> dict[str, Any]:
    """Build + persist the option graph snapshot."""
    ensure_schema(conn)
    payload = build_option_payload(
        conn, timestamp, underlying=underlying,
        news_enabled=news_enabled, news_cutoff=news_cutoff,
    )
    snapshot_id = str(uuid.uuid4())
    now = utc_now_iso()
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    conn.execute(
        "INSERT OR REPLACE INTO gnn_option_graph_snapshots "
        "(snapshot_id, timestamp, topology_version, node_count, edge_count, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (snapshot_id, timestamp, payload["topology_version"],
         len(payload["nodes"]), len(payload["edges"]), payload_json, now),
    )
    # Wipe + re-insert nodes/edges for this snapshot
    conn.execute("DELETE FROM gnn_option_graph_nodes WHERE snapshot_id = ?", (snapshot_id,))
    conn.execute("DELETE FROM gnn_option_graph_edges WHERE snapshot_id = ?", (snapshot_id,))
    if payload["nodes"]:
        conn.executemany(
            "INSERT INTO gnn_option_graph_nodes "
            "(snapshot_id, contract_symbol, underlying, strike_price, option_type, "
            " expiration_date, dte, moneyness, node_features_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (snapshot_id, n["contract_symbol"], n["underlying"], n["strike_price"],
                 n["option_type"], n["expiration_date"], n["dte"], n["moneyness"],
                 json.dumps(n["node_features"]))
                for n in payload["nodes"]
            ],
        )
    if payload["edges"]:
        conn.executemany(
            "INSERT INTO gnn_option_graph_edges "
            "(snapshot_id, source_symbol, target_symbol, reason, weight, edge_features_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (snapshot_id, e["source"], e["target"], e["reason"], float(e["weight"]),
                 json.dumps(e["edge_features"]))
                for e in payload["edges"]
            ],
        )
    conn.commit()
    return {
        "snapshot_id": snapshot_id,
        "payload_json": payload_json,
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a Phase 2 v2 option graph snapshot")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--underlying", default=None,
                   help="comma-separated underlyings (default: full universe)")
    p.add_argument("--db-path", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    conn = connect(args.db_path) if args.db_path else connect()
    try:
        und = tuple(s.strip().upper() for s in (args.underlying or "").split(",") if s.strip()) or None
        result = write_option_snapshot(conn, args.timestamp, underlying=und)
    finally:
        conn.close()
    if args.out:
        Path(args.out).write_text(result["payload_json"], encoding="utf-8")
    print(json.dumps({
        "snapshot_id": result["snapshot_id"],
        "timestamp": args.timestamp,
        "node_count": result["node_count"],
        "edge_count": result["edge_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
