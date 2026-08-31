"""Edge feature builder for the Phase 2 GNN.

Emits edges with a documented ``reason`` from these sources:

- ``sector``       - both nodes share a GICS sector (constants).
- ``supplier``     - explicit supplier relation (NVDA -> MSFT/GOOGL/META/AMZN/AMD).
- ``customer``     - explicit customer relation (the reverse of supplier).
- ``etf_membership`` - both nodes share an ETF basket (SPY/QQQ).
- ``correlation``  - data-driven, from ``v_asset_correlations`` where
                     ``|corr| > threshold`` and the bar is at or before
                     the snapshot timestamp.

All edges are produced as both directions (i -> j and j -> i) so the
message-passing graph is undirected (the GNN's GCNConv layer is
direction-agnostic and so are the human-readable edges in
``payload_json``).

Output shape (one row per directed edge)::

    {
        "source": "AAPL",
        "target": "MSFT",
        "reason": "sector" | "supplier" | "customer" | "etf_membership" | "correlation",
        "weight": float  # in [0, 1]
    }
"""
from __future__ import annotations

import sqlite3
from typing import Any

from .constants import (
    CORRELATION_THRESHOLD,
    ETF_HOLDINGS,
    SECTOR_MAP,
    SUPPLY_CHAIN,
    CUSTOMER_CHAIN,
    UNIVERSE,
)
from ..config import get_yaml


def _threshold() -> float:
    try:
        cfg = get_yaml("gnn") or {}
        return float(
            (cfg.get("edges") or {}).get("correlation", {}).get("threshold")
            or CORRELATION_THRESHOLD
        )
    except Exception:
        return CORRELATION_THRESHOLD


def _both(a: str, b: str, reason: str, weight: float) -> list[dict[str, Any]]:
    """Emit both directed edges so the resulting graph is undirected."""
    return [
        {"source": a, "target": b, "reason": reason, "weight": float(weight)},
        {"source": b, "target": a, "reason": reason, "weight": float(weight)},
    ]


def _sector_edges(universe: tuple[str, ...]) -> list[dict[str, Any]]:
    """All pairs of same-sector symbols (excluding self-loops)."""
    by_sector: dict[str, list[str]] = {}
    for sym in universe:
        sec = SECTOR_MAP.get(sym, "Other")
        by_sector.setdefault(sec, []).append(sym)
    out: list[dict[str, Any]] = []
    for members in by_sector.values():
        # Sort for byte-deterministic order; the universe tuple is already sorted.
        members = sorted(members)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                out += _both(a, b, "sector", 1.0)
    return out


def _supplier_edges() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src, dst in SUPPLY_CHAIN:
        out += _both(src, dst, "supplier", 1.0)
    return out


def _customer_edges() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src, dst in CUSTOMER_CHAIN:
        out += _both(src, dst, "customer", 1.0)
    return out


def _etf_membership_edges(universe: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _etf, members in ETF_HOLDINGS.items():
        members = [m for m in members if m in universe]
        members = sorted(set(members))
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                out += _both(a, b, "etf_membership", 1.0)
    return out


def _correlation_edges(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...],
    threshold: float,
) -> list[dict[str, Any]]:
    """Pull the most-recent bar <= snapshot for every universe pair and
    emit edges where |corr| > threshold."""
    placeholders = ",".join("?" * len(universe))
    # Per pair, take the most recent correlation row whose bar is
    # <= snapshot timestamp. The view already enforces the 64-bar
    # rolling window, so the latest row is the most informative.
    sql = (
        "SELECT symbol_a, symbol_b, correlation, timestamp "
        "FROM v_asset_correlations "
        f"WHERE symbol_a IN ({placeholders}) "
        f"AND symbol_b IN ({placeholders}) "
        "AND timestamp <= ? "
        "AND correlation IS NOT NULL "
        "ORDER BY symbol_a, symbol_b, timestamp DESC"
    )
    rows = conn.execute(
        sql, [*universe, *universe, timestamp]
    ).fetchall()
    # Reduce to most recent row per (a, b).
    latest: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["symbol_a"], row["symbol_b"])
        if key not in latest:
            latest[key] = float(row["correlation"])
    out: list[dict[str, Any]] = []
    for (a, b), c in sorted(latest.items()):
        if abs(c) > threshold:
            out += _both(a, b, "correlation", min(1.0, abs(c)))
    return out


def build_edge_features(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...] = UNIVERSE,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Return all directed edges for the snapshot at ``timestamp``.

    Sources are concatenated in the canonical order so the output list
    is byte-deterministic for a given DB state. Duplicates (e.g. an
    ETF pair that is also in the same sector) are kept - they are
    legitimate semantic duplicates the GNN can learn to weight.
    """
    th = threshold if threshold is not None else _threshold()
    out: list[dict[str, Any]] = []
    out += _sector_edges(universe)
    out += _supplier_edges()
    out += _customer_edges()
    out += _etf_membership_edges(universe)
    out += _correlation_edges(conn, timestamp, universe, th)
    # Drop self-loops if any sneak in (none should).
    out = [e for e in out if e["source"] != e["target"]]
    return out


__all__ = ["build_edge_features"]
