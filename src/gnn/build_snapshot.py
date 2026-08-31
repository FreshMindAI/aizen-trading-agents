"""Graph snapshot builder (US1).

Calls the node + edge builders, writes a row to ``gnn_graph_snapshots``
and one row per directed edge to ``gnn_graph_edges``, and stores the
full graph as ``payload_json`` on the snapshot row.

Leak guard
----------
- Every row referenced by the snapshot has ``source_timestamp <=
  snapshot.timestamp`` (enforced by the node builder's SQL).
- Every correlation edge comes from a bar ``<= snapshot.timestamp``
  (enforced by the edge builder's SQL).
- The snapshot is uniquely keyed by ``(timestamp, topology_version)``
  so re-running the same build is idempotent.

CLI
---
::

    python -m src.gnn.build_snapshot --timestamp 2026-08-25 --out snap.json
    python -m src.gnn.build_snapshot --timestamp 2026-08-25 \\
        --topology-version fixed-1
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
from .build_edge_features import build_edge_features
from .build_node_features import build_node_features
from .constants import UNIVERSE
from .dynamic_topology import build_rolling_corr_edges


def _normalize_timestamp(ts: str) -> str:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ' and return the
    canonical full ISO form. A bare date becomes end-of-day UTC."""
    if "T" in ts:
        return ts
    return f"{ts}T00:00:00Z"


def _node_to_dict(node: dict[str, Any]) -> dict[str, Any]:
    """Convert a node dict to a JSON-serializable form. None -> null."""
    return {
        "symbol": node["symbol"],
        "sector": node["sector"],
        "is_benchmark": node["is_benchmark"],
        "node_features": list(node["node_features"]),
        "feature_names": list(node["feature_names"]),
        "source_timestamp": node["source_timestamp"],
    }


def _edge_to_dict(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": edge["source"],
        "target": edge["target"],
        "reason": edge["reason"],
        "weight": float(edge["weight"]),
    }


def _load_failure_features(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...],
    window_days: int = 7,
) -> dict[str, dict[str, float]]:
    """Per-symbol failure features for the rolling ``window_days`` ending
    at ``timestamp``.

    Returns an empty dict when the ``failure_nodes`` table is missing
    (e.g. on a pre-70-failure-kg DB) so this function is safe to call
    unconditionally from build_payload.
    """
    try:
        from ..agents.failure_kg import FailureKG
        return FailureKG(conn).symbol_failure_features(timestamp, window_days=window_days)
    except Exception:
        return {}


def build_payload(
    conn: sqlite3.Connection,
    timestamp: str,
    topology_version: str = "fixed-1",
) -> dict[str, Any]:
    """Build the full snapshot payload in memory (no DB write).

    Topology routing
    ----------------
    The current `topology_version` controls which edge set is produced.
    Two values are supported today:

    - ``"fixed-1"`` (default): the static sector / supplier / customer /
      ETF / correlation edge set from :mod:`src.gnn.build_edge_features`.
    - ``"dynamic-1"``: the same static edge set PLUS the rolling-
      correlation edges from :mod:`src.gnn.dynamic_topology`. The
      ``topology_version`` is written verbatim onto the snapshot row so
      the audit trail always carries the channel mix.

    Adding a new topology version is a matter of (a) extending this
    function with the new branch, and (b) extending the schema's
    `topology_version` enum in the snapshot contract.

    Failure features
    ----------------
    When the ``failure_nodes`` / ``failure_edges`` tables are present
    (sql/70_failure_kg.sql), every payload also carries a
    ``failure_features`` sidecar: a {symbol: feature_dict} map the
    orchestrator can splice into the cycle's MarketSnapshot. The main
    ``node_features`` vector stays at its 14-dim shape so trained
    GATv2 artifacts are not invalidated; the failure channel is a
    sidecar that downstream agents (strategy selector, risk) can
    read independently.
    """
    nodes = build_node_features(conn, timestamp, universe=UNIVERSE)
    edges = build_edge_features(conn, timestamp, universe=UNIVERSE)
    if topology_version == "dynamic-1":
        # Add the rolling-correlation channel on top of the static set.
        # The new edges are tagged with `reason="rolling_corr"` so the
        # GATv2 can learn the dynamic channel as a separate semantic
        # class from the static `correlation` reason.
        dyn_edges = build_rolling_corr_edges(conn, timestamp, universe=UNIVERSE)
        edges = list(edges) + dyn_edges
    failure_features = _load_failure_features(conn, timestamp, UNIVERSE)
    return {
        "timestamp": timestamp,
        "topology_version": topology_version,
        "nodes": [_node_to_dict(n) for n in nodes],
        "edges": [_edge_to_dict(e) for e in edges],
        "failure_features": failure_features,
    }


def write_snapshot(
    conn: sqlite3.Connection,
    timestamp: str,
    topology_version: str = "fixed-1",
) -> dict[str, Any]:
    """Build + write the snapshot. Returns the new ``snapshot_id`` and
    the ``payload_json`` (byte-string) that was persisted."""
    payload = build_payload(conn, timestamp, topology_version)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    snapshot_id = str(uuid.uuid4())
    now = utc_now_iso()
    node_count = len(payload["nodes"])
    edge_count = len(payload["edges"])
    if node_count == 0:
        raise RuntimeError(
            f"no node features found at or before {timestamp!r}; "
            "is v_features_underlying_v2 populated for this date?"
        )
    conn.execute(
        "INSERT OR REPLACE INTO gnn_graph_snapshots "
        "(snapshot_id, timestamp, topology_version, node_count, edge_count, "
        " payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (snapshot_id, timestamp, topology_version, node_count, edge_count,
         payload_json, now),
    )
    # Refresh edges.
    conn.execute(
        "DELETE FROM gnn_graph_edges WHERE snapshot_id = ?",
        (snapshot_id,),
    )
    if payload["edges"]:
        conn.executemany(
            "INSERT OR REPLACE INTO gnn_graph_edges "
            "(snapshot_id, source_symbol, target_symbol, reason, weight, "
            " topology_version) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (snapshot_id, e["source"], e["target"], e["reason"],
                 float(e["weight"]), topology_version)
                for e in payload["edges"]
            ],
        )
    conn.commit()
    return {
        "snapshot_id": snapshot_id,
        "payload_json": payload_json,
        "node_count": node_count,
        "edge_count": edge_count,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a Phase 2 GNN graph snapshot")
    p.add_argument("--timestamp", required=True,
                   help="ISO timestamp; bare YYYY-MM-DD is allowed (becomes 00:00:00Z).")
    p.add_argument("--topology-version", default="fixed-1",
                   help="Topology version tag (default: fixed-1).")
    p.add_argument("--db-path", default=None,
                   help="Override SQLite path (default: data/trading.db).")
    p.add_argument("--out", default=None,
                   help="Optional path to dump payload_json verbatim.")
    args = p.parse_args(argv)

    ts = _normalize_timestamp(args.timestamp)
    conn = connect(args.db_path) if args.db_path else connect()
    try:
        result = write_snapshot(conn, ts, args.topology_version)
    finally:
        conn.close()
    if args.out:
        Path(args.out).write_text(result["payload_json"], encoding="utf-8")
    summary = {
        "snapshot_id": result["snapshot_id"],
        "timestamp": ts,
        "topology_version": args.topology_version,
        "node_count": result["node_count"],
        "edge_count": result["edge_count"],
        "out_path": args.out,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
