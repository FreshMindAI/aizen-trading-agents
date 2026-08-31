"""Node feature builder for the Phase 2 GNN.

For a given snapshot timestamp, pull the 14 GNN columns from
``v_features_underlying_v2`` for every symbol in :data:`UNIVERSE`.

The output is a deterministic list (canonical symbol order from
:data:`UNIVERSE`) of dicts that can be serialized straight into
``payload_json``. Missing symbols (cold-start case) are filled with
``NaN`` placeholders that the imputer handles at training time; the
snapshot's ``node_count`` is always ``len(UNIVERSE)`` so downstream
shape asserts are stable.

Leak guard:
    Every pulled row has ``timestamp <= snapshot_timestamp``. We use
    ``<=`` so a same-bar snapshot still gets the closing-bar features
    (the view is trailing-only - no peeking at future bars).
"""
from __future__ import annotations

import math
import sqlite3
from typing import Any

from .constants import UNIVERSE
from ..config import get_yaml

# The 14 columns. The order is canonical and MUST match
# config/gnn.yaml `features.columns`. Re-deriving it here is a
# defensible no-op (we use the YAML as the source of truth) but a
# hard-coded list keeps the builder importable without the config
# bundle in tests.
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1",
    "return_4",
    "return_16",
    "volatility_16",
    "rsi_14",
    "macd_pct",
    "hl_range",
    "atr_pct_14",
    "ma_dist_20",
    "ma_dist_50",
    "volume_ratio_20",
    "vwap_distance",
    "spy_ret_1",
    "qqq_ret_past_16",
)


def feature_columns() -> tuple[str, ...]:
    """Return the canonical 14-column list, preferring config/gnn.yaml."""
    try:
        cfg = get_yaml("gnn") or {}
        cols = tuple((cfg.get("features") or {}).get("columns") or ())
        if len(cols) == 14:
            return cols  # type: ignore[return-value]
    except Exception:
        pass
    return DEFAULT_FEATURE_COLUMNS


def _safe(v: Any) -> float | None:
    """Coerce a SQL value to a JSON-safe float (NaN/inf -> None)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f in (float("inf"), float("-inf")):
        return None
    return f


def build_node_features(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...] = UNIVERSE,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return one dict per symbol in ``universe`` (canonical order).

    Each dict is::

        {
            "symbol": "AAPL",
            "sector": "Tech",
            "node_features": [14 floats or None, ...],
            "is_benchmark": 0 | 1,
            "feature_names": [...14 names...],
            "source_timestamp": "2026-08-25T17:45:00Z"  # the bar actually pulled
        }
    """
    from .constants import NON_TRADABLE, SECTOR_MAP  # local import to avoid cycles

    cols = columns or feature_columns()
    placeholders = ",".join("?" * len(universe))
    sql = (
        f"SELECT symbol, timestamp, {','.join(cols)} "
        "FROM v_features_underlying_v2 "
        f"WHERE symbol IN ({placeholders}) AND timestamp <= ? "
        "ORDER BY symbol, timestamp DESC"
    )
    rows = conn.execute(sql, [*universe, timestamp]).fetchall()

    # Reduce to the most recent bar per symbol <= snapshot.
    latest: dict[str, sqlite3.Row] = {}
    for row in rows:
        sym = row["symbol"]
        if sym not in latest:
            latest[sym] = row

    out: list[dict[str, Any]] = []
    for sym in universe:
        row = latest.get(sym)
        if row is None:
            node_features = [None] * len(cols)
            source_ts = None
        else:
            node_features = [_safe(row[c]) for c in cols]
            source_ts = row["timestamp"]
        out.append({
            "symbol": sym,
            "sector": SECTOR_MAP.get(sym, "Other"),
            "node_features": node_features,
            "is_benchmark": 1 if sym in NON_TRADABLE else 0,
            "feature_names": list(cols),
            "source_timestamp": source_ts,
        })
    return out


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "feature_columns",
    "build_node_features",
]
