"""Rolling-correlation dynamic GNN topology.

Background
----------
The default "fixed-1" topology (see ``build_edge_features.py``) wires
nodes by static sector membership, hard-coded supplier/customer
chains, and ETF baskets. The correlation edge layer exists but is
loaded from a precomputed ``v_asset_correlations`` view that itself
uses a fixed 64-bar window. That is fine for the GATv2 model but
gives the agent no way to react when the correlation regime flips
(e.g. "in a risk-off week, AAPL/MSFT/NVDA all co-move, but the
sector relation alone does not capture this").

This module implements a *dynamic* topology: rolling-correlation
edges recomputed at snapshot time from the most recent
``underlying_bars`` rows. The intended use is the live cycle:
``InferenceService(as_of=T, topology_version="dynamic-1")`` calls
``build_rolling_corr_edges(conn, T, universe, lookback_bars=N)`` and
combines the result with the static edge set to produce a single
edge list whose dynamic-1 / fixed-1 mix is auditable via
``topology_version`` on the snapshot row.

Design
------
1. **Edge reason**: ``rolling_corr`` (distinct from the static
   ``correlation`` reason so the GNN can learn the dynamic channel as
   a separate semantic class).
2. **Lookback**: configurable ``lookback_bars`` (default 60; ~3 trading
   days at 15-min bars, ~12 days at 1D). Operator can set
   ``AIZEN_GNN_ROLLING_LOOKBACK`` to override.
3. **Threshold**: configurable ``|rho|`` cutoff (default 0.5). Uses the
   same convention as the static correlation layer so the two are
   comparable.
4. **Byte-determinism**: the SQL query is deterministic; the edge
   iteration order is ``sorted((a, b))`` and dedup is by (a, b).
5. **Leak guard**: every bar read is ``WHERE timestamp <= T``. The
   snapshot timestamp is the upper bound, so the rolling window
   is *closed* at T and never includes future bars.

Public surface
--------------
::

    from src.gnn.dynamic_topology import build_rolling_corr_edges

    edges = build_rolling_corr_edges(
        conn, timestamp="2026-08-25T13:30:00Z",
        universe=("AAPL", "MSFT", "NVDA"),
        lookback_bars=60, threshold=0.5,
    )
    # edges: [{"source": "AAPL", "target": "MSFT",
    #          "reason": "rolling_corr", "weight": 0.81}, ...]
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any

from .constants import UNIVERSE


# Defaults honor the same convention as src/gnn/build_edge_features.
DEFAULT_LOOKBACK_BARS = 60
DEFAULT_THRESHOLD = 0.5
EDGE_REASON = "rolling_corr"


def _default_lookback_bars() -> int:
    """Env override (AIZEN_GNN_ROLLING_LOOKBACK) clamped to a sane range."""
    raw = os.getenv("AIZEN_GNN_ROLLING_LOOKBACK", str(DEFAULT_LOOKBACK_BARS)).strip()
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_LOOKBACK_BARS
    return max(2, min(n, 2000))  # [2, 2000] bars


def _default_threshold() -> float:
    raw = os.getenv("AIZEN_GNN_ROLLING_THRESHOLD", str(DEFAULT_THRESHOLD)).strip()
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_THRESHOLD
    return max(0.0, min(v, 0.99))  # |rho| in [0, 1)


def _both(a: str, b: str, reason: str, weight: float) -> list[dict[str, Any]]:
    """Emit both directed edges so the resulting graph is undirected.
    Mirrors the convention in build_edge_features._both()."""
    return [
        {"source": a, "target": b, "reason": reason, "weight": float(weight)},
        {"source": b, "target": a, "reason": reason, "weight": float(weight)},
    ]


def _load_returns(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...],
    lookback_bars: int,
) -> dict[str, list[float]]:
    """Load the most recent ``lookback_bars`` 1-step returns for each
    symbol. Returns are log-returns
        r_t = ln(close_t / close_{t-1})
    computed in Python after fetching the close series. Bars are
    filtered ``timestamp <= :ts`` for leak safety.

    Output: {symbol: [r_newest, ..., r_oldest]} with length
    ``len(closes) - 1`` for each symbol. Symbols with fewer than
    ``lookback_bars + 1`` bars are kept (truncated) - we still
    compute a correlation when there are >= 5 returns; below that,
    the pair is skipped.
    """
    if not universe:
        return {}
    placeholders = ",".join("?" * len(universe))
    # Pull the most recent N+1 closes per symbol in ONE query. The
    # window-function approach keeps the SQL portable and the result
    # byte-deterministic. The +1 is so we can compute the first return.
    sql = (
        "SELECT symbol, timestamp, close FROM underlying_bars "
        f"WHERE symbol IN ({placeholders}) "
        "AND timestamp <= ? "
        "ORDER BY symbol, timestamp DESC "
    )
    # We need only the top N+1 rows per symbol. The cheap way on SQLite
    # without window functions is to do a single fetchall and then
    # truncate in Python - lookback_bars is small.
    rows = conn.execute(sql, [*universe, timestamp]).fetchall()
    by_sym: dict[str, list[tuple[str, float]]] = {s: [] for s in universe}
    for r in rows:
        sym = r["symbol"]
        if sym not in by_sym:
            continue
        by_sym[sym].append((str(r["timestamp"]), float(r["close"])))
    out: dict[str, list[float]] = {}
    for sym, series in by_sym.items():
        # series is in DESC order; the most recent is series[0].
        # Take the first lookback_bars + 1, then reverse to ASC for
        # return computation (older -> newer).
        series = series[: lookback_bars + 1]
        series_asc = list(reversed(series))
        closes = [c for _, c in series_asc]
        if len(closes) < 2:
            continue
        rets: list[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            if prev <= 0:
                continue
            # log return is more numerically stable for small changes;
            # simple return is also fine. Use log.
            import math
            rets.append(math.log(closes[i] / prev))
        out[sym] = rets
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation on two equal-length return series.
    Returns None when either series is *exactly* constant (zero
    variance). Near-constant series are still returned — the
    operator can lower the threshold to filter them out.
    """
    n = len(xs)
    if n < 5 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0.0 or dy2 == 0.0:
        # Exactly-constant series: correlation is undefined.
        return None
    denom = (dx2 * dy2) ** 0.5
    if denom <= 0.0:
        return None
    return num / denom


def _align_returns(
    rets: dict[str, list[float]],
) -> list[tuple[str, str, list[float], list[float]]]:
    """Build all (a, b, xs, ys) tuples where the two return series have
    the same length (so the correlation is well-defined). Series must
    be aligned by their position in the list, which holds because the
    loader always returns the most recent ``lookback_bars`` returns per
    symbol. The shorter series dominates so we only pair equal-length
    ones.
    """
    pairs: list[tuple[str, str, list[float], list[float]]] = []
    syms = sorted(rets.keys())
    for i, a in enumerate(syms):
        ra = rets[a]
        for b in syms[i + 1:]:
            rb = rets[b]
            n = min(len(ra), len(rb))
            if n < 5:
                continue
            # Take the most-recent n of each so the window is anchored
            # to the snapshot timestamp, not a stale earlier bar.
            pairs.append((a, b, ra[-n:], rb[-n:]))
    return pairs


def build_rolling_corr_edges(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...] = UNIVERSE,
    *,
    lookback_bars: int | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Build rolling-correlation edges for the dynamic-1 topology.

    Edges are returned as a flat list of ``{source, target, reason,
    weight}`` dicts, with both directions for every pair. The output
    is byte-deterministic for a given (timestamp, universe, lookback,
    threshold) combination.

    Args:
        conn: SQLite connection to the trading DB.
        timestamp: snapshot time; bar reads are filtered ``<= timestamp``
            so the window is closed (no future-data leakage).
        universe: tuple of symbols to correlate. Default = full UNIVERSE.
        lookback_bars: number of 1-step returns to use. Default = env
            override or 60. Series shorter than this are truncated, not
            dropped (correlation is still computable on the shorter
            window as long as the pair has >= 5 common points).
        threshold: minimum |rho| to emit an edge. Default = 0.5.
    """
    lookback = lookback_bars if lookback_bars is not None else _default_lookback_bars()
    th = threshold if threshold is not None else _default_threshold()
    rets = _load_returns(conn, timestamp, universe, lookback)
    if len(rets) < 2:
        return []
    pairs = _align_returns(rets)
    edges: list[dict[str, Any]] = []
    # Sort by (a, b) so the iteration order is byte-deterministic.
    for a, b, xs, ys in pairs:
        rho = _pearson(xs, ys)
        if rho is None:
            continue
        ar = abs(rho)
        if ar < th:
            continue
        edges += _both(a, b, EDGE_REASON, min(1.0, ar))
    return edges


__all__ = [
    "DEFAULT_LOOKBACK_BARS",
    "DEFAULT_THRESHOLD",
    "EDGE_REASON",
    "build_rolling_corr_edges",
]
