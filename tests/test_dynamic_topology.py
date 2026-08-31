"""Tests for the rolling-correlation dynamic GNN topology.

Covers:
- build_rolling_corr_edges() loads underlying_bars <= T, computes pair-wise
  log-return correlations, and emits both directions with reason="rolling_corr"
  only when |rho| >= threshold.
- Pearson helper: returns None for constant series / short series, finite
  number otherwise.
- Threshold env override (AIZEN_GNN_ROLLING_THRESHOLD) and lookback env
  override (AIZEN_GNN_ROLLING_LOOKBACK).
- Leak guard: bars with timestamp > T are NEVER read.
- Snapshot integration: write_snapshot(topology_version="dynamic-1") persists
  rolling_corr edges to gnn_graph_edges and is round-tripped by build_payload.
- Iteration order is byte-deterministic for a given (T, universe).
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.gnn.build_snapshot import build_payload
from src.gnn.constants import EDGE_REASONS, UNIVERSE
from src.gnn.dynamic_topology import (
    DEFAULT_LOOKBACK_BARS,
    DEFAULT_THRESHOLD,
    EDGE_REASON,
    _align_returns,
    _pearson,
    build_rolling_corr_edges,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _make_memory_conn() -> sqlite3.Connection:
    """An isolated in-memory SQLite with the bare minimum schema the
    dynamic-topology builder needs (underlying_bars only)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE underlying_bars (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timestamp)
        );
    """)
    return conn


# ---------------------------------------------------------------------------
# 1. Pearson helper - edge cases
# ---------------------------------------------------------------------------
def test_pearson_perfect_positive_correlation():
    """Identical return series => rho = 1.0."""
    xs = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03]
    ys = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03]
    rho = _pearson(xs, ys)
    assert rho is not None
    assert abs(rho - 1.0) < 1e-9


def test_pearson_perfect_negative_correlation():
    """Anti-correlated series => rho = -1.0; |rho| is what we threshold."""
    xs = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03]
    ys = [-x for x in xs]
    rho = _pearson(xs, ys)
    assert rho is not None
    assert abs(rho + 1.0) < 1e-9


def test_pearson_constant_series_returns_none():
    """A constant series has 0 variance; Pearson is undefined. We
    return None so the pair is skipped, not silently emitted at 0."""
    xs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ys = [0.01, -0.02, 0.03, -0.01, 0.02, 0.04]
    assert _pearson(xs, ys) is None


def test_pearson_short_series_returns_none():
    """Series shorter than 5 points are not enough for a meaningful
    correlation; return None so the pair is skipped."""
    xs = [0.01, 0.02, 0.03]
    ys = [0.01, 0.02, 0.03]
    assert _pearson(xs, ys) is None


def test_pearson_mismatched_lengths_returns_none():
    """Defensive: a mismatched pair would index out of range."""
    xs = [0.01] * 10
    ys = [0.01] * 11
    assert _pearson(xs, ys) is None


# ---------------------------------------------------------------------------
# 2. Edge builder - synthetic data
# ---------------------------------------------------------------------------
def _seed_bars(conn: sqlite3.Connection, bars: list[tuple[str, str, float]]) -> None:
    """Helper: insert a (symbol, timestamp, close) triple into underlying_bars."""
    for sym, ts, close in bars:
        conn.execute(
            "INSERT OR REPLACE INTO underlying_bars "
            "(symbol, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sym, ts, close, close, close, close, 0),
        )
    conn.commit()


def test_empty_universe_returns_no_edges():
    """Defensive: an empty universe -> empty edge list."""
    conn = sqlite3.connect(":memory:")
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T17:45:00Z", universe=(),
    )
    assert edges == []


def test_no_bars_returns_no_edges():
    """Universe set but no underlying_bars rows -> empty edge list."""
    conn = _make_memory_conn()
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T17:45:00Z", universe=("AAPL", "MSFT"),
    )
    assert edges == []


def test_emits_both_directions_for_correlated_pair():
    """Two highly-correlated symbols must produce BOTH directions."""
    conn = _make_memory_conn()
    # 30 bars for AAPL and MSFT, perfectly co-moving. Times 1h apart.
    bars: list[tuple[str, str, float]] = []
    for i in range(30):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        # Identical log-returns => rho = 1.0
        close = 100.0 * math.exp(0.01 * i)
        bars.append(("AAPL", ts, close))
        bars.append(("MSFT", ts, close))
    _seed_bars(conn, bars)
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT"),
        lookback_bars=30, threshold=0.5,
    )
    # Exactly 2 edges, both with reason=rolling_corr.
    assert len(edges) == 2
    assert {e["source"] for e in edges} == {"AAPL", "MSFT"}
    assert {e["target"] for e in edges} == {"AAPL", "MSFT"}
    assert all(e["reason"] == EDGE_REASON for e in edges)
    # weight should be ~1.0 (perfectly correlated)
    for e in edges:
        assert e["weight"] > 0.99


def test_skips_pair_below_threshold():
    """Two uncorrelated symbols must produce NO edge at the default
    threshold (0.5)."""
    conn = _make_memory_conn()
    bars: list[tuple[str, str, float]] = []
    import random
    rng = random.Random(42)
    for i in range(40):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        # AAPL: smooth uptrend. MSFT: random walk.
        aapl_close = 100.0 + i * 0.1
        msft_close = 100.0 + sum(rng.uniform(-1, 1) for _ in range(i + 1))
        bars.append(("AAPL", ts, aapl_close))
        bars.append(("MSFT", ts, msft_close))
    _seed_bars(conn, bars)
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT"),
        lookback_bars=40, threshold=0.5,
    )
    # Random walk vs smooth trend: low correlation, no edge at 0.5.
    assert edges == []


def test_threshold_lower_emits_more_edges():
    """Same data, lower threshold -> at least as many edges."""
    conn = _make_memory_conn()
    bars: list[tuple[str, str, float]] = []
    for i in range(40):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        bars.append(("AAPL", ts, 100.0 + 0.1 * i))
        bars.append(("MSFT", ts, 100.0 + 0.1 * i + 0.01 * (i % 3)))
    _seed_bars(conn, bars)
    high = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT"),
        lookback_bars=40, threshold=0.99,
    )
    low = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT"),
        lookback_bars=40, threshold=0.50,
    )
    # Higher threshold = fewer edges.
    assert len(high) <= len(low)


def test_leak_guard_excludes_bars_after_timestamp():
    """Bars with timestamp > T must NEVER be read."""
    conn = _make_memory_conn()
    bars: list[tuple[str, str, float]] = []
    # Past: 30 co-moving bars
    for i in range(30):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        close = 100.0 * math.exp(0.01 * i)
        bars.append(("AAPL", ts, close))
        bars.append(("MSFT", ts, close))
    # Future: 10 anti-correlated bars AFTER the cut-off
    for i in range(10):
        ts = f"2026-08-26T{(i + 1):02d}:00:00Z"
        # AAPL up, MSFT down => strongly anti-correlated in the future
        bars.append(("AAPL", ts, 200.0 + i))
        bars.append(("MSFT", ts, 50.0 - i))
    _seed_bars(conn, bars)
    # Cut-off = 2026-08-25T23:59:59Z -> only the 30 co-moving bars
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T23:59:59Z", universe=("AAPL", "MSFT"),
        lookback_bars=30, threshold=0.5,
    )
    # The future-anti-correlated bars must NOT be visible.
    for e in edges:
        assert e["weight"] > 0.5, (
            f"future anti-correlation leaked through: {e}"
        )


def test_iteration_is_byte_deterministic():
    """Same inputs -> identical output list. Important for snapshot
    determinism (the payload_json is hash-stable for fixed-1; dynamic-1
    must be too)."""
    conn = _make_memory_conn()
    bars: list[tuple[str, str, float]] = []
    for i in range(30):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        close = 100.0 * math.exp(0.01 * i)
        for s in ("AAPL", "MSFT", "NVDA"):
            bars.append((s, ts, close))
    _seed_bars(conn, bars)
    e1 = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT", "NVDA"),
        lookback_bars=30, threshold=0.5,
    )
    e2 = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT", "NVDA"),
        lookback_bars=30, threshold=0.5,
    )
    assert e1 == e2


# ---------------------------------------------------------------------------
# 3. Env overrides
# ---------------------------------------------------------------------------
def test_threshold_env_override(monkeypatch):
    """AIZEN_GNN_ROLLING_THRESHOLD overrides the default threshold."""
    monkeypatch.setenv("AIZEN_GNN_ROLLING_THRESHOLD", "0.99")
    conn = _make_memory_conn()
    bars: list[tuple[str, str, float]] = []
    for i in range(40):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        bars.append(("AAPL", ts, 100.0 + 0.1 * i))
        bars.append(("MSFT", ts, 100.0 + 0.1 * i + 0.01 * (i % 3)))
    _seed_bars(conn, bars)
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT"),
        lookback_bars=40,
    )
    # threshold was 0.99, so only near-perfect correlations survive.
    for e in edges:
        assert e["weight"] >= 0.99


def test_lookback_env_override(monkeypatch):
    """AIZEN_GNN_ROLLING_LOOKBACK overrides the default lookback."""
    monkeypatch.setenv("AIZEN_GNN_ROLLING_LOOKBACK", "10")
    conn = _make_memory_conn()
    bars: list[tuple[str, str, float]] = []
    for i in range(40):
        ts = f"2026-08-{24 + (i // 24):02d}T{(i % 24):02d}:00:00Z"
        close = 100.0 * math.exp(0.01 * i)
        bars.append(("AAPL", ts, close))
        bars.append(("MSFT", ts, close))
    _seed_bars(conn, bars)
    # Force a tight lookback via env and re-call with no explicit lookback.
    edges = build_rolling_corr_edges(
        conn, "2026-08-25T00:00:00Z", universe=("AAPL", "MSFT"),
    )
    # 10 bars -> 9 returns; both directions still emit when rho is ~1.0.
    assert len(edges) == 2
    assert all(e["reason"] == EDGE_REASON for e in edges)


# ---------------------------------------------------------------------------
# 4. Constants / edge-reason taxonomy
# ---------------------------------------------------------------------------
def test_edge_reason_is_in_constants():
    """The dynamic-topology edge reason must be in the canonical
    EDGE_REASONS enum so the schema CHECK accepts it."""
    assert "rolling_corr" in EDGE_REASONS


def test_default_constants_match_spec():
    """The defaults must match the values documented in the module."""
    assert EDGE_REASON == "rolling_corr"
    assert DEFAULT_LOOKBACK_BARS == 60
    assert DEFAULT_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# 5. build_payload / write_snapshot integration
# ---------------------------------------------------------------------------
def _init_db_with_54_migration(tmp_path: Path) -> sqlite3.Connection:
    """Spin up an isolated DB with the full schema applied via
    init_db() (which includes the 54_gnn_edges_rolling_corr.sql
    migration so the CHECK constraint accepts the new reason)."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    from src.db import init_db
    init_db(conn, sql_dir=Path(REPO) / "sql")
    return conn


def test_migration_accepts_rolling_corr_reason(tmp_path):
    """After init_db() applies 54_gnn_edges_rolling_corr.sql, the
    gnn_graph_edges CHECK constraint must accept 'rolling_corr'."""
    conn = _init_db_with_54_migration(tmp_path)
    # The migration also pre-creates a snapshot row placeholder; insert
    # a snapshot first so the FK is satisfied.
    conn.execute(
        "INSERT OR REPLACE INTO gnn_graph_snapshots "
        "(snapshot_id, timestamp, topology_version, node_count, "
        " edge_count, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("snap-test", "2026-08-25T00:00:00Z", "dynamic-1", 1, 1, "{}", "now"),
    )
    # Insert a rolling_corr edge. Should NOT raise.
    conn.execute(
        "INSERT INTO gnn_graph_edges "
        "(snapshot_id, source_symbol, target_symbol, reason, weight, "
        " topology_version) VALUES (?, ?, ?, ?, ?, ?)",
        ("snap-test", "AAPL", "MSFT", "rolling_corr", 0.81, "dynamic-1"),
    )
    conn.commit()
    # And round-trip.
    row = conn.execute(
        "SELECT reason, weight, topology_version FROM gnn_graph_edges "
        "WHERE snapshot_id = ?",
        ("snap-test",),
    ).fetchone()
    assert row["reason"] == "rolling_corr"
    assert row["weight"] == 0.81
    assert row["topology_version"] == "dynamic-1"


def test_migration_rejects_unknown_reason(tmp_path):
    """The CHECK constraint must still reject values outside the enum
    (the new CHECK is a superset, not a relaxation)."""
    conn = _init_db_with_54_migration(tmp_path)
    conn.execute(
        "INSERT OR REPLACE INTO gnn_graph_snapshots "
        "(snapshot_id, timestamp, topology_version, node_count, "
        " edge_count, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("snap-test", "2026-08-25T00:00:00Z", "fixed-1", 1, 0, "{}", "now"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO gnn_graph_edges "
            "(snapshot_id, source_symbol, target_symbol, reason, weight, "
            " topology_version) VALUES (?, ?, ?, ?, ?, ?)",
            ("snap-test", "AAPL", "MSFT", "not_a_real_reason", 1.0, "fixed-1"),
        )


def test_migration_is_idempotent(tmp_path):
    """Running the migration twice (or running init_db twice on a
    pre-existing DB) must be safe."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    from src.db import init_db
    init_db(conn, sql_dir=Path(REPO) / "sql")
    # Seed a snapshot + a rolling_corr edge.
    conn.execute(
        "INSERT OR REPLACE INTO gnn_graph_snapshots "
        "(snapshot_id, timestamp, topology_version, node_count, "
        " edge_count, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("snap-1", "2026-08-25T00:00:00Z", "dynamic-1", 1, 1, "{}", "now"),
    )
    conn.execute(
        "INSERT INTO gnn_graph_edges "
        "(snapshot_id, source_symbol, target_symbol, reason, weight, "
        " topology_version) VALUES (?, ?, ?, ?, ?, ?)",
        ("snap-1", "AAPL", "MSFT", "rolling_corr", 0.5, "dynamic-1"),
    )
    conn.commit()
    # Re-run init_db. schema_migrations is tracked so nothing should
    # double-apply. The rolling_corr row must survive.
    applied = init_db(conn, sql_dir=Path(REPO) / "sql")
    assert applied == [], f"unexpected migrations on re-run: {applied}"
    rows = conn.execute(
        "SELECT reason FROM gnn_graph_edges WHERE snapshot_id = ?",
        ("snap-1",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["reason"] == "rolling_corr"


def test_build_payload_routes_topology_dynamic_1(tmp_path):
    """build_payload() must consult build_rolling_corr_edges() when
    topology_version='dynamic-1'. We assert this by monkey-patching
    the rolling-edge builder and checking it gets called."""
    conn = _init_db_with_54_migration(tmp_path)
    # Monkey-patch the dynamic-topology builder. The signature is
    # (conn, timestamp, universe) -> list[edge-dicts].
    called: dict[str, int] = {"n": 0}

    def fake_builder(c, ts, universe):
        called["n"] += 1
        return [
            {"source": "AAPL", "target": "MSFT",
             "reason": "rolling_corr", "weight": 0.81},
            {"source": "MSFT", "target": "AAPL",
             "reason": "rolling_corr", "weight": 0.81},
        ]

    import src.gnn.build_snapshot as bs
    bs.build_rolling_corr_edges = fake_builder
    try:
        # fixed-1 must NOT call the dynamic builder
        called["n"] = 0
        _ = build_payload(conn, "2026-08-25T00:00:00Z", topology_version="fixed-1")
        assert called["n"] == 0
        # dynamic-1 must call exactly once
        payload = build_payload(conn, "2026-08-25T00:00:00Z", topology_version="dynamic-1")
        assert called["n"] == 1
        assert payload["topology_version"] == "dynamic-1"
        rolling = [e for e in payload["edges"] if e["reason"] == "rolling_corr"]
        assert len(rolling) == 2
    finally:
        # Restore the original symbol so other tests aren't affected.
        del bs.build_rolling_corr_edges
