"""PyG dataset contract test (T018 / US2).

For a known snapshot, the resulting ``Data`` object must have:
- ``x.shape == (n, 14)``
- ``edge_index`` is bidirectional
- no self-loops
- all nodes reachable from any other node
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.gnn.build_snapshot import build_payload
from src.gnn.constants import UNIVERSE
from src.gnn.dataset import GNNGraphDataset, build_graph_data


@pytest.fixture
def tmp_db(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src import config as cfg_mod
    cfg_mod.get_settings.cache_clear()
    cfg_mod._load_yaml_bundle.cache_clear()
    conn = sqlite3.connect("data/trading.db")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_data_object_shape_and_edges(tmp_db):
    payload = build_payload(tmp_db, "2026-08-25T17:45:00Z")
    targets = [1] * len(UNIVERSE)  # mark every node as non-flat
    data = build_graph_data(payload, targets, snapshot_id="t")
    assert data.x.shape == (len(UNIVERSE), 14)
    assert data.edge_index.dim() == 2 and data.edge_index.shape[0] == 2
    # Bidirectional: every edge i->j must have a j->i counterpart.
    ei = data.edge_index.t().tolist()
    pairs = {(s, t) for s, t in ei}
    for s, t in list(pairs):
        assert (t, s) in pairs, f"edge {s}->{t} is not bidirectional"
    # No self-loops.
    for s, t in pairs:
        assert s != t
    # All nodes reachable from node 0.
    from collections import deque
    adj: dict[int, set[int]] = {i: set() for i in range(len(UNIVERSE))}
    for s, t in pairs:
        adj[s].add(t)
    seen = {0}
    q = deque([0])
    while q:
        n = q.popleft()
        for nb in adj[n]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    assert seen == set(range(len(UNIVERSE))), (
        f"unreachable nodes: {set(range(len(UNIVERSE))) - seen}"
    )


def test_dataset_wraps_multiple_snapshots(tmp_db):
    """The InMemoryDataset should produce one Data per timestamp and
    concatenate them in order."""
    ts_list = [
        "2026-08-25T17:45:00Z",
        "2026-08-24T17:45:00Z",
        "2026-08-21T17:45:00Z",
    ]
    ds = GNNGraphDataset(tmp_db, ts_list, horizon_bars=16)
    assert len(ds) == 3
    for i in range(3):
        d = ds.get(i)
        assert d.x.shape == (len(UNIVERSE), 14)
        assert d.edge_index.dim() == 2
        # Bidirectional + no-self-loops still hold for every snapshot.
        ei = d.edge_index.t().tolist()
        pairs = {(s, t) for s, t in ei}
        for s, t in pairs:
            assert s != t
            assert (t, s) in pairs
