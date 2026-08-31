"""PyG ``InMemoryDataset`` wrapper for the Phase 2 GNN.

Loads graph snapshots from SQLite (built by :mod:`src.gnn.build_snapshot`)
and pairs each one with the per-node target from :data:`v_labels`.

Output: one :class:`torch_geometric.data.Data` per snapshot, with::

    x          : (n, 14) float tensor  - node features
    edge_index : (2, 2E) long tensor   - bidirectional edges
    y          : (n,) long tensor       - target_class in {-1, 0, 1}
    mask       : (n,) bool tensor       - True where target is non-flat
    symbol     : list[str]              - node order matches UNIVERSE
    snapshot_id: str                    - gnn_graph_snapshots.snapshot_id

For the smoke test (T019), a 1-snapshot fixture is enough to prove
the loader works end-to-end.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

import torch
from torch_geometric.data import Data, InMemoryDataset

from .constants import UNIVERSE


def _edges_to_index(edges: list[dict[str, Any]], sym_to_idx: dict[str, int]) -> torch.Tensor:
    """Build a (2, 2E) edge_index. Drops any self-loops defensively."""
    pairs: list[tuple[int, int]] = []
    for e in edges:
        s = sym_to_idx.get(e["source"])
        t = sym_to_idx.get(e["target"])
        if s is None or t is None or s == t:
            continue
        pairs.append((s, t))
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _features_to_tensor(node_features: list[Any]) -> torch.Tensor:
    """Convert a list-of-lists of (float|None) to a (N, 14) tensor.
    None -> 0.0; NaN/inf -> 0.0. The imputer is responsible for filling
    medians at training time, so the loader stays simple."""
    rows: list[list[float]] = []
    for nf in node_features:
        rows.append([0.0 if (v is None) else float(v) for v in nf])
    return torch.tensor(rows, dtype=torch.float32)


def build_graph_data(
    payload: dict[str, Any],
    targets: list[int | None] | None = None,
    snapshot_id: str | None = None,
) -> Data:
    """Convert a snapshot payload (dict) into a torch_geometric Data.

    Parameters
    ----------
    payload
        The output of :func:`src.gnn.build_snapshot.build_payload`.
    targets
        Optional per-symbol target_class in the canonical UNIVERSE order.
        ``None`` or 0 marks the node as "flat" and is excluded from the
        binary-classification loss via ``mask``.
    snapshot_id
        Carried on the Data object for traceability.
    """
    sym_to_idx = {sym: i for i, sym in enumerate(UNIVERSE)}
    x = _features_to_tensor([n["node_features"] for n in payload["nodes"]])
    edge_index = _edges_to_index(payload["edges"], sym_to_idx)
    n = len(UNIVERSE)
    if targets is None:
        y = torch.zeros(n, dtype=torch.long)
        mask = torch.zeros(n, dtype=torch.bool)
    else:
        y = torch.tensor([int(t or 0) for t in targets], dtype=torch.long)
        mask = torch.tensor([(t is not None and t != 0) for t in targets],
                            dtype=torch.bool)
    data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
    data.symbol = list(UNIVERSE)
    if snapshot_id is not None:
        data.snapshot_id = snapshot_id
    return data


def load_targets_for_timestamp(
    conn: sqlite3.Connection,
    timestamp: str,
    universe: tuple[str, ...] = UNIVERSE,
    horizon_bars: int = 16,
) -> list[int | None]:
    """Return per-symbol target_class from v_labels for the given snapshot.

    Uses the latest label row at or before ``timestamp`` for each symbol.
    """
    placeholders = ",".join("?" * len(universe))
    sql = (
        f"SELECT symbol, target_class FROM v_labels "
        f"WHERE horizon_bars = ? AND symbol IN ({placeholders}) "
        "AND timestamp <= ? "
        "ORDER BY symbol, timestamp DESC"
    )
    rows = conn.execute(sql, [horizon_bars, *universe, timestamp]).fetchall()
    latest: dict[str, int] = {}
    for row in rows:
        sym = row["symbol"]
        if sym not in latest and row["target_class"] is not None:
            latest[sym] = int(row["target_class"])
    return [latest.get(sym) for sym in universe]


class GNNGraphDataset(InMemoryDataset):
    """PyG dataset of per-snapshot graphs with per-node targets.

    Pass a list of snapshot timestamps + horizon. Each becomes one
    ``Data`` object. Useful for walk-forward training where the same
    chronological split produces different snapshot sets per fold.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        timestamps: Sequence[str],
        horizon_bars: int = 16,
    ) -> None:
        # InMemoryDataset wants a root + transform/transform_target. We
        # pass safe no-ops because we never persist to disk.
        super().__init__(".", None, None, None)
        self.conn = conn
        self.timestamps = list(timestamps)
        self.horizon_bars = horizon_bars
        self._payloads: dict[str, dict[str, Any]] = {}
        data_list = self._build_all()
        # InMemoryDataset wants self._data, self.slices.
        if data_list:
            self._data, self.slices = self.collate(data_list)
            self.data = self._data
        else:
            # Empty dataset - InMemoryDataset requires a non-empty
            # internal representation, so we leave data as None and
            # expose ``len() == 0`` for the iteration check.
            self._data = None
            self.slices = None
            self.data = None

    # ---- public helpers ----
    def get(self, idx: int) -> Data:
        if self._data is None:
            raise IndexError("empty GNNGraphDataset")
        return super().get(idx)

    def __len__(self) -> int:
        if self._data is None or self.slices is None:
            return 0
        # InMemoryDataset stores one slice per Data.
        if isinstance(self.slices, dict):
            key = next(iter(self.slices))
            return int(self.slices[key].shape[0] - 1)
        return 0

    # ---- internals ----
    def _build_all(self) -> list[Data]:
        from .build_snapshot import build_payload
        out: list[Data] = []
        for ts in self.timestamps:
            payload = build_payload(self.conn, ts)
            sid = f"inline::{ts}"
            self._payloads[sid] = payload
            targets = load_targets_for_timestamp(
                self.conn, ts, horizon_bars=self.horizon_bars,
            )
            d = build_graph_data(payload, targets, snapshot_id=sid)
            if d.mask.any():
                out.append(d)
        return out


__all__ = [
    "build_graph_data",
    "load_targets_for_timestamp",
    "GNNGraphDataset",
]
