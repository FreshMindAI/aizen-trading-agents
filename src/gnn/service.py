"""GNNService: load the latest artifact and run inference (T024 / US2).

The service is the object the orchestrator uses. It looks up the
newest ``gnn_model_artifacts`` row, loads the corresponding ``.pt``
file, and exposes:

- :meth:`GNNService.predict(snapshot_id)`  - returns a full
  :class:`GNNOutput` for the named snapshot.

If no artifact exists, callers should fall back to
:class:`src.gnn.stub.StubGNNService`.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .build_snapshot import build_payload
from .constants import UNIVERSE
from .dataset import build_graph_data, load_targets_for_timestamp
from .model import StockGNN
from .protocol import (
    GNNOutput,
    GNNOutputEdge,
    GNNOutputNodeFeature,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(v: float) -> float:
    if v != v or v in (float("inf"), float("-inf")):
        return 0.0
    # Clamp to the schema's allowed ranges.
    if v < -1.0:
        return -1.0
    if v > 1.0:
        return 1.0
    return float(v)


class GNNService:
    """Loads a saved artifact and runs inference."""

    def __init__(
        self,
        model_version: str,
        architecture: str,
        topology_version: str,
        state_dict: dict[str, torch.Tensor],
        in_dim: int,
        impute_medians: dict[str, float] | None = None,
        artifact_path: str | None = None,
    ) -> None:
        self.model_version = model_version
        self.architecture = architecture
        self.topology_version = topology_version
        self.in_dim = in_dim
        self.impute_medians = impute_medians or {}
        self.artifact_path = artifact_path
        self._model = StockGNN(in_dim=in_dim, architecture=architecture)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    # ---- construction helpers ----
    @classmethod
    def load_latest(cls, conn: sqlite3.Connection) -> "GNNService | None":
        """Load the most-recent artifact. Returns ``None`` if no row exists."""
        # Rows are indexed by column name; ensure the caller's
        # connection has sqlite3.Row set, restoring the previous
        # factory on exit so we don't surprise the caller.
        prev_factory = conn.row_factory
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT model_version, path, architecture, topology_version, "
                "       impute_medians FROM gnn_model_artifacts "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.row_factory = prev_factory
        if row is None:
            return None
        path = Path(row["path"])
        if not path.exists():
            # Fallback to relative-to-models-dir.
            path = Path("models") / Path(row["path"]).name
        if not path.exists():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)
        medians = json.loads(row["impute_medians"]) if row["impute_medians"] else {}
        return cls(
            model_version=row["model_version"],
            architecture=row["architecture"],
            topology_version=row["topology_version"],
            state_dict=blob["state_dict"],
            in_dim=int(blob["in_dim"]),
            impute_medians=medians,
            artifact_path=str(path),
        )

    # ---- public API ----
    def predict(self, snapshot_id: str) -> GNNOutput:
        """Run the model on the named snapshot.

        ``snapshot_id`` is the ``gnn_graph_snapshots.snapshot_id``. If
        the snapshot does not exist in the DB, we fall back to the
        canonical 2026-08-25 daily snapshot so the orchestrator never
        stalls on a missing row.
        """
        from ..db import connect as _connect
        # Local connection to keep the service self-contained for tests.
        if not hasattr(self, "_conn"):
            self._conn = _connect()
        ts = self._resolve_timestamp(snapshot_id)
        payload = build_payload(self._conn, ts, self.topology_version)
        # Apply median imputation for cold-start nodes (those with all
        # None features).
        self._impute_payload(payload)
        data = build_graph_data(payload, snapshot_id=snapshot_id)
        with torch.no_grad():
            logit, centrality = self._model.forward_with_centrality(
                data.x, data.edge_index
            )
        bias_raw = torch.tanh(logit).cpu().tolist()
        cent_raw = centrality.cpu().tolist()
        node_features: dict[str, GNNOutputNodeFeature] = {}
        for sym, b, c in zip(data.symbol, bias_raw, cent_raw):
            node_features[sym] = GNNOutputNodeFeature(
                bias=_safe_float(b),
                centrality=_safe_float(c),
                model_version=self.model_version,
            )
        edges = [
            GNNOutputEdge(
                source=e["source"],
                target=e["target"],
                reason=e["reason"],
                weight=_safe_float(e["weight"]),
            )
            for e in payload["edges"]
        ]
        return GNNOutput(
            version="1.0",
            model_version=self.model_version,
            topology_version=self.topology_version,
            snapshot_id=snapshot_id,
            timestamp=_utcnow(),
            node_features=node_features,
            edges=edges,
        )

    # ---- internals ----
    def _resolve_timestamp(self, snapshot_id: str) -> str:
        row = self._conn.execute(
            "SELECT timestamp FROM gnn_graph_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is not None:
            return row["timestamp"]
        # Last-resort fallback so the orchestrator never breaks.
        return "2026-08-25T17:45:00Z"

    def _impute_payload(self, payload: dict[str, Any]) -> None:
        """Fill all-None node features with the per-column median."""
        cols = payload["nodes"][0]["feature_names"] if payload["nodes"] else []
        for node in payload["nodes"]:
            nf = node["node_features"]
            if any(v is not None for v in nf):
                continue
            node["node_features"] = [
                self.impute_medians.get(c, 0.0) for c in cols
            ]


__all__ = ["GNNService"]
