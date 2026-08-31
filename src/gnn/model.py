"""Phase 2 GNN model architecture and registry.

The first iteration ships :class:`StockGNN` (GCNConv -> 32 -> 16 -> 1) as
specified in ``Phase_2_Dynamic_GNN_Trading_Model.docx`` section 5. GAT
and GraphSAGE placeholders are registered so the architecture can be
swapped without touching the dataset or training loop (spec FR5.2).

The model is registered by string key in :data:`MODEL_REGISTRY`; the
config file ``config/gnn.yaml`` carries the same keys so the same code
runs against any registered architecture.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

# Imported lazily so :mod:`src.gnn.model` loads without PyG installed at
# import time. PyG is only required when a model is actually instantiated.
try:  # pragma: no cover - import-time guard
    from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv

    _PYG_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on minimal envs
    GCNConv = GATConv = GATv2Conv = SAGEConv = None  # type: ignore[assignment]
    _PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gcn-32-16-1": {
        "type": "GCNConv",
        "hidden_dims": (32, 16),
        "out_dim": 1,
    },
    "gat-32-16-1": {
        "type": "GATConv",
        "hidden_dims": (32, 16),
        "out_dim": 1,
    },
    "sage-32-16-1": {
        "type": "SAGEConv",
        "hidden_dims": (32, 16),
        "out_dim": 1,
    },
    "gatv2-32-4h": {
        "type": "GATv2Conv",
        "hidden_dim": 32,
        "n_heads": 4,
        "edge_dim": 10,
        "out_dim": 1,
    },
    "gatv2-32-1h": {
        # Single-head GATv2 — used by train.py via the StockGNN v1
        # interface. The proper multi-head 3-head architecture lives
        # in src/gnn/gatv2.py (GATv2StockGNN) and is wired through a
        # separate training path.
        "type": "GATv2Conv",
        "hidden_dims": (32, 16),
        "out_dim": 1,
    },
}


def supported_architectures() -> tuple[str, ...]:
    """Return the registered architecture keys, in registration order."""
    return tuple(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Layer constructors
# ---------------------------------------------------------------------------
def _gcn_layer(in_dim: int, out_dim: int) -> nn.Module:
    if GCNConv is None:  # pragma: no cover - import-time guard
        raise RuntimeError("torch_geometric is not installed; cannot build GCN layer")
    return GCNConv(in_dim, out_dim)


def _gat_layer(in_dim: int, out_dim: int) -> nn.Module:
    if GATConv is None:  # pragma: no cover
        raise RuntimeError("torch_geometric is not installed; cannot build GAT layer")
    # 1 head, concat=False, average over heads implicitly
    return GATConv(in_dim, out_dim, heads=1, concat=False)


def _sage_layer(in_dim: int, out_dim: int) -> nn.Module:
    if SAGEConv is None:  # pragma: no cover
        raise RuntimeError("torch_geometric is not installed; cannot build SAGE layer")
    return SAGEConv(in_dim, out_dim)


def _gatv2_layer(in_dim: int, out_dim: int) -> nn.Module:
    """Build a 1-head GATv2Conv layer, matching the v1 StockGNN interface.

    The full multi-head GATv2 (n_heads=4, edge features, 3 prediction
    heads) lives in :class:`src.gnn.gatv2.GATv2StockGNN` and is used
    by the news-driven retraining path. This single-head builder lets
    the existing train.py register a GATv2-flavored model so we can
    run the baseline walk-forward ablation against the GCN/GAT/SAGE
    family without re-writing the v1 training loop.
    """
    if GATv2Conv is None:  # pragma: no cover
        raise RuntimeError("torch_geometric is not installed; cannot build GATv2 layer")
    return GATv2Conv(in_dim, out_dim, heads=1, concat=False)


_LAYER_BUILDERS: dict[str, Callable[[int, int], nn.Module]] = {
    "GCNConv":  _gcn_layer,
    "GATConv":  _gat_layer,
    "SAGEConv": _sage_layer,
    "GATv2Conv": _gatv2_layer,
}


# ---------------------------------------------------------------------------
# StockGNN
# ---------------------------------------------------------------------------
class StockGNN(nn.Module):
    """The Phase 2 graph model.

    Default architecture (matches the doc):
        GCNConv(in_dim, 32) -> ReLU
        GCNConv(32, 16)     -> ReLU
        Linear(16, 1)       -> squeeze

    The forward pass returns a per-node logit. For binary direction
    targets, callers apply ``torch.sigmoid`` and threshold at 0.5.
    """

    def __init__(
        self,
        in_dim: int,
        architecture: str = "gcn-32-16-1",
    ) -> None:
        super().__init__()
        if architecture not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown architecture {architecture!r}; "
                f"supported: {supported_architectures()}"
            )
        spec = MODEL_REGISTRY[architecture]
        layer_type = spec["type"]
        hidden_dims = tuple(spec["hidden_dims"])
        out_dim = int(spec["out_dim"])

        if layer_type not in _LAYER_BUILDERS:
            raise ValueError(f"no layer builder for type {layer_type!r}")

        builder = _LAYER_BUILDERS[layer_type]
        if len(hidden_dims) != 2:
            raise ValueError(
                f"architecture {architecture!r} expects 2 hidden dims, got {hidden_dims}"
            )

        self.conv1 = builder(in_dim, hidden_dims[0])
        self.conv2 = builder(hidden_dims[0], hidden_dims[1])
        self.head = nn.Linear(hidden_dims[1], out_dim)
        # Centrality head: maps the 16-dim embedding to a per-node
        # score. Sigmoid inside forward_with_centrality; stored as a
        # Linear so the optimizer can train it.
        self.centrality_head = nn.Linear(hidden_dims[1], 1)

        self.architecture = architecture
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-node logits of shape ``[num_nodes]`` for binary targets."""
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        out = self.head(h)
        return out.squeeze(-1)

    def forward_with_centrality(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logit, centrality) per node.

        ``logit``      : (N,) - bias head, used for direction classification.
        ``centrality`` : (N,) - learned per-node score in [0, 1] (after
                         sigmoid). Built from the 16-dim penultimate embedding
                         so it captures what the GNN actually learned, not
                         just the static graph degree.
        """
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        logit = self.head(h).squeeze(-1)
        cent = self.centrality_head(h).squeeze(-1)
        return logit, torch.sigmoid(cent)

    @staticmethod
    def out_per_node(num_nodes: int, out_dim: int = 1) -> tuple[int, int]:
        """Shape helper for tests."""
        return num_nodes, out_dim


__all__ = [
    "MODEL_REGISTRY",
    "StockGNN",
    "supported_architectures",
]
