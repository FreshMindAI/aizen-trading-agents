"""GATv2 model for Phase 2 v2 (per the GATv2 design doc).

Key changes from v1 (src.gnn.model.StockGNN):

- Uses GATv2Conv (the v2 formulation, more flexible than GAT).
- Multi-head attention: 4 heads, concatenated.
- Edge features supported via GATv2Conv's edge_attr.
- Dropout on attention weights (regularization, anti-oversmoothing).
- LayerNorm after each GATv2 layer (residual + normalization per doc §9).
- Three prediction heads: direction (binary), future_rv (scalar), option_opportunity (binary).
- The doc's §12 thesis: GATv2 is most useful for the option_opportunity head;
  the direction head is XGBoost's job.

Architecture::

    Input (n, in_dim) + (2, 2E) edge_index + (2E, edge_dim) edge_attr
      -> Linear projection to hidden_dim
      -> GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=4) + LayerNorm + ReLU + Dropout
      -> GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=4, concat=False) + LayerNorm + ReLU
      -> Heads:
            direction         : Linear(hidden_dim, 1) -> logit
            future_rv         : Linear(hidden_dim, 1) -> log_var
            option_opportunity: Linear(hidden_dim, 1) -> logit
      -> Centrality head: Linear(hidden_dim, 1) -> sigmoid

The model also stores attention weights (for diagnostics per doc §9:
"Store attention diagnostics for selected validation examples").
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:  # import-time guard
    from torch_geometric.nn import GATv2Conv

    _PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    GATv2Conv = None  # type: ignore[assignment]
    _PYG_AVAILABLE = False


class GATv2StockGNN(nn.Module):
    """GATv2 with multi-head attention, edge features, and three prediction heads."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int = 8,
        hidden_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.20,
        architecture: str = "gatv2-32-4h",
    ) -> None:
        super().__init__()
        if not _PYG_AVAILABLE:
            raise RuntimeError("torch_geometric is not installed; cannot build GATv2")
        if architecture not in ("gatv2-32-4h", "gatv2-16-4h", "gatv2-64-4h"):
            raise ValueError(f"unknown GATv2 architecture {architecture!r}")
        self.architecture = architecture
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gat1 = GATv2Conv(
            hidden_dim, hidden_dim, edge_dim=edge_dim,
            heads=n_heads, concat=True, dropout=dropout, add_self_loops=True,
        )
        self.ln1 = nn.LayerNorm(hidden_dim * n_heads)
        self.gat2 = GATv2Conv(
            hidden_dim * n_heads, hidden_dim, edge_dim=edge_dim,
            heads=n_heads, concat=False, dropout=dropout, add_self_loops=True,
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Three heads (doc §9: "Prediction heads - Direction, Future RV, Option Opportunity")
        self.head_direction = nn.Linear(hidden_dim, 1)
        self.head_future_rv = nn.Linear(hidden_dim, 1)
        self.head_option_opportunity = nn.Linear(hidden_dim, 1)
        # Centrality head (kept for compat with v1 service output)
        self.centrality_head = nn.Linear(hidden_dim, 1)

        # Capture attention weights on the last forward for diagnostics
        self.last_attention: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return per-node predictions as a dict."""
        h = self.proj(x)
        if return_attention:
            h, (ei1, aw1) = self.gat1(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            h = self.gat1(h, edge_index, edge_attr=edge_attr)
            ei1, aw1 = None, None
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)
        if return_attention:
            h, (ei2, aw2) = self.gat2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            h = self.gat2(h, edge_index, edge_attr=edge_attr)
            ei2, aw2 = None, None
        h = self.ln2(h)
        h = self.act(h)

        if return_attention:
            self.last_attention = (
                torch.cat([aw1, aw2], dim=0) if aw1 is not None and aw2 is not None else None,
                torch.cat([ei1, ei2], dim=1) if ei1 is not None and ei2 is not None else None,
            )

        return {
            "direction_logit": self.head_direction(h).squeeze(-1),
            "future_rv": self.head_future_rv(h).squeeze(-1),
            "option_opportunity_logit": self.head_option_opportunity(h).squeeze(-1),
            "centrality": torch.sigmoid(self.centrality_head(h).squeeze(-1)),
            "embedding": h,
        }

    # Convenience: v1-compatible interface for the existing service
    def forward_with_centrality(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backwards-compat shim: return (logit, centrality). Logit is the
        option_opportunity head — the v1 service's bias field.
        """
        out = self.forward(x, edge_index, edge_attr=edge_attr)
        return out["option_opportunity_logit"], out["centrality"]


__all__ = ["GATv2StockGNN"]
