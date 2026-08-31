"""Pydantic mirrors of the Phase 2 GNN JSON-Schema contracts.

Why a Pydantic mirror when we already have the JSON Schema?
- The orchestrator is a Python process; importing a Pydantic model is
  faster than parsing + validating JSON for every cycle.
- Pydantic gives us ``extra=forbid`` and NaN-coercion for free,
  matching the constitution principle II "schema-first" requirement.
- The JSON Schema files in ``contracts/`` remain the source of truth for
  cross-language consumers; the two are kept in sync by the round-trip
  tests in US2 / US3.

Everything here extends :class:`agents.protocol.StrictModel` semantics
(``extra=forbid``, NaN/inf -> None) where it makes sense; the strict
``GNNOutput`` and ``GNNArtifactMeta`` classes below inherit that base
directly so the orchestrator can plug them in without any glue.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import EDGE_REASONS

SCHEMA_VERSION = "1.0"
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "gcn-32-16-1",
    "gat-32-16-1",
    "sage-32-16-1",
    "gatv2-32-4h",
    "gatv2-32-1h",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _GNNBase(BaseModel):
    """Strict base for the Phase 2 protocol.

    Mirrors :class:`src.agents.protocol.StrictModel` but kept local so
    Phase 2 has no Phase 3 import dependency. NaN/inf are coerced to
    ``None`` so a partially-populated row never crashes a cycle.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    @field_validator("*", mode="before")
    @classmethod
    def _no_nans(cls, v: Any) -> Any:
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                return None
        return v


# ---------------------------------------------------------------------------
# GNNOutput - the on-the-wire and on-disk shape of
# InferenceService.gnn_output(). Mirrors contracts/gnn_output.schema.json.
# ---------------------------------------------------------------------------
EdgeReason = Literal["sector", "supplier", "customer", "etf_membership", "correlation"]


class GNNOutputNodeFeature(_GNNBase):
    """Per-symbol bias and centrality from the GNN at one snapshot."""

    bias: float = Field(ge=-1.0, le=1.0)
    centrality: float = Field(ge=0.0, le=1.0)
    model_version: str


class GNNOutputEdge(_GNNBase):
    """One edge in the graph that produced the GNN output."""

    source: str
    target: str
    reason: EdgeReason
    weight: float = Field(ge=0.0, le=1.0)

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, v: Any) -> Any:
        if isinstance(v, str) and v not in EDGE_REASONS:
            raise ValueError(
                f"reason {v!r} is not in {EDGE_REASONS}"
            )
        return v


class GNNOutput(_GNNBase):
    """The output of :func:`src.agents.inference.InferenceService.gnn_output`.

    Consumed by:
    - the Phase 3 orchestrator (Direction + Options Structure agents);
    - the decision_journal (serialized verbatim into gnn_output_json);
    - cross-process consumers via the JSON-Schema contract.

    ``model_version == "stub-1"`` is the deterministic stub used when no
    artifact has been trained yet (FR8.3). A real artifact has the form
    ``gnn-YYYY-MM-DD-N`` (see :mod:`src.gnn.service`).
    """

    version: Literal["1.0"] = "1.0"
    model_version: str
    topology_version: str = "fixed-1"
    snapshot_id: str | None = None
    timestamp: str = Field(default_factory=_utcnow)
    node_features: dict[str, GNNOutputNodeFeature] = Field(default_factory=dict)
    edges: list[GNNOutputEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GNNArtifactMeta - the JSON sidecar written next to every .pt artifact.
# Mirrors contracts/gnn_artifact_meta.schema.json.
# ---------------------------------------------------------------------------
SplitBounds = dict[str, str]
Metrics = dict[str, float]


class GNNArtifactMeta(_GNNBase):
    """Sidecar JSON describing one saved model.

    Written by :mod:`src.gnn.train`; read by :mod:`src.gnn.service`.
    The three seed fields are mandatory so a re-run is byte-deterministic
    (spec SC1).
    """

    model_version: str
    architecture: Literal[
        "gcn-32-16-1", "gat-32-16-1", "sage-32-16-1", "gatv2-32-4h", "gatv2-32-1h"
    ]
    topology_version: str = "fixed-1"
    feature_names: list[str] = Field(min_length=1)
    impute_medians: dict[str, float]
    split_bounds: SplitBounds
    test_metrics: Metrics
    created_at: str = Field(default_factory=_utcnow)
    torch_seed: int
    numpy_seed: int
    python_hash_seed: str
