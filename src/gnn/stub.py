"""Deterministic GNN stub.

Used by :func:`src.agents.inference.InferenceService.gnn_output` when no
trained artifact exists yet (FR8.3 / SC5). The stub must:

- return a fully-populated :class:`GNNOutput` with ``model_version="stub-1"``;
- never call the network or the LLM;
- be reproducible across processes (no timestamps inside the bias/centrality
  values, only the canonical ``timestamp`` field).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .constants import UNIVERSE
from .protocol import GNNOutput, GNNOutputEdge, GNNOutputNodeFeature

STUB_MODEL_VERSION = "stub-1"


def _utc_iso(ts: str | None) -> str:
    if ts:
        return ts
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stub_output(timestamp: str | None = None) -> GNNOutput:
    """Build a deterministic stub output for the full universe.

    Every symbol gets ``bias=0.0`` and ``centrality=0.0``; edges are
    empty because the stub does not know the topology. The orchestrator
    treats the stub as "no signal" and the agent prompts remain honest
    about the absence of a real GNN.
    """
    node_features: dict[str, GNNOutputNodeFeature] = {
        sym: GNNOutputNodeFeature(
            bias=0.0,
            centrality=0.0,
            model_version=STUB_MODEL_VERSION,
        )
        for sym in UNIVERSE
    }
    return GNNOutput(
        version="1.0",
        model_version=STUB_MODEL_VERSION,
        topology_version="fixed-1",
        snapshot_id=None,
        timestamp=_utc_iso(timestamp),
        node_features=node_features,
        edges=[],
    )


class StubGNNService:
    """Object-style wrapper matching the future ``GNNService`` interface.

    The Phase 3 orchestrator calls ``service.predict(snapshot_id)`` (or
    ``service.output()`` for the no-snapshot stub). Keeping the
    interface consistent means US3 can swap in the real service without
    changing the orchestrator wiring.
    """

    def __init__(self, model_version: str = STUB_MODEL_VERSION) -> None:
        self.model_version = model_version
        self.architecture = "stub"

    def output(self, timestamp: str | None = None) -> GNNOutput:
        return stub_output(timestamp=timestamp)

    def predict(self, snapshot_id: str) -> GNNOutput:
        # Snapshot is ignored: the stub is topology-free and uses the
        # full universe with zero bias. The snapshot_id is preserved
        # so journal rows from a real + stub mixed timeline can be
        # cross-referenced.
        out = stub_output()
        out.snapshot_id = snapshot_id
        return out


__all__ = ["STUB_MODEL_VERSION", "StubGNNService", "stub_output"]
