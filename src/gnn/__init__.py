"""Phase 2 GNN Trading Model.

Top-level package for the Graph Neural Network layer. Public entry points
are imported lazily by the orchestrator; the ``__main__`` block exposes
a quick debug helper (T035).

Public surface (kept small until US2/US3 land):

- :data:`UNIVERSE` - the 10-underlying + SPY benchmark universe.
- :func:`stub_output` - deterministic stub used by FR8.3 / SC5.
- :data:`TOPOLOGY_VERSION` - the fixed-1 constant; reserved interface for
  the future dynamic-topology controller.

Implementation modules (importable as a library, per constitution VI):

- :mod:`src.gnn.constants` - SECTOR_MAP, SUPPLY_CHAIN, ETF_HOLDINGS, UNIVERSE.
- :mod:`src.gnn.protocol` - Pydantic mirrors of the two JSON-Schema contracts.
- :mod:`src.gnn.model` - StockGNN and the MODEL_REGISTRY.
- :mod:`src.gnn.stub` - StubGNNService for the FR8.3 fallback path.
- :mod:`src.gnn.build_snapshot` - CLI + builder (Phase 3 / US1).
- :mod:`src.gnn.dataset` - PyG InMemoryDataset wrapper (US2).
- :mod:`src.gnn.train` - Training CLI (US2).
- :mod:`src.gnn.evaluate` - Evaluation + JSON report (US2).
- :mod:`src.gnn.service` - GNNService: load + predict (US2).
"""

from __future__ import annotations

from .constants import UNIVERSE
from .protocol import GNNOutput, GNNOutputNodeFeature, GNNOutputEdge
from .stub import StubGNNService, stub_output

TOPOLOGY_VERSION = "fixed-1"

__all__ = [
    "TOPOLOGY_VERSION",
    "UNIVERSE",
    "GNNOutput",
    "GNNOutputNodeFeature",
    "GNNOutputEdge",
    "StubGNNService",
    "stub_output",
]


def main() -> int:
    """Smoke helper: print the resolved topology_version and stub fingerprint.

    Useful for verifying the install is wired correctly (T035) without
    having to start the full orchestrator. Exit code is always 0.
    """
    out = stub_output(timestamp="1970-01-01T00:00:00Z")
    print(f"topology_version: {TOPOLOGY_VERSION}")
    print(f"stub_model_version: {out.model_version}")
    print(f"universe_size: {len(UNIVERSE)} (incl. SPY benchmark)")
    print(f"contract_version: {out.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
