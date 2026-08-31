"""Phase 2 GNN constants.

The graph builder uses three hard-coded dictionaries as the source of
truth for non-data-driven edges. These are auditable in source and never
reach the LLM prompt - they are pure Python data, not free text.

Corpus policy
-------------
- Every constant in this module is a tuple of strings so iteration order
  is stable and snapshot construction is byte-deterministic (spec SC1).
- The sector / supply-chain / ETF dictionaries are exhaustive over the
  10-underlying + SPY universe; correlation edges are the only
  data-driven edge and live in :mod:`src.gnn.build_edge_features`.
- All keys and values are upper-case tickers; case is normalized at the
  builder boundary.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# 10 underlyings from Phase 1 + a SPY benchmark node. SPY is the benchmark
# only; the orchestrator does not trade it. The order here is the canonical
# snapshot-node order, which is also the column order of the node-feature
# matrix.
UNIVERSE: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
)
BENCHMARK: str = "SPY"

# Symbols that exist in the universe but are not eligible for trade. The
# orchestrator never builds an OrderIntent for a benchmark.
NON_TRADABLE: frozenset[str] = frozenset({"SPY", "QQQ"})

# ---------------------------------------------------------------------------
# Sectors (GICS-aligned at the high-level granularity that matters for
# correlation / co-movement). Anything in the universe not listed here
# falls into "Other".
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    "SPY":   "Index",
    "QQQ":   "Index",
    "AAPL":  "Tech",
    "MSFT":  "Tech",
    "NVDA":  "Tech",
    "AMZN":  "Tech",
    "META":  "Tech",
    "GOOGL": "Tech",
    "TSLA":  "Auto",
    "AMD":   "Tech",
}

# ---------------------------------------------------------------------------
# Supplier relationships: source_symbol supplies target_symbol.
# (Hard-coded from public knowledge; auditable here.)
# ---------------------------------------------------------------------------
SUPPLY_CHAIN: tuple[tuple[str, str], ...] = (
    # NVDA supplies data-center GPUs to the hyperscalers
    ("NVDA", "MSFT"),
    ("NVDA", "GOOGL"),
    ("NVDA", "META"),
    ("NVDA", "AMZN"),
    # NVDA / AMD cross-license IP; represented as a supplier relationship
    ("NVDA", "AMD"),
)

# ---------------------------------------------------------------------------
# Customer relationships: source_symbol buys from target_symbol. The
# reverse view of SUPPLY_CHAIN. Kept explicit so both directions are
# queryable from the edge table.
# ---------------------------------------------------------------------------
CUSTOMER_CHAIN: tuple[tuple[str, str], ...] = (
    ("MSFT",  "NVDA"),
    ("GOOGL", "NVDA"),
    ("META",  "NVDA"),
    ("AMZN",  "NVDA"),
    ("AMD",   "NVDA"),
)

# ---------------------------------------------------------------------------
# ETF membership: pairs of symbols that share an ETF basket. Only SPY and
# QQQ are relevant for the 10-underlying universe.
# ---------------------------------------------------------------------------
ETF_HOLDINGS: dict[str, tuple[str, ...]] = {
    "SPY": (
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    ),
    "QQQ": (
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    ),
}

# ---------------------------------------------------------------------------
# Edge reasons (enum values for the `reason` column in gnn_graph_edges
# and the reason property in contracts/gnn_output.schema.json).
# ---------------------------------------------------------------------------
EDGE_REASONS: tuple[str, ...] = (
    "sector",
    "supplier",
    "customer",
    "etf_membership",
    "correlation",
    "rolling_corr",
)

# ---------------------------------------------------------------------------
# Correlation edge defaults (mirrors config/gnn.yaml but used at builder
# time so the constants module has zero import-time config dependency).
# ---------------------------------------------------------------------------
CORRELATION_THRESHOLD: float = 0.5
CORRELATION_WINDOW_BARS: int = 64


def is_tradable(symbol: str) -> bool:
    """True for symbols the orchestrator may emit an OrderIntent against."""
    return symbol.upper() in UNIVERSE and symbol.upper() not in NON_TRADABLE
