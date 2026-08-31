# Phase 2 — GNN: features

## What we shipped (verified working)

### Architecture
- `StockGNN` (`src/gnn/model.py:90`) — GCNConv(14 → 32) → ReLU → GCNConv(32 → 16) → ReLU → Linear(16 → 1) (logit head) + Linear(16 → 1) (centrality head).
- `MODEL_REGISTRY` has 3 entries: `gcn-32-16-1` (default), `gat-32-16-1`, `sage-32-16-1`. The other two are placeholders.
- Output: per-node logit + per-node centrality score, both via `forward_with_centrality(x, edge_index)`.

### Graph builder
- `build_edge_features.py` produces directed edges from 5 sources: sector, supplier, customer, etf_membership (all static, weight=1.0) and correlation (data-driven, weight=|corr|, threshold=0.5).
- `build_node_features.py` produces per-node feature vectors from `v_features_underlying_v2`, 14 features per node.
- `build_payload(conn, timestamp)` returns the full snapshot dict, byte-deterministic.

### Snapshot store
- 2 rows in `gnn_graph_snapshots`, 280 rows in `gnn_graph_edges`. Each snapshot is keyed by `(timestamp, topology_version)` and the full payload is stored in `payload_json` for replay.
- Idempotent: re-running the same build is a no-op (`INSERT OR REPLACE`).

### Training
- `train.py::train_model()` is the main entry. Reads config from `config/gnn.yaml`, builds train/val/test splits chronologically, runs up to 50 epochs with early stopping on val ROC-AUC (patience=10).
- Seeds: `torch=1337, numpy=1337, python_hash=1337` (single seed, deterministic).
- Persists: `{prefix}-{YYYYMMDD}-{NNNN}.pt` + `.meta.json` + a row in `gnn_model_artifacts` + a row in `gnn_model_evaluations`.
- 4 GNN artifacts on disk: `gnn-20260829-0001` through `gnn-20260829-0004`.

### Inference
- `GNNService.load_latest(conn)` returns the most-recent artifact.
- `GNNService.predict(snapshot_id)` runs the model on a given snapshot, returns a `GNNOutput` with version="1.0", per-symbol `node_features` (bias, centrality), and the edge list.
- Used by `src/agents/inference.py:131` (`gnn_output()`) which is called by every agent cycle.

### Evaluation
- `evaluate.py::evaluate_model()` runs the trained model on the test split, writes `models/report_phase2_gnn.json` with side-by-side XGB vs GNN metrics.
- The current report shows GNN at 0.4579 ROC-AUC, XGB baseline at 0.6830 ROC-AUC.

### Protocol contracts
- JSON schemas in `contracts/gnn_input.schema.json` and `contracts/gnn_output.schema.json`. Pydantic v2 models in `src/gnn/protocol.py`.
- All agent → GNN messages pass through these contracts.

## What works well
- **Deterministic builds** — same DB state + same timestamp + same topology_version = identical `payload_json` (byte-for-byte).
- **GNN service has a stub fallback** — `StubGNNService` returns `{"version": "1.0", "model_version": "stub-1", "node_features": {sym: {bias: 0, centrality: 0.5}}, ...}`. This means agents work even when no artifact has been trained.
- **Leak guard** — the SQL in `build_node_features` and `build_edge_features` enforces `source_timestamp <= snapshot.timestamp`. No look-ahead.
- **The Colab notebook works** — we can retrain on a fresh GPU in <10 min.

## What doesn't work well
- **The GNN is structurally over-parameterized for the data we feed it.** 11 nodes × 50 snapshots = 550 examples for 1,800 parameters.
- **The "centrality" head is unused.** It's a self-supervised pretext task that we set up but never wired downstream.
- **No attention.** The GCN treats all edges equally. Switching to GAT would let the model learn which edges matter.
- **Topology is not dynamic** in any meaningful sense. Only one edge type is recomputed.
- **No graph-level loss.** We only have node-level BCE.
