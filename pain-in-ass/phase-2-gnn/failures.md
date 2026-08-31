# Phase 2 — GNN: failures

## What broke

### 1. `evaluate.py` "tuple indices must be integers, not 'str'"
- **Symptom:** `evaluate_model(conn, ...)` crashed the first time it was called with a raw `sqlite3.connect()` because the connection didn't have `row_factory = sqlite3.Row` set.
- **Root cause:** the helper `_xgb_baseline_metrics` indexes rows by `r["target_class"]`. The connection passed in had the default tuple row factory.
- **Fix:** wrapped the whole `evaluate_model` body in `try/finally` that sets `conn.row_factory = sqlite3.Row` and restores on exit. Same pattern applied to `GNNService.load_latest` and `_resolve_snapshot_id`.
- **Cost:** 3 module fixes, all the same pattern. We should have had a `connect_readonly()` helper that sets the row factory at the boundary.

### 2. `gnn_model_artifacts` is empty after running `tests/test_gnn_orchestrator.py`
- **Symptom:** every test run DELETEs from the production `data/trading.db`:
  ```python
  real_conn.execute("DELETE FROM gnn_model_artifacts")
  real_conn.execute("DELETE FROM gnn_graph_snapshots")
  real_conn.execute("DELETE FROM gnn_graph_edges")
  ```
  The test never restores the originals. After the first pytest run we had 0 GNN artifacts.
- **Root cause:** the test uses a shared DB connection for setup + assertions. The "fixture" pattern is wrong.
- **Fix needed:** either copy the production DB to `tests/tmp_*.db` for the test, or use a transaction with rollback. **Not done yet** — every test run still clobbers.
- **Cost:** one full re-train + snapshot build was needed today after the test ran during the smoke test.

### 3. Standalone GNN ROC-AUC = 0.4579 (below random!)
- **Symptom:** the latest `models/report_phase2_gnn.json` shows the GNN at 0.4579 ROC-AUC vs the XGB baseline at 0.6830.
- **Root cause:** the GNN is being asked to do *binary classification* on a tiny test slice (50 snapshots × 10 nodes = 500 samples) with class-imbalanced targets (only the `target_class != 0` rows are kept). On the small val/test slice, the model's sigmoid outputs are essentially noise and the threshold is miscalibrated.
- **Cost:** the GNN's *standalone* test metric is below random. We still ship it because the orchestrator uses it as a *signed bias* (not a probability), where the relative ordering is what matters. But the metric we publish is bad.

### 4. The "dynamic topology" was a lie
- **Symptom:** the spec called for a dynamic graph topology; we shipped a static graph with one data-driven edge type.
- **Root cause:** `config/gnn.yaml:17` hard-codes `topology_version: "fixed-1"`. The `build_edge_features` module produces correlation edges (data-driven) and 4 static edge types (sector, supplier, customer, etf_membership). The static ones never change.
- **Cost:** the GNN trains on the same structural graph every day, with only the correlation subgraph slowly drifting. The "dynamic" claim in the architecture doc is misleading.

### 5. The 50-snapshot training cap
- **Symptom:** `_select_timestamps(conn, n_snapshots=50)` returns the 50 most recent timestamps where at least 6 symbols have a non-flat label. With 30-min bars over 3.5 years, that's ~10K possible daily snapshots; we use the freshest 50.
- **Root cause:** n_snapshots=50 was a smoke-test default that became the production default.
- **Cost:** the model trains on a tiny fraction of the available signal. Bumping to 250-500 is the single biggest expected lift.

### 6. Edge weights are computed but ignored
- **Symptom:** every edge has a `weight` field (correlation magnitude for correlation edges, 1.0 for the static types). The GCN layer in `model.py` does not consume edge weights — GCNConv only takes (x, edge_index).
- **Root cause:** PyG's `GCNConv` doesn't support edge weights. The alternative is `GATConv` (attention) or `TransformerConv`.
- **Cost:** the entire correlation-magnitude signal is wasted.

### 7. No centrality head actually contributes
- **Symptom:** `forward_with_centrality` returns `(logit, centrality)` where centrality is a learned per-node scalar in [0, 1]. But the only consumer is the GNN service, which stores the value in the output JSON. Nothing downstream uses it for decision-making.
- **Cost:** we trained a head we never use. Either wire it into the orchestrator or drop it.

### 8. The notebook `notebooks/phase2_train_on_colab.ipynb` is a smoke test
- **Symptom:** the Colab notebook has 7 cells and 7 code cells. It retrains on `--epochs 100` with the production data. It works but it's not a serious training pipeline.
- **Why it matters:** the Colab path is the "scale up" story. We didn't actually scale up — we use the same 50-snapshot cap.
