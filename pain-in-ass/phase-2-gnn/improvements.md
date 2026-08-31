# Phase 2 — GNN: improvements

## Highest impact

### 1. Train on more snapshots (250-500)
- Single change: bump `n_snapshots` in `config/gnn.yaml` and `_select_timestamps`.
- Use every daily bar where ≥ 60% of universe symbols have a non-flat h=16 label.
- **Expected lift:** +5-10% ROC-AUC. The model is data-starved.
- **Effort:** 30 min config + 1 hour retraining.

### 2. Switch to GAT (`gat-32-16-1`)
- The architecture is already registered. Just change `model.architecture` in `config/gnn.yaml`.
- GAT learns per-edge attention weights, which lets the model use the correlation magnitudes that GCN currently ignores.
- **Expected lift:** +2-5% ROC-AUC.
- **Effort:** 1 hour (config change + retrain).

### 3. Train on signed return as regression
- Replace `_binary_targets` (which maps {-1, 0, +1} → {0, 0.5, 1}) with a continuous target = forward return.
- Loss: Huber with delta=0.005 (covers ±0.5% moves).
- Output: continuous bias in [-1, +1], sigmoid not needed.
- **Expected lift:** +3-7% ROC-AUC because we use 100% of the data, not 62%.
- **Effort:** 1 day.

### 4. Make the topology actually dynamic
- Add a `topology_version: regime-on-1` and `topology-version: regime-off-1` snapshot family.
- When the regime agent is "risk-off", use `regime-on-1` (cross-asset correlation edges, no sector edges).
- When "risk-on", use `regime-off-1` (sector + supply-chain edges emphasized, correlation threshold raised to 0.7).
- Train one GNN per topology, ensemble at inference.
- **Effort:** 3 days. **Hackathon value:** the "creativity" pillar.

### 5. Wire the centrality head into the orchestrator
- Right now `gnn_centrality` is computed but unused.
- Use it as a portfolio risk signal: if NVDA has high centrality, treat it as a "fragility" indicator and reduce position size.
- **Effort:** 1 day.

## Medium impact

### 6. Multi-seed bagging
- Train 5 GNNs with different seeds, average the logits.
- **Effort:** 4 hours. **Expected lift:** +1-2% ROC-AUC + tighter std.

### 7. Edge dropout during training
- Randomly drop 20% of edges per epoch. Forces the model to not over-rely on any one edge type.
- **Effort:** 4 hours.

### 8. Add per-edge time decay
- The current correlation edges are `|corr| > threshold` over the last 64 bars. Weight recent correlations higher (exponential decay).
- **Effort:** 4 hours.

### 9. Node-level temporal features
- Add a 4-bar momentum delta, intraday range percentile, sector-relative return to the node features.
- **Effort:** 1 day.

### 10. Graph-level readout loss
- Add an auxiliary loss that predicts the *average* direction across the universe from the pooled graph embedding.
- This is a regularizer that should help when node-level labels are noisy.
- **Effort:** 1 day.

## Low impact

### 11. Drop `_xgb_baseline_metrics` from `evaluate.py` and replace with the real Phase 1 model
- The current XGB baseline is a hand-rolled logistic regression on 14 features. The real Phase 1 XGBoost is much stronger (0.6830 vs the logistic's ~0.55). Use the real model in the side-by-side report.
- **Effort:** 2 hours.

### 12. Move graph builds to a background job
- Right now snapshot builds are synchronous and take ~30s each. For a 250-snapshot training run that's 2 hours just on graph builds.
- **Effort:** 1 day.

### 13. Save intermediate activations for explainability
- Store the per-node 16-dim embedding after conv2 for the latest snapshot. Use it for SHAP-like analysis.
- **Effort:** 1 day.

### 14. Add a `GraphSAGE` (inductive) variant
- Right now the model is *transductive* — it can only score the 11 nodes it trained on. GraphSAGE generalizes to new nodes.
- For the hackathon this doesn't matter (universe is fixed), but it's a v2 feature.
- **Effort:** 2 days.
