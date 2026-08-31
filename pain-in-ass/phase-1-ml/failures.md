# Phase 1 — ML (XGBoost): failures

## What broke

### 1. The XGBoost feature count mismatch (caught today, 2026-08-29)
- **Symptom:** First attempt to re-score the GNN's nodes with the Phase 1 XGB model crashed with "20 features expected, 14 provided".
- **Root cause:** The 14 GNN features are a *subset* of the 20 Phase 1 features. We were using the 14-feature list instead of reading `meta.json` for the actual feature list.
- **Fix:** read `meta.json["feature_names"]` at inference time and use that. Already in `src/agents/inference.py:205`.
- **Cost:** one cycle of confusion, no production impact.

### 2. `flat_band: 0.0025` silently discards 38% of labels
- **Symptom:** For each (symbol, timestamp) pair, we compute the forward return and bucket it into {-1, 0, +1} with a ±0.25% flat band. The middle bucket is dropped during training.
- **Root cause:** the `target_class != 0` filter in `_select_timestamps` and `GNNGraphDataset`.
- **Cost:** we lose the most informative training signal — the *magnitude* of small moves. A regression target on raw returns would use all 100% of the data.

### 3. XGBoost training is single-seed
- **Symptom:** the `seeds` block in `config/gnn.yaml` and the equivalent for Phase 1 training pins `torch_seed=1337, numpy_seed=1337`. We never run an ensemble across seeds.
- **Cost:** reported metrics have unknown seed-sensitivity. A 5-seed mean would give us a real confidence interval on each metric.

### 4. Walk-forward uses 3 folds, not the 4 we initially planned
- The first walk-forward runs we did had 4 expanding-window folds. The current `models/walkforward_*.json` files have 3 folds. We changed the protocol mid-stream and didn't update the doc.
- **Why it matters:** 3 folds is a thin sample for std estimates. The 0.0075 std on direction-h4 is probably too small.

### 5. `horizon_bars=16` for the option model means "2 trading days out"
- We picked 16 because it matches the GNN's default. But options expire in calendar days, not bars. A 16-bar horizon (≈ 2 trading days) for a 30-DTE option is mostly noise.
- **Cost:** the option model's 0.5448 ROC-AUC is borderline because we're predicting over a horizon that doesn't match the contract's economic life.

### 6. Spearman correlation for `rv_h4` is `nan` for the zero-naive baseline
- `models/report_option_h4.json` shows `zero_naive spearman_rho=nan` because when the test labels are all-zero in a fold, the rank correlation is undefined.
- We handle it with a try/except in `stats.spearmanr(...)` but the warning still fires. Cosmetic, not a bug.

### 7. The `_select_timestamps` filter `HAVING n_sym >= 6` was tuned for the 11-node graph
- When we add 5 more underlyings for the hackathon, the threshold may be too loose (we'd accept timestamps where 6 of 15 symbols are labelled) or too tight (if labels are sparse, we'd reject all timestamps).
- **Fix:** re-tune as a fraction (`>= 0.6 * n_universe`).

### 8. We never validated the Phase 1 models out-of-sample
- The walk-forward reports are chronological but the splits are not blocked by symbol. If AAPL dominates fold 1, the model overfits to AAPL's regime.
- **Cost:** fold-mean metrics may overstate true out-of-sample by 1-3% ROC-AUC.
