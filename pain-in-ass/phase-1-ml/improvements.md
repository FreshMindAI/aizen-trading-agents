# Phase 1 — ML: improvements

## Highest impact

### 1. Train on signed return, not bucketed class
- Replace `_binary_targets` with a regression on raw forward return.
- Loss: Huber (robust to outliers) or quantile-pinball loss (for directional quantiles).
- Output: continuous bias in [-1, +1], thresholded at 0 for "up" decisions.
- **Expected lift:** +2-5% ROC-AUC because we use 100% of the data, not 62%.
- **Effort:** 1 day.

### 2. Add temperature scaling / isotonic calibration
- After training, fit a temperature parameter on the validation fold to make the predicted probabilities match observed frequencies.
- **Why it matters:** the orchestrator's threshold is 0.55. If XGB outputs 0.55 but the true positive rate at 0.55 is 0.48, we're trading on miscalibrated probabilities.
- **Effort:** 2 hours.

### 3. Multi-seed bagging
- Train 5 models per (task, horizon) with different seeds, average the probabilities.
- **Effort:** 4 hours (script change), +5× training time (which is 6 min × 5 = 30 min, so fine).
- **Expected lift:** +0.5-1% ROC-AUC, much tighter std.

### 4. Add the GNN's output as a Phase 1 feature
- Currently XGB trains on price/volume features only. Add `gnn_directional_bias` (lagged by 1 bar) as a 21st feature.
- The GNN captures cross-asset information the XGB doesn't see directly.
- **Effort:** 4 hours. **Expected lift:** +1-2% ROC-AUC for direction_h4.

### 5. Per-symbol or sector-blocked walk-forward
- The current walk-forward can have the same symbol in train and test (different timestamps) but never learns "this regime doesn't transfer across symbols".
- A proper purged CV with embargo gaps would be more honest.
- **Effort:** 1 day.

## Medium impact

### 6. Hyperparameter search with Optuna
- Current XGB params are `n_estimators=300, max_depth=6, lr=0.05, subsample=0.8`. These were a single guess.
- Optuna with 50 trials on a 1-year held-out window would find better.
- **Effort:** 1 day.

### 7. Add an "earnings proximity" feature
- Earnings dates explain ~30% of single-day moves. We don't have them.
- Add a `days_to_earnings` feature pulled from a static CSV / API.
- **Effort:** 4 hours.

### 8. Track metrics in a `runs/` table
- Every training run should write its config + metrics to SQLite so we can compare across seeds, dates, and feature versions.
- **Effort:** 1 day.

### 9. Add a "drift" detector on live features
- If the live feature distribution diverges from the training distribution by >3σ, pause the model and re-train.
- **Effort:** 2 days.

## Low impact

### 10. Replace XGBoost with LightGBM
- ~2× faster training, similar accuracy. Nice to have, not needed.
- **Effort:** 1 day.

### 11. Add explainability (SHAP) per prediction
- SHAP values for the top features per prediction, written to the decision_journal.
- **Effort:** 2 days.
- **Hackathon value:** this would make the "creativity" pillar very strong.

### 12. Catboost as a third ensemble member
- Different inductive bias (ordered boosting) → diversity → better ensemble.
- **Effort:** 2 days.
