# Phase 1 — ML (XGBoost): features

## What we shipped (verified working)

### Models on disk
- 8 XGBoost artifacts in `models/`, all from 2026-08-29:
  - `direction_h4_xgb_clf-20260829-152556.pkl` — binary up/down at h=4
  - `direction_h16_xgb_clf-20260829-125959.pkl` — same, h=16
  - `option_h4_xgb_clf-20260829-152605.pkl` + `_reg-20260829-152607.pkl` — profitable classifier + return regressor
  - `option_h16_xgb_clf-20260829-130037.pkl` + `_reg-20260829-130039.pkl`
  - `rv_h4_xgb_reg-20260829-152611.pkl` — realized vol regressor
  - `rv_h16_xgb_reg-20260829-130050.pkl`

### Headline metrics (3-fold walk-forward)

| Task | h | Test ROC-AUC | Test PR-AUC | Spearman ρ |
|---|---|---|---|---|
| **direction** | 4 | 0.6304 ± 0.0075 | 0.3740 ± 0.0086 | — |
| direction | 16 | 0.62 (estimated, single fold) | — | — |
| option | 4 | 0.5448 ± 0.0147 | 0.4293 ± 0.0185 | — |
| option | 16 | ~0.55 | — | — |
| **rv (regression)** | 4 | MAE 0.00136 ± 0.00006 | — | **0.6227 ± 0.0132** |
| **rv (regression)** | 16 | MAE 0.00143 ± 0.00009 | — | **0.6471 ± 0.0138** |

### What works well
- **Direction h=4 is the production signal** — 0.63 AUC, low fold variance, beats the 0.55 orchestrator gate cleanly.
- **rv regressor is strong** — Spearman 0.62-0.65, very low MAE. This is what the volatility agent uses.
- **Chronological splits are correct** — no look-ahead, no shuffle. The protocol is in `src/ml/walkforward.py`.
- **Feature store is versioned** — `v_features_underlying_v2` is the single source for the 20 features.

### What doesn't work well
- **Option model is borderline** — 0.5448 ROC-AUC, within 1 std of the random baseline. The h=4 horizon doesn't match option economics.
- **No ensemble** — single-seed XGBoost. We have the compute budget for 5-seed bagging, never did it.
- **The orchestrator's gate is XGBoost-only** — `direction_prob_min: 0.55` blocks trades when XGB says "neutral" even if the GNN is highly confident.
- **No live online learning** — we retrain on a fixed schedule, not on streaming feedback.
- **No calibration** — XGBoost probabilities are not temperature-scaled. They're used raw by the orchestrator.

### Metrics reports
- `models/report_direction_h4.json` — direction at h=4 (mostly null, see failures #1 above)
- `models/report_option_h4.json` — option h=4 with all baselines (zero_naive, logreg, xgb, ret_xgb)
- `models/report_rv_h4.json`, `models/report_rv_h16.json`
- `models/walkforward_*.json` — 3 expanding-window folds per task
