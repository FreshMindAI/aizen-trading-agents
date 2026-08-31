# ML Inference Root Cause — Why the Supervisor Sees `direction_probability=0.5`

**Date:** 2026-08-29
**Symptom:** Multi-agent pipeline produces NO_TRADE on every cycle. Direction agent's
`direction_probability=0.5` (stub value), `model_version="stub"`, `timestamp="9999-12-31T23:59:59Z"`.
GNN runs correctly with the new GATv2 model (`gnn_directional_bias` populated for all
10 symbols: SPY 0.83, QQQ 0.76, AMD 0.69, etc.) but the supervisor's
`direction_prob_min=0.55` gate fires because every underlying has prob=0.5 < 0.55.

## Root cause

The ML inference path is broken at the SQL layer. The `inference.py` loader at
`src/agents/inference.py:49-72` runs:

```sql
SELECT f.symbol, f.timestamp,
       COALESCE(d.direction_probability, 0.5)   AS direction_probability,
       COALESCE(d.predicted_future_realized_vol, NULL) AS predicted_future_realized_vol,
       COALESCE(d.gnn_directional_bias, 0.0)    AS gnn_directional_bias,
       COALESCE(d.gnn_centrality, 0.0)          AS gnn_centrality,
       d.model_version
FROM   v_features_underlying_v2 f
LEFT JOIN (
    SELECT u.symbol, u.timestamp,
           u.direction_probability,           --  ❌ not in ml_training_dataset
           u.predicted_future_realized_vol,   --  ❌ not in ml_training_dataset
           NULL AS gnn_directional_bias,
           NULL AS gnn_centrality,
           u.model_version,                   --  ❌ not in ml_training_dataset
           ROW_NUMBER() OVER (PARTITION BY u.symbol ORDER BY u.timestamp DESC) rn
    FROM   ml_training_dataset u
    WHERE  u.timestamp <= ?
) d ON d.symbol = f.symbol AND d.rn = 1
WHERE  f.timestamp <= ?
```

`ml_training_dataset` schema (verified via `PRAGMA table_info`):

| Column | Notes |
|---|---|
| `symbol`, `timestamp` | OK |
| `feature_set`, `horizon_bars` | OK |
| `return_1`, `return_4`, `return_16`, `volatility_16`, `rsi_14` | v1 features |
| `macd`, `macd_signal`, `macd_hist`, `atr_14` | v1 features |
| `volume_ratio_20`, `vwap_distance` | v1 features |
| `n_contracts`, `avg_moneyness`, `avg_option_return_1`, `avg_option_volatility_16`, `min_days_to_expiry` | option-side features |
| `future_return`, `future_realized_vol`, `target_class` | **forward labels** (correctly excluded from the inference path) |
| ❌ `direction_probability` | **does not exist** |
| ❌ `predicted_future_realized_vol` | **does not exist** |
| ❌ `model_version` | **does not exist** |

`v_features_underlying_v2` view (`sql/14_view_features_underlying_v2.sql`) computes
the v2 features (`hl_range`, `co_return`, `ma_dist_20`, `ma_dist_50`, `atr_pct_14`,
`macd_pct`, market context) ON THE FLY from `v_features_underlying` and
`underlying_bars`. The trained XGBoost models in `models/*.pkl` were fit on the
**v2 feature set** (confirmed by reading `direction_h16_xgb_clf-20260829-125959.meta.json`).

So the design intent was:
- `ml_training_dataset` would have the **features used at training time**
  (i.e. v2 names) AND the **predictions** written back for the inference path.
- The inference path would `SELECT direction_probability, predicted_future_realized_vol`
  from the latest row per symbol.

What actually happened:
- The `ml_training_dataset` table is a *materialized feature+label table* but
  only with **v1 features** and **labels**. There is no write-back step that
  populates prediction columns. The training pipeline reads features, fits a
  model, writes the model to `models/*.pkl`, and stops. Nothing writes
  predictions back into the table.
- The inference SQL was written *as if* the write-back step existed.

Result: `SELECT u.direction_probability FROM ml_training_dataset u` raises
`sqlite3.OperationalError: no such column: u.direction_probability`. The
inference loader at `src/agents/inference.py:324` catches that in a
bare `except Exception` and falls back to a stub universe. The stub sets
`direction_probability=0.5, predicted_future_realized_vol=0.20,
model_version="stub", timestamp="9999-12-31T23:59:59Z"`.

That's why every cycle is NO_TRADE: the supervisor gates at 0.55, every
candidate sits at 0.5.

## Why GNN works but ML doesn't

The GNN path uses a different loader (`_load_gnn_output` and `GNNService`).
It loads the trained `*.pt` artifact from disk on the **first** inference
call, runs the model forward on the **freshly-built** graph snapshot, and
populates `gnn_directional_bias` and `gnn_centrality` for every node. No
SQL prediction-table lookup is needed because the GNN service runs the
model at inference time.

The ML path was architected as a "precompute at training, lookup at
inference" pipeline. The precompute step was never wired up.

## Fix options (pick one)

### Option A — Run the model at inference time (recommended)
- Change `_LATEST_UNDERLYING_SQL` to drop the broken subquery. Keep the
  `v_features_underlying_v2` left side.
- Load the trained XGBoost artifacts in `InferenceService.__init__` (lazy,
  once per process). The artifacts in `models/direction_h*_xgb_clf-*.pkl`
  and `rv_h*_xgb_reg-*.pkl` are already trained and loadable — verified by
  running `python -m src.ml.predict` on Aug 17 12:30 PM ET earlier.
- For each (symbol, timestamp) row in the snapshot, build the v2 feature
  vector in the right column order, call the XGBoost `predict_proba` /
  `predict` method, and write the result into the `UnderlyingScore`.
- Pros: matches the GNN pattern; one source of truth (the trained model);
  no write-back table to keep in sync.
- Cons: inference time grows by O(symbols * features) per cycle — still
  sub-millisecond for 10 symbols × 14 features.

### Option B — Write predictions back at training time
- Add a write-back step at the end of every ML training run that calls
  `model.predict_proba` on the training rows and inserts the results into
  a new `ml_predictions` table (or a few columns on `ml_training_dataset`).
- Update `_LATEST_UNDERLYING_SQL` to read from the new table.
- Pros: keeps the precompute/lookup pattern; fits if we later add new
  models.
- Cons: doubles the storage; the predictions are stale until the next
  training run (15-min cadence here, so not a problem in practice).

### Option C — Change the training pipeline to write the v2 features into ml_training_dataset AND predictions
- Same as Option B but also migrate `ml_training_dataset` to v2 column
  names so the training and inference pipelines share one schema.
- Pros: closes the schema gap permanently.
- Cons: requires a backfill migration; existing models may not be
  retrainable from the v2 materialized table without re-running the
  feature pipeline on the full 509k rows.

## Recommendation

**Option A** for the v1 inference path. It is the smallest correct change
that makes the existing trained models actually drive the supervisor.
The precompute path (Option B/C) is the right design for a multi-model
production system with strict latency requirements, but adds storage
and migration cost we don't need for a 10-symbol, 15-min-cadence
system.

## Implementation outline (Option A)

1. Add a small loader in `src/agents/inference.py` (or a sibling
   `src/ml/loader.py`) that:
   - Reads `models/direction_h4_xgb_clf-*.meta.json` (latest mtime) →
     `feature_names`, `impute_medians`, `model_path`.
   - Calls `joblib.load(model_path)` to get the XGBClassifier.
   - Caches the loaded model on the `InferenceService` instance.
   - Same for `direction_h16_*`, `rv_h4_*`, `rv_h16_*`,
     `option_h4_*`, `option_h16_*` (whichever the inference path needs).
2. Replace the broken subquery in `_LATEST_UNDERLYING_SQL` with a
   CTE that pulls the v2 feature columns in the order the model
   expects, then in Python (after the SQL fetch) call
   `model.predict_proba(features_df)` and merge the probabilities
   back into the row dicts.
3. Drop the `COALESCE(..., 0.5)` fallbacks and the `except Exception`
   that swallows the SQL error. Add a test that asserts the
   `model_version` field on the `UnderlyingScore` is the real artifact
   name (e.g. `direction_h4_xgb_clf-20260829-125959`), not `"stub"`.
4. The supervisor's `direction_prob_min=0.55` gate should now fire
   only when the model is genuinely uncertain, not because the SQL
   silently returned 0.5.

## What this changes downstream

- `direction_probability` becomes a real probability in [0, 1] from
  the trained XGBoost. PROCEED/NO_TRADE decisions will start firing
  on actual signal instead of always being NO_TRADE.
- `predicted_future_realized_vol` becomes a real regression output.
- `model_version` is the real artifact name, so we can trace any
  decision back to the exact model that made it.
- `timestamp` is the real as-of timestamp of the feature row, not
  the 9999-12-31 sentinel.

## Validation

1. `python -m src.agents.run --once --mode dry-run --log-level INFO`
   should produce a `direction_probability` per symbol that varies
   across symbols and is close to the model's training behavior.
2. Inspect the latest `decision_journal` row's `ml_prediction_json`
   and confirm `model_version` is the real artifact name.
3. Re-run the Aug 17 12:30 PM ET sim. If the supervisor still
   NO_TRADEs, the issue is the supervisor's threshold, not the ML
   inference.
4. Run the backtester (`python -m src.agents.cli.backtest
   --start 2026-08-04 --end 2026-08-29 --interval weekly`) and
   confirm `n_proceed > 0` and `hit_rate_h4` is meaningful.
