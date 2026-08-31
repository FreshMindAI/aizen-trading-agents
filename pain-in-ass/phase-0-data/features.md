# Phase 0 — Data: features

## What we shipped (verified working)

### Storage
- **SQLite** as the single system of record. 14 tables; full schema in `sql/01_schema.sql`. Row counts as of 2026-08-29:
  - `underlying_bars`: 254,939
  - `option_bars`: 195,466
  - `option_contracts`: 14,892
  - `features`: 254,939
  - `labels`: 509,678
  - `ml_training_dataset`: 509,678
  - `gnn_graph_snapshots`: 2
  - `gnn_graph_edges`: 280
  - `gnn_model_artifacts`: 2 (after test cleanups)
  - `gnn_model_evaluations`: 2
  - `decision_journal`: 11
- **Views**: `v_features_underlying_v2`, `v_labels`, `v_asset_correlations`, `v_options_chain_snapshot`, `v_ml_training_dataset`. Views keep the SQL on disk and out of code.

### Ingestion
- `python -m src.download_bars --symbols AAPL,SPY,...` — pulls 30-min underlying bars from `data.alpaca.markets/v2/stocks/bars`.
- `python -m src.download_option_contracts --symbols AAPL,SPY --select-atm --cap 12` — pulls near-the-money contracts from `paper-api.alpaca.markets/v2/options/contracts`.
- `python -m src.download_option_bars --all-selected --days-back 7` — pulls historical option bars.
- All three are idempotent (use `INSERT OR IGNORE` semantics via the `data.run_log` table).

### Universe
- 10 underlyings hard-coded in `src/gnn/constants.py:27-38` and mirrored in `config/gnn.yaml:22-32` and `config/risk.yaml:23`.
- 2 ETFs (SPY, QQQ) are in the GNN graph but marked `is_benchmark=True` / `NON_TRADABLE` — the orchestrator never trades them.
- Option universe is keyed off the same 10 underlyings: 14,892 contracts, balanced 7,446 calls / 7,446 puts.

### Feature engineering
- 14 features per underlying bar, computed in `v_features_underlying_v2`:
  - `return_1, return_4, return_16` (log returns over 1/4/16 bars)
  - `volatility_16` (16-bar rolling std of log returns)
  - `rsi_14, macd_pct, hl_range, atr_pct_14`
  - `ma_dist_20, ma_dist_50` (distance to 20/50-bar moving averages)
  - `volume_ratio_20, vwap_distance`
  - `spy_ret_1, qqq_ret_past_16` (cross-asset)
- All features are NULL-safe and stored as REAL.

### Correlation
- `v_asset_correlations` view computes pairwise rolling 64-bar Pearson correlation between every universe pair, anchored to the latest bar.
- The 64-bar window = 32 hours of 30-min bars ≈ 1.3 trading days. Captures intraday co-movement.

## What works well

- **Idempotency** — running the same pull twice produces the same DB state. This is what makes retraining reproducible.
- **Single source of truth** — SQLite means we can always inspect what the model trained on, no hidden parquet files.
- **Schema versioning** — every "shape-changing" view has a `_v2` suffix; old versions stay around until the new one is validated.

## What doesn't work well

- **Single DB file** — no replication, no point-in-time recovery. We rely on gitignore + manual copies.
- **No incremental refresh for the GNN graph** — every snapshot build reads the full correlation view.
- **No catalog / lineage** — if we change a feature in `v_features_underlying_v2`, we have no automated way to know which downstream model was trained on the old version.
