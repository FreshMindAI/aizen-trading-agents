# DATASETS.md - Why the database looks like this, and what's inside it

Companion to README.md. Two halves: **(A)** the reasoning behind each dataset,
**(B)** a concrete catalog of every table/view with live row counts.

---

## A. Reasoning behind the dataset creation

### A.1 The goal this data serves

Train the Phase-1 ML models of *Options Alpha Agents*: predict short-horizon
direction/magnitude moves of 10 liquid US equities, optionally enriched with
options-market context. Phase 2 (GNN) additionally consumes cross-asset
correlation structure. Everything in the database exists to feed that, or to
make it reproducible/auditable.

### A.2 Governing principles (from the spec doc)

| Principle | Why |
|---|---|
| SQLite as local system of record | Free-first, zero-infra research setup; one file, easy snapshots |
| Raw observations are **immutable** (`INSERT OR IGNORE` only) | Re-downloads can never corrupt or silently rewrite history; duplicate detection falls out of PK skip counts |
| All analytics are **derived** (SQL views) | Features/labels stay recomputable and tunable (e.g. change the flat band, rebuild) without ever touching raw data |
| Every pull logged in `data_runs` | Each dataset is traceable to endpoint, params, date range, feed, and row counts |

### A.3 Why each raw dataset exists

- **`underlying_bars`** - the primary signal. 15-minute bars because the doc's
  first experiment (§4.1) specifies 15Min; `feed=iex` is the free tier (recorded
  per row); `adjustment=split` because unadjusted prices would plant fake
  return spikes at split dates (e.g. NVDA's 2024 10:1) and poison every
  return/volatility feature.
- **`symbols`** - registry of the tradable universe (SPY, QQQ, AAPL, MSFT,
  NVDA, AMZN, META, GOOGL, TSLA, AMD - chosen for liquidity + deep option
  chains).
- **`option_contracts`** - defines which option instruments exist and were
  selectable. Fetched `status=active` inside an expiry window (7-550 DTE) and a
  ±10% strike band around spot so SPY/QQQ's thousands of strikes stay within
  request budget.
- **`contract_selection`** - the exact, reproducible pick list (with selection
  spot price and run_id) that option-bar pulls follow. Without it, "why did we
  download THESE 300 contracts?" would be unanswerable.
- **`option_bars`** - option price paths powering option-context features
  (moneyness premium, option-side volatility, option-return aggregates).
  Uses the free delayed `indicative` feed; recorded per row/run.
- **`option_quotes`, `option_snapshots`** - deliberately EMPTY landing zones.
  They belong to the live-agent phase (doc §4.5-4.6). Kept in schema from day
  one so later phases have a home; training must never mix current snapshots
  into past observations anyway.

### A.4 Why the derived datasets look the way they do

- **`features` / `v_features_underlying`** (`feature_set='u15m_v1'`) -
  equity feature vector matching the doc §8 column contract: multi-horizon
  returns (1/4/16 bars), realized vol (16), Cutler RSI(14), SMA-MACD(12/26/9),
  ATR(14), volume ratio(20), VWAP distance. Indicators use SMA/Cutler variants
  on purpose: SQLite window functions can't express recursive EMA/Wilder
  smoothing without fragile chained CTEs, and rankings at 15-min horizons are
  near-identical.
- **`labels` / `v_labels`** - supervision targets:
  - Horizons **4 and 16 bars** (= 1h and 4h on the 15-min grid): one intraday
    regime, one multi-hour regime.
  - **3-class target** with a **±0.25% flat band** (from `v_params`,
    editable in SQL): raw 15-min returns are mostly noise; without a dead-zone
    the classifier learns coin flips. `future_return` and
    `future_realized_vol` stay available for regression heads.
  - **Leak guard**: labels whose forward window extends past available data do
    not exist (`LEAD(close,h)` IS NULL filtered out) - no row can peek into
    the future, and train/test splits can't leak through missing tails.
- **`ml_training_dataset` / `v_ml_training_dataset`** - the single flat SELECT
  models consume: equity features ⋈ labels LEFT JOIN option-market context
  (n_contracts, avg moneyness, avg option return/vol, min DTE per
  symbol-timestamp). Context is LEFT-joined so equity-only training still works
  where option prints are missing.
- **`asset_correlations` / `v_asset_correlations`** - rolling 64-bar pairwise
  correlation/covariance of bar returns: future GNN edge weights. Expensive
  (O(pairs x bars x window)) -> materialized on demand only.
- **`v_params`** - central SQL-side knobs (flat_threshold, lookbacks, corr
  window) so retuning never touches Python.
- **`data_runs`** - provenance for every download and materialization.

### A.5 Why option data cannot reach 2023 (accepted limitations)

1. **API floor**: Alpaca historical option data begins **Feb 2024** (spec §5).
   No entitlement changes this.
2. **Survivorship**: only `status=active` contracts are listed by the API, so
   expired contracts' histories are unrecoverable. Since standard cycles list
   ~12 months out, the deepest obtainable option history comes from the
   furthest-dated active expiries - hence the `--far` round-robin selector.
   Practical effect: option context starts Mar 2024 (AAPL/AMZN) to Oct 2025
   (AMD) depending on listing dates.
3. **Feed reality**: `indicative` option data has gaps (some contracts print
   only a handful of bars/day); IEX equity data occasionally misses slots and
   shows market half-days short. Validators WARN, never silently drop.
4. **Option bars end "yesterday"**: pulls default `end` to today 00:00Z
   because ranges touching the live session require a signed OPRA agreement
   (HTTP 403 otherwise).

### A.6 The Phase-1 ML layer (training doc)

Three outputs feed the agents: **direction** P(future_return > tau), **future
realized volatility**, and **option opportunity** (expected return + P(profit)).
Design choices carried over from the training doc:

- Underlying tasks train on `(symbol, timestamp)` rows; the option task on
  `(contract_symbol, symbol, timestamp)` rows - kept as SEPARATE datasets
  (doc §7) so equity-only training never inherits the Feb-2024 option floor.
- Option labels use bar CLOSE because historical quotes need the paid OPRA feed;
  doc §21 explicitly permits this if recorded (it is, in the view header).
  Historical IV/Greeks would require reconstruction from timestamp-aligned bars
  (doc §23) - not backfilled snapshots - so they're absent for now.
- Chronological TRAIN→VAL→TEST splits, walk-forward after baseline; scalers fit
  on train only; hyperparameters frozen in `src/ml/train.py`; baselines always
  reported next to XGBoost. Multi-task MLP deferred by design (doc §3).

---

## B. What's in the database (snapshot: 2026-08-26)

### B.1 Tables (physical storage)

| Table | Rows | Grain (PK) | Coverage / purpose |
|---|---:|---|---|
| `underlying_bars` | 253,995 | (symbol, timestamp) | 10 symbols, 15Min iex/split, **2023-01-03 .. today** |
| `option_contracts` | 12,210 | contract_symbol | 6,105 calls + 6,105 puts, expiries 2026-09 .. 2028-01 |
| `contract_selection` | 300 | contract_symbol | far-dated ATM picks (cap 30/symbol) feeding bar pulls |
| `option_bars` | 186,281 | (contract_symbol, timestamp) | 300 contracts, Feb-2024 .. Aug-24, feed=indicative |
| `symbols` | 10 | symbol | universe registry |
| `features` | 253,995 | (symbol, timestamp, feature_set) | materialized equity features ('u15m_v1') |
| `labels` | 507,790 | (symbol, timestamp, horizon_bars) | h=4: 253,955 · h=16: 253,835 |
| `ml_training_dataset` | 507,790 | cache table | flat model-ready join; 118,022 rows carry option context |
| `asset_correlations` | 0 | (timestamp, sym_a, sym_b, window) | materialize on demand (view computes ~1.08M rows) |
| `option_quotes` | 0 | (contract_symbol, timestamp) | landing zone - live phase |
| `option_snapshots` | 0 | (contract_symbol, timestamp) | landing zone - live phase |
| `data_runs` | 11+ | run_id | provenance for every download/materialization |

### B.2 Views (recomputable logic)

| View | Rows when queried | Role |
|---|---:|---|
| `v_features_underlying` | 253,995 | equity feature vector ('u15m_v1'), doc column contract |
| `v_features_underlying_v2` | 253,995 | + hl_range, close-open, MA distances, volume change, SPY/QQQ context ('u15m_v2') - ML input |
| `v_market_context` | 30,161 | past-only SPY/QQQ returns/vol per timestamp |
| `v_features_option` | 185,988 | per-contract features joined to underlying at same timestamp |
| `v_option_context` | 59,011 | option context collapsed to (symbol, timestamp) |
| `v_labels` | 507,790 | future_return / forward vol / target_class @ h=4,16 |
| `v_ml_training_dataset` | 507,790 | underlying-level training SELECT |
| `v_option_training` | 365,947 | contract-level rows: features at t + y_option_return/profit @ h=4,16 (bars-only labels) |
| `v_asset_correlations` | ~1,083,380 | rolling 64-bar pairwise correlations |
| `v_params` | 13 | central SQL-side parameters incl. direction tau & round-trip cost |

New tables from the ML doc §26: `data_sources` (endpoint/feed provenance),
`option_greeks_history` (placeholder until IV/Greeks are reconstructed from
timestamp-aligned bars - snapshots are never backfilled into history).

### B.3 Option-context history depth per underlying

| Underlying | Option bars | Earliest context | Contracts |
|---|---:|---|---:|
| AAPL | 25,450 | 2024-03-26 | 40 |
| AMZN | 28,411 | 2024-04-02 | 29 |
| MSFT | 17,454 | 2024-02-07 | 29 |
| NVDA | 35,551 | 2024-06-10 | 29 |
| QQQ | 11,147 | 2024-06-18 | 28 |
| SPY | 12,850 | 2024-03-06 | 40 |
| META | 9,488 | 2024-09-19 | 29 |
| TSLA | 29,413 | 2024-09-20 | 29 |
| GOOGL | 8,469 | 2025-02-27 | 29 |
| AMD | 8,048 | 2025-10-10 | 29 |

### B.4 Class balance (target_class distribution)

| Horizon | -1 (down) | 0 (flat, +/-0.25%) | +1 (up) |
|---|---:|---:|---:|
| h=4 (1h) | 67,611 | 112,950 | 73,394 |
| h=16 (4h) | 92,859 | 53,435 | 107,541 |

### B.5 Using it

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/trading.db")
df = pd.read_sql("""
    SELECT * FROM ml_training_dataset
    WHERE horizon_bars = 16 AND return_16 IS NOT NULL
""", conn)
```

Warmup note: filter rows until indicators are fully warmed (~32 bars/symbol;
`return_16 IS NOT NULL` covers the strictest case). MACD/ATR use partial-frame
averages during their warmup and surface no NULL - filter explicitly if that
matters for your model.

### B.6 Keeping it fresh

```bash
python -m src.download_stocks --universe --start 2023-01-01T05:00:00Z   # idempotent top-up
python -m src.download_option_bars --all-selected --start 2024-02-01T05:00:00Z
python -m src.materialize && python -m src.validate_data --stage all
```
