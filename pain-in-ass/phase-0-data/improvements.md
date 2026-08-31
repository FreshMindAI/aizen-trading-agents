# Phase 0 — Data: improvements

## Highest impact (do first)

### 1. Add a live spot override before pulling contracts
- `src/download_option_contracts.py:30` reads the latest close from `underlying_bars`. For intraday runs after a market open, the band is off by 0.5-1.5%.
- **Fix:** add a `get_live_spot(client, symbol)` helper that calls `GET /v2/stocks/{symbol}/snapshot` from Alpaca (free tier, no rate limit issue at single-call granularity) and falls back to the DB close.
- **Effort:** 30 min. **Impact:** ~5% better strike centering for the ATM subset.

### 2. Add a `feed: sip` config switch with a clear warning
- Currently `feed: iex` is the default. The IEX feed is free but consolidated only at the IEX venue. For the hackathon it doesn't matter (paper account), but for any live deployment we need `sip` for accurate fills.
- **Fix:** make `feed` an env var, default to `iex` for paper, require explicit `feed=sip` for live.
- **Effort:** 10 min.

### 3. Wire the rate limiter into the downloader
- `config/alpaca.yaml:12` says `rate_limit_per_min: 190` but nothing enforces it. Add a token-bucket in `src/alpaca_client.py:AlpacaClient.request()`.
- **Effort:** 1 hour. **Impact:** zero failed refreshes.

### 4. Nightly VACUUM + backup snapshot
- Add a cron-style task: `sqlite3 data/trading.db "VACUUM INTO 'data/snapshots/trading-$(date +%Y%m%d).db'"` after the daily refresh.
- Keep 7 daily snapshots, 4 weekly, 12 monthly. Auto-prune.
- **Effort:** 30 min. **Impact:** catastrophe-proof.

### 5. Drop `TWO_PASS_TYPES`
- It's dead. The flag is in `download_option_contracts.py:25`. Remove the flag and the comment.
- **Effort:** 5 min.

## Medium impact

### 6. Add a feature catalog table
```sql
CREATE TABLE feature_catalog (
    feature_name    TEXT PRIMARY KEY,
    definition      TEXT NOT NULL,
    sql_expression  TEXT NOT NULL,
    version         TEXT NOT NULL,
    deprecated_at   TEXT
);
```
On any change to `v_features_underlying_v2`, insert the new version and mark the old one deprecated. Training code reads from the latest non-deprecated version.
- **Effort:** 2 hours.

### 7. Materialize the correlation view
- `v_asset_correlations` is recomputed on every query. For the 10-symbol universe × 64-bar window × 254K bars it's ~1.3M cell recomputations per call. We do this on every GNN snapshot build.
- **Fix:** add a scheduled materialization to a table `asset_correlations` (not a view) refreshed every 4 hours.
- **Effort:** 1 hour. **Impact:** ~10× faster snapshot builds.

### 8. Add per-symbol feature stats (mean / std / null count)
- For drift detection. If `qqq_ret_past_16.mean` shifts by 2σ, alert.
- **Effort:** 1 hour.

## Low impact / nice to have

### 9. Move secrets out of `.env` into Windows Credential Manager
- The `.env` file is fine for paper but a liability for live. Use `keyring` lib to read/write.
- **Effort:** 2 hours.

### 10. Add a `data_dictionary.md` that documents every column in every table
- Currently spread across SQL comments. Centralize.

### 11. Test data integrity with `sql/90_validate_checks.sql` on every refresh
- We have the SQL but it doesn't run by default. Wire it as a post-refresh hook.
