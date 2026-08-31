# Phase 0 — Data: failures

## What broke

### 1. `spot_from_db` returned a stale close on the first call of the day
- **Symptom:** `option_contracts` download picked strike bands centered on yesterday's close. With gappy symbols (AAPL on a half-day, NVDA around earnings) the band was 1-3% off the live mark, so the "ATM" strikes weren't actually ATM.
- **Root cause:** `spot_from_db()` reads `underlying_bars` ordered by `timestamp DESC LIMIT 1` with no lookback. If the last bar is from yesterday's close and the market hasn't opened yet, the "spot" is yesterday's mark.
- **Fix attempted:** use the most recent bar ≤ now (already what the SQL does). The real fix is to add a "live spot override" from Alpaca's snapshot endpoint before the contracts pull. We never wired it.
- **Cost:** ~20% of "ATM" strikes were off by 1 strike width (~0.5-1.5%). Not catastrophic but visible in moneyness distributions.

### 2. The two-pass type filter (`TWO_PASS_TYPES`) is a code-smell we left in
- **Symptom:** A defensive flag `TWO_PASS_TYPES = False` in `download_option_contracts.py:25` exists because at one point the API returned only calls unless we explicitly filtered for `type=put`. The flag is a permanent scar; we never removed the guard.
- **Root cause:** Alpaca's options contracts endpoint *does* return both calls and puts by default today. The flag is dead code with a comment that says "flip to True only if…".

### 3. Pagination and rate limit
- **Symptom:** On the first big refresh (Aug 2024), we hit `429 Too Many Requests` from `paper-api.alpaca.markets/v2/options/contracts` because we asked for 10 symbols × 30 expiries in one shot.
- **Root cause:** the puller has no client-side rate limiter. `data.rate_limit_per_min: 190` in `config/alpaca.yaml` is documented but never enforced.
- **Fix:** we manually added a `time.sleep(0.4)` between requests in the first refresh script. That's the only place the limit is respected; everywhere else we trust the server.
- **Cost:** during a single refresh on 2026-08-29 we got 2 × 429 responses and silently dropped a page. ~50 contracts missing for QQQ. We didn't re-run.

### 4. The `vwap_distance` and `qqq_ret_past_16` features don't have unit tests
- **Symptom:** silent NaNs in some bars for symbols with no QQQ trading that day (rare but happens on holidays when the etf publishes a quote but the underlying doesn't).
- **Root cause:** SQL `vwap` window is the trading day; if the day has zero bars, the window is empty and the feature is NULL. The pipeline doesn't validate non-null before training.
- **Cost:** the XGBoost model silently treats NULL as 0 (because pandas coerces). So one holiday a year, every symbol's `qqq_ret_past_16` is 0. Negligible at 1-day resolution, but should be caught.

### 5. `option_bars` ingest uses the *free* feed
- **Symptom:** the `feed: iex` config in `alpaca.yaml` is the free IEX feed, which is fine for paper but means our backtest on `option_bars` doesn't match the consolidated tape. The free feed drops some prints.
- **Cost:** walk-forward metrics on the option model may be optimistic by 0.5-2% ROC-AUC because we under-count downside prints.

### 6. `.env` / secrets handling
- We never built a proper secrets loader. The `.env` file is read by `python-dotenv` at app boot, but the keys also leak into `data/run_log` when the alpaca client logs request URLs. We added a `[REDACTED]` filter in `setup_logging` only after the first leak. **The Alpaca keys are still on disk in `data/.env` — must gitignore, never commit.**

### 7. The DB has no backup
- A single `data/trading.db` file with 254K underlying rows + 195K option rows + 4 GNN artifacts. If it corrupts we lose everything. We have no `VACUUM INTO` snapshot, no nightly backup. SQLite is durable but not crash-proof.

## Things we didn't notice in time

- The `ml_training_dataset` view joins on `(symbol, timestamp)` and the timestamps in `option_bars` are at 1-min resolution while `underlying_bars` are at 30-min. The join silently drops 29/30 option bars per underlying bar. We didn't realize until we saw option row counts ~30× higher than expected.
