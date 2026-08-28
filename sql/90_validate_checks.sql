-- 90_validate_checks.sql
-- Reusable validation queries. This file intentionally contains ONLY comments -
-- it is applied harmlessly by init_db and kept here so analysts can copy-paste.
-- src/validate_data.py automates all of them.

-- OHLC sanity (expect 0):
-- SELECT COUNT(*) FROM underlying_bars
--   WHERE high < MAX(open, close) OR low > MIN(open, close)
--      OR MIN(open, low, close) <= 0 OR high <= 0;
-- (same shape for option_bars)

-- Timestamp format check (expect 0):
-- SELECT COUNT(*) FROM underlying_bars WHERE timestamp NOT GLOB
--   '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z';

-- Grid heuristic: bars per symbol per day outside [20, 26] are IEX-thin days or
-- half sessions - WARN only, not failures:
-- SELECT symbol, date(timestamp) AS d, COUNT(*) AS n
--   FROM underlying_bars GROUP BY symbol, d HAVING n NOT BETWEEN 20 AND 26;

-- Orphan option bars (must be 0):
-- SELECT COUNT(*) FROM option_bars b
--   LEFT JOIN option_contracts oc USING (contract_symbol)
--   WHERE oc.contract_symbol IS NULL;

-- Per-contract coverage vs sessions in range (WARN below ~80%; indicative gaps normal):
-- SELECT contract_symbol, COUNT(DISTINCT date(timestamp)) AS sessions, COUNT(*) AS bars
--   FROM option_bars GROUP BY contract_symbol ORDER BY sessions;

-- Duplicate proof: rerun any downloader; the newest data_runs row must show
-- rows_inserted = 0 and rows_skipped equal to the previous total:
-- SELECT dataset_type, status, rows_inserted, rows_skipped, started_at
--   FROM data_runs ORDER BY started_at DESC LIMIT 5;

-- Class balance across horizons (should be bell-shaped around 0, never ~99% one class):
-- SELECT horizon_bars, target_class, COUNT(*) FROM v_labels GROUP BY 1, 2 ORDER BY 1, 2;

-- Leak-guard proof: max labeled timestamp must be < max bar timestamp per symbol:
-- SELECT l.symbol, MAX(l.timestamp), MAX(b.timestamp)
--   FROM v_labels l JOIN underlying_bars b USING (symbol) GROUP BY l.symbol;

-- Sample training rows (core features NULL-free, target_class in {-1,0,1}):
-- SELECT * FROM v_ml_training_dataset LIMIT 5;

-- Correlation sanity: SPY/QQQ rolling correlation should sit high most days:
-- SELECT * FROM v_asset_correlations
--   WHERE symbol_a IN ('SPY','QQQ') AND symbol_b IN ('SPY','QQQ')
--   ORDER BY timestamp DESC LIMIT 5;
