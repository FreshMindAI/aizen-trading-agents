-- 02_params.sql
-- Central SQL-side parameters.
--
-- flat_threshold is READ LIVE by v_labels (sql/20_view_labels.sql), so changing
-- it here immediately redefines the 3-class target.
--
-- Lookbacks are mirrored here for documentation/reference, BUT window-frame
-- bounds must be literals in SQLite, so if you change a lookback you must also
-- update the matching frame in sql/10_view_features_underlying.sql /
-- sql/30_view_asset_correlations.sql.
DROP VIEW IF EXISTS v_params;
CREATE VIEW v_params(key, value) AS
    SELECT 'flat_threshold' AS key, 0.0025 AS value UNION ALL   -- +/- band around 0 => target_class 0
    SELECT 'vol_lookback',          16 UNION ALL                -- bars, volatility_16 window
    SELECT 'rsi_period',            14 UNION ALL                -- bars, Cutler RSI
    SELECT 'macd_fast',             12 UNION ALL                -- SMA periods
    SELECT 'macd_slow',             26 UNION ALL
    SELECT 'macd_signal',            9 UNION ALL
    SELECT 'atr_period',            14 UNION ALL
    SELECT 'volume_window',         20 UNION ALL
    SELECT 'corr_window',           64 UNION ALL                -- asset-correlation rolling window
    SELECT 'horizon_fast',           4 UNION ALL                -- label horizons, 15Min bars
    SELECT 'horizon_slow',          16 UNION ALL
    -- Phase-1 ML (training doc section 2/5): direction tau and option round-trip
    -- cost fraction applied to y_option_profit. Read live by v_option_training;
    -- trainers also read these as their default thresholds.
    SELECT 'direction_threshold',   0.0025 UNION ALL            -- tau: "sufficiently positive"
    SELECT 'option_cost_roundtrip', 0.010;                      -- 1% round-trip cost assumption
