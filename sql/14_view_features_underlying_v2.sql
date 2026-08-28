-- 14_view_features_underlying_v2.sql
-- Extended equity feature set 'u15m_v2' (training doc section 4 "Underlying"):
-- everything in u15m_v1 PLUS intrabar shape, moving-average distances, volume
-- change and market-context columns.
--
-- Scale-free variants (atr_pct_14, macd_pct) exist because Phase-1 models pool
-- all 10 symbols into ONE training set - raw dollar-scale indicators are not
-- comparable across a $120 stock and a $900 one. Trainers should prefer the
-- _pct columns for pooled fits.
--
-- All new windows are trailing-only -> causality preserved (doc section 6).

DROP VIEW IF EXISTS v_features_underlying_v2;
CREATE VIEW v_features_underlying_v2 AS
WITH bars AS (
    SELECT
        f.symbol, f.timestamp,
        f.return_1, f.return_4, f.return_16,
        f.volatility_16, f.rsi_14, f.macd, f.macd_signal, f.macd_hist,
        f.atr_14, f.volume_ratio_20, f.vwap_distance,
        b.open, b.high, b.low, b.close, b.volume, b.trade_count
    FROM v_features_underlying f
    JOIN underlying_bars b USING (symbol, timestamp)
),
ind AS (
    SELECT
        *,
        (high - low) * 1.0 / NULLIF(close, 0)              AS hl_range,
        close * 1.0 / NULLIF(open, 0) - 1                  AS co_return,
        close * 1.0 / NULLIF(AVG(close) OVER m20, 0) - 1   AS ma_dist_20,
        close * 1.0 / NULLIF(AVG(close) OVER m50, 0) - 1   AS ma_dist_50,
        volume * 1.0 / NULLIF(LAG(volume) OVER w, 0) - 1   AS volume_change_1,
        trade_count * 1.0 / NULLIF(AVG(trade_count) OVER t20, 0)
                                                           AS trade_count_ratio_20,
        atr_14 * 1.0 / NULLIF(close, 0)                    AS atr_pct_14,
        macd * 1.0 / NULLIF(close, 0)                      AS macd_pct
    FROM bars
    WINDOW
        w   AS (PARTITION BY symbol ORDER BY timestamp),
        m20 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        m50 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
        t20 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT
    ind.symbol,
    ind.timestamp,
    'u15m_v2'                                          AS feature_set,
    -- v1 core
    ind.return_1, ind.return_4, ind.return_16,
    ind.volatility_16, ind.rsi_14, ind.macd, ind.atr_14,
    ind.volume_ratio_20, ind.vwap_distance,
    -- v2 additions: intrabar shape
    ind.hl_range,
    ind.co_return,
    -- trend distance
    ind.ma_dist_20,
    ind.ma_dist_50,
    -- flow
    ind.volume_change_1,
    ind.trade_count_ratio_20,
    -- scale-free pooled-model variants
    ind.atr_pct_14,
    ind.macd_pct,
    -- market context (LEFT JOIN: IEX gaps on SPY/QQQ must not drop rows)
    ctx.spy_ret_1,
    ctx.spy_ret_past_16,
    ctx.spy_volatility_16,
    ctx.qqq_ret_1,
    ctx.qqq_ret_past_16,
    ctx.qqq_volatility_16,
    NULL                                               AS iv,
    NULL                                               AS iv_minus_rv,
    NULL                                               AS iv_div_rv
FROM ind
LEFT JOIN v_market_context ctx USING (timestamp);
