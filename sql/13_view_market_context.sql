-- 13_view_market_context.sql
-- Market-regime context per 15-minute timestamp (doc section 4 "Market context"):
-- benchmark returns/volatility computed from PAST bars only (LAG / trailing frames).
-- One row per distinct SPY/QQQ timestamp; other symbols LEFT JOIN on timestamp so an
-- IEX gap degrades gracefully to NULL instead of dropping the row.
--
-- VIX is deliberately absent: no permitted free historical source is wired up yet and
-- the doc says the pipeline must not depend on it (section 22).

DROP VIEW IF EXISTS v_market_context;
CREATE VIEW v_market_context AS
WITH b AS (
    SELECT symbol,
           timestamp,
           close * 1.0 / NULLIF(LAG(close) OVER wp, 0) - 1 AS r1
    FROM underlying_bars
    WHERE symbol IN ('SPY', 'QQQ')
    WINDOW wp AS (PARTITION BY symbol ORDER BY timestamp)
),
ctx AS (
    SELECT
        symbol,
        timestamp,
        r1,
        CASE WHEN COUNT(r1) OVER v16 = 16 THEN SQRT(
                 MAX(SUM(r1 * r1) OVER v16
                     - SUM(r1) OVER v16 * SUM(r1) OVER v16 / 16.0, 0) / 16.0)
        END AS realized_vol_16,
        SUM(r1) OVER p16 AS cum_ret_16   -- past 16-bar compounded-ish return proxy
    FROM b
    WINDOW
        v16 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 15 PRECEDING AND CURRENT ROW),
        p16 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 15 PRECEDING AND 1 PRECEDING)
)
SELECT
    timestamp,
    MAX(CASE WHEN symbol = 'SPY' THEN r1 END)             AS spy_ret_1,
    MAX(CASE WHEN symbol = 'SPY' THEN cum_ret_16 END)     AS spy_ret_past_16,
    MAX(CASE WHEN symbol = 'SPY' THEN realized_vol_16 END) AS spy_volatility_16,
    MAX(CASE WHEN symbol = 'QQQ' THEN r1 END)             AS qqq_ret_1,
    MAX(CASE WHEN symbol = 'QQQ' THEN cum_ret_16 END)     AS qqq_ret_past_16,
    MAX(CASE WHEN symbol = 'QQQ' THEN realized_vol_16 END) AS qqq_volatility_16
FROM ctx
GROUP BY timestamp;
