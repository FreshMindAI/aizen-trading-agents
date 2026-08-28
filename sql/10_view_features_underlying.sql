-- 10_view_features_underlying.sql
-- Phase-1 equity features as ONE query over raw underlying_bars. feature_set = 'u15m_v1'.
--
-- Methodology notes (deliberate, documented deviations):
--   * Pure window functions only - no recursive smoothers.
--       rsi_14   : Cutler's RSI (SMA of gains/losses over 14 returns), not Wilder's.
--       macd     : SMA12 - SMA26 of close; signal = SMA9 of the line (not EMA).
--       atr_14   : SMA of true range over 14 bars (not Wilder smoothing).
--   * SQLite has no STDDEV aggregate, so dispersion uses the identity
--       Var = E[x^2] - E[x]^2  ==  SUM(x*x)/n - (SUM(x)/n)^2
--     computed directly over each window frame.
--   * Warmup: indicators surface NULL until their frame is full (COUNT checks),
--     so no half-window artifacts leak into training rows.
--
-- Column order matches the doc's features table through option_volume, with
-- extra diagnostic columns (macd_signal/macd_hist/...) appended AFTER them;
-- src/materialize.py inserts by explicit column names, never positionally.
DROP VIEW IF EXISTS v_features_underlying;
CREATE VIEW v_features_underlying AS
WITH base AS (
    SELECT
        symbol,
        timestamp,
        -- doc schema spells OHLC long; alias once here so the rest stays terse
        open  AS o,
        high  AS h,
        low   AS l,
        close AS c,
        volume, vwap, trade_count,
        -- NB: must reference `close`, not the sibling alias `c` - SQLite does not
        -- let one select-list expression see another's alias.
        close * 1.0 / NULLIF(LAG(close) OVER w, 0) - 1 AS r1,   -- simple 1-bar return
        LAG(close) OVER w                              AS pc
    FROM underlying_bars
    WINDOW w AS (PARTITION BY symbol ORDER BY timestamp)
),
ind AS (
    SELECT
        *,
        -- realized volatility over the last 16 returns (frame must be FULL)
        CASE WHEN COUNT(r1) OVER v16 = 16 THEN SQRT(
                 MAX(SUM(r1 * r1) OVER v16
                     - SUM(r1) OVER v16 * SUM(r1) OVER v16 / 16.0, 0) / 16.0)
        END                                           AS volatility_16,
        -- Cutler RSI: 100 * avgGain / (avgGain + avgLoss); avgGain+avgLoss == AVG(ABS(r1))
        CASE WHEN COUNT(r1) OVER r14 = 14 AND AVG(ABS(r1)) OVER r14 > 0
             THEN 100.0 * AVG(CASE WHEN r1 > 0 THEN r1 ELSE 0 END) OVER r14
                  / AVG(ABS(r1)) OVER r14
             WHEN COUNT(r1) OVER r14 = 14 THEN 50.0   -- perfectly flat window
        END                                           AS rsi_14,
        MAX(h - l, ABS(h - pc), ABS(l - pc))          AS tr,   -- true range
        volume * 1.0 / NULLIF(AVG(volume) OVER v20, 0) AS volume_ratio_20,
        (c - vwap) / NULLIF(vwap, 0)                   AS vwap_distance,
        AVG(c) OVER m12 - AVG(c) OVER m26              AS macd_line_sma
    FROM base
    WINDOW
        v16 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 15 PRECEDING AND CURRENT ROW),
        r14 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
        v20 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        m12 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
        m26 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
)
SELECT
    symbol,
    timestamp,
    'u15m_v1'                                        AS feature_set,
    r1                                               AS return_1,
    c * 1.0 / NULLIF(LAG(c, 4)  OVER w2, 0) - 1      AS return_4,
    c * 1.0 / NULLIF(LAG(c, 16) OVER w2, 0) - 1      AS return_16,
    volatility_16,
    rsi_14,
    macd_line_sma                                    AS macd,
    AVG(macd_line_sma) OVER s9                       AS macd_signal,
    macd_line_sma - AVG(macd_line_sma) OVER s9       AS macd_hist,
    AVG(tr) OVER a14                                 AS atr_14,
    volume_ratio_20,
    vwap_distance,
    -- Option/live-context columns stay NULL in the equity view; they exist so the
    -- view stays column-compatible with the doc's features table and so the same
    -- shape extends to option-context joins later.
    NULL                                             AS iv,
    NULL                                             AS iv_minus_rv,
    NULL                                             AS iv_div_rv,
    NULL                                             AS days_to_expiry,
    NULL                                             AS moneyness,
    NULL                                             AS bid_ask_spread,
    NULL                                             AS open_interest,
    NULL                                             AS option_volume
FROM ind
WINDOW
    w2  AS (PARTITION BY symbol ORDER BY timestamp),
    s9  AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 8 PRECEDING AND CURRENT ROW),
    a14 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 13 PRECEDING AND CURRENT ROW);
