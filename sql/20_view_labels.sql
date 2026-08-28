-- 20_view_labels.sql
-- Supervision targets: future return, forward realized volatility and a 3-class
-- direction label at horizons 4 and 16 bars (1h / 4h on the 15Min grid).
--
-- Design notes:
--   * SQLite requires CONSTANT offsets in LEAD/window frames, so the two
--     horizons are written as two UNION ALL branches with literals.
--   * future_realized_vol is stddev over EXACTLY the h returns in (t, t+h],
--     computed as a window frame [1 FOLLOWING, h FOLLOWING] - one linear scan,
--     no self-join. (An earlier rn-subscript join version was quadratic at
--     scale: its forward-window aggregate forced a range scan per row.)
--   * Leak guard: LEAD(close, h) IS NULL for the final h bars of each symbol,
--     and those rows are filtered out - no label can peek past available data.
--   * Flat band comes from v_params('flat_threshold') and applies live.

DROP VIEW IF EXISTS v_labels;
CREATE VIEW v_labels AS
WITH b AS (
    SELECT symbol,
           timestamp,
           close,
           close * 1.0 / NULLIF(LAG(close) OVER w, 0) - 1 AS ret
    FROM underlying_bars
    WINDOW w AS (PARTITION BY symbol ORDER BY timestamp)
),
h4 AS (
    SELECT symbol, timestamp,
           4                                            AS horizon_bars,
           close                                        AS close_t,
           LEAD(close, 4) OVER wb                       AS future_close,
           CASE WHEN COUNT(ret) OVER f4 = 4 THEN SQRT(
                    MAX(SUM(ret * ret) OVER f4
                        - SUM(ret) OVER f4 * SUM(ret) OVER f4 / 4.0, 0) / 4.0)
           END                                          AS future_realized_vol
    FROM b
    WINDOW wb AS (PARTITION BY symbol ORDER BY timestamp),
           f4  AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 1 FOLLOWING AND 4 FOLLOWING)
),
h16 AS (
    SELECT symbol, timestamp,
           16                                           AS horizon_bars,
           close                                        AS close_t,
           LEAD(close, 16) OVER wb                      AS future_close,
           CASE WHEN COUNT(ret) OVER f16 = 16 THEN SQRT(
                    MAX(SUM(ret * ret) OVER f16
                        - SUM(ret) OVER f16 * SUM(ret) OVER f16 / 16.0, 0) / 16.0)
           END                                          AS future_realized_vol
    FROM b
    WINDOW wb AS (PARTITION BY symbol ORDER BY timestamp),
           f16 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 1 FOLLOWING AND 16 FOLLOWING)
),
scored AS (
    SELECT symbol, timestamp, horizon_bars, close_t, future_close, future_realized_vol FROM h4
    UNION ALL
    SELECT symbol, timestamp, horizon_bars, close_t, future_close, future_realized_vol FROM h16
)
SELECT
    symbol,
    timestamp,
    horizon_bars,
    (future_close * 1.0 / close_t) - 1               AS future_return,
    future_realized_vol,
    CASE
        WHEN future_close > close_t * (1 + (SELECT value FROM v_params WHERE key = 'flat_threshold'))
            THEN 1
        WHEN future_close < close_t * (1 - (SELECT value FROM v_params WHERE key = 'flat_threshold'))
            THEN -1
        ELSE 0
    END                                              AS target_class
FROM scored
WHERE future_close IS NOT NULL;                       -- leak guard
