-- 30_view_asset_correlations.sql
-- Rolling pairwise Pearson correlation + covariance of bar returns (GNN input
-- for Phase 2). window = 64 bars, from v_params('corr_window').
--
-- O(pairs x bars x window) - do NOT select this view casually over long
-- histories; materialize it over bounded ranges instead:
--     python -m src.materialize --tables asset_correlations

DROP VIEW IF EXISTS v_asset_correlations;
CREATE VIEW v_asset_correlations AS
WITH rr AS (
    SELECT symbol,
           timestamp,
           close * 1.0 / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp), 0) - 1 AS r
    FROM underlying_bars
),
p AS (
    SELECT a.timestamp,
           a.symbol AS symbol_a,
           b.symbol AS symbol_b,
           a.r      AS ra,
           b.r      AS rb
    FROM rr a
    JOIN rr b ON b.symbol > a.symbol AND b.timestamp = a.timestamp
),
cov AS (
    SELECT *,
        AVG(ra * rb) OVER w - AVG(ra) OVER w * AVG(rb) OVER w       AS cov_ab,
        SQRT(MAX(AVG(ra * ra) OVER w - AVG(ra) OVER w * AVG(ra) OVER w, 0)) AS sa,
        SQRT(MAX(AVG(rb * rb) OVER w - AVG(rb) OVER w * AVG(rb) OVER w, 0)) AS sb
    FROM p
    WINDOW w AS (PARTITION BY symbol_a, symbol_b ORDER BY timestamp
                 ROWS BETWEEN 63 PRECEDING AND CURRENT ROW)
)
SELECT
    timestamp,
    symbol_a,
    symbol_b,
    (SELECT value FROM v_params WHERE key = 'corr_window') AS window_bars,
    cov_ab                                             AS covariance,
    cov_ab / NULLIF(sa * sb, 0)                        AS correlation
FROM cov
WHERE sa > 0 AND sb > 0;
