-- 12_view_option_context.sql
-- Option-market context collapsed to ONE row per (symbol, timestamp) so it can
-- be joined onto the equity training set without exploding rows.

DROP VIEW IF EXISTS v_option_context;
CREATE VIEW v_option_context AS
SELECT
    symbol,
    timestamp,
    COUNT(*)                                   AS n_contracts,
    AVG(moneyness)                             AS avg_moneyness,
    AVG(option_return_1)                       AS avg_option_return_1,
    MIN(days_to_expiry)                        AS min_days_to_expiry,
    AVG(option_volatility_16)                  AS avg_option_volatility_16
FROM v_features_option
GROUP BY symbol, timestamp;
