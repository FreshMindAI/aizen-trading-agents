-- 22_view_option_training.sql
-- Option-opportunity training rows (training doc sections 2C, 5, 7): one row per
-- (timestamp, underlying, contract, horizon) with option features at t and the
-- FORWARD option targets:
--     y_option_return = opt_close[t+H] / opt_close[t] - 1
--     y_option_profit = 1 if y_option_return - option_cost_roundtrip > 0
--
-- LABEL PRICE SOURCE (doc section 5 mandates recording this): historical option
-- QUOTES are unavailable on our free feed (option_quotes is empty), so labels use
-- the option CLOSE from the indicative bar feed. Doc section 21 explicitly allows
-- this as long as it is recorded - it is, here and in DATASETS.md/README.
--
-- Causality: every feature is computed at or before t; LEAD(...) targets are the
-- only forward references and rows without a full forward window are filtered
-- (leak guard, same pattern as v_labels). SQLite needs constant LEAD offsets,
-- hence one branch per horizon. Constant per-row attributes (is_call, strike,
-- DTE, ...) must ride THROUGH both UNION ALL branches.

DROP VIEW IF EXISTS v_option_training;
CREATE VIEW v_option_training AS
WITH ob AS (
    SELECT
        b.contract_symbol,
        b.timestamp,
        b.close                                        AS opt_close,
        b.volume                                       AS option_volume,
        b.trade_count                                  AS option_trade_count,
        oc.underlying_symbol                           AS symbol,
        oc.strike_price,
        CASE WHEN oc.option_type = 'call' THEN 1 ELSE 0 END AS is_call,
        julianday(oc.expiration_date)
            - julianday(substr(b.timestamp, 1, 10))    AS days_to_expiry,
        b.close * 1.0 / NULLIF(LAG(b.close) OVER wo, 0) - 1 AS opt_r1
    FROM option_bars b
    JOIN option_contracts oc USING (contract_symbol)
    WINDOW wo AS (PARTITION BY b.contract_symbol ORDER BY b.timestamp)
),
h4 AS (
    SELECT contract_symbol, symbol, timestamp,
           4                                        AS horizon_bars,
           opt_close                                AS option_close_t,
           LEAD(opt_close, 4) OVER wob              AS future_option_close,
           opt_r1                                   AS option_return_1,
           -- option realized vol over the PAST 16 returns (full frame required)
           CASE WHEN COUNT(opt_r1) OVER v16 = 16 THEN SQRT(
                    MAX(SUM(opt_r1 * opt_r1) OVER v16
                        - SUM(opt_r1) OVER v16 * SUM(opt_r1) OVER v16 / 16.0, 0) / 16.0)
           END                                      AS option_volatility_16,
           strike_price, is_call, days_to_expiry,
           option_volume, option_trade_count
    FROM ob
    WINDOW
        wob AS (PARTITION BY contract_symbol ORDER BY timestamp),
        v16 AS (PARTITION BY contract_symbol ORDER BY timestamp ROWS BETWEEN 15 PRECEDING AND CURRENT ROW)
),
h16 AS (
    SELECT contract_symbol, symbol, timestamp,
           16                                       AS horizon_bars,
           opt_close                                AS option_close_t,
           LEAD(opt_close, 16) OVER wob             AS future_option_close,
           opt_r1                                   AS option_return_1,
           CASE WHEN COUNT(opt_r1) OVER v16 = 16 THEN SQRT(
                    MAX(SUM(opt_r1 * opt_r1) OVER v16
                        - SUM(opt_r1) OVER v16 * SUM(opt_r1) OVER v16 / 16.0, 0) / 16.0)
           END                                      AS option_volatility_16,
           strike_price, is_call, days_to_expiry,
           option_volume, option_trade_count
    FROM ob
    WINDOW
        wob AS (PARTITION BY contract_symbol ORDER BY timestamp),
        v16 AS (PARTITION BY contract_symbol ORDER BY timestamp ROWS BETWEEN 15 PRECEDING AND CURRENT ROW)
),
scored AS (
    SELECT * FROM h4
    UNION ALL
    SELECT * FROM h16
)
SELECT
    s.contract_symbol,
    s.symbol,
    s.timestamp,
    s.horizon_bars,
    s.future_option_close * 1.0 / NULLIF(s.option_close_t, 0) - 1 AS y_option_return,
    CASE WHEN s.future_option_close * 1.0 / NULLIF(s.option_close_t, 0) - 1
              - (SELECT value FROM v_params WHERE key = 'option_cost_roundtrip') > 0
         THEN 1 ELSE 0 END                          AS y_option_profit,
    s.option_return_1,
    s.option_volatility_16,
    s.option_close_t,
    s.strike_price * 1.0 / NULLIF(ub.close, 0)      AS moneyness,
    LN(s.strike_price * 1.0 / NULLIF(ub.close, 0))  AS log_moneyness,
    s.is_call,
    s.days_to_expiry,
    s.option_volume,
    s.option_trade_count,
    ub.close                                        AS underlying_close,
    fu.return_1, fu.return_4, fu.return_16, fu.volatility_16,
    fu.rsi_14, fu.macd_pct, fu.atr_pct_14, fu.volume_ratio_20, fu.vwap_distance,
    fu.hl_range, fu.co_return, fu.ma_dist_20, fu.ma_dist_50,
    fu.volume_change_1, fu.trade_count_ratio_20,
    fu.spy_ret_1, fu.spy_ret_past_16, fu.spy_volatility_16,
    fu.qqq_ret_past_16, fu.qqq_volatility_16
FROM scored s
JOIN underlying_bars ub
     ON ub.symbol = s.symbol AND ub.timestamp = s.timestamp
LEFT JOIN v_features_underlying_v2 fu
     ON fu.symbol = s.symbol AND fu.timestamp = s.timestamp
WHERE s.future_option_close IS NOT NULL;            -- leak guard
