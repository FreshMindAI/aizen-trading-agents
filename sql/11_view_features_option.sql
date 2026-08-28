-- 11_view_features_option.sql
-- Per-contract option features joined to their underlying at the SAME timestamp.
-- feature_set = 'opt15m_v1'.
--
-- The INNER join on underlying_bars(symbol, timestamp) is intentional: option
-- bars whose underlying bar is missing (free IEX feed gaps) are dropped instead
-- of emitting moneyness against a stale spot. Use LEFT JOIN if recall ever
-- matters more than correctness - the doc's guidance is correctness first.

DROP VIEW IF EXISTS v_features_option;
CREATE VIEW v_features_option AS
WITH ob AS (
    SELECT
        contract_symbol,
        timestamp,
        -- doc schema spells OHLC long; alias once here so the rest stays terse
        open  AS o,
        high  AS h,
        low   AS l,
        close AS c,
        volume, vwap, trade_count,
        close * 1.0 / NULLIF(LAG(close) OVER w, 0) - 1 AS r1   -- sibling alias `c` not usable here
    FROM option_bars
    WINDOW w AS (PARTITION BY contract_symbol ORDER BY timestamp)
)
SELECT
    ob.contract_symbol,
    oc.underlying_symbol                             AS symbol,
    ob.timestamp,
    'opt15m_v1'                                      AS feature_set,
    oc.strike_price,
    oc.option_type,
    oc.expiration_date,
    CAST(julianday(oc.expiration_date) - julianday(ob.timestamp) AS INTEGER)
                                                     AS days_to_expiry,
    oc.strike_price * 1.0 / NULLIF(ub.close, 0)      AS moneyness,
    ob.c                                             AS opt_price,
    ob.r1                                            AS option_return_1,
    CASE WHEN COUNT(ob.r1) OVER v16 = 16 THEN SQRT(
             MAX(SUM(ob.r1 * ob.r1) OVER v16
                 - SUM(ob.r1) OVER v16 * SUM(ob.r1) OVER v16 / 16.0, 0) / 16.0)
    END                                              AS option_volatility_16,
    (ob.c - ob.vwap) / NULLIF(ob.vwap, 0)            AS opt_vwap_distance,
    ub.close * 1.0 / NULLIF(LAG(ub.close) OVER uw, 0) - 1
                                                     AS underlying_return_1,
    -- Populated once snapshots/latest quotes go live (live-agent phase):
    NULL                                             AS iv,
    NULL                                             AS bid_ask_spread,
    NULL                                             AS open_interest
FROM ob
JOIN option_contracts oc USING (contract_symbol)
JOIN underlying_bars ub
     ON ub.symbol = oc.underlying_symbol AND ub.timestamp = ob.timestamp
WINDOW
    v16 AS (PARTITION BY ob.contract_symbol ORDER BY ob.timestamp
            ROWS BETWEEN 15 PRECEDING AND CURRENT ROW),
    uw  AS (PARTITION BY oc.underlying_symbol ORDER BY ob.timestamp);
