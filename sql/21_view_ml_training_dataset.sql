-- 21_view_ml_training_dataset.sql
-- THE deliverable for model training: equity features + labels + option-market
-- context in a single flat SELECT. Consume directly:
--
--     pd.read_sql("SELECT * FROM v_ml_training_dataset", conn)
--
-- Leak safety is inherited from v_labels: rows without a full forward window do
-- not exist there, so the inner join drops them here too.
-- target_class ∈ {-1, 0, +1}; horizon_bars ∈ {4, 16}.

DROP VIEW IF EXISTS v_ml_training_dataset;
CREATE VIEW v_ml_training_dataset AS
SELECT
    u.symbol,
    u.timestamp,
    u.feature_set,
    l.horizon_bars,
    -- equity features
    u.return_1,
    u.return_4,
    u.return_16,
    u.volatility_16,
    u.rsi_14,
    u.macd,
    u.macd_signal,
    u.macd_hist,
    u.atr_14,
    u.volume_ratio_20,
    u.vwap_distance,
    -- option-market context (NULL until option bars exist for that timestamp)
    oc.n_contracts,
    oc.avg_moneyness,
    oc.avg_option_return_1,
    oc.avg_option_volatility_16,
    oc.min_days_to_expiry,
    -- targets
    l.future_return,
    l.future_realized_vol,
    l.target_class
FROM v_features_underlying u
JOIN v_labels l
     ON l.symbol = u.symbol AND l.timestamp = u.timestamp
LEFT JOIN v_option_context oc
     ON oc.symbol = u.symbol AND oc.timestamp = u.timestamp;
