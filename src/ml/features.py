"""Feature-set definitions per task.

Columns must exist on the source view. Scale-free variants (_pct) are preferred
for pooled cross-symbol fits; raw dollar-scale columns (macd, atr_14,
underlying_close) are deliberately NOT model inputs.
"""

# Underlying dataset (v_features_underlying_v2 JOIN v_labels) ----------------
UNDERLYING_FEATURES = [
    # returns / dispersion
    "return_1", "return_4", "return_16", "volatility_16",
    # momentum
    "rsi_14", "macd_pct",
    # range / intrabar shape
    "hl_range", "co_return", "atr_pct_14",
    # trend distance
    "ma_dist_20", "ma_dist_50",
    # flow
    "volume_ratio_20", "volume_change_1", "trade_count_ratio_20", "vwap_distance",
    # market context
    "spy_ret_1", "spy_ret_past_16", "spy_volatility_16",
    "qqq_ret_past_16", "qqq_volatility_16",
]

# Columns that must be non-NULL to accept a row: full-indicator warmup gate.
# Everything else goes through a train-fitted median imputer instead of dropping
# rows, so sporadic IEX gaps (vwap/trade_count/SPY slot misses) don't shrink data.
UNDERLYING_WARMUP_COLS = ["volatility_16", "rsi_14", "ma_dist_50"]

# Option dataset (v_option_training) -----------------------------------------
OPTION_FEATURES = UNDERLYING_FEATURES + [
    # contract-specific, all causal at t
    "option_return_1", "option_volatility_16",
    "moneyness", "log_moneyness", "is_call", "days_to_expiry",
]

OPTION_WARMUP_COLS = UNDERLYING_WARMUP_COLS + ["option_volatility_16"]

# Keys kept alongside features for splitting/metrics/backtests (never trained on).
UNDERLYING_KEYS = ["symbol", "timestamp"]
OPTION_KEYS = ["contract_symbol", "symbol", "timestamp"]
