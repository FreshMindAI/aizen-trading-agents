-- 01_schema.sql
-- Raw layer DDL, verbatim from "Alpaca Data Extraction & SQLite Data Layer" (doc sections 7-9).
-- Raw observation tables are append-only via INSERT OR IGNORE - never updated in place.
-- Derived tables (features / labels / asset_correlations) are declared here per the doc;
-- their content is produced purely by the sql/ views (10_*..30_*) and materialized into
-- tables by src/materialize.py.

-- ---------------------------------------------------------------------------
-- Raw: symbols metadata
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    exchange TEXT,
    active INTEGER,
    updated_at TEXT
);

-- ---------------------------------------------------------------------------
-- Raw: equity bars (15Min default)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS underlying_bars (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER,
    vwap REAL,
    trade_count INTEGER,
    feed TEXT,
    adjustment TEXT,
    PRIMARY KEY (symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_underlying_time ON underlying_bars(timestamp);

-- ---------------------------------------------------------------------------
-- Raw: option contracts
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS option_contracts (
    contract_symbol TEXT PRIMARY KEY,
    contract_id TEXT,
    underlying_symbol TEXT NOT NULL,
    expiration_date TEXT NOT NULL,
    strike_price REAL NOT NULL,
    option_type TEXT NOT NULL,
    style TEXT,
    status TEXT,
    tradable INTEGER,
    root_symbol TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_options_underlying_expiry
    ON option_contracts(underlying_symbol, expiration_date);

-- ---------------------------------------------------------------------------
-- Raw: option bars / quotes / snapshots
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS option_bars (
    contract_symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    vwap REAL,
    trade_count INTEGER,
    feed TEXT,
    PRIMARY KEY (contract_symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_option_bars_time ON option_bars(timestamp);
CREATE INDEX IF NOT EXISTS idx_option_bars_contract ON option_bars(contract_symbol);

-- Landing zone for the live agent phase; created now so later phases have a home.
CREATE TABLE IF NOT EXISTS option_quotes (
    contract_symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    bid REAL,
    ask REAL,
    bid_size INTEGER,
    ask_size INTEGER,
    feed TEXT,
    PRIMARY KEY (contract_symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS option_snapshots (
    contract_symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    last_price REAL,
    bid REAL,
    ask REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    implied_volatility REAL,
    PRIMARY KEY (contract_symbol, timestamp)
);

-- ---------------------------------------------------------------------------
-- Derived Phase-1 tables (doc section 8). Content = materialized views.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    feature_set TEXT NOT NULL,
    return_1 REAL,
    return_4 REAL,
    return_16 REAL,
    volatility_16 REAL,
    rsi_14 REAL,
    macd REAL,
    atr_14 REAL,
    volume_ratio_20 REAL,
    vwap_distance REAL,
    iv REAL,
    iv_minus_rv REAL,
    iv_div_rv REAL,
    days_to_expiry REAL,
    moneyness REAL,
    bid_ask_spread REAL,
    open_interest REAL,
    option_volume REAL,
    PRIMARY KEY(symbol, timestamp, feature_set)
);

CREATE TABLE IF NOT EXISTS labels (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    horizon_bars INTEGER NOT NULL,
    future_return REAL,
    future_realized_vol REAL,
    target_class INTEGER,
    PRIMARY KEY(symbol, timestamp, horizon_bars)
);

CREATE TABLE IF NOT EXISTS asset_correlations (
    timestamp TEXT NOT NULL,
    symbol_a TEXT NOT NULL,
    symbol_b TEXT NOT NULL,
    window_bars INTEGER NOT NULL,
    correlation REAL,
    covariance REAL,
    PRIMARY KEY(timestamp, symbol_a, symbol_b, window_bars)
);

-- ---------------------------------------------------------------------------
-- Data-run metadata (doc section 9): every download is reproducible.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT,
    completed_at TEXT,
    dataset_type TEXT,
    symbols TEXT,
    timeframe TEXT,
    start_time TEXT,
    end_time TEXT,
    feed TEXT,
    adjustment TEXT,
    api_endpoint TEXT,
    rows_inserted INTEGER,
    rows_skipped INTEGER,
    status TEXT,
    error_message TEXT
);

-- ---------------------------------------------------------------------------
-- Pipeline additions beyond the doc
-- ---------------------------------------------------------------------------
-- Deterministic ATM contract picks made by src/download_option_contracts.py,
-- kept for reproducibility of every downstream option-bar pull.
CREATE TABLE IF NOT EXISTS contract_selection (
    run_id TEXT NOT NULL,
    contract_symbol TEXT PRIMARY KEY,
    underlying_symbol TEXT NOT NULL,
    rank INTEGER NOT NULL,
    spot_at_selection REAL NOT NULL
);
