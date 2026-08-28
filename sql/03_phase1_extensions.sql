-- 03_phase1_extensions.sql
-- Schema extensions recommended by "Phase 1 - Options Alpha ML Model" doc section 26.
--   * option_bars.feed already exists in this schema (01_schema.sql), so the doc's
--     ALTER TABLE is intentionally omitted here.
--   * data_sources        : exact data provenance beyond data_runs.
--   * option_greeks_history : placeholder for reconstructed historical IV/Greeks.
--     Doc section 23 rule enforced elsewhere: today's snapshots must NEVER be written
--     into historical timestamps, so this table stays empty until IV/Greeks are
--     reconstructed from timestamp-aligned bars with a documented calculation.

CREATE TABLE IF NOT EXISTS data_sources (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    feed TEXT,
    timeframe TEXT,
    adjustment TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS option_greeks_history (
    contract_symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    implied_volatility REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    rho REAL,
    calculation_method TEXT,
    PRIMARY KEY (contract_symbol, timestamp)
);

INSERT OR IGNORE INTO data_sources (source_id, source_name, endpoint, feed, timeframe, adjustment, notes)
VALUES
    ('stock_bars_iex', 'Alpaca historical stock bars',
     'GET /v2/stocks/{symbol}/bars', 'iex', '15Min', 'split',
     'Free IEX feed; single-exchange prints, extended-hours slots possible'),
    ('option_contracts_active', 'Alpaca option contract metadata',
     'GET /v2/options/contracts (paper host)', 'standard', NULL, NULL,
     'status=active only -> survivorship limitation'),
    ('option_bars_indicative', 'Alpaca historical option bars',
     'GET /v1beta1/options/bars', 'indicative', '15Min', NULL,
     'History floor Feb 2024; end clamped to today 00:00Z (OPRA entitlement)');
