-- 60_backtest_cycles.sql
-- Per-cycle results of the point-in-time backtest (spec 003 / T046).
-- One row per replayed cycle: the model's predicted action + the realized
-- forward outcome measured by the labeler (short-horizon via v_labels and
-- option-structure payoff at expiry). Kept in a dedicated table so the
-- live decision_journal stays clean and queryable separately.
--
-- This table is written by ``src.agents.backtest.BacktestRunner`` and
-- read by aggregate reporting. Idempotent: safe to re-run init_db.
--
-- The CHECK on ``final_action`` mirrors the contract enum from the
-- multi-agent supervisor (``src/agents/nodes/supervisor.py``).
-- ``coverage_*`` is a 0/1 flag indicating whether the labeler was able
-- to compute that particular label for this cycle — useful for the
-- "coverage" aggregate metric.

CREATE TABLE IF NOT EXISTS backtest_cycles (
    decision_id            TEXT PRIMARY KEY,
    cycle_as_of            TEXT NOT NULL,
    final_action           TEXT NOT NULL
        CHECK (final_action IN ('PROCEED','NO_TRADE','REJECTED_BY_RISK','DRY_RUN','ERROR')),
    predicted_underlying   TEXT,
    predicted_strategy_id  TEXT,
    predicted_legs_json    TEXT,           -- JSON array of legs (NULL for NO_TRADE)
    forward_return_h1      REAL,           -- 1h forward return from v_labels
    forward_return_h4      REAL,           -- 4h forward return from v_labels
    target_class           TEXT,           -- 3-class band from v_labels at h=16
    option_payoff          REAL,           -- held-to-expiry structure payoff (or NULL)
    hit_h4                 INTEGER,        -- 1 if predicted direction == sign(forward_return_h4), 0 if not, NULL if undeterminable
    hit_h1                 INTEGER,        -- same at h=4
    coverage_h1            INTEGER NOT NULL DEFAULT 0,
    coverage_h4            INTEGER NOT NULL DEFAULT 0,
    coverage_payoff        INTEGER NOT NULL DEFAULT 0,
    run_at                 TEXT NOT NULL,
    model_version          TEXT,
    feature_flag_state     TEXT NOT NULL,  -- 'news-on' or 'news-off'
    notes                  TEXT
);

CREATE INDEX IF NOT EXISTS ix_backtest_cycles_as_of
    ON backtest_cycles(cycle_as_of);

CREATE INDEX IF NOT EXISTS ix_backtest_cycles_underlying
    ON backtest_cycles(predicted_underlying);
