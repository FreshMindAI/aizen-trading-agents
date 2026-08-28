-- Phase 3 decision journal. One row per orchestrator cycle.
-- Every JSON column is the serialized Pydantic model from src.agents.protocol.
-- Keep this table narrow on the index path (decision_id, timestamp) and wide
-- on the analysis path (the JSON blobs). No FK to data_runs: journal is the
-- system of record for agent reasoning, not data ingestion.

CREATE TABLE IF NOT EXISTS decision_journal (
    decision_id          TEXT PRIMARY KEY,
    timestamp            TEXT NOT NULL,                -- cycle started
    completed_at         TEXT,
    market_state_hash    TEXT NOT NULL DEFAULT '',
    schema_version       TEXT NOT NULL DEFAULT '1.0',

    run_mode             TEXT NOT NULL DEFAULT 'paper',  -- paper | dry-run | live
    underlying_focus     TEXT,                            -- the symbol being traded
    final_action         TEXT NOT NULL DEFAULT 'NO_TRADE', -- PROCEED|REDUCE|REJECT|NO_TRADE
    outcome_label        TEXT,                            -- filled on RECONCILE
    realized_pnl         REAL,

    market_snapshot_json TEXT NOT NULL,
    ml_prediction_json   TEXT NOT NULL,
    gnn_output_json      TEXT NOT NULL DEFAULT '{}',
    topology_version     TEXT,

    agent_messages_json  TEXT NOT NULL DEFAULT '[]',     -- envelope log
    agent_observations_json TEXT NOT NULL DEFAULT '[]',
    strategy_proposal_json TEXT NOT NULL DEFAULT 'null',
    selected_strategy_json TEXT NOT NULL DEFAULT 'null',

    risk_decision_json   TEXT NOT NULL DEFAULT 'null',
    order_intent_json    TEXT NOT NULL DEFAULT 'null',
    execution_result_json TEXT NOT NULL DEFAULT 'null',

    model_versions       TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_journal_timestamp
    ON decision_journal(timestamp);
CREATE INDEX IF NOT EXISTS idx_journal_underlying
    ON decision_journal(underlying_focus, timestamp);
CREATE INDEX IF NOT EXISTS idx_journal_outcome
    ON decision_journal(final_action, outcome_label);
