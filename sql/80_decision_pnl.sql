-- Trade P&L analysis (Loop 3 of the autonomous-loops plan).
-- One row per (broker_order_id, filled_at) — the natural key of an
-- Alpaca fill, since the same broker_order_id can have multiple fill
-- events (partial fills). Applied by `python -m src.db init` in
-- sorted filename order. Idempotent.

CREATE TABLE IF NOT EXISTS decision_pnl (
    decision_id        TEXT NOT NULL,        -- FK to decision_journal.decision_id (soft FK; we JSON-extract)
    client_order_id    TEXT NOT NULL,        -- what we stamped on submit_order()
    broker_order_id    TEXT NOT NULL,        -- Alpaca's id
    filled_at          TEXT NOT NULL,        -- ISO-8601 UTC, the fill timestamp
    symbol             TEXT NOT NULL,        -- OCC option symbol OR underlying
    asset_class        TEXT NOT NULL CHECK (asset_class IN ('option','equity')),
    side               TEXT NOT NULL CHECK (side IN ('buy','sell')),
    quantity           INTEGER NOT NULL,
    entry_price        REAL NOT NULL,        -- for paired rows: open fill price; for unpaired: fill price
    exit_price         REAL NOT NULL,        -- for paired rows: close fill price; for unpaired: 0.0
    realized_pnl       REAL NOT NULL,        -- signed USD; 0.0 for unpaired opens
    classification     TEXT NOT NULL CHECK (classification IN ('win','loss','breakeven','open')),
    computed_at        TEXT NOT NULL,
    PRIMARY KEY (broker_order_id, filled_at)
);

CREATE INDEX IF NOT EXISTS idx_decision_pnl_decision
    ON decision_pnl(decision_id);
CREATE INDEX IF NOT EXISTS idx_decision_pnl_filled_at
    ON decision_pnl(filled_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_pnl_classification
    ON decision_pnl(classification);
