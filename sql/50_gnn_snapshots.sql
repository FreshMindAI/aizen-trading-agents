-- Phase 2 GNN graph snapshot + model artifact storage.
-- Applied by `python -m src.db init`, which globs every sql/*.sql file in
-- sorted order. Idempotent (CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT
-- EXISTS). The data model and contracts are documented under
-- specs/002-phase2-gnn/data-model.md.

-- ---------------------------------------------------------------------------
-- One row per (timestamp, topology_version). The payload_json is the full
-- {nodes, edges} graph; the normalized tables below carry the same data
-- for queryability (FR7.2 / SC2: "every edge has a documented reason").
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gnn_graph_snapshots (
    snapshot_id       TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    topology_version  TEXT NOT NULL,
    node_count        INTEGER NOT NULL CHECK (node_count > 0),
    edge_count        INTEGER NOT NULL CHECK (edge_count >= 0),
    payload_json      TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    UNIQUE (timestamp, topology_version)
);

CREATE INDEX IF NOT EXISTS ix_gnn_snapshots_timestamp
    ON gnn_graph_snapshots(timestamp);

-- ---------------------------------------------------------------------------
-- Normalized edge table. CHECK constraint enforces the reason enum from
-- contracts/gnn_output.schema.json and spec FR1.2.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gnn_graph_edges (
    snapshot_id        TEXT NOT NULL,
    source_symbol      TEXT NOT NULL,
    target_symbol      TEXT NOT NULL,
    reason             TEXT NOT NULL CHECK (reason IN
                          ('sector','supplier','customer','etf_membership','correlation')),
    weight             REAL NOT NULL,
    topology_version   TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, source_symbol, target_symbol, reason),
    FOREIGN KEY (snapshot_id) REFERENCES gnn_graph_snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_gnn_edges_reason
    ON gnn_graph_edges(reason);

CREATE INDEX IF NOT EXISTS ix_gnn_edges_snapshot
    ON gnn_graph_edges(snapshot_id);

-- ---------------------------------------------------------------------------
-- Saved models. Path is relative to models/ (gitignored). meta.json carries
-- the versioned sidecar; this table is the index.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gnn_model_artifacts (
    model_version     TEXT PRIMARY KEY,
    path              TEXT NOT NULL,
    architecture      TEXT NOT NULL,
    topology_version  TEXT NOT NULL,
    feature_names     TEXT NOT NULL,   -- JSON array
    impute_medians    TEXT NOT NULL,   -- JSON object
    split_bounds      TEXT NOT NULL,   -- JSON object: {train_end, val_end, test_end}
    test_metrics      TEXT NOT NULL,   -- JSON object: {roc_auc, pr_auc, log_loss, brier}
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_gnn_artifacts_created_at
    ON gnn_model_artifacts(created_at);

-- ---------------------------------------------------------------------------
-- Per-split evaluation records. Multiple rows per model_version
-- (train / val / test / wf-...).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gnn_model_evaluations (
    model_version   TEXT NOT NULL,
    split           TEXT NOT NULL,
    metrics_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (model_version, split),
    FOREIGN KEY (model_version) REFERENCES gnn_model_artifacts(model_version) ON DELETE CASCADE
);
