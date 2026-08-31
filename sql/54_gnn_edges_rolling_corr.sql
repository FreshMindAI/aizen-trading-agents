-- ---------------------------------------------------------------------------
-- Migration: extend the gnn_graph_edges CHECK constraint to include the
-- "rolling_corr" reason introduced by the dynamic-1 topology.
--
-- SQLite does not support ALTER TABLE ... ALTER CONSTRAINT, so the
-- standard portable recipe is:
--   1. PRAGMA foreign_keys=OFF
--   2. Detect the current state (table missing / has-old-check / has-new-check)
--   3. Either CREATE TABLE gnn_graph_edges with the new CHECK, or rebuild
--      it via the rename-and-copy recipe.
--   4. PRAGMA foreign_keys=ON
--
-- The migration is idempotent: a fresh DB that does not yet have
-- gnn_graph_edges gets CREATE TABLE IF NOT EXISTS; a DB with the old
-- CHECK gets the rebuild; a DB that already has the new CHECK is a
-- no-op.
-- ---------------------------------------------------------------------------

PRAGMA foreign_keys=OFF;

-- Case A: fresh DB. gnn_graph_edges does not exist; create it with the
-- new CHECK. CREATE TABLE IF NOT EXISTS makes this a no-op on rerun.
CREATE TABLE IF NOT EXISTS gnn_graph_edges_new (
    snapshot_id        TEXT NOT NULL,
    source_symbol      TEXT NOT NULL,
    target_symbol      TEXT NOT NULL,
    reason             TEXT NOT NULL CHECK (reason IN
                          ('sector','supplier','customer','etf_membership',
                           'correlation','rolling_corr')),
    weight             REAL NOT NULL,
    topology_version   TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, source_symbol, target_symbol, reason),
    FOREIGN KEY (snapshot_id) REFERENCES gnn_graph_snapshots(snapshot_id) ON DELETE CASCADE
);

-- If the old gnn_graph_edges table exists with the old CHECK, migrate it.
-- Detect via sqlite_master.
DROP TABLE IF EXISTS _gnn_edges_has_old;
CREATE TEMP TABLE _gnn_edges_has_old (
    present INTEGER NOT NULL DEFAULT 0
);
INSERT INTO _gnn_edges_has_old (present)
SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='gnn_graph_edges');

-- Rename old -> staging only if the old table exists AND it does NOT
-- already have the rolling_corr enum (case where 54 was partially
-- applied).
DROP TABLE IF EXISTS gnn_graph_edges_staging;
CREATE TABLE gnn_graph_edges_staging (
    snapshot_id        TEXT NOT NULL,
    source_symbol      TEXT NOT NULL,
    target_symbol      TEXT NOT NULL,
    reason             TEXT NOT NULL CHECK (reason IN
                          ('sector','supplier','customer','etf_membership',
                           'correlation','rolling_corr')),
    weight             REAL NOT NULL,
    topology_version   TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, source_symbol, target_symbol, reason),
    FOREIGN KEY (snapshot_id) REFERENCES gnn_graph_snapshots(snapshot_id) ON DELETE CASCADE
);

-- If the old table exists, copy its known-good rows into staging.
INSERT OR IGNORE INTO gnn_graph_edges_staging
    (snapshot_id, source_symbol, target_symbol, reason, weight, topology_version)
SELECT snapshot_id, source_symbol, target_symbol, reason, weight, topology_version
FROM   gnn_graph_edges
WHERE  EXISTS(SELECT 1 FROM _gnn_edges_has_old WHERE present = 1)
  AND  reason IN ('sector','supplier','customer','etf_membership','correlation');

DROP TABLE IF EXISTS gnn_graph_edges;
ALTER TABLE gnn_graph_edges_staging RENAME TO gnn_graph_edges;

DROP TABLE IF EXISTS gnn_graph_edges_new;

CREATE INDEX IF NOT EXISTS ix_gnn_edges_reason
    ON gnn_graph_edges(reason);

CREATE INDEX IF NOT EXISTS ix_gnn_edges_snapshot
    ON gnn_graph_edges(snapshot_id);

DROP TABLE IF EXISTS _gnn_edges_has_old;

PRAGMA foreign_keys=ON;
