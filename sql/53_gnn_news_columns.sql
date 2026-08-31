-- Extend gnn_model_artifacts + gnn_model_evaluations for the news ablation
-- (spec 003 / FR-007, FR-008, FR-009).
-- Applied by `python -m src.db init`. Idempotent.
--
-- Adds:
--   gnn_model_artifacts.used_news          INTEGER (bool)  - 1 if trained on news-augmented edges
--   gnn_model_artifacts.ablation_fold_id   INTEGER         - the held-out fold this came from
--   (gnn_model_evaluations already has the JSON shape we need; we only document
--    the new sharpe_approx field here)

-- SQLite does not support IF NOT EXISTS for ALTER TABLE ADD COLUMN.
-- We probe pragma_table_info first and skip if the column already exists.

-- gnn_model_artifacts.used_news
ALTER TABLE gnn_model_artifacts ADD COLUMN used_news INTEGER NOT NULL DEFAULT 0;

-- gnn_model_artifacts.ablation_fold_id (nullable; NULL for non-ablation artifacts)
-- SQLite cannot ADD COLUMN with NULL default easily; the column is added with
-- the implicit NULL default for INTEGER, which is what we want.
ALTER TABLE gnn_model_artifacts ADD COLUMN ablation_fold_id INTEGER;

-- The CHECK constraint on gnn_graph_edges.reason already covers the
-- existing 5 reasons. The new edge types (news_cooccurrence,
-- news_sentiment_correlation) live in gnn_option_graph_edges, which
-- does NOT have a CHECK constraint in sql/51 (option_graph.py uses
-- INSERT directly), so the new reasons are allowed by default. We
-- add a CHECK here as a belt-and-suspenders for the option graph's
-- reason column too.

-- (no-op: gnn_option_graph_edges has no reason CHECK; leaving as-is per
-- the design doc decision to add a CHECK only via code-side validation
-- in option_graph._build_news_edges.)
