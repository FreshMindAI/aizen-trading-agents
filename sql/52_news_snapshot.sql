-- News snapshot table (spec 003 / FR-001).
-- Applied by `python -m src.db init`, which globs every sql/*.sql file in
-- sorted order. Idempotent (CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS).
-- The data model and contracts are documented under
-- specs/003-news-driven-gnn-retrain/data-model.md and
-- specs/003-news-driven-gnn-retrain/contracts/news-snapshot.json.

-- ---------------------------------------------------------------------------
-- One row per (article_published_at, symbol). `timestamp` is the article's
-- published_at from Alpaca (NOT the wall-clock write time). This is the
-- single most important guarantee for news-time-leakage protection
-- (failure-analysis §2.1 row 2).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_snapshot (
    timestamp       TEXT NOT NULL,            -- article's published_at, ISO-8601 UTC
    symbol          TEXT NOT NULL,            -- ticker from Alpaca payload's symbols list
    sentiment       REAL NOT NULL CHECK (sentiment BETWEEN -1.0 AND 1.0),
    article_count   INTEGER NOT NULL CHECK (article_count >= 1),
    topics_json     TEXT NOT NULL,            -- JSON array of 1-3 topic strings
    raw_json        TEXT NOT NULL,            -- full Alpaca payload for replay
    created_at      TEXT NOT NULL,            -- wall-clock write time, ISO-8601 UTC
    PRIMARY KEY (timestamp, symbol),
    CHECK (timestamp <= created_at)           -- article cannot be written before it is published
);

CREATE INDEX IF NOT EXISTS idx_news_snapshot_symbol_ts
    ON news_snapshot(symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_news_snapshot_ts
    ON news_snapshot(timestamp DESC);
