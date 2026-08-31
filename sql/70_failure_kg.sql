-- ---------------------------------------------------------------------------
-- 70_failure_kg.sql
-- Knowledge graph of agent / broker / LLM failures.
--
-- Why a knowledge graph?
-- ----------------------
-- The user diagnosed that the multi-agent pipeline silently degraded
-- on a larger universe: agents were not taking "the rest" of the
-- failure data into account, and broker/LLM errors were lost to logs.
-- Surfacing this as a graph lets the GNN consume the failure channel
-- alongside the market channel: when AAPL's execution node has been
-- failing all morning, that knowledge should propagate to the
-- direction / risk / portfolio nodes via weighted edges in the
-- snapshot.
--
-- The graph has three node types and one edge type:
--
--   SymbolFailure  : one node per (symbol, day) when a failure touched
--                    the symbol. Holds error count, severity, last
--                    error string, and a `kind` enum so the GNN can
--                    distinguish API / LLM / option-broker / option-chain
--                    failures.
--
--   AgentFailure   : one node per (agent_id, day) when an agent
--                    returned an error. Holds the agent's display
--                    name, error string, and the underlying symbol the
--                    agent was processing (if any).
--
--   CycleFailure   : one node per (decision_id) when the cycle as a
--                    whole failed. Holds the cycle timestamp, the
--                    final action, the exception class, and the error
--                    message.
--
--   FailureEdge    : a directed edge from one node to another, with
--                    a `relation` enum (caused_by, similar_to, co_occurs)
--                    and a weight in [0, 1]. The graph is bipartite
--                    (Symbol / Agent / Cycle) and homogeneous within a
--                    relation type so the GNN can learn each channel
--                    separately.
--
-- Per-symbol node features
-- ------------------------
-- A companion view ``v_symbol_failure_features`` exposes one row per
-- symbol with a 4-element feature vector:
--   [failure_count_7d, llm_failure_count_7d, broker_failure_count_7d,
--    weighted_failure_score_7d]
-- The strategy-selector / risk node joins this into the snapshot
-- so the GNN sees a per-symbol failure score as an additional
-- node feature.
--
-- The migration is idempotent: every CREATE uses IF NOT EXISTS and
-- every INSERT uses OR IGNORE, so re-running on a pre-existing DB
-- is a no-op.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS failure_nodes (
    -- Stable id derived from the node key, e.g. "sym:AAPL:2026-08-25",
    -- "agent:direction:2026-08-25", "cycle:abc-123".
    -- The kind/foreign-key triplet below is unique and is what the
    -- writer uses to dedup; node_id is for graph consumers.
    node_id            TEXT PRIMARY KEY,
    kind               TEXT NOT NULL CHECK (kind IN
                          ('symbol_failure','agent_failure','cycle_failure')),
    symbol             TEXT,                       -- nullable for cycle-level
    agent_id           TEXT,                       -- nullable for symbol-level
    decision_id        TEXT,                       -- nullable for symbol/agent
    occurred_at        TEXT NOT NULL,
    day                TEXT NOT NULL,              -- YYYY-MM-DD, for windowing
    severity           TEXT NOT NULL DEFAULT 'info'
                          CHECK (severity IN ('info','warn','error','critical')),
    error_class        TEXT,                       -- exception class name
    error_message      TEXT,                       -- short, structured
    error_count        INTEGER NOT NULL DEFAULT 1, -- incrementable
    metadata_json      TEXT NOT NULL DEFAULT '{}', -- free-form context
    created_at         TEXT NOT NULL,
    UNIQUE (kind, symbol, agent_id, decision_id, occurred_at)
);

CREATE INDEX IF NOT EXISTS ix_failure_nodes_day
    ON failure_nodes(day);
CREATE INDEX IF NOT EXISTS ix_failure_nodes_symbol
    ON failure_nodes(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_failure_nodes_agent
    ON failure_nodes(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_failure_nodes_decision
    ON failure_nodes(decision_id) WHERE decision_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Edges: connect two nodes with a typed relation and a [0, 1] weight.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS failure_edges (
    edge_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id     TEXT NOT NULL,
    target_node_id     TEXT NOT NULL,
    relation           TEXT NOT NULL CHECK (relation IN
                          ('caused_by','similar_to','co_occurs')),
    weight             REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    day                TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE (source_node_id, target_node_id, relation, day),
    FOREIGN KEY (source_node_id) REFERENCES failure_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES failure_nodes(node_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_failure_edges_source
    ON failure_edges(source_node_id);
CREATE INDEX IF NOT EXISTS ix_failure_edges_target
    ON failure_edges(target_node_id);
CREATE INDEX IF NOT EXISTS ix_failure_edges_day
    ON failure_edges(day);
CREATE INDEX IF NOT EXISTS ix_failure_edges_relation
    ON failure_edges(relation);

-- ---------------------------------------------------------------------------
-- Per-symbol feature view: 7-day rolling failure score.
--
-- The window is anchored to the snapshot timestamp. Symbols with
-- zero failures get a zero vector so the GNN's node-feature order
-- is uniform across the universe.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_symbol_failure_features AS
WITH window_start AS (
    -- The "7 days back" anchor is computed in the writer via parameter;
    -- this view is a fallback that takes the most-recent day in the
    -- failure_nodes table and walks back 6 days.
    SELECT date(MAX(day), '-6 days') AS start_day
    FROM failure_nodes
),
per_symbol AS (
    SELECT fn.symbol,
           COALESCE(SUM(fn.error_count), 0) AS failure_count_7d,
           COALESCE(SUM(CASE WHEN fn.error_class IN
                                ('LLMError','LLMTransportError','requests.RequestException')
                            THEN fn.error_count ELSE 0 END), 0) AS llm_failure_count_7d,
           COALESCE(SUM(CASE WHEN fn.error_class IN
                                ('AlpacaAPIError','BrokerError','OrderRejectedError')
                            THEN fn.error_count ELSE 0 END), 0) AS broker_failure_count_7d,
           COALESCE(SUM(fn.error_count *
                        CASE fn.severity
                            WHEN 'critical' THEN 1.0
                            WHEN 'error'    THEN 0.7
                            WHEN 'warn'     THEN 0.3
                            ELSE 0.1
                        END), 0.0) AS weighted_failure_score_7d
    FROM   failure_nodes fn, window_start ws
    WHERE  fn.kind = 'symbol_failure'
      AND  fn.day >= ws.start_day
    GROUP BY fn.symbol
)
SELECT s.symbol AS symbol,
       COALESCE(p.failure_count_7d, 0)         AS failure_count_7d,
       COALESCE(p.llm_failure_count_7d, 0)    AS llm_failure_count_7d,
       COALESCE(p.broker_failure_count_7d, 0)  AS broker_failure_count_7d,
       COALESCE(p.weighted_failure_score_7d, 0.0) AS weighted_failure_score_7d
FROM   symbols s
LEFT JOIN per_symbol p ON p.symbol = s.symbol;
