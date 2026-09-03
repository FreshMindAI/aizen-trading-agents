-- ---------------------------------------------------------------------------
-- 71_kill_switch_latch.sql
-- Latches the daily-loss kill-switch for the rest of the calendar day (UTC)
-- so a later tick on the same day cannot re-enter after a breach.
--
-- Why a latch (not just re-checking pnl each cycle)?
-- -------------------------------------------------
-- The kill-switch cap is "total P/L vs -2% of capital". If a tick at
-- 11:00 ET trips it (realized -$2000 + unrealized -$1000 = -$3000),
-- and a 15:00 ET tick finds a different mix of positions whose total
-- is back above -$2000 (because we auto-closed the worst ones at
-- 11:00), the kill-switch would NOT re-fire on a pure-pnl check.
-- But the operator's intent was "no new entries for the rest of today".
-- A latch row in this table makes that intent persistent: as long as
-- today's row exists, the orchestrator returns NO_TRADE before even
-- running the agents.
--
-- Schema:
--   day_utc    TEXT PRIMARY KEY,   -- 'YYYY-MM-DD' (UTC)
--   breached_at TEXT NOT NULL,     -- ISO-8601 of the trip
--   total_pnl  REAL NOT NULL,      -- snapshot at trip time
--   threshold_usd REAL NOT NULL,   -- capital * pct
--   pct        REAL NOT NULL       -- the pct that was in force
--
-- Cleared implicitly by the date-based primary key (a new day =
-- new row; old row is never read). No DELETE statements needed.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kill_switch_latch (
    day_utc        TEXT PRIMARY KEY,
    breached_at    TEXT NOT NULL,
    total_pnl      REAL NOT NULL,
    threshold_usd  REAL NOT NULL,
    pct            REAL NOT NULL
);
