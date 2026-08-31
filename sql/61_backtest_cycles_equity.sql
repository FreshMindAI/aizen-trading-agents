-- 61_backtest_cycles_equity.sql
-- Equity-path extension to the backtest_cycles table (parallel options+stocks).
-- For cycles where the selected strategy is a long-equity leg, the labeler
-- also computes:
--   * equity_payoff_h1 / equity_payoff_h4: realized P&L of the long stock
--     at 1h / 4h horizon. entry = underlying_bars.close at-or-before
--     cycle_as_of, exit = underlying_bars.close at-or-before cycle_as_of + h.
--   * equity_hit_h4: 1 if price went up, 0 if it went down, NULL if no bar.
-- These columns are NULL for option-only cycles, so the existing option
-- payoff reporting is untouched. The aggregate report
-- (``BacktestRunner._aggregate``) computes hit_rate_equity_h4, mean
-- equity PnL, and equity coverage.

ALTER TABLE backtest_cycles ADD COLUMN equity_payoff_h1 REAL;
ALTER TABLE backtest_cycles ADD COLUMN equity_payoff_h4 REAL;
ALTER TABLE backtest_cycles ADD COLUMN equity_hit_h4   INTEGER;
ALTER TABLE backtest_cycles ADD COLUMN equity_hit_h1   INTEGER;
ALTER TABLE backtest_cycles ADD COLUMN coverage_equity_h1 INTEGER NOT NULL DEFAULT 0;
ALTER TABLE backtest_cycles ADD COLUMN coverage_equity_h4 INTEGER NOT NULL DEFAULT 0;
