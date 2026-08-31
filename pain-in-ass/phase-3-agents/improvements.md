# Phase 3 — Agents: improvements

## Highest impact (do for the hackathon)

### 1. Wire the Alpaca account + positions into `inference.py`
- `inference._load_portfolio()` always returns `[]`. Replace with a call to `AlpacaTradingClient.list_positions()`.
- `inference._account_equity()` / `_account_cash()` should return real values from `AlpacaTradingClient.get_account()`.
- **Effort:** 1 hour. **Impact:** the risk agent can now reason about position sizing in dollars, and the portfolio agent knows what it already holds.

### 2. Add a GNN-confirmation override on the XGB gate
- New rule: `if xgb_prob in [0.45, 0.55] and gnn_bias > 0.5 → allow trade` (currently blocked by `direction_prob_min: 0.55`).
- Toggle via `thresholds.gnn_override_enabled: true` in `config/agents.yaml`.
- **Effort:** 4 hours (config + supervisor change + tests).
- **Impact:** unlocks 5-15 trades per day that the current gate would block.

### 3. Add a "strategy type" selector to the supervisor
- Today the supervisor picks one of {long_stock, options_structure, no_trade}. Add a fourth: `short_put_spread` for IV-rich environments.
- Pick based on: `iv_rv_gap > 0.05` → short_put_spread; `gnn_bias > 0.5 and xgb > 0.55` → long_stock (or covered call); `xgb > 0.55` only → options_structure.
- **Effort:** 1 day. **Impact:** 3× the trade surface.

### 4. Add a "research" agent
- A new agent that fetches the latest news for the universe (Alpaca news API) and emits a sentiment signal.
- Output: per-symbol `news_sentiment` in [-1, +1], per-symbol `news_volume` (count of articles).
- **Effort:** 1 day (new agent + tests).
- **Hackathon value:** the creativity pillar.

### 5. Make the GNN the primary signal (per user's request)
- Currently `gnn_confirmation: 0.10` in scoring. Bump to `0.30` and lower `direction_edge: 0.30 → 0.15`.
- Add a regime-aware weight: when the regime agent says "high volatility", trust the GNN more (graph captures cross-asset contagion).
- **Effort:** 4 hours. **Impact:** GNN becomes the dominant signal.

## Medium impact

### 6. Switch the LLM provider to a real model for live runs
- The user has explicit rules: DO NOT call the GMI / MiniMaxAI / MiniMax-M3 endpoint. DO NOT print the ANTHROPIC_AUTH_TOKEN.
- For the hackathon we can use a local LLM (Ollama) or a non-Anthropic endpoint (OpenAI) if the user wants real reasoning.
- **Effort:** 4 hours.

### 7. Add a regime-conditioned graph topology
- When the regime agent says "risk-off", switch the GNN graph to `topology_version: regime-on-1` (cross-asset correlation edges, no sector edges).
- Train a separate GNN for each regime.
- **Effort:** 2 days.

### 8. Per-symbol cooldown
- Currently `cooldown_seconds: 60` is global. Make it per-symbol with a `cooldown_seconds_map` config.
- **Effort:** 2 hours.

### 9. End-of-day P&L reconciliation
- A scheduled job at 4:05 PM ET that:
  1. Reads all fills from the day's `decision_journal` rows.
  2. Reads current positions from Alpaca.
  3. Computes realized + unrealized P&L.
  4. Writes a `daily_pnl` row to a new table.
- **Effort:** 1 day. **Hackathon value:** P&L is the primary judging criterion.

### 10. Replace the static scoring formula with a learned aggregator
- Train a small MLP on the 5 sub-scores → final action. Backtest-proven, not hand-tuned.
- **Effort:** 3 days.

## Low impact

### 11. Add an "explain" agent that summarizes the supervisor's reasoning in plain English
- Uses the LLM. Optional for the hackathon.
- **Effort:** 1 day.

### 12. Add a "compliance" agent that checks for wash-sale violations, PDT, etc.
- Hardcoded rules. Reads positions + recent fills.
- **Effort:** 2 days.

### 13. Add a backtest mode to the orchestrator
- Run a backtest loop over historical timestamps using the same agents.
- **Effort:** 3 days.
