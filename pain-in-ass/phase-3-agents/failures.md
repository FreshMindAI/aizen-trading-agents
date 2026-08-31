# Phase 3 — Agents: failures

## What broke

### 1. The orchestrator's NO_TRADE gate is too strict for a 1-week hackathon
- **Symptom:** in the most recent cycle, all 10 XGBoost direction probabilities were in 0.21-0.49. The orchestrator's gate is `direction_prob_min: 0.55`, so every symbol was filtered. Final action = `NO_TRADE`.
- **Root cause:** the XGBoost model is well-calibrated for long-term Sharpe, not for week-long P&L. The threshold 0.55 is too high for a competitive trading week.
- **Cost:** the system would trade zero this week. We're already in the "creativity" lane; we have to be in the "P&L" lane too.
- **Fix:** add a GNN-confirmation override (`if gnn_bias > 0.5 and xgb_prob in [0.45, 0.55] → allow trade`) and lower the threshold to 0.52.

### 2. The `OrderIntent` Pydantic model requires `option_type`, `strike`, `expiry` on every leg
- **Symptom:** when we tried to submit a stock market order via `client.submit_order(intent)`, the model validation failed because the `Leg` schema is options-only.
- **Root cause:** `Leg` was designed for multi-leg options strategies. Stock orders went through `OrderIntent` as a 1-leg options contract, which is wrong.
- **Fix:** bypassed `OrderIntent` for the paper-trade and used `client._request('POST', '/v2/orders', json_body={...})` directly with a stock-shaped body.
- **Cost:** the orchestration layer doesn't have a first-class "stock" path. We should add one.

### 3. The supervisor agent's reasoning is fully mocked
- **Symptom:** every cycle, the supervisor's output is `mock-1`'s deterministic JSON. No real reasoning happens.
- **Root cause:** `llm.provider: mock` in `config/agents.yaml:35`. The supervisor is essentially a no-op.
- **Why it matters for the hackathon:** the "creativity" pillar of the judging rewards novel agent design. A mock LLM looks like a placeholder.
- **Hackathon fix:** switch `llm.provider` to `mock` for tests but enable a real LLM for live hackathon runs. **DO NOT enable the GMI/MiniMaxAI endpoint** — the user has explicit security rules against it.

### 4. The `_resolve_snapshot_id` returns `"stub-snapshot"` when no graph exists
- **Symptom:** if `gnn_graph_snapshots` is empty, the inference service falls back to `"stub-snapshot"` and the GNN service handles it via the last-resort timestamp.
- **Root cause:** defensive design, but the fallback timestamp is hard-coded to `"2026-08-25T17:45:00Z"`. If we run a year from now, the fallback is stale.
- **Cost:** cosmetic. Will fix in v2.

### 5. The portfolio agent returns no positions
- **Symptom:** `inference._load_portfolio()` always returns `[]`. The portfolio agent never sees the actual Alpaca book.
- **Root cause:** we never wired the Alpaca positions endpoint into the inference layer.
- **Cost:** the agent layer trades "blind" — it doesn't know what it already holds. This is a real risk for the hackathon: we could over-trade a name we already own.

### 6. The "no_trade_if_disagreement" default is on
- **Symptom:** when XGB says "neutral" and the GNN says "very bullish", the orchestrator returns NO_TRADE.
- **Root cause:** the disagreement gate is `true` in `config/agents.yaml:52`.
- **Cost:** the GNN-confirmation override (failure #1) won't fire unless we either flip the disagreement gate to `false` or add a specific exception.

### 7. `account_equity` and `account_cash` are always `None`
- The agent protocol has these fields, but `inference._account_equity()` and `_account_cash()` return `None`.
- **Root cause:** we never connected the Alpaca account endpoint to the inference service.
- **Cost:** the risk agent can't reason about position sizing in dollar terms because it doesn't know the account size.

### 8. The decision_journal `agent_messages_json` is fine but `notes` doesn't exist as a column
- **Symptom:** the watch script's first version tried to UPDATE `notes`, but the schema has no such column. The actual place to append free-text is `agent_messages_json` (a JSON array).
- **Fix:** updated `watch_nvda_paper_fill.py` to use `agent_messages_json`.
- **Cost:** one debugging cycle.

### 9. The candidates ranking uses both XGB and GNN but XGB dominates
- The candidate scoring formula is:
  ```
  score = 0.30 * direction_edge
        + 0.20 * volatility_edge
        + 0.25 * option_expected_return
        + 0.20 * probability_profit
        + 0.10 * gnn_confirmation    ← only 10% weight
        - 0.10 * spread_penalty
        - 0.05 * liquidity_penalty
        - 0.15 * portfolio_risk_penalty
  ```
- GNN is a 10% tie-breaker, not a primary signal. **The user wants the GNN to be the primary signal** — this is the architectural change for v2.

### 10. The orchestrator loop has a 60s cooldown per symbol
- `cooldown_seconds: 60` in `config/agents.yaml:6`. Reasonable, but the loop interval is 300s. So we get one decision per symbol per 5 min. In a 6.5-hour trading day, that's 78 decisions across the universe.
- **Cost:** fine, but we miss intraday swings that resolve in 30 min.
