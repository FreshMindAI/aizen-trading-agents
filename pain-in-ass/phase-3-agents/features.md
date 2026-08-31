# Phase 3 — Agents: features

## What we shipped (verified working)

### 8 specialized agents

| Agent | File | Job |
|---|---|---|
| `regime` | `src/agents/nodes/regime.py` | Classifies market regime (trending / mean-reverting / volatile) |
| `direction` | `src/agents/nodes/direction.py` | Reads XGBoost direction probabilities |
| `volatility` | `src/agents/nodes/volatility.py` | Reads XGBoost rv regressor |
| `options_structure` | `src/agents/nodes/options_structure.py` | Picks the top-N option structures per symbol |
| `portfolio` | `src/agents/nodes/portfolio.py` | Reasoning about current positions |
| `risk` | `src/agents/risk.py` | Hard limits, REJECT is final |
| `supervisor` | `src/agents/supervisor.py` | Aggregates scores, picks the final action |
| `execution` | `src/agents/execution.py` | Maps the final action to an `OrderIntent` |

All eight are registered in `config/agents.yaml:11-27` with individual `enabled: true` toggles.

### Orchestrator
- `src/agents/orchestrator.py` — the main loop. Each cycle:
  1. Build a `MarketSnapshot` via `inference.build_snapshot()`
  2. Run each agent in dependency order
  3. Supervisor picks the final action
  4. Risk agent approves or rejects
  5. Execution agent maps to `OrderIntent`
  6. `decision_journal` row is written
- 11 journal rows on disk as of today (10 NO_TRADE + 1 paper trade).

### Decision protocol
- `src/agents/protocol.py` — Pydantic v2 strict models (`extra=forbid`):
  - `MarketSnapshot` (top-level container)
  - `UnderlyingScore`, `OptionScore` (per-instrument ML scores)
  - `PortfolioPosition` (current holdings)
  - `Side`, `OrderIntent`, `Leg` (order shape)
- JSON schemas in `contracts/`.

### Inference service
- `src/agents/inference.py:90` — `InferenceService` reads from SQLite via the canonical SQL in `_LATEST_UNDERLYING_SQL` and `_LATEST_OPTION_SQL`. Joins on the latest `ml_training_dataset` row per symbol/contract.
- Returns a `MarketSnapshot` ready for the agents.
- The `gnn_output()` method loads the latest GNN artifact and runs `GNNService.predict()`. Has a `StubGNNService` fallback.

### Risk agent
- `src/agents/risk.py` — deterministic, no LLM. Reads `config/risk.yaml`:
  - Per-trade: `max_leg_quantity: 10`, `max_order_notional_usd: 2000`, `max_loss_per_trade_usd: 300`
  - Portfolio: `max_open_positions: 6`, `max_concentration_per_underlying: 0.40`, `max_gross_exposure_usd: 8000`
  - Universe: 10 names, DTE 7-60
- REJECT is final. The risk agent's decision cannot be overridden by the LLM.

### Decision journal
- `data/trading.db:decision_journal` — every cycle writes a row with:
  - `decision_id, timestamp, final_action, market_snapshot_json, ml_prediction_json, gnn_output_json, agent_messages_json, agent_observations_json, strategy_proposal_json, selected_strategy_json, risk_decision_json, order_intent_json, execution_result_json, model_versions`
- Auditable. The hackathon judges can replay any decision.

## What works well
- **Strict Pydantic models** mean a malformed message from one agent can't crash the orchestrator.
- **Risk is authoritative** — no LLM can override a REJECT. This is the right governance.
- **Every cycle is journaled** — we can always answer "what was the model thinking when it made this trade?"
- **The mock LLM is deterministic** — tests are reproducible.

## What doesn't work well
- **Mock LLM is a placeholder.** The supervisor's "reasoning" is just template substitution. For the hackathon's creativity pillar, this is a real weakness.
- **Portfolio agent is blind** — no live Alpaca positions wired in.
- **Account equity/cash are None** — risk sizing is qualitative.
- **The XGB gate dominates** — GNN is 10% weight, XGB direction prob drives the gate. Need to invert the hierarchy per the user's request.
- **Options-structure agent returns top-5 candidates** but the supervisor doesn't always pick one (it can pick stock if the option signal is weak).
