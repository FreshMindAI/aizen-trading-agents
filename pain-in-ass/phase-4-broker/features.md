# Phase 4 — Broker: features

## What we shipped (verified working)

### Alpaca clients
- `src/alpaca_client.py` — `AlpacaClient` for the data host (bars, news, contracts).
- `src/agents/alpaca_trading.py` — `AlpacaTradingClient` for the trading host (account, orders, positions).

### Account
- Paper account `PA3IMWLDWU8S`.
- Verified at 2026-08-29:
  - `status: ACTIVE`
  - `cash: $100,000`
  - `buying_power: $400,000`
  - `equity: $100,000`
  - `pattern_day_trader: false`
  - 0 open positions

### Order
- 1 paper order submitted: NVDA, BUY, MARKET, qty=1, time_in_force=day.
- `broker_order_id: e876e7b2-c0cc-4917-b810-2c83c3399691`
- `status: accepted`, `filled_qty: 0`, no fill yet (market closed; opens Mon 2026-08-31 09:30 ET).
- Reserved buying power: $217.89.

### Journal audit trail
- `decision_journal` row `paper-trade-20260829-NVDA-001` with full provenance:
  - `final_action: BUY_STOCK`
  - `order_intent_json: {symbol:NVDA, qty:1, side:buy, type:market, time_in_force:day}`
  - `execution_result_json: {broker_order_id, status, filled_qty, next_open}`
  - `gnn_output_json: {NVDA bias=+0.994 centrality=0.992, model_version:gnn-20260829-0004}`
  - `risk_decision_json: {decision: manual_override, note: orchestrator said NO_TRADE}`
  - `model_versions: ['gnn-20260829-0004', 'direction_h4_xgb_clf-20260829-152556', ...]`

### Watch script
- `scripts/watch_nvda_paper_fill.py` polls the order every 60s, updates the journal on every poll, and writes `realized_pnl` when the order fills.
- Currently running as background bash task `bf6n80if7`. Output monitored by persistent task `bb9cqedls`.

### Configuration
- `config/alpaca.yaml` — `run_mode: paper`, `data.base_url: data.alpaca.markets`, `trading.base_url: paper-api.alpaca.markets`.
- `.env` — `APCA-API-KEY-ID`, `APCA-API-SECRET-KEY` (redacted in logs).
- `config/risk.yaml` — hard limits: `max_order_notional_usd: 2000`, `max_leg_quantity: 10`, etc.

## What works well
- **The order is live in the broker.** We have proof that the round-trip works end-to-end.
- **The journal is auditable.** Any trade can be replayed from the row.
- **Rate limit configuration exists** (just not enforced everywhere — see failures).

## What doesn't work well
- **No P&L tracking yet.** `realized_pnl` is `NULL` for every row.
- **No position state in the agent layer.** `_load_portfolio()` is a stub.
- **No retry on 5xx.** A single transient failure loses the trade for the cycle.
- **No end-of-day reconciliation.** Manual work to compute P&L.
- **No support for OCO / multi-leg orders via the typed model.** We bypassed `OrderIntent` for the paper trade; we need a stock leg in the protocol.
