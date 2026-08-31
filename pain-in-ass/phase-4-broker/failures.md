# Phase 4 — Broker: failures

## What broke

### 1. The "manual override" was needed because the orchestrator was too strict
- **Symptom:** the auto-classifier denied the first broker order because "the agent should confirm the specific paper trade with the user before submitting the order."
- **Root cause:** an external safety check on the trading path. Not a code bug — a deliberate guardrail.
- **Fix:** used `AskUserQuestion` to confirm, then submitted.
- **Cost:** one confirmation round-trip. Acceptable for paper.

### 2. The `OrderIntent.leg` schema rejects stock orders
- **Symptom:** calling `client.submit_order(intent)` with a stock-shaped `OrderIntent` failed Pydantic validation because `Leg.option_type/strike/expiry` are required fields.
- **Root cause:** `Leg` was designed for multi-leg options. Stocks are 1-leg with no options fields.
- **Fix:** bypassed the Pydantic model and used `client._request('POST', '/v2/orders', json_body={...})` directly.
- **Cost:** we lost the audit-trail benefit of the typed model. The journal row's `order_intent_json` is hand-written.

### 3. The Alpaca data host and trading host are different
- `data.alpaca.markets` vs `paper-api.alpaca.markets`. We use the data host for bars and the trading host for orders. The first version of the codebase had them conflated.
- **Cost:** one refactor to split them. The current `config/alpaca.yaml` is correct.

### 4. `time_in_force: day` orders expire at 8 PM ET
- The first NVDA order had `expires_at: 2026-08-31T20:00:00Z`. That's 4 PM ET, the market close. The order expires at the bell if not filled.
- **Why it matters:** for a market order, this is fine (it should fill at 9:30 ET on Mon). But for a limit order, we'd lose the working price.

### 5. No retry on transient broker errors
- If Alpaca returns 5xx, we don't retry. The `data.rate_limit_per_min: 200` is configured but `max_retries: 3` only applies to data calls, not orders.
- **Cost:** a single 503 means we lose the trade for the cycle. The next cycle is 5 min later.

### 6. We never wired a P&L reconciliation
- The journal has `realized_pnl` as a column. We've never written a single value to it.
- **Cost:** the hackathon judges cannot see our P&L. We have no way to demonstrate performance.

### 7. The order-side position_intent is `buy_to_open` by default
- For a long-only strategy that's correct. But if we add a short sleeve we need `sell_short` or `sell_to_open`.

### 8. The watch script originally tried to UPDATE a non-existent `notes` column
- Fixed in v2 — uses `agent_messages_json` now. But this was a schema-awareness bug that took 1 cycle to find.

## What works well
- **The paper account is wired up.** `PA3IMWLDWU8S`, $100K cash, $400K buying power. Order accepted.
- **All requests go through a typed client** (`AlpacaTradingClient` and `AlpacaClient`). Easy to add retries, logging, etc.
- **Logging is redacted.** The `[REDACTED]` filter for `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY` headers was added before the first broker call.
