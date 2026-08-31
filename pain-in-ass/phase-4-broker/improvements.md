# Phase 4 — Broker: improvements

## Highest impact (do for the hackathon)

### 1. Add a stock leg to the OrderIntent protocol
- New variant: `OrderIntent(legs=[Leg(side=buy, quantity=1, symbol="NVDA", asset_class="us_equity")])` where the `option_type/strike/expiry` fields are optional when `asset_class="us_equity"`.
- **Effort:** 2 hours. **Impact:** every agent that emits a stock intent now goes through the typed model.

### 2. End-of-day P&L reconciliation
- New file: `scripts/eod_pnl.py`.
- Reads the day's `decision_journal` rows with `execution_result_json.status='filled'`, joins to current positions, writes a `daily_pnl` row.
- **Effort:** 1 day. **Hackathon value:** the primary judging criterion (P&L).

### 3. Add retry + circuit breaker to `AlpacaTradingClient`
- On 5xx, retry up to 3 times with exponential backoff (1s, 2s, 4s).
- On repeated failures, trip a circuit breaker for 60s.
- **Effort:** 4 hours.

### 4. Real-time positions in the inference layer
- `inference._load_portfolio()` should call `AlpacaTradingClient.list_positions()` and return `PortfolioPosition` objects.
- **Effort:** 2 hours.

### 5. Real-time account equity in the inference layer
- `inference._account_equity()` and `_account_cash()` should call `AlpacaTradingClient.get_account()`.
- **Effort:** 2 hours.

## Medium impact

### 6. Add a "kill switch" to the risk agent
- If equity drops below 95% of starting equity, halt all new orders.
- If a single position drops below -X% in a day, auto-close.
- **Effort:** 4 hours.

### 7. Add support for OCO and bracket orders
- For a covered call, attach a take-profit + stop-loss bracket to the long stock.
- **Effort:** 1 day.

### 8. Add a `dry-run` mode that simulates fills at the last close
- For backtests and strategy validation without touching the broker.
- **Effort:** 1 day.

### 9. Add a `live` run mode that requires explicit env confirmation
- Currently the only thing protecting us from going live is the `paper` default in `config/alpaca.yaml`.
- Add a runtime check: if `run_mode: live`, require `LIVE_TRADING_CONFIRMED=true` env var.
- **Effort:** 2 hours.

## Low impact

### 10. Move the .env secrets to OS keyring
- `keyring.set_password("aizen", "APCA-API-KEY-ID", key)` on setup, `keyring.get_password(...)` on read.
- **Effort:** 1 day.

### 11. Add per-symbol trade history export
- CSV export of all fills per symbol for tax accounting.
- **Effort:** 4 hours.

### 12. Add a `trade` audit log table separate from `decision_journal`
- The decision_journal is for decisions. A separate `trade_log` table for fills (one row per execution) would make P&L calculations easier.
- **Effort:** 1 day.
