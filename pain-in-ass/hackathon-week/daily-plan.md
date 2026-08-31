# Hackathon week — Mon-Fri plan

> **Window:** Mon 2026-08-31 09:30 ET → Fri 2026-09-04 16:00 ET
> **Goal:** end the week with positive P&L on the paper account + a submission-ready architecture write-up.

## State entering the week (as of Sat 2026-08-29)

- 1 paper trade pending fill (NVDA market BUY, accepted, awaiting Mon open).
- 10 underlying bars universe, 14,892 option contracts.
- Phase 1 XGBoost direction-h4 at 0.6304 ROC-AUC (3-fold walk-forward).
- Phase 2 GNN `gnn-20260829-0004` loaded.
- Orchestrator: 8 agents, mock LLM, NO_TRADE gate too strict.
- 11 decision_journal rows, 0 P&L values.

## Sat 30 Aug — preparation day

- [ ] Extend universe to 15 names: add JPM, GS, COIN, NFLX, BA.
- [ ] Refresh data: bars for 5 new names, option contracts, option bars.
- [ ] Update `config/risk.yaml:23` `allowed_underlyings`.
- [ ] Update `config/gnn.yaml:22-32` + `src/gnn/constants.py:27-38`.
- [ ] Retrain Phase 1 XGBoost with new universe (~30 min).
- [ ] Retrain Phase 2 GNN with `--n-snapshots 250 --architecture gat-32-16-1` (~30 min).
- [ ] Write `pain-in-ass/hackathon-week/news-research-agent.md` (done Sat 29).
- [ ] Build `src/agents/nodes/research.py` (skeleton, mock data).
- [ ] Test the orchestrator with the new universe + scoring weights.

## Sun 31 Aug — research agent + GNN news edges

- [ ] Add `news_snapshot` table + migration.
- [ ] Implement `fetch_news()` against Alpaca's `/v1beta1/news` endpoint.
- [ ] Wire the research agent into the orchestrator dependency graph.
- [ ] Add `news_cooccurrence` and `news_sentiment_correlation` edge types.
- [ ] Retrain the GNN with the new edges (~30 min).
- [ ] Walk-forward backtest the new model on the last 60 days (~15 min).
- [ ] Adjust scoring weights based on backtest P&L.

## Mon 1 Sept — first live day

- [ ] 09:30 ET: NVDA order fills (if market opens normally).
- [ ] 09:35 ET: orchestrator runs first cycle of the week.
- [ ] Submit 2-3 directional longs (XGB + GNN + news agree).
- [ ] 12:00 ET: mid-cycle review. Adjust thresholds if too many / too few signals.
- [ ] 16:05 ET: end-of-day P&L reconciliation. Write `daily_pnl` row.
- [ ] 16:30 ET: review + adjust.

## Tue 2 Sept — options sleeve

- [ ] If directional sleeve is working: add the short-put-spread strategy type.
- [ ] Pick 1-2 names where `iv_rv_gap > 0.05`.
- [ ] Sell 1-2 put spreads, 30-45 DTE, $5 wide.
- [ ] 16:05 ET: EOD P&L.
- [ ] Document what worked, what didn't.

## Wed 3 Sept — iteration

- [ ] Walk-forward backtest over Tue + Wed to validate the new sleeve.
- [ ] Adjust risk limits if needed (currently `max_order_notional_usd: 2000` may be too tight).
- [ ] Add the third sleeve: protective puts on directional longs.

## Thu 4 Sept — stabilization

- [ ] All 3 sleeves running. No new strategy types.
- [ ] 16:05 ET: EOD P&L.
- [ ] Write the submission write-up: `ARCHITECTURE.md`, `RESULTS.md`, `REPRODUCIBILITY.md`.
- [ ] Export `decision_journal` as a CSV for the judges.

## Fri 5 Sept — submission

- [ ] 09:30 ET: open. Run the orchestrator.
- [ ] 12:00 ET: stop opening new positions. Let existing ones run.
- [ ] 15:45 ET: close any open positions we don't want to hold over the weekend.
- [ ] 16:05 ET: final EOD P&L.
- [ ] 16:30 ET: final write-up, push to lablab.ai.

## P&L target

| Scenario | Trades | Win rate | Avg P&L/trade | Total P&L |
|---|---|---|---|---|
| Bear | 10 | 50% | -$50 | -$250 |
| Base | 30 | 60% | +$30 | +$540 |
| Bull | 60 | 65% | +$50 | +$1,950 |

The bear case is bad. The base case gets us a respectable return on $100K. The bull case puts us in the top 3 for the creativity + P&L combination.

## What we will NOT do

- ❌ Switch to live trading (paper only for the hackathon).
- ❌ Use the GMI/MiniMaxAI/MiniMax-M3 endpoint.
- ❌ Print the ANTHROPIC_AUTH_TOKEN.
- ❌ Commit anything to git (per the user's standing rule).
- ❌ Use a real LLM provider that costs money (mock + lexicon only).

## Open questions for the user

1. **Should we extend the universe to 15 names, or keep it at 10 for the first day and add names as we confirm they work?**
2. **Should we enable a real LLM (OpenAI, local Ollama) for the supervisor, or stick with mock?**
3. **Do we want the news agent to be lexicon-based (free, deterministic) or FinBERT-based (better, ~400MB download)?**
4. **What is the maximum drawdown you are willing to accept on the paper account during the week?**
