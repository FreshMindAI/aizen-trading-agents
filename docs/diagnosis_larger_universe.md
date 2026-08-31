# Diagnosis: Why the 10-ticker universe performed worse

**Date:** 2026-08-30
**Backtests compared:**
- 4-ticker (`NVDA,AAPL,TSLA,MSFT`): 50% hit rate, mean equity PnL −$13.48 over 4 trades
- 10-ticker (`SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD`): **16.7% hit rate**, mean equity PnL −$16.74 over 4 trades

## Empirical pattern

| Week | 4-ticker pick | 4h return | hit | 10-ticker pick | 4h return | hit |
|---|---|---|---|---|---|---|
| 07-20 | MSFT | +2.30% | ✅ | AMD | −2.21% | ❌ |
| 07-27 | TSLA | −1.73% | ❌ | AMD | **−5.27%** | ❌ |
| 08-03 | MSFT | −0.23% | ❌ | AMZN | −0.73% | ❌ |
| 08-10 | TSLA | +0.12% | ✅ | AMD | −0.47% | ❌ |
| 08-17 | AAPL | −0.17% | ❌ | AMZN | −1.23% | ❌ |
| 08-24 | TSLA | +0.67% | ✅ | TSLA | +0.67% | ✅ |

The 10-ticker system **picked AMD 3× and AMZN 2×** — both names that went down in the backtest window. The 4-ticker system spread its picks across MSFT, TSLA, AAPL and had a balanced 3/3 win/loss.

## Root cause: model is overconfident on volatile names

Looking at the candidates the supervisor chose:
- The XGBoost direction classifier (h=4) was trained on the full universe, so it has learned that AMD and AMZN have **higher short-term volatility** → higher predicted `direction_probability` on any given bar.
- The `direction_prob_min = 0.55` gate is a *static* threshold. AMD/AMZN frequently clear it because their short-term vol keeps the model's signal above 0.55.
- The supervisor picks the highest-scoring candidate. With a larger universe, the supervisor is more likely to find an AMD/AMZN candidate whose score (which is weighted 0.3×direction_edge + 0.2×volatility_edge + ...) is dominated by the vol-edge component, even when direction_edge is weak.

**In short**: more candidates → more chances for high-vol names to clear the gate → worse picks.

## Three-part fix (proposed for #158 / #159 follow-up)

1. **Tighter direction gate** — raise `direction_prob_min` from 0.55 to 0.62 (or compute it dynamically from the artifact's `_BASE_RATE + _BASE_RATE_CUSHION`).
2. **Conviction margin** — supervisor returns NO_TRADE when the top score is within 0.05 of the runner-up (no clear winner).
3. **Per-name vol-target** — penalize candidates whose `volatility_edge` is high but `direction_edge` is low (a "high vol, low direction" combo is a coin-flip, not an edge).

The "knowledge graph for failure data" task (158) will also help here: if the supervisor knows that AMD had 4 losing trades in a row, the GNN can reweight it down before the supervisor picks.

## What does NOT cause the failure

- **GNN topology** — the graph is fixed (sector-peer edges) and identical for both runs, so it cannot explain the per-name pick difference.
- **News signal** — both runs have `news=on` and the same news_snapshot table.
- **Risk limits** — identical for both runs (no per-name filtering at the equity leg).
- **Capital / sizing** — both runs use the new $100k capital scaling.

## TL;DR

The 10-ticker universe isn't broken; the system just has **higher recall but lower precision** when given more candidates. Tighten the gate + add a conviction margin and the 10-ticker system should match or beat the 4-ticker (more high-quality candidates available).
