# News research agent — design

> **Goal:** add a news-driven signal to the orchestrator and to the GNN topology, so the model "thinks about news" the way a human trader does. This is the single highest-value add for the Alpaca AI Trading Agents Hackathon's creativity pillar.

## What we have today

- The orchestrator has 8 agents: `regime, direction, volatility, options_structure, portfolio, risk, supervisor, execution`.
- None of them read news.
- The GNN topology has 5 edge types: `sector, supplier, customer, etf_membership, correlation`. None of them are news-driven.

## What we add

### A. A new `research` agent

**File:** `src/agents/nodes/research.py`

**Inputs:** the universe (10-15 symbols), the current `MarketSnapshot` timestamp.

**Outputs:** a `ResearchOutput` Pydantic model:
```python
class ResearchOutput(BaseModel):
    version: Literal["1.0"]
    timestamp: str
    per_symbol: dict[str, SymbolResearch]

class SymbolResearch(BaseModel):
    sentiment: float          # in [-1, +1], 0 = neutral
    volume: int               # article count in last 24h
    topics: list[str]         # top 3 extracted topics
    last_article_at: str | None
```

**How it works:**

1. Calls Alpaca's news endpoint: `GET https://data.alpaca.markets/v1beta1/news?symbols=AAPL,MSFT,NVDA&start=YYYY-MM-DD&end=YYYY-MM-DD&limit=50`.
2. For each article, count the symbols mentioned (the API returns a list per article).
3. Compute a simple lexicon-based sentiment score: positive words − negative words / total words. Use a finance-tuned lexicon (e.g. Loughran-McDonald).
4. Output the per-symbol aggregates.

**Why lexicon-based, not LLM-based:** the user has explicit rules against the GMI/MiniMaxAI/MiniMax-M3 endpoint, and the LLM provider is `mock` by default. A lexicon is deterministic, free, and runs in <1s for 50 articles. We can layer an LLM on top later.

### B. Two new GNN edge types

**File:** `src/gnn/build_edge_features.py` — add two new functions.

**`news_cooccurrence` edge:** for every pair (A, B) of symbols that appear together in ≥ 2 articles in the last 24h, emit an edge with `weight = log(1 + count)`.

**`news_sentiment_correlation` edge:** for every pair (A, B) where the rolling 5-day correlation of daily sentiment is |ρ| > 0.5, emit an edge with `weight = |ρ|`.

**Add to `EDGE_REASONS`:**
```python
EDGE_REASONS: tuple[str, ...] = (
    "sector",
    "supplier",
    "customer",
    "etf_membership",
    "correlation",
    "news_cooccurrence",
    "news_sentiment_correlation",   # ← new
)
```

### C. A daily news snapshot table

**File:** `sql/news_snapshot.sql` (new migration)

```sql
CREATE TABLE news_snapshot (
    timestamp   TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    sentiment   REAL,
    article_count INTEGER,
    topics_json TEXT,
    raw_json    TEXT,           -- full article payload for replay
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (timestamp, symbol)
);
CREATE INDEX idx_news_symbol_ts ON news_snapshot(symbol, timestamp DESC);
```

The research agent writes one row per (timestamp, symbol) per day. The GNN edge builder reads from this table.

### D. Wire the research agent into the orchestrator

**File:** `src/agents/orchestrator.py` — add the research node to the dependency graph.

```
regime -> direction -> volatility -> research -> options_structure -> portfolio -> risk -> supervisor -> execution
```

The research agent runs *after* the ML agents (so it can correlate sentiment with the XGBoost direction) and *before* the options-structure agent (so the candidate scoring can include news sentiment).

**New scoring weight:**
```yaml
scoring:
  weights:
    direction_edge: 0.20            # was 0.30
    volatility_edge: 0.15           # was 0.20
    option_expected_return: 0.20    # was 0.25
    probability_profit: 0.15        # was 0.20
    gnn_confirmation: 0.20          # was 0.10
    news_sentiment: 0.10            # ← new
    spread_penalty: 0.10
    liquidity_penalty: 0.05
    portfolio_risk_penalty: 0.15
```

The news sentiment is a positive bias on the candidate's score when sentiment > 0 and a negative bias when sentiment < 0.

### E. Use the GNN news edges in the topology

**File:** `src/gnn/build_snapshot.py` — when building the snapshot, include the news edges from the last 24h of `news_snapshot` rows.

This is the "dynamic topology" the spec promised: every snapshot, the graph re-wires based on the news of the day. Two stocks mentioned together in 3 articles today get a stronger edge than they had yesterday.

## Implementation order

1. **`sql/news_snapshot.sql` + migration runner** — 1 hour
2. **`src/agents/alpaca_data.py::fetch_news(symbols, days_back)`** — 2 hours
3. **`src/agents/nodes/research.py`** — 4 hours
4. **`src/gnn/build_edge_features.py` — add news edges** — 4 hours
5. **Update `EDGE_REASONS` + retrain GNN** — 1 day (incl. retraining)
6. **Wire research into orchestrator** — 4 hours
7. **Update scoring weights** — 1 hour
8. **Tests** — 1 day

**Total: ~4 days of focused work.** Achievable in the hackathon week if we start Sun 30 Aug.

## Hackathon-week daily plan

| Day | Goal | Deliverable |
|---|---|---|
| **Sat 29 Aug (today)** | Design + folder structure (this doc + folder) | `pain-in-ass/hackathon-week/news-research-agent.md` |
| **Sun 30 Aug** | Implement the research agent + news table + Alpaca fetch | `src/agents/nodes/research.py` working with mock data |
| **Mon 31 Aug (market open)** | Add news edges to GNN, retrain, wire orchestrator | GNN trained with news edges, 1-3 trades submitted |
| **Tue 1 Sept** | Backtest the new topology on the last 30 days | Walk-forward report with news edges |
| **Wed 2 Sept** | Iterate on scoring weights based on P&L so far | Optimized weights |
| **Thu 3 Sept** | Stabilize, document, prep submission | `ARCHITECTURE.md` + `RESULTS.md` |
| **Fri 4 Sept** | Final P&L snapshot, submit | `daily_pnl` table populated |

## Why this wins the hackathon

1. **Creativity pillar (50% of judging):** "we added a news-driven dynamic GNN topology that re-wires the graph daily based on real-time Alpaca news." This is a sentence the judges will remember. No other team is doing this.
2. **P&L pillar (50% of judging):** news sentiment is a real, measurable edge. A lexicon-based sentiment on Alpaca news, combined with our existing ML signal, should give a 1-3% lift on the win rate. Over 50 trades, that's the difference between +$200 and +$800.
3. **Risk pillar (implicit):** the news agent runs *before* the risk agent, so the risk layer is unchanged. The news signal is bounded in [-1, +1], so it cannot dominate the score.

## Risks

1. **The Alpaca news API may not have data for all 10 symbols.** Test with a quick `curl` before committing.
2. **The lexicon sentiment is crude.** A better alternative is a local FinBERT model (~400 MB), but that requires PyTorch + a download. For the hackathon, lexicon is the right tradeoff.
3. **News edges may be noisy.** If two stocks are mentioned together 1 time, that's noise. Threshold at ≥ 2 co-occurrences.
4. **The retraining step is slow.** Bumping `n_snapshots` to 250 will take ~10 min. We should run it in the background and not block the market open.
