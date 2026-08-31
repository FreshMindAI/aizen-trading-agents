# Vertus / Vernus AI — reference architecture

> **Note on naming:** The user mentioned "Vertus AI" or "Vertus AI agent" as the reference architecture this stack is patterned on. There is no widely-documented product under that exact name in public sources. The closest reference points in the agentic-trading literature are: **Google's Vertex AI Agent Builder**, **LangGraph multi-agent orchestrators**, and the general pattern of **LLM-tool-using multi-agent systems for finance** (e.g. the building blocks described in [Alpaca's "Building a Multi-Agent AI Trading System on Alpaca"](https://alpaca.markets/learn/building-a-multi-agent-ai-trading-system-on-alpaca)). This document captures the design pattern the user is pointing at, and how our stack compares.

## The reference pattern (5 layers)

A "Vertus-style" multi-agent trading system is typically organized as:

```
┌──────────────────────────────────────────────┐
│ L0  Orchestrator (supervisor / planner)       │  ← LLM, plans the cycle
├──────────────────────────────────────────────┤
│ L1  Specialized agents                       │  ← regime, direction, risk, ...
│      (each is an LLM with tools)             │
├──────────────────────────────────────────────┤
│ L2  Memory / state                           │  ← vector store, episodic, journal
├──────────────────────────────────────────────┤
│ L3  Tool layer                               │  ← data APIs, broker APIs, model
│      (data, ML, GNN, news)                   │     inference endpoints
├──────────────────────────────────────────────┤
│ L4  Execution / governance                   │  ← hard risk limits, kill switch
└──────────────────────────────────────────────┘
```

The defining properties of this pattern:

1. **The LLM is the planner, not the trader.** The supervisor LLM decides *which* tools to call, in what order, and synthesizes the answer. It does not place orders.
2. **Each agent has its own tool set.** A "news" agent has news APIs. A "data" agent has data APIs. A "risk" agent has the broker account + position endpoints.
3. **Memory is per-agent.** Each agent can read its own history (episodic) and a shared journal (semantic).
4. **Governance is a hard layer, not a soft suggestion.** The risk agent is a deterministic check, not an LLM opinion.
5. **Reasoning is observable.** Every agent's chain-of-thought is logged to the journal for replay.

## How our stack maps

| Reference layer | Our equivalent | Notes |
|---|---|---|
| L0 Orchestrator (LLM supervisor) | `src/agents/supervisor.py` | Currently **mock LLM** — placeholder for a real one. |
| L1 Specialized agents | `src/agents/nodes/{regime,direction,volatility,options_structure,portfolio,risk,supervisor,execution}.py` | 8 agents; close to the reference count (5-7 typical). |
| L2 Memory / state | `data/trading.db:decision_journal` (episodic) + `v_features_underlying_v2` (semantic) | No vector store yet. **The decision_journal is the killer feature** — better than most reference systems. |
| L3 Tool layer | `src/alpaca_client.py` (data) + `src/agents/alpaca_trading.py` (broker) + `src/gnn/service.py` (GNN) + `src/ml/predict.py` (ML) | All typed. Each agent can call them. |
| L4 Execution / governance | `src/agents/risk.py` + `config/risk.yaml` | Deterministic REJECT, no LLM override. Matches the reference. |

## Where we diverge from the reference

1. **The supervisor is a placeholder.** A real Vertus-style system uses an LLM (Claude, GPT-4, Llama) for the orchestrator. We use `mock-1`. **This is the single biggest gap for the hackathon's creativity pillar.**
2. **No news agent.** Every multi-agent trading system I have seen includes a news/sentiment agent. We don't have one yet. **This is the highest-value add for the hackathon.**
3. **No vector store.** The reference systems use embeddings (e.g. Chroma, Pinecone) for episodic memory. We use SQLite + JSON columns.
4. **No LLM-as-judge.** Most reference systems have a "critic" agent that evaluates the supervisor's reasoning. We don't.
5. **GNN topology is fixed.** The reference systems (e.g. FinGPT, Alpha-GNN) use dynamic, learned topologies. Ours is mostly static.

## What the reference does better than us

- **Real reasoning.** The LLM can synthesize news + data + position into a single decision narrative.
- **Tool-use loops.** The LLM can call a tool, see the result, and decide to call another tool. Our orchestrator is a fixed pipeline.
- **Episodic memory.** Past decisions are searchable by semantic similarity, not just by `decision_id`.
- **Explainability.** The LLM's chain-of-thought is the explanation. We have structured JSON.

## What we do better than the reference

- **Deterministic risk governance.** Our `risk.py` is not an LLM; it is a hard-coded rule set. The reference systems sometimes put risk as an LLM opinion, which can be gamed.
- **Single source of truth.** One SQLite file, one schema, no parquet/CSV/JSON sprawl.
- **Auditability.** Every decision is one row. The reference systems often have 5+ files per decision.
- **Production-shaped code.** Pydantic strict models, JSON-schema contracts, typed clients. Many reference systems are notebooks.
- **A live paper trade.** We have one order in the broker. Most reference systems are backtest-only.

## The Verdict

Our architecture is **structurally aligned with the multi-agent reference pattern**. The pieces are in the right places. The biggest gap is that the LLM layer is a placeholder. The biggest opportunity is the missing news agent + dynamic news-driven GNN topology.

If we shipped three things in the next 48 hours, we would close most of the gap to the reference:

1. A real news research agent (see `pain-in-ass/hackathon-week/news-research-agent.md`).
2. A learned (GAT) GNN topology with `news_cooccurrence` and `news_sentiment` edges.
3. A live LLM supervisor (using a non-Anthropic, non-GMI endpoint — per the user's security rules).

Those three changes would put us at parity with the reference on the creativity axis, while keeping our lead on the governance + auditability axis.
