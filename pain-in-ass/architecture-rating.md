# Architecture rating

**Reviewed:** 2026-08-29
**Reviewer:** self (post-mortem of 4-phase build for the Alpaca AI Trading Agents Hackathon)
**Stack:** Data → Phase 1 ML (XGBoost) → Phase 2 GNN → Phase 3 Agents → Phase 4 Broker (Alpaca paper)

---

## TL;DR

The architecture is **structurally correct** and **demos well**, but it is **under-fed, under-calibrated, and over-gated** for the hackathon's P&L criterion. The data and agent layers are production-grade. The ML and GNN layers are research-grade and have not been pressure-tested on the 7-day live window yet. The biggest *structural* gap is that the GNN was meant to be a primary signal in the user's intent, but the current scoring weights make it a 10% tie-breaker. Flipping that hierarchy is the single most impactful change for the hackathon.

---

## Per-axis rating (1 = weak, 5 = strong)

| Axis | Score | Why |
|---|---|---|
| **Reproducibility** | 5 | Single SQLite DB, deterministic seeds, byte-deterministic snapshot builds. Walk-forward splits are chronological. Any decision can be replayed. |
| **Auditability** | 5 | Every cycle writes a `decision_journal` row with full provenance. Pydantic v2 strict models. JSON-schema contracts. |
| **Risk governance** | 5 | Risk agent is deterministic, REJECT is final, no LLM can override. Hard limits in `config/risk.yaml`. |
| **Data quality** | 4 | 254K underlying bars, 195K option bars, 14,892 contracts. IEX feed is the only weak point. No nightly backup. |
| **ML accuracy** | 3 | Direction-h4 is solid (0.63 AUC, low std). Option-h4 is borderline (0.545). rv regressor is strong (Spearman 0.62-0.65). No calibration, no ensemble, no online learning. |
| **GNN signal quality** | 2 | Standalone ROC-AUC 0.46 (below random) on the test slice. The "dynamic topology" claim is mostly cosmetic. Architecture is right; data and targets are wrong. |
| **Agent reasoning** | 3 | 8 specialized agents with a supervisor and a deterministic risk veto. Mock LLM is a placeholder. Portfolio agent is blind (no live positions). |
| **Live trading** | 3 | Paper account wired up, 1 order accepted, journal row written. No P&L tracking, no kill switch, no retry on 5xx, no stock leg in `OrderIntent`. |
| **Extensibility** | 4 | Plugin architecture for agents, model registry for GNN, contract-based GNN I/O. New symbols, agents, and topologies are config changes. |
| **Hackathon fit (P&L)** | 2 | Conservative gates block most trades. No multi-leg options, no theta sleeve, no daily P&L summary. Will end the week with 1-3 trades and 0 P&L without intervention. |
| **Hackathon fit (creativity)** | 5 | 8-agent multi-agent system, GAT/GCN/SAGE GNN registry, news-driven edges (planned), risk-as-veto governance. This is the part that wins judging. |

**Composite:** 41 / 55 ≈ **7.5 / 10** for the architecture-as-designed. **5.5 / 10** for the architecture-as-shipped for the hackathon.

---

## What the architecture does right

1. **Strict layered separation.** The orchestrator never reads `.pkl` files. The ML layer never reads the broker. The GNN service has a stub fallback. Each layer is independently testable.
2. **JSON-schema contracts at the boundaries.** The `MarketSnapshot` and `GNNOutput` are typed Pydantic models, validated at every agent → GNN and agent → broker hop.
3. **Risk as governance, not as an LLM suggestion.** This is the right call and is rare in agentic trading systems.
4. **SQLite as system of record.** One file, one source of truth, easy to inspect, easy to back up. No parquet graveyard.
5. **Decision journal is the killer feature.** Any trade is replayable from one row. This is the auditability story the hackathon judges will care about.
6. **Topology-versioned snapshots.** The `topology_version: "fixed-1"` tag is the seed of the dynamic-topology story, even if the dynamics aren't real yet.

## What the architecture does wrong

1. **The scoring formula buries the GNN.** `gnn_confirmation: 0.10` is a tie-breaker. The user wanted the GNN to be the primary signal. Fix: bump to 0.30, lower `direction_edge` to 0.15.
2. **No portfolio state in the inference layer.** Every cycle starts with "what does the world look like" but never "what do I already own". This is a fundamental design hole for any system that holds multi-day positions.
3. **No P&L reconciliation.** `realized_pnl` is in the schema, no code writes to it. We have 11 journal rows, 0 P&L values.
4. **The GNN's "dynamic topology" is not dynamic.** The static edges never change. Only one edge type is recomputed. Call this what it is.
5. **No options-structure sleeve.** The supervisor can pick long_stock or options_structure, but no theta-positive strategies (short puts, iron condors, etc.). For a 1-week P&L window, theta is the highest-EV source of return.
6. **The mock LLM is a placeholder.** The supervisor's "reasoning" is template substitution. For the creativity pillar, this is a real weakness.
7. **Risk limits are too tight for the hackathon.** `max_open_positions: 6` × `max_order_notional_usd: 2000` = $12K max deployment. We have $100K cash. We can only deploy 12% of capital.

## Compared to the reference architectures

See `reference-architectures/vertus-ai.md` for the side-by-side. The short version:

| Dimension | Our stack | Reference (multi-agent trading) |
|---|---|---|
| Agents | 8 specialized, deterministic risk | 5-7 specialized, sometimes LLM-as-risk |
| Memory | SQLite + decision journal | Vector store + episodic memory |
| Tool use | Alpaca data + trading clients | MCP servers, function calling |
| Reasoning | Mock LLM (placeholder) | Real LLM with tool use |
| Topology | Fixed graph + 1 dynamic edge type | Learned attention (GAT) over full graph |
| Evaluation | Walk-forward + paper trading | Backtest + paper + live A/B |

Our stack is **comparable in structure** but **behind in execution**. The reference architectures have real LLM reasoning, learned attention, and online learning. We have clean code and a paper account.

## The 3 things I would change tomorrow if I had a day

1. **Add a `news_cooccurrence` + `news_sentiment` edge type to the GNN.** Pull daily news from `data.alpaca.markets/v1beta1/news?symbols=AAPL,MSFT,NVDA&start=YYYY-MM-DD&end=YYYY-MM-DD`, compute a per-pair co-occurrence count and a sentiment-correlation matrix, add as two new edge types. Retrain. The hackathon judges will see a graph that *thinks about news*. (See `pain-in-ass/hackathon-week/news-research-agent.md` for the design.)

2. **Flip the GNN hierarchy.** `gnn_confirmation: 0.10 → 0.30`, `direction_edge: 0.30 → 0.15`. The user wants the GNN to be the primary signal. The XGBoost direction probability becomes a confirmation, not the gate.

3. **Add a short-put-spread strategy type to the supervisor.** When `iv_rv_gap > 0.05`, the supervisor can pick a 30-45 DTE short put spread. This is a positive-theta trade that doesn't need the directional signal to be right. Over a 1-week window, theta is more reliable than direction.

If we ship those three changes by Mon 09:30 ET, we go from "1 trade, 0 P&L" to "10-20 trades, positive theta + direction" within the first 3 days of the hackathon.
