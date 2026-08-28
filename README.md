# Aizen Trading - Alpaca Options Alpha Agents

Local research pipeline for the **Phase-1 ML** stack and the **Phase-3
multi-agent** options decision system, against the Alpaca paper API.

```
Alpaca market data -> SQLite (immutable raw) -> pure-SQL features/labels
        |                       |                       |
        v                       v                       v
   option_contracts      v_features_underlying    ml_training_dataset
        |                       |                       |
        v                       v                       v
   option_bars           src.ml.train (XGBoost)   src.ml.predict
                                                        |
                                                        v
                                              src.agents.inference
                                                        |
                                                        v
                                  LangGraph state machine (Phase 3)
                                                        |
                                            Regime / Direction / Volatility /
                                            Options Structure / Portfolio /
                                            Supervisor / Risk / Execution
                                                        |
                                                        v
                                                decision_journal
                                                        |
                                                        v
                                          Alpaca Trading API (paper)
```

## Layout

```
sql/                  schema + view scripts (applied in sorted order)
src/                  collectors, ML trainers, multi-agent package
  agents/             Phase 3 multi-agent system
    protocol.py       Pydantic message contracts (versioned)
    llm/              LLM provider abstraction (anthropic, openai, mock)
    inference.py      ML/GNN -> MarketSnapshot bridge
    risk.py           deterministic risk engine
    scoring.py        candidate-strategy scoring (linear, weights in YAML)
    alpaca_trading.py thin Alpaca Trading API client
    journal.py        SQLite decision_journal persistence
    graph.py          LangGraph orchestrator + sequential fallback
    nodes/            one file per specialized agent
config/               YAML: alpaca, agents, risk, settings
tests/                pytest suite
data/                 trading.db (gitignored)
models/               XGBoost artifacts (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env       # add your paper Alpaca keys
python -m src.db init      # apply sql/*.sql
```

### Phase-1 ML (data + train + predict)

```bash
python -m src.download_stocks --symbols AAPL,SPY --start 2026-08-17T00:00:00Z
python -m src.ml.train --task direction --horizon 4
python -m src.ml.predict --underlyings NVDA,AAPL
```

### Phase-3 multi-agent (orchestrator)

```bash
# single decision cycle in dry-run mode (no broker calls)
python -m src.agents.run --once --mode dry-run

# paper trading loop: every 5 minutes, real Alpaca paper API
python -m src.agents.run --loop --interval 300 --mode paper

# Anthropic as the LLM provider (ANTHROPIC_API_KEY must be set)
AIZEN_LLM_PROVIDER=anthropic python -m src.agents.run --once --mode dry-run

# OpenAI as the LLM provider
AIZEN_LLM_PROVIDER=openai python -m src.agents.run --once --mode dry-run
```

## Configuration

All non-secret config is in `config/*.yaml`:

| File | Purpose |
|---|---|
| `settings.yaml` | Project metadata, paths, logging |
| `alpaca.yaml`   | Data + trading base URLs, rate limits, run mode |
| `agents.yaml`   | Agent switches, scoring weights, thresholds, LLM provider |
| `risk.yaml`     | Hard risk limits (per-trade, portfolio, universe) |

Endpoints and provider keys are env-driven so the same code path flips
between providers without code edits:

| Variable | Used by |
|---|---|
| `AIZEN_LLM_PROVIDER`, `LLM_PROVIDER` | Selects `mock` / `anthropic` / `openai` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` | Anthropic provider |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_ORG_ID` | OpenAI provider |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | Both data + trading clients |
| `ALPACA_DATA_URL`, `ALPACA_TRADING_URL` | Override hosts (e.g. internal gateway) |
| `RUN_MODE` | `paper` / `dry-run` / `live` (one-shot override) |

## Phase-3 architecture (doc section 2)

```
DATA / MARKET STATE
        |
  Phase 1 ML   Phase 2 GNN
  dir / RV /   graph embeddings /
  option score topology signals
        |
  MARKET STATE BUS  (MarketSnapshot)
        |
  LangGraph orchestrator
        |
  Regime -> Direction -> Volatility -> Options Structure -> Portfolio
                                                       -> Supervisor
                                                       -> Risk (deterministic)
                                                       -> Execution
        |
  decision_journal  +  Alpaca Trading API
```

Key design points (doc section 3):

* Supervisor/orchestrator pattern; agents do not call each other directly.
* ML/GNN are numerical services, not LLM agents.
* The risk engine is a pure function of (config, portfolio, OrderIntent) -
  the LLM cannot override a `REJECT`.
* OrderIntent is the only payload the broker ever sees; the execution node
  validates it twice (defense in depth).
* LLM explains scores; the math is deterministic and lives in `scoring.py`.

## LLM provider abstraction

The canonical request shape is **Anthropic Messages API** (the same shape
the Claude Agent SDK and Claude Code use internally). Each provider is a
thin adapter:

| Provider | Backend |
|---|---|
| `AnthropicProvider` | Native `POST /v1/messages` (Anthropic) |
| `OpenAIProvider`    | Translates to `POST /v1/chat/completions` (OpenAI) |
| `MockProvider`      | Deterministic (no network); default for paper / tests |

The orchestrator only ever calls `provider.complete(request)` and gets back
a normalized `LLMResponse { text, tool_calls, stop_reason, usage }`. The
provider is selected at process start; runtime swapping isn't supported
because the canonical shape stays identical.

## Tests

```bash
python -m pytest tests/      # 30 tests, <1s
```

Covers: Pydantic protocol roundtrip, deterministic risk engine, scoring
formula, LLM provider factory + init-failure modes, end-to-end orchestrator
(sequential fallback and LangGraph path).

## Decision journal

Every cycle writes one row to `decision_journal` with:

* the full `MarketSnapshot` (underlyings + options + portfolio)
* every `AgentMessage` envelope (sender, receiver, payload)
* every `AgentObservation` (agent_id, confidence, signal, evidence)
* the `selected_strategy`, `risk_decision`, `order_intent`
* the `execution_result` (broker response or dry-run summary)
* `model_versions`, `topology_version`, `market_state_hash`

The journal is the only place reasoning outlives the process. `NO_TRADE` is
a first-class outcome.

## Security

Keys live only in `.env` (gitignored). Paper credentials only. Never log
secrets; the LLM never sees an API key - the LLM only sees structured
inputs and emits structured outputs. Rotate immediately in the Alpaca
dashboard if exposed.
