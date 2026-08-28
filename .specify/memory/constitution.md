# Aizen Trading Constitution

## Core Principles

### I. Determinism over Autonomy
Trading decisions are reproducible and auditable. The risk engine is a pure
function of (config, portfolio, OrderIntent) - never an LLM. ML features
are pure SQL; agent observations and proposals are Pydantic models. The
LLM is allowed to EXPLAIN, never to OVERRIDE.

### II. Schema-First Communication
Every inter-agent message conforms to a versioned Pydantic contract in
`src/agents/protocol.py`. Unknown fields are rejected (extra=forbid);
NaN/inf are coerced to None. The decision_journal stores the same JSON
that flows over the wire, so reasoning outlives the process.

### III. Defense in Depth at the Execution Boundary
OrderIntents are validated three times: by the supervisor (semantic),
by the risk engine (hard limits), and by the execution node (shape +
broker safety). The broker never sees an intent that bypassed any layer.

### IV. NO_TRADE is a First-Class Outcome
The system is not required to trade every cycle. The supervisor may emit
NO_TRADE on agent conflict, low confidence, empty candidate set, or risk
rejection. NO_TRADE is logged identically to PROCEED/REJECT.

### V. Config over Code
Endpoints, agent switches, scoring weights, risk limits, and run mode all
live in `config/*.yaml` or environment variables. The same code runs
against paper, dry-run, or (with explicit override) live environments.
Secrets live only in `.env` (gitignored).

### VI. Library-First, CLI-Wrapped
Every Phase-3 module is importable and testable as a library. The CLI in
`src/agents/run.py` is a thin wrapper that wires the orchestrator, journal,
and broker into a single command.

## Architectural Constraints

* **Storage**: SQLite is the system of record for both raw market data and
  agent reasoning. No external database is required.
* **Models**: Phase-1 XGBoost artifacts in `models/` (gitignored). Phase-2
  GNN ships as a stub (`InferenceService.gnn_output`).
* **Broker**: Alpaca only. MCP server not required. The Trading API is the
  only execution path; the Data API is the only data path.
* **LLM providers**: Anthropic (default, native Messages API), OpenAI
  (Chat Completions translation), Mock (deterministic). The canonical
  request shape is Anthropic-shaped for parity with Claude Code.

## Development Workflow

* **Test-first**: every agent, the risk engine, the scoring formula, and
  the LLM provider factory have pytest coverage. New behavior ships with
  tests in the same commit.
* **Decision trace**: every cycle writes a `decision_journal` row with
  market_state_hash, all agent_messages, the selected_strategy, the
  risk_decision, and the execution_result. The journal is the only
  source of truth for post-hoc analysis.
* **Run modes**: `--mode dry-run` for local development and CI,
  `--mode paper` for broker integration, `--mode live` only with explicit
  override and a separate review.

## Governance

* Constitution supersedes ad-hoc decisions in code, config, and PRs.
* Amendments require a written rationale, an updated constitution file,
  and a migration plan for any in-flight decisions.
* The LLM provider, the broker endpoint, and the risk limits are the
  three knobs that change most often - keep them config-only.

**Version**: 1.0.0 | **Ratified**: 2026-08-29 | **Last Amended**: 2026-08-29
