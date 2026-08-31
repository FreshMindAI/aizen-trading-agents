# Plan: Harden LLM API Retry + Surface Failure Causes in Cycle Trace

## Context

The user asked: *"api failures for llm call are need to handling with retries and tell me what is failure cause of failing on large universe."* Two requirements:
1. **Bounded retry with exponential backoff** for transient LLM transport errors (429, 5xx, timeout, connection reset) — 3–5 attempts.
2. **Visible failure cause** — the operator must be able to read why a cycle's LLM call failed without grepping the log. The cycle trace (JSONL per cycle) is the canonical per-cycle artifact; failure reasons belong there.

Today both `AnthropicProvider` and `OpenAIProvider` carry their own duplicate retry loop (`max_retries=2` default = 3 attempts). The loops are identical in structure (try → on `RequestException`/`5xx`/`408|429` retry with backoff → on `4xx` give up → on parse error raise). The trace step at `Orchestrator.run_cycle` in `src/agents/graph.py` does not capture LLM call attempts or failure modes — failures land as `cycle_error: LLMError: ...` on the `error` step but the trace does not show *which* provider, *how many* attempts, or *what* the last error was.

**Intended outcome:** after this change, an operator inspecting `models/cycle_traces/cycle_{decision_id}.jsonl` sees a `llm_call` step with `provider`, `model`, `attempts`, `last_status`, `last_error`, and either `success: true` or a populated `reasons` list when the call failed permanently.

## Approach

Three layers, each independently testable.

### Layer 1 — Shared retry helper in `src/agents/llm/base.py`

Add `_call_with_retry(provider_name, fn, *, max_attempts, base_delay_s, max_delay_s, retry_after_max_s)` to `LLMProvider`. The helper:

- Calls `fn()` (which returns `(status_code, body_or_error)`).
- Retries on: transport errors (`requests.RequestException` — `Timeout`, `ConnectionError`, `ChunkedEncodingError`), HTTP 5xx, HTTP 408, HTTP 429, and HTTP 5xx-class responses. Does NOT retry 4xx (except 408/429) — those are the caller's fault.
- Backoff: `min(base_delay * 2 ** (attempt-1), max_delay)`, with `Retry-After` honored up to `retry_after_max_s` (default 30s).
- Returns the parsed response on success, raises `LLMError` after the final attempt.
- Calls an optional `on_retry(attempt, exc_or_status, sleep_s)` callback between attempts so callers can record telemetry.

Default for the helper: `max_attempts=4` (3 retries after the first attempt = 4 total — middle of the 3-5 range the user asked for; bumpable via `AIZEN_LLM_MAX_ATTEMPTS` env var).

### Layer 2 — Wire `AnthropicProvider.complete` and `OpenAIProvider.complete` through the helper

Both providers currently inline their retry loop. Refactor each `complete()` to:

1. Capture per-attempt telemetry in a `LLMCallTelemetry` dataclass (`provider`, `model`, `attempts`, `last_status`, `last_error`, `total_sleep_s`, `started_at`, `ended_at`).
2. Delegate the retry loop to `_call_with_retry`, passing `on_retry=telemetry.record_attempt`.
3. Stash the `LLMCallTelemetry` on `self.last_call` after the call returns (or raises) so the orchestrator can read it.

This keeps the per-provider request/response translation intact (which is the bulk of each file) while removing the duplicated retry code.

`MockProvider` is unchanged — it has no network so no retry. The base class helper lives in `base.py`; `MockProvider` simply doesn't call it.

### Layer 3 — `llm_call` trace step + orchestrator integration

`src/agents/trace.py`:

- Add `build_llm_step(telemetry: LLMCallTelemetry) -> StepRecord` that returns:
  - `step="llm_call"`
  - `fields={"provider", "model", "attempts", "last_status", "total_sleep_s", "duration_ms"}`
  - `success=True` only if `attempts == 1` and HTTP 2xx; `success=False` on retry or final failure
  - `reasons=[]` on success, otherwise `["attempt N: <status or exc type>: <message>"]` for each attempt plus `["giving up after N attempts: <last error>"]` on permanent failure.
- Move the `LLMCallTelemetry` dataclass definition into `trace.py` (or a small `src/agents/llm/telemetry.py` so `base.py` and `trace.py` both import it without circular deps — pick the small-module option to avoid touching `base.py`'s import block).

`src/agents/graph.py` `Orchestrator.run_cycle`:

- After the graph invoke (or sequential driver), collect the `last_call` telemetry from each LLM-using node into a list. For LangGraph, the simplest approach is to read `self.llm.last_call` after the cycle (this only covers the *last* agent's call — useful but not full-fidelity). For full fidelity, each `build_node` would need to push its own `LLMCallTelemetry` into a cycle-scoped list on the state. **Recommendation:** take the simpler path first — record `self.llm.last_call` once after the cycle (the dominant pattern is one LLM call per agent, and a single `llm_call` step is more useful than zero). Document the limitation in the docstring; full per-agent telemetry is a follow-up.
- Add a single `trace.add(build_llm_step(self.llm.last_call))` after the existing per-agent step loop. When no LLM call was made (e.g. short-circuited on missing data, or mock provider), `last_call is None` and the step is skipped.

### Layer 4 — Tests in `tests/test_llm_retry.py`

Six tests, all `AIZEN_LLM_PROVIDER=mock` for the unrelated paths:

1. `test_retry_succeeds_on_transient_5xx`: 2× 503 then 200 → succeeds on attempt 3, telemetry.attempt_count == 3, success=True.
2. `test_retry_succeeds_on_429_with_retry_after`: 429 with `Retry-After: 0.05` then 200 → succeeds, `total_sleep_s >= 0.05`.
3. `test_retry_gives_up_after_max_attempts`: 4× 503 → raises `LLMError`, telemetry.attempt_count == 4, last_status == 503.
4. `test_no_retry_on_4xx`: 400 → raises immediately, telemetry.attempt_count == 1.
5. `test_retry_on_connection_error`: 1× `ConnectionError` then 200 → succeeds on attempt 2.
6. `test_llm_call_step_in_cycle_trace`: end-to-end — drive a `MockProvider` (already retries N/A), then a stub `AnthropicProvider` with a mocked `requests.Session` that returns 503/200; assert the per-cycle trace file contains a `llm_call` step with `success=True` and `attempts=2` on the success path, and `success=False` with `reasons` on the failure path.

Tests use a `MockTransport` (a class with `.post(url, **kw)` returning a `_FakeResponse`) injected into `provider._session` so we don't need `requests_mock` (which isn't in `requirements.txt`).

## Critical files to modify

| File | Change |
|---|---|
| `src/agents/llm/base.py` | Add `_call_with_retry(...)` classmethod to `LLMProvider`. Add `LLMError` already exists; add `LLMTransportError` subclass for retryable transport errors. |
| `src/agents/llm/anthropic_provider.py` | Replace inline retry loop with `_call_with_retry`; instantiate `LLMCallTelemetry`; stash `self.last_call`. Keep request/response translation. |
| `src/agents/llm/openai_provider.py` | Same shape as anthropic. |
| `src/agents/llm/telemetry.py` | **New.** `LLMCallTelemetry` dataclass + `LLMTransportError`. |
| `src/agents/trace.py` | Add `build_llm_step(telemetry)` builder. |
| `src/agents/graph.py` | `Orchestrator.run_cycle` reads `self.llm.last_call` after the cycle and emits one `llm_call` step. |
| `tests/test_llm_retry.py` | **New.** 6 tests as above. |

## Reusable helpers already in the repo

- `LLMError` in `src/agents/llm/base.py:88` — already raised on transport / parse failure. Keep the existing contract; add `LLMTransportError` as a *subclass* so existing `except LLMError` catches still work.
- `StepRecord` in `src/agents/trace.py:55` — already supports `success`, `reasons`, `fields`, `duration_ms`. Reuse directly; no shape change.
- `CycleTraceBuilder.add` in `src/agents/trace.py:116` — already appends arbitrary `StepRecord`; the new `build_llm_step` plugs into the same `trace.add(...)` pattern.
- `MockProvider` in `src/agents/llm/mock_provider.py` — unchanged. Mock has no `last_call` (no real call), so the orchestrator's `last_call` check naturally skips the trace step for mock cycles.

## Verification (end-to-end)

1. `python -m pytest tests/test_llm_retry.py -v` — 6 tests pass.
2. `python -m pytest tests/test_llm_providers.py -v` — existing tests still pass (the refactor must not change observed behavior on the happy path).
3. `python -m pytest tests/test_alpaca_mcp.py tests/test_research_agent.py -v` — the orchestrator + trace path is still green.
4. Hand-trace a cycle: with `AIZEN_LLM_PROVIDER=mock` and `AIZEN_TRACE=1`, run `python -m src.agents.cli.run_cycle`. Open `models/cycle_traces/cycle_{decision_id}.jsonl` and verify the `llm_call` step is absent (mock has no call) and the `final` step is present.
5. Hand-trace a cycle with `AIZEN_LLM_PROVIDER=anthropic` pointed at a local fake server (or skip — the unit tests in step 1 already cover this). For a real run, point `ANTHROPIC_BASE_URL` at a controlled stub that returns 503 twice; confirm the cycle trace records `attempts=3, last_status=503, success=true` (the call ultimately succeeded on the 3rd attempt).

## Scope boundary

This plan does **not** address:
- Per-agent telemetry (only one `llm_call` step per cycle, sourced from `self.llm.last_call`). Follow-up.
- Backoff jitter — pure deterministic exponential backoff for now; jitter is a one-line addition (`random.uniform(0, base)`) when concurrency is added.
- Persistent retry-attempt counters across cycles — out of scope; per-cycle is enough.
- The "knowledge graph for failure data" (#158) and "dynamic GNN topology" (#157) — separate tasks.
