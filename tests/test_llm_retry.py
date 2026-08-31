"""Tests for the LLM API retry hardening.

Covers:
- Shared ``_call_with_retry`` in base.py: retry on 5xx, retry on 429 with
  Retry-After, give up after max_attempts, no retry on 4xx, retry on
  ConnectionError.
- Telemetry dataclass: attempt count, last_status, per_attempt records.
- ``build_llm_step`` in trace.py: builds a step on success / retry /
  failure / no-telemetry.
- End-to-end: an AnthropicProvider with a mocked Session produces a
  ``llm_call`` step in the cycle trace with the right shape.

The MockProvider is unchanged: it has no last_call, so the orchestrator
skips emitting the step. That is exercised in test_alpaca_mcp / cycle
tests already, not here.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import pytest
import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.llm import (  # noqa: E402
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
)
from src.agents.llm.telemetry import (  # noqa: E402
    LLMCallTelemetry,
    LLMTransportError,
)
from src.agents.trace import build_llm_step  # noqa: E402


# ---------------------------------------------------------------------------
# Fake transport - injects whatever status / exception sequence the test needs
# ---------------------------------------------------------------------------
@dataclass
class _FakeResponse:
    status_code: int
    _json: Any
    text: str = ""
    headers: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

    def json(self) -> Any:
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _MockTransport:
    """A stand-in for `provider._session`. Yields a sequence of responses
    or raises a sequence of exceptions on each `.post()` call."""

    def __init__(self, sequence: list[Any]):
        # Each entry is either a _FakeResponse or an Exception class/instance.
        self._sequence = list(sequence)
        self._call_count = 0
        self.calls: list[dict] = []

    def post(self, url: str, **kw) -> _FakeResponse:
        self.calls.append({"url": url, **kw})
        if self._call_count >= len(self._sequence):
            raise IndexError(f"transport exhausted after {self._call_count} calls")
        item = self._sequence[self._call_count]
        self._call_count += 1
        if isinstance(item, BaseException) or (
            isinstance(item, type) and issubclass(item, BaseException)
        ):
            if isinstance(item, type):
                raise item()
            raise item
        return item


def _make_provider_for_retry_tests() -> LLMProvider:
    """Build a no-op provider instance we can call _call_with_retry on.
    The helper is an instance method on the LLMProvider class so we need
    an actual instance (or class) to invoke it through."""
    from src.agents.llm.mock_provider import MockProvider
    return MockProvider()


# ---------------------------------------------------------------------------
# 1. Retry succeeds on transient 5xx
# ---------------------------------------------------------------------------
def test_retry_succeeds_on_transient_5xx(monkeypatch):
    """A 503 followed by a 200 should succeed on attempt 2. Telemetry
    records both attempts; success=True on the final state; total_attempts=2."""
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "4")
    transport = _MockTransport([
        _FakeResponse(503, {"error": "service unavailable"}),
        _FakeResponse(200, {"content": [{"type": "text", "text": "ok"}],
                            "model": "claude-haiku-4-5-20251001",
                            "stop_reason": "end_turn", "usage": {}}),
    ])
    telemetry = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    provider = _make_provider_for_retry_tests()

    def perform():
        return transport.post("https://api.test/v1/messages")

    def parse(resp):
        return resp.json()

    result = provider._call_with_retry(
        telemetry=telemetry, perform=perform, parse=parse,
        base_delay_s=0.001, max_delay_s=0.01,
    )
    assert result["content"][0]["text"] == "ok"
    assert telemetry.succeeded is True
    assert telemetry.attempt_count == 2
    assert telemetry.attempts[0].status == 503
    assert telemetry.attempts[1].status == 200
    assert telemetry.final_status == 200
    # The 503 attempt's "slept_before_s" should record the schedule
    # delay that preceded attempt 2.
    assert telemetry.attempts[0].slept_before_s >= 0.0
    assert transport._call_count == 2


# ---------------------------------------------------------------------------
# 2. Retry on 429 with Retry-After
# ---------------------------------------------------------------------------
def test_retry_succeeds_on_429_with_retry_after(monkeypatch):
    """A 429 with Retry-After=0.05 should retry, and the sleep should be
    at least the Retry-After value."""
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "4")
    transport = _MockTransport([
        _FakeResponse(429, {"error": "rate limited"},
                      headers={"retry-after": "0.05"}),
        _FakeResponse(200, {"content": [{"type": "text", "text": "ok"}],
                            "model": "claude-haiku-4-5-20251001",
                            "stop_reason": "end_turn", "usage": {}}),
    ])
    telemetry = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    provider = _make_provider_for_retry_tests()

    def perform():
        return transport.post("https://api.test/v1/messages")

    def parse(resp):
        return resp.json()

    def retry_after_for(resp):
        ra = resp.headers.get("retry-after")
        return float(ra) if ra else None

    provider._call_with_retry(
        telemetry=telemetry, perform=perform, parse=parse,
        base_delay_s=0.5, max_delay_s=30.0,
        retry_after_for=retry_after_for,
    )
    assert telemetry.succeeded is True
    assert telemetry.attempt_count == 2
    # The schedule delay should be capped to Retry-After (0.05s).
    assert 0.04 <= telemetry.attempts[0].slept_before_s <= 0.2
    assert telemetry.attempts[0].status == 429


# ---------------------------------------------------------------------------
# 3. Retry gives up after max_attempts
# ---------------------------------------------------------------------------
def test_retry_gives_up_after_max_attempts(monkeypatch):
    """Four 503s in a row with max_attempts=4 must raise LLMError after
    attempt 4. Telemetry records 4 attempts and final_status=503."""
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "4")
    transport = _MockTransport([
        _FakeResponse(503, {"error": "service unavailable"}),
        _FakeResponse(503, {"error": "service unavailable"}),
        _FakeResponse(503, {"error": "service unavailable"}),
        _FakeResponse(503, {"error": "service unavailable"}),
    ])
    telemetry = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    provider = _make_provider_for_retry_tests()

    def perform():
        return transport.post("https://api.test/v1/messages")

    def parse(resp):
        return resp.json()

    with pytest.raises(LLMError) as exc_info:
        provider._call_with_retry(
            telemetry=telemetry, perform=perform, parse=parse,
            base_delay_s=0.001, max_delay_s=0.01,
        )
    assert "giving up after 4 attempts" in str(exc_info.value)
    assert telemetry.succeeded is False
    assert telemetry.attempt_count == 4
    assert telemetry.final_status == 503
    # All 4 attempts should have been made - the transport proves it.
    assert transport._call_count == 4


# ---------------------------------------------------------------------------
# 4. No retry on permanent 4xx
# ---------------------------------------------------------------------------
def test_no_retry_on_4xx(monkeypatch):
    """A 400 must NOT be retried. Telemetry records 1 attempt and
    final_status=400. The LLMError raised is NOT an LLMTransportError."""
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "4")
    transport = _MockTransport([
        _FakeResponse(400, {"error": {"type": "invalid_request_error",
                                      "message": "bad model"}}),
    ])
    telemetry = LLMCallTelemetry(provider="anthropic", model="bad-model")
    provider = _make_provider_for_retry_tests()

    def perform():
        return transport.post("https://api.test/v1/messages")

    def parse(resp):
        return resp.json()

    with pytest.raises(LLMError) as exc_info:
        provider._call_with_retry(
            telemetry=telemetry, perform=perform, parse=parse,
            base_delay_s=0.001, max_delay_s=0.01,
        )
    # 400 is not a transport error - it's a permanent client error.
    assert not isinstance(exc_info.value, LLMTransportError)
    assert telemetry.attempt_count == 1
    assert telemetry.final_status == 400
    assert telemetry.attempts[0].error and "400" in telemetry.attempts[0].error
    assert transport._call_count == 1


# ---------------------------------------------------------------------------
# 5. Retry on ConnectionError
# ---------------------------------------------------------------------------
def test_retry_on_connection_error(monkeypatch):
    """A requests.ConnectionError on the first attempt must trigger a
    retry, then succeed on the second attempt."""
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "4")
    transport = _MockSequence = type("_S", (), {
        "_calls": 0,
        "calls": [],
    })()

    def perform():
        transport.calls.append(1)
        transport._calls += 1
        if transport._calls == 1:
            raise requests.ConnectionError("DNS resolution failed")
        return _FakeResponse(200, {"content": [{"type": "text", "text": "ok"}],
                                   "model": "claude-haiku-4-5-20251001",
                                   "stop_reason": "end_turn", "usage": {}})

    telemetry = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    provider = _make_provider_for_retry_tests()

    def parse(resp):
        return resp.json()

    result = provider._call_with_retry(
        telemetry=telemetry, perform=perform, parse=parse,
        base_delay_s=0.001, max_delay_s=0.01,
    )
    assert result["content"][0]["text"] == "ok"
    assert telemetry.succeeded is True
    assert telemetry.attempt_count == 2
    assert telemetry.attempts[0].status is None
    assert "ConnectionError" in (telemetry.attempts[0].error or "")
    assert telemetry.attempts[1].status == 200
    assert transport._calls == 2


# ---------------------------------------------------------------------------
# 6. build_llm_step - end-to-end visibility
# ---------------------------------------------------------------------------
def test_build_llm_step_success_on_first_attempt():
    """build_llm_step returns a StepRecord with success=True, attempts=1,
    no reasons when the call succeeded on the first try."""
    tel = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    tel.record_attempt(1, status=200, error=None, slept_before_s=0.0)
    tel.finalize(succeeded=True)

    step = build_llm_step(tel)
    assert step is not None
    assert step.step == "llm_call"
    assert step.success is True
    assert step.fields["provider"] == "anthropic"
    assert step.fields["attempts"] == 1
    assert step.fields["final_status"] == 200
    assert step.reasons == []


def test_build_llm_step_succeeds_after_retry_flags_as_not_first_try():
    """A call that needed a retry reports success=False (visibility for
    operators) even though the final HTTP response was 2xx."""
    tel = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    tel.record_attempt(1, status=503, error="http 503", slept_before_s=0.5)
    tel.record_attempt(2, status=200, error=None, slept_before_s=0.0)
    tel.finalize(succeeded=True)

    step = build_llm_step(tel)
    assert step is not None
    assert step.success is False
    assert step.fields["attempts"] == 2
    assert step.fields["succeeded"] is True
    assert step.fields["final_status"] == 200
    # Should mention the retry in reasons.
    assert any("retry" in r.lower() for r in step.reasons)


def test_build_llm_step_permanent_failure():
    """A 4-attempt failure surfaces attempts, last_status, and a 'giving
    up' reason. success=False."""
    tel = LLMCallTelemetry(provider="anthropic", model="claude-haiku-4-5-20251001")
    for i in range(4):
        tel.record_attempt(i + 1, status=503, error="http 503", slept_before_s=0.5)
    tel.finalize(succeeded=False)

    step = build_llm_step(tel)
    assert step is not None
    assert step.success is False
    assert step.fields["attempts"] == 4
    assert step.fields["succeeded"] is False
    assert step.fields["final_status"] == 503
    assert any("giving up" in r for r in step.reasons)
    # Per-attempt detail should be present.
    assert len(step.fields["per_attempt"]) == 4
    assert step.fields["per_attempt"][0]["status"] == 503


def test_build_llm_step_returns_none_when_no_telemetry():
    """When the provider has no last_call (e.g. MockProvider, or a
    short-circuited cycle), build_llm_step returns None so the caller
    can skip the trace append."""
    assert build_llm_step(None) is None


# ---------------------------------------------------------------------------
# 7. End-to-end: AnthropicProvider with mocked transport publishes last_call
# ---------------------------------------------------------------------------
def test_anthropic_provider_publishes_last_call_on_retry_then_success(monkeypatch):
    """Wire a mock transport into a real AnthropicProvider, drive a call
    that retries once, and confirm self.last_call is populated with the
    correct attempt count and final_status=200."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890")
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "4")
    from src.agents.llm.anthropic_provider import AnthropicProvider
    provider = AnthropicProvider(base_url="https://api.test", timeout_s=5)

    transport = _MockTransport([
        _FakeResponse(503, {"error": "service unavailable"}),
        _FakeResponse(200, {"content": [{"type": "text", "text": "hello"}],
                            "model": "claude-haiku-4-5-20251001",
                            "stop_reason": "end_turn", "usage": {}}),
    ])
    provider._session = transport  # type: ignore[assignment]
    req = LLMRequest(
        system="You are the test_agent in a multi-agent trading system.",
        messages=[LLMMessage(role="user", content="hi")],
        max_tokens=16,
    )
    out = provider.complete(req)
    assert out.text == "hello"
    assert provider.last_call is not None
    assert provider.last_call.succeeded is True
    assert provider.last_call.attempt_count == 2
    assert provider.last_call.final_status == 200
    # Telemetry should be JSON-serializable.
    blob = provider.last_call.to_dict()
    assert blob["attempts"] == 2
    assert blob["per_attempt"][0]["status"] == 503
    assert blob["per_attempt"][1]["status"] == 200


def test_anthropic_provider_publishes_last_call_on_permanent_failure(monkeypatch):
    """A provider that runs out of attempts must still publish a
    last_call so the orchestrator can surface the failure in the trace."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890")
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "2")
    from src.agents.llm.anthropic_provider import AnthropicProvider
    provider = AnthropicProvider(base_url="https://api.test", timeout_s=5)

    transport = _MockTransport([
        _FakeResponse(503, {"error": "service unavailable"}),
        _FakeResponse(503, {"error": "service unavailable"}),
    ])
    provider._session = transport  # type: ignore[assignment]
    req = LLMRequest(
        system="You are the test_agent in a multi-agent trading system.",
        messages=[LLMMessage(role="user", content="hi")],
        max_tokens=16,
    )
    with pytest.raises(LLMError) as exc_info:
        provider.complete(req)
    assert "giving up" in str(exc_info.value).lower()
    assert provider.last_call is not None
    assert provider.last_call.succeeded is False
    assert provider.last_call.attempt_count == 2
    assert provider.last_call.final_status == 503
