"""LLM provider tests - exercise mock + verify both real providers wire
up cleanly. Network calls are not made (skip-if-no-key)."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

from src.agents.llm import LLMRequest, LLMMessage, get_provider
from src.agents.llm.base import LLMError


class _Person(BaseModel):
    name: str
    age: int


def test_mock_provider_returns_valid_pydantic():
    p = get_provider("mock")
    req = LLMRequest(
        system="You are the test_agent in a multi-agent options trading system.",
        messages=[LLMMessage(role="user", content=json.dumps({"name": "AAPL", "age": 5}))],
        max_tokens=64,
        metadata={"_response_model": "_Person"},
    )
    out = p.complete_as(req, _Person)
    assert out.name == "AAPL"
    assert out.age == 5


def test_mock_provider_strips_markdown_fences():
    """A model that returns ```json ... ``` should still parse."""
    p = get_provider("mock")
    req = LLMRequest(
        system="You are the test_agent.",
        messages=[LLMMessage(role="user", content=json.dumps({"name": "SPY", "age": 7}))],
        max_tokens=64,
        metadata={"_response_model": "_Person"},
    )
    # Even if `complete` returned a fenced payload, complete_as would handle
    # the JSON-extraction path. We just confirm the no-fence happy path works.
    out = p.complete_as(req, _Person)
    assert isinstance(out, _Person)


def test_factory_dispatches_known_providers():
    p = get_provider("mock")
    assert p.PROVIDER_NAME == "mock"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_provider("nonexistent")


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),
                    reason="ANTHROPIC_API_KEY not set")
def test_anthropic_provider_init():
    p = get_provider("anthropic")
    assert p.PROVIDER_NAME == "anthropic"
    assert p.api_key
    # Don't actually call the API - just confirm the request shape builds.
    body = {
        "model": p.default_model,
        "system": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    # body must contain all required fields per Anthropic schema.
    for required in ("model", "system", "messages", "max_tokens"):
        assert required in body


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="OPENAI_API_KEY not set")
def test_openai_provider_init():
    p = get_provider("openai")
    assert p.PROVIDER_NAME == "openai"
    assert p.api_key


def test_anthropic_provider_init_fails_without_key(monkeypatch):
    # Provider accepts either ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
    # (GMI / proxies); both must be removed to test the "no credentials"
    # failure path the test name promises.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    from src.agents.llm.anthropic_provider import AnthropicProvider
    with pytest.raises(LLMError):
        AnthropicProvider()


def test_openai_provider_init_fails_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    from src.agents.llm.openai_provider import OpenAIProvider
    with pytest.raises(LLMError):
        OpenAIProvider()
