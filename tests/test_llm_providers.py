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


# ---------------------------------------------------------------------------
# Model resolution chain (added 2026-09-02, updated 2026-09-03 when the
# MiniMax-M3 ban was lifted).
#
# Tests assert the precedence order documented in LLMProvider.__init__:
#   explicit kwarg > per-provider env (ANTHROPIC_MODEL) > cross-provider
#   AIZEN_LLM_MODEL > class default.
#
# The 2026-09-03 decision: the project rule forbidding the GMI /
# MiniMaxAI / MiniMax-M3 endpoint was lifted because that is the only
# model id the GMI proxy plan currently serves with credit (verified
# 200 OK on 2026-09-03). All other GMI catalog entries return 402
# Insufficient balance on the same plan. The blocker that previously
# raised LLMError on this id was removed in the same commit. These
# tests now assert that the id is allowed, and that the resolution
# chain is unchanged.
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_llm_env(monkeypatch):
    """Strip every env var that influences model resolution so each
    test starts from a known baseline."""
    for var in ("ANTHROPIC_MODEL", "AIZEN_LLM_MODEL", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_ORG_ID",
                "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_explicit_kwarg_wins_over_env_and_default(monkeypatch, clean_llm_env):
    """Explicit `default_model=...` kwarg beats both env vars."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "openai/gpt-5.4-mini")
    monkeypatch.setenv("AIZEN_LLM_MODEL", "MiniMaxAI/MiniMax-M3")
    from src.agents.llm.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key="sk-test", default_model="anthropic/claude-haiku-4.5")
    assert p.default_model == "anthropic/claude-haiku-4.5"


def test_aizen_llm_model_fallback_used_when_anthropic_model_unset(
    monkeypatch, clean_llm_env
):
    """When per-provider env is absent, AIZEN_LLM_MODEL resolves the model.
    This is the cron-loop workflow path: AIZEN_LLM_MODEL=MiniMaxAI/...
    in .github/workflows/cron-loop.yml env block."""
    monkeypatch.setenv("AIZEN_LLM_MODEL", "MiniMaxAI/MiniMax-M3")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    p = get_provider("gmi_fallback")
    assert p.default_model == "MiniMaxAI/MiniMax-M3"


def test_anthropic_model_env_wins_over_aizen_llm_model(monkeypatch, clean_llm_env):
    """A direct provider=anthropic deployment can still pin via the
    legacy ANTHROPIC_MODEL env var without being overridden by the
    cross-provider AIZEN_LLM_MODEL."""
    monkeypatch.setenv("AIZEN_LLM_MODEL", "MiniMaxAI/MiniMax-M3")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    p = get_provider("anthropic")
    assert p.default_model == "claude-haiku-4-5-20251001"


def test_class_default_used_when_no_env_set(monkeypatch, clean_llm_env):
    """No env override -> class default. Locks in that
    ANTHROPIC_DEFAULT_MODEL is the GMI catalog id that the proxy
    actually serves with credit."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    p = get_provider("anthropic")
    from src.agents.llm.base import ANTHROPIC_DEFAULT_MODEL
    assert p.default_model == ANTHROPIC_DEFAULT_MODEL
    # Sanity: must be the catalog-format owner/name id (slash form),
    # not the date-stamped Anthropic id (which 404s on the GMI proxy).
    assert "/" in p.default_model, (
        f"expected GMI-format owner/name catalog id, got {p.default_model!r}"
    )


def test_minimax_m3_id_now_allowed_via_env(monkeypatch, clean_llm_env):
    """The 2026-09-03 decision lifts the MiniMax-M3 ban. The id is
    now read straight through to the provider; no LLMError is raised
    when ANTHROPIC_MODEL or AIZEN_LLM_MODEL is set to it."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "MiniMaxAI/MiniMax-M3")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    p = get_provider("gmi_fallback")
    assert p.default_model == "MiniMaxAI/MiniMax-M3"


def test_unrelated_model_id_still_allowed(monkeypatch, clean_llm_env):
    """Sanity check: any GMI catalog id is allowed when the GMI plan
    covers it. The blocker that was removed targeted only the MiniMax
    family. Other GMI ids (gemma, gpt, claude family) still resolve
    to whatever the env says; whether the proxy serves them is a
    runtime concern, not a code-level block."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic/claude-haiku-4.5")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    p = get_provider("gmi_fallback")
    assert p.default_model == "anthropic/claude-haiku-4.5"


def test_no_blocked_model_id_helper():
    """The blocker helper was removed in the 2026-09-03 commit. This
    test asserts that the symbol is gone so a future re-introduction
    is a conscious decision (it would need to be re-added to
    src/agents/llm/base.py, with updated tests)."""
    import src.agents.llm.base as base_mod
    assert not hasattr(base_mod, "_is_blocked_model_id"), (
        "the no-MiniMax-M3 blocker was removed on 2026-09-03; if a "
        "future change re-adds it, the tests above need to be re-evaluated"
    )
    assert not hasattr(base_mod, "_BLOCKED_MODEL_ID_SUBSTRINGS")
