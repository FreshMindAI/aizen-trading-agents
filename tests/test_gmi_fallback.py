"""Unit tests for the GMI fallback provider.

These tests do NOT hit the network - we patch `requests.Session.post`
to simulate primary-token failure (401) and fallback-token success.

Run with:
    AIZEN_LLM_PROVIDER=mock python -m pytest tests/test_gmi_fallback.py -v
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Force-mock so test discovery doesn't try to reach a real LLM.
os.environ.setdefault("AIZEN_LLM_PROVIDER", "mock")


def _build_response(status: int, body: dict[str, Any] | str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    if isinstance(body, dict):
        resp.json.return_value = body
    else:
        resp.json.side_effect = ValueError("not json")
    resp.text = body if isinstance(body, str) else str(body)[:300]
    resp.headers = {}
    return resp


def _build_request() -> Any:
    from src.agents.llm import LLMRequest, LLMMessage
    return LLMRequest(
        system="test",
        messages=[LLMMessage(role="user", content="hi")],
        max_tokens=16,
    )


def test_provider_resolves_via_factory() -> None:
    from src.agents.llm import get_provider
    p = get_provider("gmi_fallback")
    assert p.PROVIDER_NAME == "gmi_fallback"
    assert p.fallback_token is None  # no env var set in this test


def test_gmi_fallback_default_model_is_anthropic_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression pin: when an operator flips AIZEN_LLM_PROVIDER to
    ``gmi_fallback`` in the cron env, the provider must inherit
    ANTHROPIC_DEFAULT_MODEL (claude-haiku-4-5-20251001) — NOT the
    MockProvider's "mock-1" sentinel. The cron workflow does not set
    ANTHROPIC_MODEL, so any mock-1 leak here would 404 the upstream
    GMI endpoint. Pin both paths: factory call and bare constructor.
    """
    from src.agents.llm import get_provider
    from src.agents.llm.base import ANTHROPIC_DEFAULT_MODEL
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    p = get_provider("gmi_fallback")
    assert p.default_model == ANTHROPIC_DEFAULT_MODEL, (
        f"gmi_fallback inherited {p.default_model!r}, expected "
        f"{ANTHROPIC_DEFAULT_MODEL!r}. The MockProvider's 'mock-1' "
        f"must not leak into the gmi_fallback default_model."
    )


def test_gmi_fallback_rejects_mock_sentinel_in_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Regression pin for the cron-loop 404: when ``config/agents.yaml``
    sets ``llm.model: mock-1`` (originally written for the MockProvider)
    and the resolved provider is a real one, the orchestrator must drop
    the override and use the provider's class default. Without the guard,
    the gmi_fallback provider would send ``model: mock-1`` to the GMI
    endpoint and 404.
    """
    import sqlite3
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "gmi_fallback")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-jwt")
    # The test runner may inherit an ANTHROPIC_MODEL from the host
    # environment; we don't care what specific model is selected, only
    # that the orchestrator's guard drops the mock-only sentinel.
    # Patch the YAML loader to simulate a stale config.
    from src.agents import graph as graph_mod
    monkeypatch.setattr(
        graph_mod, "get_yaml",
        lambda *_a, **_kw: {"llm": {
            "provider": "mock", "model": "mock-1", "timeout_s": 30,
        }},
    )
    from src.agents.graph import Orchestrator
    c = sqlite3.connect(":memory:")
    orch = Orchestrator(conn=c, config=None)
    # The guard must have dropped "mock-1" — assert the *negative*
    # property (no mock-only sentinel) rather than pinning a specific
    # model id, since the host environment may set ANTHROPIC_MODEL.
    assert orch.llm.PROVIDER_NAME == "gmi_fallback"
    assert orch.llm.default_model not in ("mock-1", None, ""), (
        f"orchestrator's guard failed to drop the mock-only sentinel; "
        f"gmi_fallback.default_model={orch.llm.default_model!r} would 404 "
        f"the upstream endpoint."
    )
    assert not str(orch.llm.default_model).startswith("mock-"), (
        f"gmi_fallback.default_model={orch.llm.default_model!r} starts "
        f"with 'mock-' — the guard let a mock-only sentinel through."
    )


def test_fallback_token_picked_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "primary-jwt")
    monkeypatch.setenv("GMI_FALLBACK_TOKEN", "fallback-jwt")
    from src.agents.llm.gmi_fallback import GMIFallbackProvider
    p = GMIFallbackProvider()
    assert p.fallback_token == "fallback-jwt"
    assert p.auth_token == "primary-jwt"


def test_primary_401_triggers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "primary-jwt")
    monkeypatch.setenv("GMI_FALLBACK_TOKEN", "fallback-jwt")
    from src.agents.llm.gmi_fallback import GMIFallbackProvider
    p = GMIFallbackProvider()
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "model": "test",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    call_count = {"n": 0}
    auths_seen: list[str] = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        call_count["n"] += 1
        # The primary's session has a fixed header; the fallback uses
        # a one-off session. We can read the Authorization off
        # request headers in real life, but here we just inspect
        # call order.
        if call_count["n"] == 1:
            return _build_response(401, "unauthorized")
        return _build_response(200, payload)

    with patch.object(p, "_session") as primary_sess, \
         patch("requests.Session") as SessCls:
        primary_sess.post.side_effect = fake_post
        # Build a fallback session that mirrors the construction
        # inside the wrapper.
        fb_sess = MagicMock()
        fb_sess.post.side_effect = fake_post
        fb_sess.headers = {}
        SessCls.return_value = fb_sess

        resp = p.complete(_build_request())
        assert resp.text == "ok"
        # Two HTTP calls total: one 401 on primary, one 200 on fallback.
        assert call_count["n"] == 2


def test_primary_5xx_does_not_trigger_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """5xx should be retried within the primary's own loop, not
    swapped to a different token. We expect LLMError to be raised
    after the primary's retry budget is exhausted (no fallback swap)."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "primary-jwt")
    monkeypatch.setenv("GMI_FALLBACK_TOKEN", "fallback-jwt")
    from src.agents.llm.gmi_fallback import GMIFallbackProvider
    # Tighten the retry budget so this test doesn't take 30s.
    monkeypatch.setenv("AIZEN_LLM_MAX_ATTEMPTS", "2")
    p = GMIFallbackProvider()

    with patch.object(p, "_session") as primary_sess, \
         patch("requests.Session") as SessCls:
        primary_sess.post.return_value = _build_response(503, "boom")
        SessCls.return_value = MagicMock()  # fallback should NOT be hit
        from src.agents.llm.base import LLMError
        with pytest.raises(LLMError):
            p.complete(_build_request())
        # Fallback session was never constructed.
        assert not SessCls.called
