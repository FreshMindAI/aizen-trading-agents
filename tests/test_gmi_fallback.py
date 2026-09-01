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
