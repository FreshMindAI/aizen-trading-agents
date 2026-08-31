"""Anthropic provider - native Messages API client.

Mirrors how the Claude Agent SDK / Claude Code wire requests:

  POST {base_url}/v1/messages
  Headers: x-api-key, anthropic-version
  Body:    { model, system, messages, max_tokens, temperature, tools? }

Translates the response into the canonical LLMResponse. Supports tool use
(text + tool_calls in one turn), which the orchestrator can route back into
the agent loop.

Retry behaviour is delegated to ``LLMProvider._call_with_retry`` so the
exact same bounded exponential-backoff schedule is shared with
``OpenAIProvider``. The provider's only responsibility here is request
shape + response parsing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from .base import (
    ANTHROPIC_API_VERSION,
    ANTHROPIC_DEFAULT_BASE_URL,
    ANTHROPIC_DEFAULT_MODEL,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ToolSpec,
)
from .telemetry import LLMCallTelemetry

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    PROVIDER_NAME = "anthropic"
    DEFAULT_BASE_URL = ANTHROPIC_DEFAULT_BASE_URL
    DEFAULT_MODEL = ANTHROPIC_DEFAULT_MODEL
    ENV_BASE_URL = "ANTHROPIC_BASE_URL"
    ENV_MODEL = "ANTHROPIC_MODEL"
    ENV_API_KEY = "ANTHROPIC_API_KEY"

    def __init__(self, base_url: str | None = None, default_model: str | None = None,
                 api_key: str | None = None, timeout_s: int = 60,
                 max_retries: int | None = None, auth_token: str | None = None,
                 extra_headers: dict[str, str] | None = None) -> None:
        # ANTHROPIC_AUTH_TOKEN (GMI / proxies) overrides ANTHROPIC_API_KEY when
        # set. When auth_token is present we send `Authorization: Bearer ...`
        # instead of `x-api-key: ...` because GMI expects bearer auth.
        env_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        self.auth_token = auth_token or env_token
        super().__init__(base_url=base_url, default_model=default_model,
                         api_key=api_key, timeout_s=timeout_s)
        if not (self.api_key or self.auth_token):
            raise LLMError(
                "Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set. Export one "
                "(or pass api_key=.../auth_token=...) before constructing AnthropicProvider."
            )
        # ``max_retries`` is the legacy kwarg name - we keep it for callers
        # that still pass it but it now controls the *total* attempt count
        # via ``_call_with_retry`` (1 initial + max_retries retries == max_retries+1
        # total). Default falls back to ``_default_max_attempts()`` (env override,
        # else 4). We accept max_retries=2 (old default) as 3 total attempts for
        # backward compatibility.
        if max_retries is not None:
            self._max_attempts_override = max(1, int(max_retries) + 1)
        else:
            self._max_attempts_override = None
        self._session = requests.Session()
        headers = {
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        else:
            headers["x-api-key"] = self.api_key
        if extra_headers:
            headers.update(extra_headers)
        self._session.headers.update(headers)

    # ---- public ---------------------------------------------------------
    def complete(self, request: LLMRequest) -> LLMResponse:
        body = self._build_body(request)
        # Path is configurable so GMI / LiteLLM proxies with non-standard
        # routes (e.g. /anthropic/v1/messages) can be used.
        path = os.getenv("ANTHROPIC_MESSAGES_PATH", "/v1/messages")
        url = f"{self.base_url}{path}"
        model = request.model or self.default_model
        telemetry = LLMCallTelemetry(
            provider=self.PROVIDER_NAME, model=model,
        )
        # Always publish a telemetry record, even on the error path. We use
        # a try/finally so the orchestrator can read ``last_call`` whether
        # the call succeeded or raised.
        try:
            payload = self._call_with_retry(
                telemetry=telemetry,
                perform=lambda: self._session.post(url, json=body, timeout=self.timeout_s),
                parse=lambda r: r.json(),
                max_attempts=self._max_attempts_override,
                retry_after_for=self._retry_after_for,
            )
        except Exception:
            # Telemetry already finalized by the helper; just publish.
            self.last_call = telemetry
            raise
        self.last_call = telemetry
        return self._parse(payload)

    # ---- internal -------------------------------------------------------
    def _build_body(self, req: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model or self.default_model,
            "system": req.system,
            "messages": [self._to_anthropic_msg(m) for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.tools:
            body["tools"] = [self._to_anthropic_tool(t) for t in req.tools]
        if req.metadata:
            # Anthropic allows user_id for abuse tracking; pass through.
            if "user_id" in req.metadata:
                body["metadata"] = {"user_id": str(req.metadata["user_id"])[:64]}
        return body

    @staticmethod
    def _to_anthropic_msg(m: Any) -> dict[str, Any]:
        return {"role": m.role, "content": m.content}

    @staticmethod
    def _to_anthropic_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }

    @staticmethod
    def _parse(payload: dict[str, Any]) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in payload.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })
        stop = payload.get("stop_reason") or "end_turn"
        usage = payload.get("usage", {}) or {}
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            usage={
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            },
            model=payload.get("model", ""),
            raw=payload,
        )

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        # Kept for backwards compatibility with any external callers that
        # imported the private method. New retry logic lives in the
        # base class's _call_with_retry; this is a no-op shim.
        return

    @staticmethod
    def _retry_after_for(resp: "requests.Response") -> float | None:
        """Parse the Retry-After header (seconds). Returns None if absent
        or unparseable. Mirrors how Claude Code reads the Anthropic
        Messages API rate-limit response."""
        ra = resp.headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except (TypeError, ValueError):
            return None
