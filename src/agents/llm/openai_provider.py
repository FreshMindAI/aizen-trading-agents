"""OpenAI provider - Chat Completions API.

The canonical LLMRequest is Anthropic-shaped. We translate to OpenAI's
chat.completions schema:

  POST {base_url}/v1/chat/completions
  Headers: Authorization: Bearer {api_key}
  Body:    { model, messages: [{role, content}], max_tokens, temperature,
            tools?: [{type: function, function: {name, description, parameters}}],
            response_format?: {type: json_object} }

Then we map the response back into LLMResponse. `tool_calls` carries the
function name + JSON args; `text` is the assistant message content.

Retry behaviour is delegated to ``LLMProvider._call_with_retry`` so the
exact same bounded exponential-backoff schedule is shared with
``AnthropicProvider``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from .base import (
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_DEFAULT_MODEL,
    ToolSpec,
)
from .telemetry import LLMCallTelemetry

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    PROVIDER_NAME = "openai"
    DEFAULT_BASE_URL = OPENAI_DEFAULT_BASE_URL
    DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    ENV_BASE_URL = "OPENAI_BASE_URL"
    ENV_MODEL = "OPENAI_MODEL"
    ENV_API_KEY = "OPENAI_API_KEY"

    def __init__(self, base_url: str | None = None, default_model: str | None = None,
                 api_key: str | None = None, timeout_s: int = 60,
                 max_retries: int | None = None, organization: str | None = None,
                 force_json_response: bool = True) -> None:
        super().__init__(base_url=base_url, default_model=default_model,
                         api_key=api_key, timeout_s=timeout_s)
        if not self.api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Export it (or pass api_key=...) before "
                "constructing OpenAIProvider."
            )
        # Legacy ``max_retries`` kwarg controls total attempt count via
        # _call_with_retry. None falls back to the env / 4 default.
        if max_retries is not None:
            self._max_attempts_override = max(1, int(max_retries) + 1)
        else:
            self._max_attempts_override = None
        import os as _os
        self.organization = organization or _os.getenv("OPENAI_ORG_ID")
        self.force_json_response = force_json_response
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        if self.organization:
            self._session.headers["OpenAI-Organization"] = self.organization

    # ---- public ---------------------------------------------------------
    def complete(self, request: LLMRequest) -> LLMResponse:
        body = self._build_body(request)
        url = f"{self.base_url}/v1/chat/completions"
        model = request.model or self.default_model
        telemetry = LLMCallTelemetry(
            provider=self.PROVIDER_NAME, model=model,
        )
        try:
            payload = self._call_with_retry(
                telemetry=telemetry,
                perform=lambda: self._session.post(url, json=body, timeout=self.timeout_s),
                parse=lambda r: r.json(),
                max_attempts=self._max_attempts_override,
                retry_after_for=self._retry_after_for,
            )
        except Exception:
            self.last_call = telemetry
            raise
        self.last_call = telemetry
        return self._parse(payload)

    # ---- request translation -------------------------------------------
    def _build_body(self, req: LLMRequest) -> dict[str, Any]:
        # Anthropic's top-level `system` -> OpenAI's first user/system turn.
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content})

        body: dict[str, Any] = {
            "model": req.model or self.default_model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.tools:
            body["tools"] = [self._to_openai_tool(t) for t in req.tools]
            body["tool_choice"] = "auto"
        # Only force JSON-object response when there are no tools (the two
        # flags are mutually exclusive in OpenAI's schema).
        if self.force_json_response and not req.tools:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _to_openai_tool(t: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }

    # ---- response translation ------------------------------------------
    @staticmethod
    def _parse(payload: dict[str, Any]) -> LLMResponse:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("openai response had no choices")
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        tool_calls: list[dict[str, Any]] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": args,
            })
        # OpenAI finish_reason: stop, length, tool_calls, content_filter
        finish = choices[0].get("finish_reason") or "stop"
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }.get(finish, finish)
        usage = payload.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
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
        or unparseable. OpenAI emits Retry-After on 429 responses; we
        cap it via max_delay_s in the helper."""
        ra = resp.headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except (TypeError, ValueError):
            return None
