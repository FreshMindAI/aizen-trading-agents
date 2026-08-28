"""Anthropic provider - native Messages API client.

Mirrors how the Claude Agent SDK / Claude Code wire requests:

  POST {base_url}/v1/messages
  Headers: x-api-key, anthropic-version
  Body:    { model, system, messages, max_tokens, temperature, tools? }

Translates the response into the canonical LLMResponse. Supports tool use
(text + tool_calls in one turn), which the orchestrator can route back into
the agent loop.
"""

from __future__ import annotations

import json
import logging
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
                 max_retries: int = 2) -> None:
        super().__init__(base_url=base_url, default_model=default_model,
                         api_key=api_key, timeout_s=timeout_s)
        if not self.api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Export it (or pass api_key=...) before "
                "constructing AnthropicProvider."
            )
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        })

    # ---- public ---------------------------------------------------------
    def complete(self, request: LLMRequest) -> LLMResponse:
        body = self._build_body(request)
        url = f"{self.base_url}/v1/messages"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, json=body, timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_err = exc
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                raise LLMError(f"anthropic transport error: {exc}") from exc

            if resp.status_code >= 500 and attempt < self.max_retries:
                last_err = RuntimeError(f"http {resp.status_code}")
                self._backoff(attempt)
                continue
            if resp.status_code in (408, 429) and attempt < self.max_retries:
                last_err = RuntimeError(f"http {resp.status_code}")
                self._backoff(attempt, retry_after=resp.headers.get("retry-after"))
                continue
            if resp.status_code >= 400:
                # Don't retry client errors - the request itself is bad.
                raise LLMError(
                    f"anthropic {resp.status_code}: {resp.text[:500]}"
                )
            try:
                return self._parse(resp.json())
            except (ValueError, KeyError) as exc:
                raise LLMError(f"anthropic returned unparseable body: {exc}") from exc
        raise LLMError(f"anthropic giving up after retries: {last_err}")

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
        import time
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        time.sleep(min(0.5 * (2 ** attempt), 10.0))
