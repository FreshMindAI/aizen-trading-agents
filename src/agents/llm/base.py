"""LLM provider abstraction - Anthropic-shaped canonical request.

The canonical request mirrors the Claude Agent SDK / Anthropic Messages API
so the agents never see provider-specific fields. Each provider is a thin
adapter that translates to its native HTTP shape:

  * AnthropicProvider  -> native (Messages API)
  * OpenAIProvider     -> translates to Chat Completions
  * MockProvider       -> deterministic for tests / paper

Design notes (carried over from how Claude Code surfaces models):
  * `system` is a top-level string, not a message.
  * `messages` is the conversation history; alternating `user` / `assistant`.
  * `tools` are optional function-calling specs - the provider either
    supports them natively (Anthropic, OpenAI) or ignores them (Mock).
  * `max_tokens` is REQUIRED (Anthropic rejects requests without it).
  * Returns a normalized `LLMResponse` with `text`, `tool_calls`, `usage`
    and `stop_reason` - never provider-native objects.

Configuration is purely env-driven (mirrors Claude Code's `ANTHROPIC_*` /
`OPENAI_*` env vars) so the same code path can flip providers without code
edits.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Default Anthropic Messages API host. Override with ANTHROPIC_BASE_URL when
# proxying (LiteLLM, internal gateways, etc.).
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# OpenAI Chat Completions host. OpenAIProvider hits /v1/chat/completions.
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Canonical request / response
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class LLMRequest:
    """Anthropic-shaped canonical request."""
    system: str
    messages: list["LLMMessage"]
    max_tokens: int = 1024
    temperature: float = 0.0
    model: str | None = None          # provider default if None
    tools: list[ToolSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMMessage:
    """A single conversation turn. `content` is plain text for now;
    multimodal blocks can be added later without breaking the protocol."""
    role: Literal["user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Provider-neutral response. Agents consume only these fields."""
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"     # end_turn | tool_use | max_tokens | error
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised on transport / parse failure. Orchestrator catches and falls back."""


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """One provider, one base URL, many model ids."""

    PROVIDER_NAME: str = "abstract"

    def __init__(self, base_url: str | None = None, default_model: str | None = None,
                 api_key: str | None = None, timeout_s: int = 60) -> None:
        self.base_url = (base_url or self._env_base_url() or self.DEFAULT_BASE_URL).rstrip("/")
        self.default_model = default_model or self._env_model() or self.DEFAULT_MODEL
        self.api_key = api_key or self._env_api_key()
        self.timeout_s = timeout_s

    # ---- env hooks subclasses override ----------------------------------
    DEFAULT_BASE_URL: str = ""
    DEFAULT_MODEL: str = ""
    ENV_BASE_URL: str = ""
    ENV_MODEL: str = ""
    ENV_API_KEY: str = ""

    def _env_base_url(self) -> str | None:
        return os.getenv(self.ENV_BASE_URL) if self.ENV_BASE_URL else None

    def _env_model(self) -> str | None:
        return os.getenv(self.ENV_MODEL) if self.ENV_MODEL else None

    def _env_api_key(self) -> str | None:
        return os.getenv(self.ENV_API_KEY) if self.ENV_API_KEY else None

    # ---- core API -------------------------------------------------------
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a canonical request, return a normalized response."""

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def system_prompt(agent_id: str, role: str, json_schema_hint: str) -> "LLMMessage":
        return LLMMessage(
            role="system",
            content=(
                f"You are the {agent_id} in a multi-agent options trading system. "
                f"Your role: {role}. "
                "Reason only over the structured inputs provided. "
                "Respond with a single JSON object matching this schema:\n"
                f"{json_schema_hint}\n"
                "Do not add prose, do not add fields outside the schema. "
                "Never invent option symbols, strikes, expiries or quantities."
            ),
        )

    @staticmethod
    def user_payload(payload: dict[str, Any]) -> "LLMMessage":
        import json
        return LLMMessage(role="user", content=json.dumps(payload, default=str, indent=2))

    # ---- high-level helper: parse text into a Pydantic model ------------
    def complete_as(self, request: LLMRequest, response_model: Type[T]) -> T:
        """Convenience: complete() + parse `text` as JSON -> response_model.

        Providers without a native tool-use API still satisfy the
        structured-output contract because every provider has to return
        `text`; we just JSON-decode it here.
        """
        import json

        # Ask the model for JSON-only output via system prompt.
        augmented = LLMRequest(
            system=(
                request.system
                + "\n\nRespond with a SINGLE JSON object that conforms to the schema. "
                  "No prose, no markdown, no code fences."
            ),
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model=request.model,
            tools=request.tools,
            metadata={**request.metadata, "_response_model": response_model.__name__},
        )
        resp = self.complete(augmented)
        text = resp.text.strip()
        # Strip accidental ```json fences.
        if text.startswith("```"):
            text = text.split("```", 2)[1] if text.count("```") >= 2 else text
            text = text.removeprefix("json").strip()
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{self.PROVIDER_NAME} returned invalid JSON for "
                f"{response_model.__name__}: {exc}; raw={text[:300]}"
            ) from exc
        try:
            return response_model.model_validate(data)
        except Exception as exc:
            raise LLMError(
                f"{self.PROVIDER_NAME} JSON did not match {response_model.__name__}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Factory - reads config or env, returns the configured provider
# ---------------------------------------------------------------------------
def get_provider(name: str | None = None, **overrides: Any) -> LLMProvider:
    """Resolve a provider by name. Order of precedence:
        1. explicit `name` arg
        2. AIZEN_LLM_PROVIDER env var
        3. LLM_PROVIDER env var
        4. `anthropic` (Phase-3 default - mirrors Claude Code's default)
    """
    resolved = (name
                or os.getenv("AIZEN_LLM_PROVIDER")
                or os.getenv("LLM_PROVIDER")
                or "anthropic").lower()
    if resolved in ("anthropic", "claude"):
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(**overrides)
    if resolved in ("openai", "gpt", "oai"):
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(**overrides)
    if resolved in ("mock", "deterministic", "stub"):
        from .mock_provider import MockProvider
        return MockProvider(**overrides)
    raise ValueError(f"Unknown LLM provider: {resolved!r}")


# Backwards-compat shim: existing callers used `LLMMessage` / `LLMProvider`
# with a different shape. Re-export here so older code keeps compiling.
__all__ = [
    "ANTHROPIC_DEFAULT_BASE_URL",
    "ANTHROPIC_DEFAULT_MODEL",
    "ANTHROPIC_API_VERSION",
    "OPENAI_DEFAULT_BASE_URL",
    "OPENAI_DEFAULT_MODEL",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ToolSpec",
    "get_provider",
]
