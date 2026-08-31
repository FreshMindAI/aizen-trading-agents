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

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Type, TypeVar

import requests
from pydantic import BaseModel

from .telemetry import LLMCallAttempt, LLMCallTelemetry, LLMTransportError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

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
        # Telemetry slot for the most recent LLM call. Subclasses that go
        # over the wire (AnthropicProvider, OpenAIProvider) populate this
        # on every call. MockProvider leaves it None - the orchestrator
        # detects that and skips emitting an llm_call step in the trace.
        self.last_call: LLMCallTelemetry | None = None

    # ---- env hooks subclasses override ----------------------------------
    DEFAULT_BASE_URL: str = ""
    DEFAULT_MODEL: str = ""
    ENV_BASE_URL: str = ""
    ENV_MODEL: str = ""
    ENV_API_KEY: str = ""

    # ---- shared retry helper -------------------------------------------
    @staticmethod
    def _default_max_attempts() -> int:
        """Bounded retry count. Default 4 (= 1 initial + 3 retries). Honoured
        as long as a provider has not overridden the kwarg explicitly. Ops
        can bump via AIZEN_LLM_MAX_ATTEMPTS=5 etc."""
        raw = os.getenv("AIZEN_LLM_MAX_ATTEMPTS", "4").strip()
        try:
            n = int(raw)
        except ValueError:
            return 4
        return max(1, min(n, 10))  # clamp to [1, 10] to bound test waits

    def _call_with_retry(
        cls_or_self,
        *,
        telemetry: LLMCallTelemetry,
        perform: Callable[[], "requests.Response | None",
        ],
        parse: Callable[["requests.Response"], Any],
        max_attempts: int | None = None,
        base_delay_s: float = 0.5,
        max_delay_s: float = 30.0,
        retry_after_for: Callable[["requests.Response"], float | None] | None = None,
    ) -> Any:
        """Bounded exponential backoff wrapper for HTTP LLM calls.

        Args:
            telemetry: populated on every attempt; stashed on self.last_call
                by the caller before this helper returns.
            perform: callable that issues ONE HTTP request. Returns the
                `requests.Response` (any status code) or raises a
                `requests.RequestException` on transport failure. Returning
                `None` is treated like a transport error and retried.
            parse: callable that takes a 2xx response and returns the parsed
                payload. If it raises, the error is treated as a permanent
                client error (no retry) and re-raised as LLMError.
            max_attempts: total attempts including the first. Default = env
                override or 4. Clamped to >= 1.
            base_delay_s: initial sleep; doubles each retry up to max_delay_s.
            max_delay_s: cap on the exponential schedule (Retry-After is
                honored above this, so providers can still ask for longer).

        Returns the parsed payload on success. Raises `LLMError` on a
        permanent client error (4xx other than 408/429, parse failure) or
        on exhaustion of `max_attempts` for transport / 5xx / 408 / 429.
        """
        if max_attempts is None:
            # ``cls_or_self`` may be a class (when called as
            # ``LLMProvider._call_with_retry(...)``) or an instance
            # (when called as ``self._call_with_retry(...)``). Either
            # way ``_default_max_attempts`` is a staticmethod so we
            # can reach it via the class.
            cls = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)
            max_attempts = cls._default_max_attempts()
        max_attempts = max(1, max_attempts)

        # Per-attempt helper: run, classify, sleep, record telemetry.
        def _attempt(idx: int, last_resp_holder: list) -> Any:
            # The sleep that PRECEDED this attempt. Attempt 1 has no sleep.
            # The retry loop mutates `pending_sleep_s` on the just-recorded
            # attempt after raising LLMTransportError; here we read it.
            slept = 0.0
            if idx > 1 and telemetry.attempts:
                slept = telemetry.attempts[-1].slept_before_s
            try:
                resp = perform()
            except requests.RequestException as exc:
                telemetry.record_attempt(
                    idx, status=None,
                    error=f"{type(exc).__name__}: {exc}",
                    slept_before_s=slept,
                )
                raise LLMTransportError(
                    f"{telemetry.provider} transport error on attempt {idx}: {exc}"
                ) from exc
            if resp is None:
                telemetry.record_attempt(
                    idx, status=None,
                    error="perform() returned None",
                    slept_before_s=slept,
                )
                raise LLMTransportError(
                    f"{telemetry.provider} perform() returned None on attempt {idx}"
                )

            status = int(getattr(resp, "status_code", 0) or 0)
            # Stash the last response so the retry loop can read headers
            # (Retry-After) to drive the next sleep.
            last_resp_holder[0] = resp

            if 500 <= status < 600:
                telemetry.record_attempt(
                    idx, status=status,
                    error=("" if status == 0 else f"http {status}"),
                    slept_before_s=slept,
                )
                raise LLMTransportError(
                    f"{telemetry.provider} {status} on attempt {idx}"
                )
            if status in (408, 429):
                telemetry.record_attempt(
                    idx, status=status,
                    error=f"http {status}",
                    slept_before_s=slept,
                )
                raise LLMTransportError(
                    f"{telemetry.provider} {status} on attempt {idx}"
                )
            if 400 <= status < 500:
                # Permanent client error: parse the body for context, then
                # record the attempt and raise LLMError. Do NOT retry.
                body = ""
                try:
                    body = (getattr(resp, "text", "") or "")[:300]
                except Exception:  # pragma: no cover - defensive
                    body = "<body unreadable>"
                telemetry.record_attempt(
                    idx, status=status,
                    error=f"http {status}: {body}",
                    slept_before_s=slept,
                )
                raise LLMError(
                    f"{telemetry.provider} client error {status}: {body}"
                )
            # 2xx (or 3xx) - hand off to the parser.
            try:
                payload = parse(resp)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                # Malformed body is a permanent error for this attempt.
                telemetry.record_attempt(
                    idx, status=status,
                    error=f"parse error: {type(exc).__name__}: {exc}",
                    slept_before_s=slept,
                )
                raise LLMError(
                    f"{telemetry.provider} unparseable response: {exc}"
                ) from exc
            telemetry.record_attempt(
                idx, status=status, error=None, slept_before_s=slept,
            )
            return payload

        # Main loop.
        last_exc: Exception | None = None
        last_resp_holder: list = [None]  # mutable single-cell container
        for attempt_idx in range(1, max_attempts + 1):
            try:
                result = _attempt(attempt_idx, last_resp_holder)
            except LLMTransportError as exc:
                last_exc = exc
                if attempt_idx >= max_attempts:
                    break
                # Compute sleep for the *next* attempt and stash it on
                # the just-recorded attempt so the trace knows what
                # backoff was used. The next attempt reads this back via
                # `telemetry.attempts[-1].slept_before_s`.
                delay = min(base_delay_s * (2 ** (attempt_idx - 1)), max_delay_s)
                # Retry-After header can override the schedule (capped at
                # max_delay_s). Provider passes a parser via retry_after_for.
                last_resp = last_resp_holder[0]
                if retry_after_for is not None and last_resp is not None:
                    try:
                        ra = retry_after_for(last_resp)
                    except Exception:  # pragma: no cover - defensive
                        ra = None
                    if ra is not None and ra > 0:
                        delay = min(float(ra), max_delay_s)
                if telemetry.attempts:
                    telemetry.attempts[-1].slept_before_s = delay
                time.sleep(delay)
                last_resp_holder[0] = None
                continue
            except LLMError:
                # Permanent client error / parse error. The caller is
                # expected to have finalized or will finalize telemetry
                # before re-raising; we just bubble up.
                raise
            # Success.
            telemetry.finalize(succeeded=True)
            return result

        # Exhausted retries. Finalize telemetry and re-raise.
        telemetry.finalize(succeeded=False)
        raise LLMError(
            f"{telemetry.provider} giving up after {max_attempts} attempts: "
            f"{type(last_exc).__name__ if last_exc else 'unknown'}: "
            f"{last_exc if last_exc else ''}"
        ) from last_exc

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
    "LLMCallAttempt",
    "LLMCallTelemetry",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMTransportError",
    "ToolSpec",
    "get_provider",
]
