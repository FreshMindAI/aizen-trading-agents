"""Pluggable LLM provider abstraction.

Canonical request shape mirrors the Anthropic Messages API (matches how
Claude Code wires its model calls). Providers are thin adapters:

  * AnthropicProvider - native
  * OpenAIProvider    - Chat Completions translation
  * MockProvider      - deterministic for tests / paper

Default is `anthropic`; override with AIZEN_LLM_PROVIDER or LLM_PROVIDER
env vars. API keys come from ANTHROPIC_API_KEY / OPENAI_API_KEY. Custom
endpoints (LiteLLM, internal gateways) via ANTHROPIC_BASE_URL /
OPENAI_BASE_URL.
"""

from .base import (  # noqa: F401
    ANTHROPIC_API_VERSION,
    ANTHROPIC_DEFAULT_BASE_URL,
    ANTHROPIC_DEFAULT_MODEL,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_DEFAULT_MODEL,
    ToolSpec,
    get_provider,
)
from .gmi_fallback import GMIFallbackProvider  # noqa: F401
from .telemetry import (  # noqa: F401
    LLMCallAttempt,
    LLMCallTelemetry,
    LLMTransportError,
)

__all__ = [
    "ANTHROPIC_API_VERSION",
    "ANTHROPIC_DEFAULT_BASE_URL",
    "ANTHROPIC_DEFAULT_MODEL",
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
