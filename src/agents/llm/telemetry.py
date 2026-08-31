"""LLM call telemetry + retryable transport error.

Why a separate module?
----------------------
Both `AnthropicProvider.complete` and `OpenAIProvider.complete` need to
share two things:

  1. A bounded retry loop with exponential backoff (lives on
     `LLMProvider._call_with_retry` in `base.py`).
  2. A small record of what happened on the wire so the cycle trace can
     surface *why* a call failed (status code, last error, attempt count).

The record is the `LLMCallTelemetry` dataclass below. It is created per
LLM call, populated by the retry loop on each attempt, and stashed on
`provider.last_call` so the orchestrator can read it after the cycle
without the provider needing to know about traces or vice versa.

`LLMTransportError` is a subclass of `LLMError` so existing
`except LLMError:` catches keep working, while the retry loop can
distinguish "this is a transport failure, retry me" from "this is a
permanent client error, do not retry me".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ``LLMTransportError`` is intentionally NOT a subclass of ``LLMError``
# in this module to avoid a circular import between ``base.py`` and
# ``telemetry.py``. Callers that need to catch both should use
# ``except (LLMError, LLMTransportError)`` or just catch the
# transport error first then re-raise as LLMError if they want a
# single point. The retry loop in ``base._call_with_retry`` always
# raises ``LLMTransportError`` for transport-class failures and
# ``LLMError`` for permanent client / parse errors.


class LLMTransportError(Exception):
    """Retryable transport / HTTP failure. Raised by the retry loop in
    ``LLMProvider._call_with_retry`` for transport errors, 5xx, 408,
    and 429 responses - all of which are candidates for another attempt.
    """


@dataclass
class LLMCallAttempt:
    """One HTTP attempt inside a retried LLM call."""
    attempt: int              # 1-indexed
    status: int | None        # HTTP status (None for transport error)
    error: str | None         # error string (None for 2xx)
    slept_before_s: float     # how long we slept before this attempt


@dataclass
class LLMCallTelemetry:
    """End-to-end record of one logical LLM call (one or more HTTP attempts).

    Created at the start of `provider.complete()`, mutated on every attempt,
    and stashed on `provider.last_call` regardless of whether the call
    succeeded or raised. The orchestrator reads `last_call` after the cycle
    to emit the `llm_call` step in the cycle trace.
    """
    provider: str
    model: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    attempts: list[LLMCallAttempt] = field(default_factory=list)
    succeeded: bool = False
    final_status: int | None = None
    final_error: str | None = None
    total_sleep_s: float = 0.0

    def record_attempt(
        self,
        attempt: int,
        *,
        status: int | None,
        error: str | None,
        slept_before_s: float,
    ) -> None:
        self.attempts.append(LLMCallAttempt(
            attempt=attempt,
            status=status,
            error=error,
            slept_before_s=slept_before_s,
        ))
        self.final_status = status
        self.final_error = error

    def finalize(self, *, succeeded: bool) -> None:
        self.ended_at = time.time()
        self.succeeded = succeeded
        self.total_sleep_s = sum(a.slept_before_s for a in self.attempts)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return (end - self.started_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary used by the cycle trace. Stable shape."""
        return {
            "provider": self.provider,
            "model": self.model,
            "attempts": self.attempt_count,
            "succeeded": self.succeeded,
            "final_status": self.final_status,
            "final_error": self.final_error,
            "total_sleep_s": round(self.total_sleep_s, 4),
            "duration_ms": round(self.duration_ms, 3),
            "per_attempt": [
                {
                    "attempt": a.attempt,
                    "status": a.status,
                    "error": a.error,
                    "slept_before_s": round(a.slept_before_s, 4),
                }
                for a in self.attempts
            ],
        }


__all__ = ["LLMCallAttempt", "LLMCallTelemetry", "LLMTransportError"]
