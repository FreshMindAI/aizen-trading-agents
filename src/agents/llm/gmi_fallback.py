"""GMI-serving fallback wrapper.

Wraps an `AnthropicProvider` (GMI) with a bounded, single-step
fallback to a second GMI JWT (e.g. when the primary token is
rate-limited or rejected with 401/403/429).

Why a wrapper rather than editing `AnthropicProvider` directly:

* Keeps the provider's transport-retry logic untouched (it already
  handles transient 5xx/429 with bounded exponential backoff within a
  single token).
* Lets the fallback be opt-in: pass `auth_token` (primary) and
  ``fallback_token`` (secondary), or set
  ``ANTHROPIC_AUTH_TOKEN`` + ``GMI_FALLBACK_TOKEN`` env vars.
* The fallback is *one* extra attempt, not a multi-step retry loop -
  the primary already retries on transient transport errors. A
  second full retry would just amplify the same failure mode.

Activation:
    1. Set ``AIZEN_LLM_PROVIDER=gmi_fallback`` in env (or pass
       ``name='gmi_fallback'`` to ``get_provider``).
    2. Set ``ANTHROPIC_AUTH_TOKEN`` to the primary JWT.
    3. Set ``GMI_FALLBACK_TOKEN`` to the secondary JWT.
    4. Optional: ``ANTHROPIC_BASE_URL`` to the GMI host
       (default ``https://api.gmi-serving.com``).

If only the primary token is set, this wrapper degrades to a plain
``AnthropicProvider`` (no fallback, no extra latency).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

from .anthropic_provider import AnthropicProvider
from .base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTransportError,
)
from .telemetry import LLMCallTelemetry

logger = logging.getLogger(__name__)

# Status codes that indicate the token is bad / rate-limited, where
# swapping to the secondary token is more useful than a same-token
# retry. 5xx is NOT in here because the primary's own retry loop
# already handles transient server errors within the same token.
_FALLBACK_TRIGGER_STATUSES = frozenset({401, 403, 429})


class GMIFallbackProvider(AnthropicProvider):
    """AnthropicProvider + bounded single-step fallback token swap.

    On 401/403/429 (bad/expired token, or rate-limited) the wrapper
    swaps the bearer token to the fallback and retries the request
    ONCE. The fallback call uses fresh telemetry so the trace shows
    the swap explicitly. After one fallback attempt we surface the
    fallback's outcome (success or final error).
    """

    PROVIDER_NAME = "gmi_fallback"

    def __init__(self, *, fallback_token: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fallback_token = fallback_token or os.getenv("GMI_FALLBACK_TOKEN")
        if not self.fallback_token:
            logger.warning(
                "gmi_fallback constructed without GMI_FALLBACK_TOKEN; "
                "degrading to plain AnthropicProvider behavior."
            )

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            return super().complete(request)
        except (LLMTransportError, LLMError) as primary_exc:
            if not self.fallback_token:
                raise
            if not self._should_fallback(primary_exc):
                raise
            logger.warning(
                "primary GMI token failed (%s); retrying with fallback token",
                primary_exc,
            )
            # Swap the bearer token in the session, then issue exactly
            # ONE more call (the primary's own retry loop is bypassed
            # by using a fresh _session.post() at low level).
            return self._call_with_fallback_token(request, primary_exc)

    # ---- internal -------------------------------------------------------
    @staticmethod
    def _should_fallback(exc: BaseException) -> bool:
        # We only swap tokens for 401/403/429 - these point at the
        # token itself, not at the request shape. 5xx is left to the
        # primary's own retry budget.
        if isinstance(exc, LLMTransportError):
            msg = str(exc)
            for code in _FALLBACK_TRIGGER_STATUSES:
                if str(code) in msg:
                    return True
            return False
        if isinstance(exc, LLMError):
            msg = str(exc)
            # The primary raises LLMError for permanent 4xx - a 401/403
            # *is* a permanent 4xx but for token reasons we want to
            # try the fallback. Look for the status code in the message.
            for code in _FALLBACK_TRIGGER_STATUSES:
                if f"client error {code}" in msg or f"http {code}" in msg:
                    return True
        return False

    def _call_with_fallback_token(
        self,
        request: LLMRequest,
        primary_exc: BaseException,
    ) -> LLMResponse:
        """Issue a single, low-level call using the fallback token.

        We do NOT reuse ``_call_with_retry`` here - the primary just
        exhausted its budget and we want exactly one shot. We still
        log a clean telemetry record so the trace captures the
        fallback attempt.
        """
        body = self._build_body(request)
        path = os.getenv("ANTHROPIC_MESSAGES_PATH", "/v1/messages")
        url = f"{self.base_url}{path}"
        model = request.model or self.default_model

        # Build a one-off session with the fallback bearer.
        session = requests.Session()
        headers = {
            "anthropic-version": self._session.headers.get(
                "anthropic-version", "2023-06-01"
            ),
            "content-type": "application/json",
            "Authorization": f"Bearer {self.fallback_token}",
        }
        session.headers.update(headers)

        fallback_telemetry = LLMCallTelemetry(
            provider=f"{self.PROVIDER_NAME}-fallback", model=model,
        )
        try:
            resp = session.post(url, json=body, timeout=self.timeout_s)
            status = int(getattr(resp, "status_code", 0) or 0)
            if 200 <= status < 300:
                payload = resp.json()
                fallback_telemetry.record_attempt(
                    1, status=status, error=None, slept_before_s=0.0,
                )
                fallback_telemetry.finalize(succeeded=True)
                self.last_call = fallback_telemetry
                return self._parse(payload)
            # Final failure - record and raise with both errors in the
            # message so the orchestrator can log it.
            body_text = (getattr(resp, "text", "") or "")[:300]
            fallback_telemetry.record_attempt(
                1, status=status,
                error=f"http {status}: {body_text}",
                slept_before_s=0.0,
            )
            fallback_telemetry.finalize(succeeded=False)
            self.last_call = fallback_telemetry
            raise LLMError(
                f"gmi_fallback: fallback token also failed "
                f"(http {status}: {body_text}); primary error was: {primary_exc}"
            ) from primary_exc
        except requests.RequestException as exc:
            fallback_telemetry.record_attempt(
                1, status=None,
                error=f"{type(exc).__name__}: {exc}",
                slept_before_s=0.0,
            )
            fallback_telemetry.finalize(succeeded=False)
            self.last_call = fallback_telemetry
            raise LLMError(
                f"gmi_fallback: fallback token transport error: {exc}; "
                f"primary error was: {primary_exc}"
            ) from primary_exc
        finally:
            session.close()
