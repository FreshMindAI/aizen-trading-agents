"""Thin Alpaca REST client mirroring the spec doc exactly: explicit hosts,
header auth, request pacing, exponential backoff, next_page_token pagination.

Secrets exist only inside the requests.Session - they are never logged.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import requests

from .config import get_settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class AlpacaError(Exception):
    """Base class for Alpaca API failures."""


class AlpacaAuthError(AlpacaError):
    """401/403 - missing, invalid or insufficient credentials. Never retried."""


class AlpacaEntitlementError(AlpacaError):
    """403 caused by a missing data entitlement/agreement rather than bad keys,
    e.g. 'OPRA agreement is not signed' when a range touches the live feed."""


class AlpacaClientError(AlpacaError):
    """Any other 4xx - our request was wrong. Never retried."""


class AlpacaServerError(AlpacaError):
    """5xx still failing after retries."""


class AlpacaRateLimitError(AlpacaError):
    """429 still failing after retries."""


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s")


def normalize_ts(raw: str) -> str:
    """RFC3339 -> 'YYYY-MM-DDTHH:MM:SSZ'.

    Fixed-width UTC text sorts lexicographically == chronologically, which every
    SQL view relies on. Naive timestamps are assumed to be UTC already.
    """
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AlpacaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = requests.Session()
        # Set once; never logged, never echoed.
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.settings.api_key_id,
                "APCA-API-SECRET-KEY": self.settings.api_secret,
            }
        )
        self._min_interval = 60.0 / self.settings.rate_limit_per_min
        self._last_request_monotonic = 0.0

    # -- low-level -----------------------------------------------------------

    def _pace(self) -> None:
        """Keep average request rate under the configured budget."""
        wait = self._last_request_monotonic + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_monotonic = time.monotonic()

    def _sleep_backoff(self, attempt: int, response: requests.Response | None) -> None:
        delay = min(self.settings.backoff_cap_s, self.settings.backoff_base_s * (2**attempt))
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
        delay *= random.uniform(0.5, 1.5)  # jitter avoids thundering-herd retries
        time.sleep(min(delay, self.settings.backoff_cap_s * 2))

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Single GET with pacing + retry. Returns decoded JSON."""
        max_attempts = self.settings.max_retries + 1
        last_response: requests.Response | None = None
        for attempt in range(max_attempts):
            self._pace()
            try:
                response = self.session.get(url, params=params, timeout=self.settings.timeout_s)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt + 1 < max_attempts:
                    logger.warning(
                        "network %s on GET %s (attempt %d/%d), backing off",
                        type(exc).__name__, url, attempt + 1, max_attempts,
                    )
                    self._sleep_backoff(attempt, None)
                    continue
                raise AlpacaError(f"network failure after {max_attempts} attempts: {exc}") from exc

            status = response.status_code
            logger.info("GET %s -> %s", url, status)

            if status == 403 and ("not signed" in response.text.lower() or
                                  "entitlement" in response.text.lower()):
                raise AlpacaEntitlementError(
                    f"403 from {url}: {response.text[:200]} - the account lacks this "
                    "live-data entitlement. For historical pulls keep `end` fully in "
                    "the past (settled sessions); sign agreements in the Alpaca "
                    "dashboard for live feeds."
                )
            if status in (401, 403):
                raise AlpacaAuthError(
                    f"{status} from {url} - check ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY "
                    "and data-feed entitlements"
                )
            if status < 400:
                return response.json()
            if status in RETRYABLE_STATUSES:
                last_response = response
                if attempt + 1 < max_attempts:
                    self._sleep_backoff(attempt, response)
                    continue
                cls = AlpacaRateLimitError if status == 429 else AlpacaServerError
                raise cls(f"{status} from {url} after {max_attempts} attempts")
            raise AlpacaClientError(
                f"{status} from {url}: {response.text[:300]}"
            )
        raise AlpacaServerError(f"giving up on {url}")  # pragma: no cover - loop always returns/raises

    # -- pagination ------------------------------------------------------------

    def paginate(self, url: str, params: dict[str, Any] | None = None,
                 token_param: str = "page_token") -> Iterator[Any]:
        """Yield one JSON payload per page, following the pagination cursor until absent.

        Alpaca responses expose the cursor as `next_page_token`, but every endpoint
        we use takes it back as the REQUEST param named by token_param
        ('page_token' for stock bars v2 and options endpoints - verified
        empirically; sending 'next_page_token' gets HTTP 400).
        Tokens are opaque: passed via params so requests URL-encodes them -
        manual string interpolation corrupts them.
        """
        params = dict(params or {})
        while True:
            payload = self.get(url, params)
            yield payload
            token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not token:
                return
            params[token_param] = token

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "AlpacaClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
