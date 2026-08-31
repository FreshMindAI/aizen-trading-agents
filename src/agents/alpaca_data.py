"""Alpaca data API client (news endpoint).

Spec 003 / T008. The trading client lives in ``alpaca_trading.py``; this
module is the data-side counterpart, currently scoped to the news endpoint
that the research agent uses.

Per spec FR-014: the news pipeline MUST NOT call any LLM. This client is
pure HTTP + JSON; the lexicon sentiment runs in-process.

Constitution §I (Determinism over Autonomy): retries are bounded and logged;
on unrecoverable failure the caller (research node) falls back to an empty
``ResearchOutput`` rather than raising.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.config import get_settings

logger = logging.getLogger(__name__)


class AlpacaDataError(RuntimeError):
    """Raised on any non-retryable data API failure."""


class AlpacaDataClient:
    """Thin wrapper around the Alpaca news endpoint.

    Uses the same API key pair as the trading client. The base URL for the
    data API is fixed at ``https://data.alpaca.markets`` (no paper/live split
    for the data API).
    """

    BASE_URL = "https://data.alpaca.markets"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.max_retries = 3
        self.timeout = (5.0, 15.0)
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.settings.api_key_id,
            "APCA-API-SECRET-KEY": self.settings.api_secret,
            "Content-Type": "application/json",
        })

    # ---- public ---------------------------------------------------------
    def fetch_news(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        limit: int = 50,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch news articles for ``symbols`` between ``start`` and ``end``.

        ``start`` and ``end`` are ISO-8601 dates (YYYY-MM-DD); Alpaca accepts
        date-only and datetime forms. ``limit`` is per-page; we paginate via
        ``page_token`` if present in the response, until we either hit
        ``limit`` total articles or run out of pages.

        Returns a list of raw article dicts. On any non-retryable failure
        (HTTP 4xx other than 429, schema drift, etc.) returns ``[]`` and
        logs a single warning - the research node treats that as "no news".

        Conforms to spec FR-014 + US1 acceptance scenario 4 (retry with
        exponential backoff on 429 / 5xx; single non-fatal warning on
        unrecoverable failure).
        """
        all_articles: list[dict[str, Any]] = []
        page_token: str | None = None
        attempts = 0

        while len(all_articles) < limit and attempts <= self.max_retries:
            params: dict[str, Any] = {
                "symbols": ",".join(symbols),
                "start": start,
                "end": end,
                "limit": min(50, limit - len(all_articles)),
                "include_content": "true" if include_content else "false",
                "sort": "desc",
            }
            if page_token:
                params["page_token"] = page_token
            try:
                resp = self._request_with_retry("GET", "/v1beta1/news", params=params)
            except AlpacaDataError as exc:
                logger.warning("alpaca news fetch failed: %s", exc)
                return []
            articles = resp.get("news") or []
            if not articles:
                break
            all_articles.extend(articles)
            page_token = resp.get("next_page_token")
            if not page_token:
                break
            attempts += 1
        return all_articles[:limit]

    # ---- low-level ------------------------------------------------------
    def _request_with_retry(
        self, method: str, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        last_status = 0
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(method, url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_status = -1
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise AlpacaDataError(f"transport error: {exc}") from exc
            last_status = resp.status_code
            if resp.status_code < 400:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 30.0))
                    except ValueError:
                        time.sleep(0.5 * (2 ** attempt))
                else:
                    time.sleep(0.5 * (2 ** attempt))
                continue
            raise AlpacaDataError(
                f"alpaca data {resp.status_code} on {method} {path}: {resp.text[:500]}"
            )
        raise AlpacaDataError(
            f"alpaca data {method} {path} gave up: last_status={last_status}"
        )


__all__ = ["AlpacaDataClient", "AlpacaDataError"]
