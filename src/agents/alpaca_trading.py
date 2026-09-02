"""Thin Alpaca Trading API client.

Mirrors the existing AlpacaClient in src/alpaca_client.py but targets the
*trading* host (paper-api / api). Methods:

  * submit_order(intent)        - POST /v2/orders, returns broker response
  * get_account()               - GET  /v2/account
  * list_positions()            - GET  /v2/positions
  * cancel_order(id)            - DELETE /v2/orders/{id}
  * close_position(symbol)      - DELETE /v2/positions/{symbol}  (used by the
                                  per-position stop-loss added 2026-09-02)

End-of-day behavior, retries, and pacing are governed by config/alpaca.yaml
(trading section). Secrets are loaded once and never logged.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from ..config import get_settings
from .protocol import OrderIntent, Side

logger = logging.getLogger(__name__)

# Resolve config/alpaca.yaml -> trading section once at import.
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "alpaca.yaml"
try:
    import yaml as _yaml
    _ALPACA_CFG = _yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
except FileNotFoundError:
    _ALPACA_CFG = {}


class AlpacaTradingError(RuntimeError):
    """Raised on any non-retryable trading API failure."""


class AlpacaTradingClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        # Configurable: YAML > Settings > paper-api default.
        yaml_trading = (_ALPACA_CFG.get("trading") or {})
        self.base_url = (
            self.settings.trading_base_url
            or yaml_trading.get("base_url")
            or "https://paper-api.alpaca.markets"
        ).rstrip("/")
        self.rate_limit_per_min = int(
            yaml_trading.get("rate_limit_per_min", 200)
        )
        self.max_retries = int(yaml_trading.get("max_retries", 3))
        timeout = yaml_trading.get("timeout_s") or [5, 15]
        self.timeout = (float(timeout[0]), float(timeout[-1]))

        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.settings.api_key_id,
            "APCA-API-SECRET-KEY": self.settings.api_secret,
            "Content-Type": "application/json",
        })
        self._min_interval = 60.0 / max(1, self.rate_limit_per_min)
        self._last_request_monotonic = 0.0

    # ---- low-level ------------------------------------------------------
    def _pace(self) -> None:
        wait = self._last_request_monotonic + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_monotonic = time.monotonic()

    def _request(self, method: str, path: str, json_body: dict | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_status = 0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                resp = self._session.request(
                    method, url, json=json_body, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise AlpacaTradingError(f"transport error: {exc}") from exc
            last_status = resp.status_code
            if resp.status_code < 400:
                return resp.json() if resp.content else {}
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
            raise AlpacaTradingError(
                f"alpaca trading {resp.status_code} on {method} {path}: "
                f"{resp.text[:500]}"
            )
        raise AlpacaTradingError(
            f"alpaca trading {method} {path} gave up: last_status={last_status}, exc={last_exc}"
        )

    # ---- public ---------------------------------------------------------
    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/v2/account")

    def list_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/positions") or []

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v2/orders/{order_id}")

    def close_position(self, symbol: str) -> dict[str, Any]:
        """Market-close an open position via ``DELETE /v2/positions/{symbol}``.

        Used by the per-position stop-loss pre-flight added 2026-09-02
        (see :mod:`src.agents.position_management`). Alpaca returns the
        closing order that was created; we surface the standard envelope
        so callers can log ``broker_order_id`` and ``status`` alongside
        the rest of the cycle's execution_result. The qty defaults to
        the full position; a partial close is not supported today
        (the cycle has one decision per symbol, so a partial close
        adds no value for our use case).
        """
        return self._request("DELETE", f"/v2/positions/{symbol}")

    def list_fills(
        self,
        after: str | None = None,
        until: str | None = None,
        *,
        status: str = "filled",
        limit: int = 500,
        side: str | None = None,
    ) -> list[dict[str, Any]]:
        """Page through ``/v2/orders`` and return every order matching
        the filters.

        ``after`` and ``until`` are RFC-3339 timestamps (e.g. from
        :func:`src.db.utc_now_iso`). ``side`` is optional and filters
        on ``buy`` / ``sell`` server-side. We paginate via the
        ``page_token`` query param; loop terminates when the response
        is empty or the next page would be empty.

        Used by the daily-P&L loop to grade the previous session's
        trades.
        """
        all_orders: list[dict[str, Any]] = []
        page_token: str | None = None
        attempts = 0
        while len(all_orders) < limit and attempts <= self.max_retries:
            params: dict[str, Any] = {
                "status": status,
                "limit": min(500, limit - len(all_orders)),
                "direction": "desc",
            }
            if after:
                params["after"] = after
            if until:
                params["until"] = until
            if side:
                params["side"] = side
            if page_token:
                params["page_token"] = page_token
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            try:
                resp = self._request("GET", f"/v2/orders?{qs}")
            except AlpacaTradingError as exc:
                logger.warning("alpaca orders fetch failed: %s", exc)
                return all_orders
            orders = resp if isinstance(resp, list) else []
            if not orders:
                break
            all_orders.extend(orders)
            if len(all_orders) >= limit:
                break
            # Alpaca's pagination uses a cursor in the response; the
            # official pattern is to pass `page_token` from the previous
            # response. If the API didn't return one we're done.
            page_token = (
                orders[-1].get("id") if isinstance(orders[-1], dict) else None
            )
            if not page_token:
                break
            attempts += 1
        return all_orders[:limit]

    def submit_order(self, intent: OrderIntent) -> dict[str, Any]:
        """Convert a multi-leg OrderIntent to one or more broker orders.

        Multi-leg option orders in Alpaca use the `legs` parameter on
        /v2/orders. For the MVP, we send them as a single multi-leg order.
        """
        body = self._intent_to_broker_request(intent)
        result = self._request("POST", "/v2/orders", json_body=body)
        return {
            "status": "submitted",
            "broker_order_id": result.get("id"),
            "client_order_id": result.get("client_order_id"),
            "submitted_at": result.get("submitted_at"),
            "raw": result,
        }

    # ---- conversion -----------------------------------------------------
    @staticmethod
    def _intent_to_broker_request(intent: OrderIntent) -> dict[str, Any]:
        symbols = [leg.contract_symbol for leg in intent.legs]
        if len(symbols) == 1:
            # Single-leg order.
            leg = intent.legs[0]
            order: dict[str, Any] = {
                "symbol": leg.contract_symbol,
                "qty": intent.quantity,
                "side": leg.side.value,
                "type": "limit" if leg.limit_price else "market",
                "time_in_force": intent.time_in_force,
            }
            if leg.limit_price:
                order["limit_price"] = float(leg.limit_price)
            return order
        # Multi-leg: mleg order with explicit legs.
        return {
            "symbol": symbols[0],
            "qty": intent.quantity,
            "side": intent.legs[0].side.value,
            "type": "limit" if intent.legs[0].limit_price else "market",
            "time_in_force": intent.time_in_force,
            "order_class": "mleg",
            "legs": [
                {
                    "symbol": leg.contract_symbol,
                    "ratio_qty": leg.quantity,
                    "side": leg.side.value,
                    **({"limit_price": float(leg.limit_price)}
                       if leg.limit_price else {}),
                }
                for leg in intent.legs
            ],
        }
