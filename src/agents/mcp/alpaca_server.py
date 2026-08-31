"""Alpaca MCP server (in-process).

Exposes Alpaca broker + market-data APIs as named, JSON-schema-described
tools. Each tool is a thin wrapper around the existing client in
:mod:`src.agents.alpaca_trading` (broker) and :mod:`src.agents.alpaca_data`
(news). The server is deterministic: it does not cache, and every call
hits the broker (so an outage is visible immediately to the agent
rather than hidden behind a stale cache).

The shape mirrors what an MCP server would produce: ``tools/list``
returns ``ToolDef`` objects; ``tools/call`` returns a JSON-serialisable
result. An external MCP client could be pointed at this code by
replacing the call site — no API surface change is needed for that
migration.

Run-mode behaviour
  * ``paper`` (default) — calls go to the Alpaca paper endpoint.
  * ``live``            — calls go to the Alpaca live endpoint.
  * ``dry-run``         — ``submit_order`` returns a fake fill without
    hitting the broker. Other tools still call the broker (read-only
    data is OK on a free Render instance when paper keys are present;
    without keys, they raise a clean :class:`ToolError`).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..alpaca_trading import AlpacaTradingClient

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """Raised when an MCP tool call fails (broker error, bad args, etc.).
    The message is short and safe to surface to an LLM agent."""


@dataclass
class ToolDef:
    """MCP-shaped tool definition (name + JSON-schema input)."""
    name: str
    description: str
    input_schema: dict[str, Any]   # JSON Schema object
    handler: Callable[["AlpacaMCPServer", dict[str, Any]], Any] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _require_str(args: dict, key: str) -> str:
    v = args.get(key)
    if not isinstance(v, str) or not v:
        raise ToolError(f"{key!r} must be a non-empty string")
    return v


def _opt_int(args: dict, key: str, default: int) -> int:
    v = args.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int):
        raise ToolError(f"{key!r} must be an integer")
    return v


def _opt_str(args: dict, key: str, default: str | None = None) -> str | None:
    v = args.get(key)
    if v is None:
        return default
    if not isinstance(v, str):
        raise ToolError(f"{key!r} must be a string")
    return v


def _opt_float(args: dict, key: str) -> float | None:
    v = args.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ToolError(f"{key!r} must be a number")
    return float(v)


class AlpacaMCPServer:
    """In-process MCP server for Alpaca.

    The constructor takes a ``run_mode`` (``"paper"`` / ``"live"`` /
    ``"dry-run"``); in ``dry-run`` mode the submit/cancel tools return
    a fake fill without hitting the broker so the orchestrator can be
    exercised in tests + Render free tier without an Alpaca key.
    """

    def __init__(self, *, run_mode: str | None = None,
                 trading_client: "AlpacaTradingClient | None" = None) -> None:
        self.run_mode = (run_mode or os.getenv("RUN_MODE") or "paper").lower()
        self._trading = trading_client  # lazy-built on first use
        self._tools: dict[str, ToolDef] = {}
        self._register_default_tools()

    # ---- public surface (MCP-shaped) -----------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """Dispatch one tool call. Returns a JSON-serialisable value."""
        args = args or {}
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name!r}")
        try:
            return tool.handler(self, args)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool %s failed", name)
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    # ---- lazy broker client --------------------------------------------
    def _get_trading_client(self) -> "AlpacaTradingClient":
        if self._trading is None:
            from ..alpaca_trading import AlpacaTradingClient
            self._trading = AlpacaTradingClient()
        return self._trading

    # ---- tool registration ---------------------------------------------
    def _register_default_tools(self) -> None:
        for tool in (
            # ---- account / portfolio ----
            self._tool_get_account(),
            self._tool_get_positions(),
            self._tool_get_order(),
            self._tool_list_orders(),
            # ---- orders ----
            self._tool_submit_order(),
            self._tool_cancel_order(),
            # ---- research / market data ----
            self._tool_get_news(),
        ):
            self._tools[tool.name] = tool

    # ---- tool builders -------------------------------------------------
    def _tool_get_account(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            return server._get_trading_client().get_account()
        return ToolDef(
            name="get_account",
            description="Account snapshot (equity, cash, buying power, pattern-day-trader flag).",
            input_schema={"type": "object", "properties": {}},
            handler=_handler,
        )

    def _tool_get_positions(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            return server._get_trading_client().list_positions()
        return ToolDef(
            name="get_positions",
            description="List of open positions (symbol, qty, avg_entry_price, market_value, unrealized_pl).",
            input_schema={"type": "object", "properties": {}},
            handler=_handler,
        )

    def _tool_get_order(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            oid = _require_str(args, "order_id")
            client = server._get_trading_client()
            # The trading client has no get_order method today (it only
            # submit/cancel/list). Build a tiny request via _request.
            return client._request("GET", f"/v2/orders/{oid}")
        return ToolDef(
            name="get_order",
            description="Status of a single order by id.",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=_handler,
        )

    def _tool_list_orders(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            status = _opt_str(args, "status", "all") or "all"
            limit = _opt_int(args, "limit", 50)
            client = server._get_trading_client()
            return client._request("GET", f"/v2/orders?status={status}&limit={limit}")
        return ToolDef(
            name="list_orders",
            description="Recent orders (default: all, last 50).",
            input_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "default": "all"},
                    "limit": {"type": "integer", "default": 50},
                },
            },
            handler=_handler,
        )

    def _tool_submit_order(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            from ..protocol import Leg, OrderIntent, Side
            sym = _require_str(args, "symbol")
            qty = args.get("qty")
            if qty is None or (isinstance(qty, bool)) or not isinstance(qty, (int, float)):
                raise ToolError("'qty' must be a number")
            side_str = _require_str(args, "side").lower()
            if side_str not in ("buy", "sell"):
                raise ToolError("'side' must be 'buy' or 'sell'")
            tif = (_opt_str(args, "time_in_force", "day") or "day").lower()
            typ = (_opt_str(args, "type", "market") or "market").lower()
            limit = _opt_float(args, "limit_price")
            client_order_id = _opt_str(args, "client_order_id")
            if self.run_mode == "dry-run":
                return {
                    "id": f"dryrun-{sym}-{qty}",
                    "client_order_id": client_order_id,
                    "symbol": sym, "qty": qty, "side": side_str, "type": typ,
                    "time_in_force": tif,
                    "status": "filled" if typ == "market" else "accepted",
                    "filled_qty": qty if typ == "market" else 0,
                    "filled_avg_price": limit,
                    "dry_run": True,
                }
            leg = Leg(
                contract_symbol=sym,
                side=Side.BUY if side_str == "buy" else Side.SELL,
                quantity=int(qty),
                limit_price=limit,
            )
            intent = OrderIntent(
                strategy_id="mcp",
                underlying=sym,
                legs=[leg],
                quantity=int(qty),
                time_in_force=tif,
            )
            return server._get_trading_client().submit_order(intent)
        return ToolDef(
            name="submit_order",
            description=(
                "Place an order (market / limit / stop). In dry-run mode, "
                "returns a fake fill without hitting the broker. The "
                "``side`` is 'buy' or 'sell'; ``type`` is 'market', 'limit', "
                "'stop', or 'stop_limit'; ``time_in_force`` is 'day', 'gtc', "
                "'ioc', or 'fok'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "qty": {"type": "number"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"], "default": "market"},
                    "time_in_force": {"type": "string", "enum": ["day", "gtc", "ioc", "fok"], "default": "day"},
                    "limit_price": {"type": "number"},
                    "client_order_id": {"type": "string"},
                },
                "required": ["symbol", "qty", "side"],
            },
            handler=_handler,
        )

    def _tool_cancel_order(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            oid = _require_str(args, "order_id")
            if server.run_mode == "dry-run":
                return {"id": oid, "status": "canceled", "dry_run": True}
            return server._get_trading_client().cancel_order(oid)
        return ToolDef(
            name="cancel_order",
            description="Cancel an open order by id. In dry-run mode, returns a fake cancel.",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            handler=_handler,
        )

    def _tool_get_news(self) -> ToolDef:
        def _handler(server: "AlpacaMCPServer", args: dict) -> Any:
            sym = args.get("symbol")
            limit = _opt_int(args, "limit", 20)
            from ..alpaca_data import AlpacaDataClient
            from datetime import datetime, timedelta, timezone
            client = AlpacaDataClient()
            end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            symbols = [sym.upper()] if sym else []
            try:
                return client.fetch_news(
                    symbols=symbols, start=start, end=end, limit=limit,
                )
            except Exception as exc:  # noqa: BLE001
                # News is best-effort; don't fail the cycle on it.
                logger.warning("get_news failed: %s", exc)
                return []
        return ToolDef(
            name="get_news",
            description="Recent Alpaca news for one symbol (or all symbols) over the last 7 days.",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
            handler=_handler,
        )
