"""In-process MCP server for Alpaca (hackathon deployment).

A minimal Model-Context-Protocol-shaped server that exposes the Alpaca
broker + market-data APIs as callable tools. Agents import
``AlpacaMCPServer`` and call ``server.call("get_quote", {"symbol": "NVDA"})``
instead of touching the REST client directly.

Why in-process?
  * The hackathon runs the orchestrator on Render's free tier (no
    separate process, no long-lived sockets). An in-process registry
    gives us the same tool-discovery + typed-args shape without the
    operational cost of a real stdio MCP server.
  * When the orchestrator later moves to a Claude-orchestrated agent
    loop, the same ``tools`` dict here can be wrapped behind the
    official MCP SDK with no API changes — the tool schemas are
    already MCP-compatible (name, description, JSON-schema input).

The tool surface covers what each agent in the multi-agent graph needs:

  market data
    - get_quote(symbol)         → latest NBBO quote
    - get_snapshot(symbol)      → full snapshot (quote + bar + fundamentals)
    - get_bars(symbol, tf, n)   → recent OHLCV bars
    - get_option_chain(symbol)  → option chain (calls + puts)
  account / portfolio
    - get_account()             → equity, cash, buying power
    - get_positions()           → list of open positions
    - get_portfolio_history()   → equity curve
  order management
    - submit_order(...)         → place an order
    - get_order(id)             → status of an order
    - cancel_order(id)          → cancel an open order
    - list_orders(...)          → recent orders
"""
from .alpaca_server import AlpacaMCPServer, ToolDef, ToolError

__all__ = ["AlpacaMCPServer", "ToolDef", "ToolError"]
