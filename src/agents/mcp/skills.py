"""Agent skills registry: which MCP tools each agent can call.

Each agent in the multi-agent graph has a narrow, declared set of
"skills" — the MCP tools it is allowed to call. The orchestrator hands
each agent a ``SkillRegistry`` scoped to that subset, so a research
agent cannot accidentally submit an order and a risk agent cannot
fetch the option chain. The skills map is static (defined in code)
because the tool-to-agent mapping is part of the system's safety
contract, not a runtime configuration.

Why a per-agent registry rather than a single global one?
  * The risk agent should NOT be able to submit an order (it only
    evaluates). The execution agent should NOT be able to call
    ``get_quote`` (it only submits). Pre-binding the allowed tools
    keeps each agent's blast radius small.
  * When the orchestrator moves to a Claude-orchestrated loop, the
    per-agent registry becomes the ``tools=`` parameter to the
    sub-agent's prompt; no further filtering is needed.
  * Tests can construct a tiny registry and assert that the right
    tools are exposed to the right agent.

The map mirrors the graph in :mod:`src.agents.graph`:
  - research       → read-only market data
  - regime         → read-only market data + account
  - direction      → read-only market data
  - volatility     → read-only market data
  - options_structure → read-only market data + option chain
  - portfolio      → read-only account + positions + history
  - supervisor     → read-only account + positions + orders
  - risk           → read-only account + positions
  - execution      → write: submit_order, cancel_order; read: get_order, list_orders, get_positions, get_account
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .alpaca_server import AlpacaMCPServer, ToolDef, ToolError

# Static skill map. Each key is the graph node name (matches
# Orchestrator._compose). Each value is the list of MCP tool names the
# agent is allowed to call.
AGENT_SKILLS: dict[str, tuple[str, ...]] = {
    "research": (
        "get_news",          # the only tool wired today; will grow
    ),
    "regime": (
        "get_news", "get_account",
    ),
    "direction": (
        "get_news", "get_account",
    ),
    "volatility": (
        "get_news", "get_account",
    ),
    "options_structure": (
        "get_news", "get_account",
    ),
    "portfolio": (
        "get_account", "get_positions",
    ),
    "supervisor": (
        "get_account", "get_positions", "list_orders",
    ),
    "risk": (
        "get_account", "get_positions",
    ),
    "execution": (
        # The execution agent is the ONLY agent allowed to submit or
        # cancel orders. It can also read account/positions and
        # order status to confirm the fill.
        "get_account", "get_positions", "get_order", "list_orders",
        "submit_order", "cancel_order",
    ),
}


@dataclass
class SkillRegistry:
    """Per-agent view of the MCP server.

    Wraps an :class:`AlpacaMCPServer` and exposes only the tools the
    owning agent is allowed to call. Unknown / disallowed tool names
    raise :class:`ToolError` so the LLM gets a clear refusal.
    """
    agent_name: str
    server: AlpacaMCPServer
    allowed: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.allowed:
            self.allowed = AGENT_SKILLS.get(self.agent_name, ())

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            self.server._tools[name].to_dict()
            for name in self.allowed
            if name in self.server._tools
        ]

    def call(self, tool_name: str, args: dict[str, Any] | None = None) -> Any:
        if tool_name not in self.allowed:
            raise ToolError(
                f"agent {self.agent_name!r} is not allowed to call "
                f"{tool_name!r}; allowed: {list(self.allowed)}"
            )
        return self.server.call(tool_name, args)

    def allowed_tool_names(self) -> tuple[str, ...]:
        return tuple(self.allowed)


def build_skill_registry(
    agent_name: str,
    server: AlpacaMCPServer,
    *,
    override: Iterable[str] | None = None,
) -> SkillRegistry:
    """Build a :class:`SkillRegistry` for one agent.

    ``override`` (optional) replaces the default AGENT_SKILLS entry —
    use sparingly, only for tests. Production code should rely on the
    static map so the safety contract is enforced at code-review time.
    """
    allowed = tuple(override) if override is not None else AGENT_SKILLS.get(agent_name, ())
    return SkillRegistry(agent_name=agent_name, server=server, allowed=allowed)
