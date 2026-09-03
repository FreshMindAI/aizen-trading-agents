"""Shared helpers for the agent node factories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel

from ..llm import LLMMessage, LLMProvider, LLMRequest
from ..protocol import (
    AgentMessage,
    AgentObservation,
    DecisionState,
    MessageType,
)

# Single module-level logger so every agent's per-cycle reasoning lands on
# the cron stdout in chronological order. The operator sees one INFO line
# per agent per cycle in the GH Actions log; the full observation payload
# stays in the per-cycle JSONL trace under models/cycle_traces/.
_AGENT_LOG = logging.getLogger("aizen.agent")


def _to_message(obs: AgentObservation, decision_id: str,
                sender: str, receiver: str) -> AgentMessage:
    return AgentMessage(
        decision_id=decision_id,
        sender=sender,
        receiver=receiver,
        message_type=obs.message_type,
        payload=obs.model_dump(mode="json"),
    )


def _llm_call(
    llm: LLMProvider,
    agent_id: str,
    role: str,
    payload: dict[str, Any],
    response_model: Type[BaseModel],
) -> BaseModel:
    """Build the canonical request and call the LLM. Falls back to a
    deterministic default if the provider raises (so the loop never stalls)."""
    schema_hint = response_model.model_json_schema()
    system = LLMProvider.system_prompt(agent_id, role,
        f"JSON-schema for response: {schema_hint}").content
    try:
        req = LLMRequest(
            system=system,
            messages=[LLMMessage(role="user", content=_json(payload))],
            max_tokens=1024,
            temperature=0.0,
        )
        return llm.complete_as(req, response_model)
    except Exception:
        # Deterministic fallback: ask the mock path even if the configured
        # provider failed. The orchestrator decides what to do with the
        # resulting observation (typically: treat as low confidence).
        from ..llm import get_provider
        stub = get_provider("mock")
        req = LLMRequest(
            system=system,
            messages=[LLMMessage(role="user", content=_json(payload))],
            max_tokens=1024,
            temperature=0.0,
            metadata={"_response_model": response_model.__name__},
        )
        return stub.complete_as(req, response_model)


def _json(payload: Any) -> str:
    import json
    return json.dumps(payload, default=str)


@dataclass
class AgentResult:
    observations: list[AgentObservation]
    messages: list[AgentMessage]

    def as_update(self) -> dict[str, Any]:
        return {
            "agent_observations": self.observations,
            "agent_messages": self.messages,
        }


# ---------------------------------------------------------------------------
# Per-cycle agent-reasoning log
# ---------------------------------------------------------------------------
# Cap the signal-dict size we print. Most agent signals are 3-6 keys; the
# options_structure_agent's signal is a small {candidates_returned,
# dte_fallback} dict. The cap is just defensive against future agents
# dumping the whole snapshot into ``signal``.
_MAX_SIGNAL_KEYS = 8
_MAX_EVIDENCE = 3
_MAX_RISKS = 2


def _compact(value: Any) -> str:
    """Render a value compactly for one-line logs. Floats are fixed-2,
    long strings are truncated, dicts/lists are JSON-serialised up to
    ~160 chars. The point is human-readability in the cron stdout, not
    fidelity (the JSONL trace has the full payload)."""
    try:
        if isinstance(value, float):
            return f"{value:.2f}"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            preview = ", ".join(_compact(v) for v in value[:3])
            return f"[{preview}{'...' if len(value) > 3 else ''}]"
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = list(value.items())[:_MAX_SIGNAL_KEYS]
            rendered = ", ".join(
                f"{k}={_compact(v)}" for k, v in items
            )
            return rendered[:160] + ("..." if len(rendered) > 160 else "")
        s = str(value)
        return s[:80] + ("..." if len(s) > 80 else "")
    except Exception:  # noqa: BLE001
        return repr(value)[:80]


def _log_agent_observation(agent_id: str, obs: AgentObservation) -> None:
    """Emit one INFO line per agent observation so the cron stdout
    (visible in the GH Actions log) shows what each agent decided.

    The full observation is also written to the per-cycle JSONL trace;
    this helper is the operator-friendly one-liner. Format::

        [agent_id] confidence=0.45 signal={candidates_returned=0, dte_fallback=no_dte_data} evidence="..." risks="..."

    Args:
        agent_id: short identifier (e.g. ``regime_agent``). Used as the
            log-line prefix; also used downstream to look up which agent
            said what in the trace.
        obs: the AgentObservation that the LLM (or the mock fallback)
            just produced. ``obs.signal`` and ``obs.evidence`` are
            rendered compactly.
    """
    try:
        signal = obs.signal or {}
        signal_str = _compact(signal) if signal else "(no signal)"
        evidence = obs.evidence or []
        evidence_str = " | ".join(evidence[:_MAX_EVIDENCE]) if evidence else ""
        risks = obs.risks or []
        risks_str = " | ".join(risks[:_MAX_RISKS]) if risks else ""
        conf = float(getattr(obs, "confidence", 0.0) or 0.0)
        parts = [
            f"[{agent_id}]",
            f"confidence={conf:.2f}",
            f"signal={signal_str}",
        ]
        if evidence_str:
            parts.append(f"evidence={evidence_str!r}")
        if risks_str:
            parts.append(f"risks={risks_str!r}")
        _AGENT_LOG.info(" ".join(parts))
    except Exception as exc:  # noqa: BLE001
        # Logging must never break the cycle. If the rendering itself
        # raises (e.g. on an exotic signal payload), fall back to a
        # one-liner with the agent name + the exception class.
        _AGENT_LOG.warning("[%s] _log_agent_observation failed: %s",
                           agent_id, type(exc).__name__)
