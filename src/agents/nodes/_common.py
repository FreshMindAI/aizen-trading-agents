"""Shared helpers for the agent node factories."""

from __future__ import annotations

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
