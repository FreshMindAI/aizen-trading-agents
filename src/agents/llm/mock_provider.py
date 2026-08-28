"""Deterministic mock LLM provider.

Default for paper / dry-run / tests. Returns a fully-populated Pydantic model
derived from the canonical LLMRequest so reasoning is auditable and
reproducible. Mirrors how Claude Code's stub mode returns canned text: the
model never pretends to reason - it inspects the input and emits a
defensible default.
"""

from __future__ import annotations

import statistics
from typing import Type

from pydantic import BaseModel

from .base import LLMError, LLMProvider, LLMRequest


class MockProvider(LLMProvider):
    PROVIDER_NAME = "mock"
    DEFAULT_BASE_URL = "mock://local"
    DEFAULT_MODEL = "mock-1"
    # No env vars - mock never needs network or keys.
    ENV_BASE_URL = ""
    ENV_MODEL = ""
    ENV_API_KEY = ""

    def __init__(self, **_: object) -> None:
        # Skip super().__init__'s env-key requirement entirely.
        self.base_url = "mock://local"
        self.default_model = "mock-1"
        self.api_key = None
        self.timeout_s = 1

    def complete(self, request: LLMRequest) -> "LLMResponse":  # type: ignore[override]
        from .base import LLMResponse
        # Heuristic: the request asks for a JSON schema of `response_model`.
        target = request.metadata.get("_response_model") if request.metadata else None
        text = ""
        if target:
            text = self._draft_for(target, request)
        else:
            text = "[mock] " + (request.messages[-1].content if request.messages else "")
        return LLMResponse(
            text=text,
            stop_reason="end_turn",
            usage={"input_tokens": 0, "output_tokens": 0},
            model=self.default_model,
            raw={"mock": True},
        )

    def complete_as(self, request: LLMRequest, response_model: Type[BaseModel]) -> BaseModel:  # type: ignore[override]
        """Generate a default Pydantic instance from the request's user payload."""
        user_payload: dict = {}
        for m in request.messages:
            if m.role == "user":
                import json
                try:
                    user_payload = json.loads(m.content)
                except (TypeError, ValueError):
                    user_payload = {"raw": m.content}
                break

        if response_model.__name__ == "AgentObservation":
            return self._observation(user_payload, request)
        if response_model.__name__ == "StrategyProposal":
            return self._proposal(user_payload, response_model)
        # Generic fallback
        kwargs = {f: user_payload.get(f) for f in response_model.model_fields if f in user_payload}
        return response_model(**kwargs)

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _draft_for(target: str, req: LLMRequest) -> str:
        # Very small JSON templates for the common cases.
        if target == "AgentObservation":
            return ('{"agent_id":"mock","message_type":"SUPERVISOR_DECISION",'
                    '"confidence":0.5,"signal":{},"evidence":[],"risks":[],'
                    '"data_version":"mock-1","model_versions":["mock-1"]}')
        if target == "StrategyProposal":
            return ('{"underlying":"SPY","legs":[{"contract_symbol":"SPY",'
                    '"side":"buy","quantity":1,"option_type":"call",'
                    '"strike":500,"expiry":"2026-12-31"}],'
                    '"thesis":"mock","expected_return":0.0,'
                    '"probability_profit":0.5,"confidence":0.5,'
                    '"max_loss":100,"expiry":"2026-12-31"}')
        return "{}"

    @staticmethod
    def _observation(payload: dict, req: LLMRequest) -> "AgentObservation":
        # Confidence = mean of any 0..1 numeric field.
        candidates: list[float] = []
        for v in payload.values():
            if isinstance(v, (int, float)) and 0 <= v <= 1:
                candidates.append(float(v))
        conf = statistics.mean(candidates) if candidates else 0.5
        conf = max(0.05, min(0.95, conf))

        signal: dict = {k: v for k, v in payload.items()
                        if isinstance(v, (int, float, str, bool, dict, list))}
        if len(signal) > 40:
            signal = dict(list(signal.items())[:40])

        # Agent id from system prompt.
        agent_id = "mock_agent"
        if req.system and "You are the " in req.system:
            try:
                agent_id = req.system.split("You are the ")[1].split(" ")[0]
            except (IndexError, ValueError):
                pass

        from ..protocol import AgentObservation, MessageType
        return AgentObservation(
            agent_id=agent_id,
            message_type=MessageType.SUPERVISOR_DECISION,
            confidence=conf,
            signal=signal,
            evidence=["mock reasoning", f"saw {len(payload)} inputs"],
            risks=[],
            data_version="mock-1",
            model_versions=["mock-1"],
        )

    @staticmethod
    def _proposal(payload: dict, response_model: Type[BaseModel]):
        from ..protocol import Leg, OptionType, Side, StrategyProposal
        candidates = payload.get("candidates") or []
        if not candidates:
            # Return a tiny no-op proposal.
            return StrategyProposal(
                underlying="SPY",
                legs=[Leg(contract_symbol="SPY", side=Side.BUY, quantity=1,
                          option_type=OptionType.CALL, strike=500.0,
                          expiry="2026-12-31")],
                thesis="mock provider: no candidates",
                expected_return=0.0,
                probability_profit=0.5,
                confidence=0.1,
                max_loss=100.0,
                expiry="2026-12-31",
            )
        ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
        top = ranked[0]
        return StrategyProposal(
            underlying=top.get("underlying", "SPY"),
            legs=[Leg(contract_symbol=top.get("contract_symbol", "SPY"),
                      side=Side.BUY, quantity=1,
                      option_type=OptionType.CALL,
                      strike=float(top.get("strike", 500.0)),
                      expiry=top.get("expiry", "2026-12-31"))],
            thesis=top.get("thesis", "mock-selected highest-score candidate"),
            expected_return=float(top.get("expected_return", 0.0)),
            probability_profit=float(top.get("probability_profit", 0.5)),
            confidence=float(top.get("confidence", 0.5)),
            max_loss=float(top.get("max_loss", 100.0)),
            expiry=top.get("expiry", "2026-12-31"),
            score=float(top.get("score", 0.0)),
        )
