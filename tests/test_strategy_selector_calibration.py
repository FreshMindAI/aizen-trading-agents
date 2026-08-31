"""Regression tests for the XGBoost-gate re-centering (Option A).

Background
----------
The original ``direction_prob_min: 0.55`` threshold in
``config/agents.yaml`` was set with the implicit assumption of a 50/50 market.
The trained ``direction_h4_xgb_clf`` model has a test base rate of ~0.29
(only 29% of 4-hour windows closed up in the test period), so the model's
output mass is concentrated in 0.05-0.45. The hard 0.55 gate was therefore
unreachable and the long_stock branch was dead.

The fix (``src/agents/nodes/strategy_selector.py``) reads the trained
artifact's ``test.base_rate`` at import time and re-centers:
  XGB_NEUTRAL_LOW   = base_rate - 0.05
  XGB_NEUTRAL_HIGH  = base_rate + 0.20
  XGB_LONG_THRESHOLD = base_rate + 0.10

These tests pin that the re-centering actually happens, falls back to the
legacy constants when no artifact is present, and is documented.

Why these matter: a future refactor that drops the import-time artifact read
would silently re-introduce the 0.55 gate and the long_stock branch would
stop firing. This test catches that.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.inference import (  # noqa: E402
    MODEL_DIR,
    _latest_artifact,
)


def _meta() -> dict | None:
    """Read the newest direction_h4_xgb_clf meta, if any."""
    p = _latest_artifact(MODEL_DIR, "direction", 4, "clf")
    if p is None:
        return None
    meta = p.with_suffix(".meta.json")
    if not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8"))


def test_xgb_long_threshold_below_or_equal_to_artifact_base_rate_plus_cushion():
    """The re-centered long threshold must be reachable for the trained
    model: XGB_LONG_THRESHOLD <= base_rate + 0.15 (cushion is 0.10 + 5%
    float tolerance). When no artifact is present the legacy 0.55 fallback
    is used; we accept that case too.
    """
    meta = _meta()
    if meta is None:
        pytest.skip("no trained direction_h4_xgb_clf artifact")
    br = meta["test_metrics"]["test"]["base_rate"]
    # Import here so the artifact read has already happened
    from src.agents.nodes import strategy_selector as sel
    # Re-centered gate = base_rate + 0.10
    assert sel.XGB_LONG_THRESHOLD <= br + 0.15, (
        f"XGB_LONG_THRESHOLD={sel.XGB_LONG_THRESHOLD:.3f} is set too high "
        f"for base_rate={br:.3f}; expected <= {br + 0.15:.3f}"
    )
    # And it should be > base_rate (positive cushion).
    assert sel.XGB_LONG_THRESHOLD >= br, (
        f"XGB_LONG_THRESHOLD={sel.XGB_LONG_THRESHOLD:.3f} must be >= "
        f"base_rate={br:.3f}"
    )


def test_neutral_low_below_neutral_high_below_one():
    """The neutral band must be ordered: NEUTRAL_LOW < NEUTRAL_HIGH < 1.0."""
    from src.agents.nodes import strategy_selector as sel
    assert sel.XGB_NEUTRAL_LOW < sel.XGB_NEUTRAL_HIGH
    assert sel.XGB_NEUTRAL_HIGH < 1.0
    assert sel.XGB_NEUTRAL_LOW > 0.0


def test_long_threshold_inside_neutral_band():
    """The long threshold must sit inside (or at the upper edge of) the
    neutral band: NEUTRAL_LOW <= LONG_THRESHOLD <= NEUTRAL_HIGH. This
    keeps the "long" branch reachable for any xgb reading that the
    neutral band considers at least neutral.
    """
    from src.agents.nodes import strategy_selector as sel
    assert sel.XGB_LONG_THRESHOLD >= sel.XGB_NEUTRAL_LOW, (
        f"XGB_LONG_THRESHOLD={sel.XGB_LONG_THRESHOLD} < NEUTRAL_LOW="
        f"{sel.XGB_NEUTRAL_LOW}; long branch fires for bearish xgb"
    )
    assert sel.XGB_LONG_THRESHOLD <= sel.XGB_NEUTRAL_HIGH + 0.001, (
        f"XGB_LONG_THRESHOLD={sel.XGB_LONG_THRESHOLD} > NEUTRAL_HIGH="
        f"{sel.XGB_NEUTRAL_HIGH}; long branch doesn't reach bullish"
    )


def test_effective_threshold_propagates_to_direction_node(monkeypatch):
    """The direction node must report the *re-centered* effective threshold
    in its payload (so the LLM sees the real gate, not the legacy 0.55)."""
    from src.agents.protocol import MarketSnapshot, UnderlyingScore
    from src.agents.nodes import direction as dir_mod
    from src.agents.nodes import strategy_selector as sel

    captured = {}

    class _FakeObs:
        message_type = None
        def model_copy(self, update):
            captured["update"] = update
            # Apply the update so downstream code sees the new message_type
            new = _FakeObs()
            new.message_type = update.get("message_type", self.message_type)
            return new

        def model_dump(self, mode=None):
            return {}

    class _FakeLLM:
        def invoke(self, *a, **kw):
            return _FakeObs()

    def _fake_llm_call(llm, agent_id, role, payload, out_cls):
        captured.setdefault("payloads", []).append(payload)
        return _FakeObs()

    # Need a non-None snapshot with at least one underlying for the node
    # to reach the _llm_call.
    snap = MarketSnapshot(
        timestamp="t",
        underlyings=[UnderlyingScore(symbol="X", timestamp="t", horizon_bars=4)],
    )

    class _State:
        market_snapshot = snap
        gnn_output = None
        decision_id = "x"
        cycle_started_at = "t"

    monkeypatch.setattr(dir_mod, "_llm_call", _fake_llm_call)
    node = dir_mod.build_node(_FakeLLM(), {"thresholds": {"direction_prob_min": 0.55}}, None)
    node(_State())
    assert captured.get("payloads"), "direction node did not call _llm_call"
    payload = captured["payloads"][0]
    # The reported threshold must NOT be the legacy 0.55 when the artifact
    # is present. It must be the re-centered value.
    reported = payload.get("direction_prob_min")
    assert reported is not None
    if sel._BASE_RATE is not None:
        expected = min(0.95, sel._BASE_RATE + sel._BASE_RATE_CUSHION)
        assert reported == pytest.approx(expected, abs=1e-9), (
            f"direction node reported legacy gate {reported}; expected "
            f"re-centered {expected}"
        )
    else:
        # No artifact -> legacy fallback is allowed.
        assert reported == pytest.approx(0.55, abs=1e-9)
