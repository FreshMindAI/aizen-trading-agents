"""Tests for the 2026-09-03 position-management hardening.

Four independent failure-mode fixes, each covered by a focused test:

  1. Fail-CLOSED on broker fetch error
     test_orchestrator_fail_closed_on_broker_error
     The orchestrator must short-circuit the cycle to NO_TRADE/BLOCKED
     when Alpaca's list_positions call raises. Previous behavior
     returned [] which silently disabled every pre-flight check.

  2. -25% early warning
     test_early_warning_returns_positions_in_warning_zone
     test_early_warning_excludes_stop_loss_zone
     Positions between -25% and -50% loss are returned; positions
     past -50% (which would be auto-closed) are NOT also returned as
     warnings (the louder signal wins).

  3. 404 from close_position is success
     test_auto_close_stop_loss_404_is_already_closed
     When the broker returns 404 (position was closed by another
     path between our snapshot and the call), the result is
     status="already_closed", not "error".

  4. Kill-switch latch table
     test_kill_switch_latch_persists_for_today
     test_kill_switch_latch_clears_on_new_day
     test_orchestrator_short_circuits_on_latched_kill_switch
     A kill-switch trip is written to kill_switch_latch; subsequent
     ticks on the same UTC day return NO_TRADE without re-running
     the math. The next UTC day has a clean slate.

The orchestrator-level tests use a stubbed AlpacaTradingClient and
the public ``Orchestrator.run_cycle`` API so we exercise the wiring
end-to-end, not just the helper functions in isolation.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents import position_management as pm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pos(symbol: str, qty: float, entry: float, mark: float) -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry_price": entry,
        "current_price": mark,
        "unrealized_pl": (mark - entry) * qty,
    }


def _fresh_db() -> sqlite3.Connection:
    """Build an in-memory DB with the kill_switch_latch table present.

    Mirrors what ``init_db`` does on a normal DB. The 71_* migration
    adds the latch table; we execute it explicitly here so tests
    don't depend on running the full migration set.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    sql_path = REPO / "sql" / "71_kill_switch_latch.sql"
    conn.executescript(sql_path.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def _full_db(tmp_path: Path) -> sqlite3.Connection:
    """Build a file-backed DB with the full schema applied. Used by
    orchestrator-level tests that need the decision_journal table to
    exist for the journal upsert path."""
    from src.db import init_db
    db_path = tmp_path / "trading.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# 2. -25% early warning
# ---------------------------------------------------------------------------
def test_early_warning_returns_positions_in_warning_zone():
    """A position at -20% loss is in the early-warning zone (between
    -15% warning and -30% stop-loss) and must be returned with a
    warning entry."""
    pos = _make_pos("AAPL", 10, 100, 80)  # -20%
    out = pm.early_warning_positions([pos], pct=-0.15, stop_loss_pct=-0.30)
    assert len(out) == 1
    assert out[0]["symbol"] == "AAPL"
    assert out[0]["loss_pct"] == pytest.approx(-0.20)


def test_early_warning_excludes_stop_loss_zone():
    """A position at -60% loss is past the -30% stop-loss; it should
    NOT also appear in the warning list (the auto-close is the
    louder signal — the warning would be redundant noise)."""
    pos = _make_pos("NVDA", 1, 200, 80)  # -60%
    out = pm.early_warning_positions([pos], pct=-0.15, stop_loss_pct=-0.30)
    assert out == []


def test_early_warning_excludes_safe_positions():
    """A position at -5% loss is well below the warning threshold;
    no warning, no entry."""
    pos = _make_pos("MSFT", 5, 400, 380)  # -5%
    out = pm.early_warning_positions([pos], pct=-0.15, stop_loss_pct=-0.30)
    assert out == []


# ---------------------------------------------------------------------------
# 3. 404 from close_position is success
# ---------------------------------------------------------------------------
def test_auto_close_stop_loss_404_is_already_closed():
    """When close_position raises an error containing '404' the
    function returns status='already_closed', not 'error'."""
    fake_pos = _make_pos("AAPL", 1, 100, 30)  # -70% past -50%
    client = MagicMock()
    client.close_position.side_effect = RuntimeError(
        "alpaca trading 404 on DELETE /v2/positions/AAPL: ..."
    )
    results = pm.auto_close_stop_loss(client, [fake_pos], pct=-0.50)
    assert len(results) == 1
    assert results[0]["symbol"] == "AAPL"
    assert results[0]["status"] == "already_closed"
    # No error field on a success path.
    assert "error" not in results[0]


def test_auto_close_stop_loss_real_error_stays_error():
    """A non-404 broker error (e.g. 503) must still surface as
    'error' so the operator sees the failure mode."""
    fake_pos = _make_pos("NVDA", 1, 200, 50)
    client = MagicMock()
    client.close_position.side_effect = RuntimeError(
        "alpaca trading 503 on DELETE /v2/positions/NVDA: ..."
    )
    results = pm.auto_close_stop_loss(client, [fake_pos], pct=-0.50)
    assert results[0]["status"] == "error"
    assert "503" in results[0]["error"]


def test_auto_close_stop_loss_happy_path():
    """The successful close path still works and returns
    status='closed' with the broker_order_id."""
    fake_pos = _make_pos("AAPL", 1, 100, 30)
    client = MagicMock()
    client.close_position.return_value = {"id": "order-123"}
    results = pm.auto_close_stop_loss(client, [fake_pos], pct=-0.50)
    assert results[0]["status"] == "closed"
    assert results[0]["broker_order_id"] == "order-123"


# ---------------------------------------------------------------------------
# 4. Kill-switch latch table
# ---------------------------------------------------------------------------
def test_kill_switch_latch_persists_for_today():
    """After record_kill_switch_latch, is_kill_switch_latched_today
    must return True for the same UTC day and False the day after."""
    conn = _fresh_db()
    now = datetime(2026, 9, 3, 14, 30, 0, tzinfo=timezone.utc)
    assert pm.is_kill_switch_latched_today(conn, now=now) is False
    pm.record_kill_switch_latch(
        conn, total_pnl=-3000.0, threshold_usd=-2000.0, pct=-0.02, now=now,
    )
    assert pm.is_kill_switch_latched_today(conn, now=now) is True
    # Day +1: latch should NOT apply (different UTC date).
    tomorrow = now.replace(day=now.day + 1)
    assert pm.is_kill_switch_latched_today(conn, now=tomorrow) is False


def test_kill_switch_latch_clears_on_new_day():
    """Recording for a different day does not affect a query for
    today. This is the implicit-clear-via-PK behavior we depend on."""
    conn = _fresh_db()
    yesterday = datetime(2026, 9, 2, 11, 0, 0, tzinfo=timezone.utc)
    today = datetime(2026, 9, 3, 11, 0, 0, tzinfo=timezone.utc)
    pm.record_kill_switch_latch(
        conn, total_pnl=-3000.0, threshold_usd=-2000.0, pct=-0.02, now=yesterday,
    )
    assert pm.is_kill_switch_latched_today(conn, now=yesterday) is True
    assert pm.is_kill_switch_latched_today(conn, now=today) is False


def test_kill_switch_latch_no_conn_is_noop():
    """None conn: helper returns False on read, no-op on write. Used
    by tests that don't have a real DB."""
    assert pm.is_kill_switch_latched_today(None) is False
    # Write must not raise.
    pm.record_kill_switch_latch(None, total_pnl=-1.0, threshold_usd=-2.0, pct=-0.02)
    # A DB without the latch table is also a clean no-op.
    bare = sqlite3.connect(":memory:")
    assert pm.is_kill_switch_latched_today(bare) is False
    pm.record_kill_switch_latch(
        bare, total_pnl=-1.0, threshold_usd=-2.0, pct=-0.02,
    )
    # And reading again stays False.
    assert pm.is_kill_switch_latched_today(bare) is False


# ---------------------------------------------------------------------------
# 1. Fail-closed on broker fetch error (orchestrator-level)
# ---------------------------------------------------------------------------
def test_orchestrator_fail_closed_on_broker_error(monkeypatch, tmp_path):
    """If Alpaca list_positions raises, the cycle must return
    NO_TRADE with status='BLOCKED' and reason='broker_unreachable'.
    Previous behavior (return []) would have let the cycle proceed
    to open new positions even with no visibility into existing ones.
    """
    # Force the position-management threshold loader to defaults (no
    # config to read in this minimal test).
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "mock")
    # Make sure no env-var overrides push the kill-switch before the
    # broker error short-circuits the cycle.
    monkeypatch.delenv("AIZEN_DAILY_LOSS_KILL_SWITCH_PCT", raising=False)
    monkeypatch.delenv("AIZEN_EARLY_WARNING_PCT", raising=False)

    # Build a DB with the full schema applied. The orchestrator's
    # journal upsert path needs the decision_journal table to exist.
    conn = _full_db(tmp_path)

    from src.agents.graph import Orchestrator

    # Build the orchestrator with a stubbed broker. We patch the
    # AlpacaTradingClient.list_positions to raise.
    with patch("src.agents.alpaca_trading.AlpacaTradingClient") as MockClient:
        instance = MagicMock()
        instance.list_positions.side_effect = RuntimeError("network unreachable")
        MockClient.return_value = instance

        orch = Orchestrator(conn=conn)
        state = orch.run_cycle()
        # Cycle returned BLOCKED, not PROCEED.
        assert state.final_action == "NO_TRADE"
        assert state.execution_result["status"] == "BLOCKED"
        assert state.execution_result["reason"] == "broker_unreachable"
        # The error is recorded in supplementary for the journal trace.
        assert "network unreachable" in state.execution_result["detail"]["error"]
        # A journal row was written.
        row = orch.journal.get(state.decision_id)
        assert row is not None
        assert row["final_action"] == "NO_TRADE"
    conn.close()


def test_orchestrator_short_circuits_on_latched_kill_switch(monkeypatch, tmp_path):
    """If a previous tick today already tripped the kill-switch, the
    next tick must short-circuit to NO_TRADE without even calling the
    broker. This is the operator-intent persistence the latch exists
    for.
    """
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "mock")
    monkeypatch.delenv("AIZEN_DAILY_LOSS_KILL_SWITCH_PCT", raising=False)
    conn = _full_db(tmp_path)
    # Latch today's date BEFORE the orchestrator runs.
    pm.record_kill_switch_latch(
        conn, total_pnl=-3000.0, threshold_usd=-2000.0, pct=-0.02,
    )

    from src.agents.graph import Orchestrator

    with patch("src.agents.alpaca_trading.AlpacaTradingClient") as MockClient:
        instance = MagicMock()
        # If list_positions is called, return an empty list (it must
        # NOT be called - the latch should short-circuit first).
        instance.list_positions.return_value = []
        MockClient.return_value = instance

        orch = Orchestrator(conn=conn)
        state = orch.run_cycle()
        assert state.final_action == "NO_TRADE"
        assert state.execution_result["status"] == "BLOCKED"
        assert state.execution_result["reason"] == "daily_loss_kill_switch_latched"
        # Broker was never queried (latch short-circuits first).
        instance.list_positions.assert_not_called()
    conn.close()


# ---------------------------------------------------------------------------
# 2026-09-03 threshold + profit-take additions
# ---------------------------------------------------------------------------
def test_early_warning_excludes_stop_loss_zone_with_param():
    """The -15% warning must exclude any position already past the
    -30% stop-loss. Test with an explicit stop_loss_pct=-0.30 to lock
    in the parameterised behavior."""
    pos = _make_pos("NVDA", 1, 200, 80)  # -60% past -30%
    out = pm.early_warning_positions([pos], pct=-0.15, stop_loss_pct=-0.30)
    assert out == []


def test_early_warning_default_stop_loss_30():
    """Default stop_loss_pct param is -0.30; a -20% position is in
    the warning zone, a -40% is past the stop-loss."""
    warning_pos = _make_pos("AAPL", 1, 100, 80)  # -20%
    stop_pos = _make_pos("NVDA", 1, 100, 50)  # -50%
    out = pm.early_warning_positions([warning_pos, stop_pos])
    assert len(out) == 1
    assert out[0]["symbol"] == "AAPL"


def test_profit_take_returns_positions_above_threshold():
    """A position at +60% gain is past the +50% profit-take; returned."""
    pos = _make_pos("AAPL", 1, 100, 160)  # +60%
    out = pm.profit_take_positions([pos], pct=0.50)
    assert len(out) == 1
    assert out[0]["symbol"] == "AAPL"
    assert out[0]["gain_pct"] == pytest.approx(0.60)


def test_profit_take_excludes_positions_below_threshold():
    """A position at +20% gain is below the +50% profit-take; not returned."""
    pos = _make_pos("AAPL", 1, 100, 120)  # +20%
    out = pm.profit_take_positions([pos], pct=0.50)
    assert out == []


def test_profit_take_excludes_losers():
    """A losing position is never profit-taken, even at -50%."""
    pos = _make_pos("NVDA", 1, 100, 50)  # -50%
    out = pm.profit_take_positions([pos], pct=0.50)
    assert out == []


def test_auto_close_profit_take_happy_path():
    """A +60% position is closed; status='closed', broker_order_id set."""
    pos = _make_pos("AAPL", 1, 100, 160)
    client = MagicMock()
    client.close_position.return_value = {"id": "order-999"}
    results = pm.auto_close_profit_take(client, [pos], pct=0.50)
    assert results[0]["status"] == "closed"
    assert results[0]["broker_order_id"] == "order-999"


def test_auto_close_profit_take_404_is_already_closed():
    """404 from the broker on a profit-take is 'already_closed', not error."""
    pos = _make_pos("AAPL", 1, 100, 160)
    client = MagicMock()
    client.close_position.side_effect = RuntimeError(
        "alpaca trading 404 on DELETE /v2/positions/AAPL: ..."
    )
    results = pm.auto_close_profit_take(client, [pos], pct=0.50)
    assert results[0]["status"] == "already_closed"


def test_load_thresholds_default_stop_loss_is_30():
    """Pin the new default stop_loss_pct at -0.30 (tightened from
    -0.50 on 2026-09-03). A regression here would mean the old -50%
    is back, which the COIN/META loss review explicitly rejected."""
    from src.agents import position_management as pm_mod
    # Pass no agents_cfg so the code's own default is the answer.
    thr = pm_mod._load_thresholds(None)
    assert thr["stop_loss_pct"] == pytest.approx(-0.30)


def test_load_thresholds_default_daily_cap_is_3pct():
    """Pin the new default daily_loss_kill_switch_pct at -0.03
    (loosened from -0.02 on 2026-09-03)."""
    from src.agents import position_management as pm_mod
    thr = pm_mod._load_thresholds(None)
    assert thr["daily_loss_kill_switch_pct"] == pytest.approx(-0.03)


def test_load_thresholds_default_profit_take_is_50pct():
    """Pin the new default profit_take_pct at +0.50."""
    from src.agents import position_management as pm_mod
    thr = pm_mod._load_thresholds(None)
    assert thr["profit_take_pct"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Regression: 2026-09-03 GH Actions log — cents-vs-dollars unit mismatch
# ---------------------------------------------------------------------------
def test_loss_pct_ignores_broker_unrealized_pl_unit_mismatch():
    """Pin the unit-consistent loss_pct math for option positions.

    The 2026-09-03 cron run logged
    ``stop-loss closed COIN260911P00175000 @ loss_pct=-5721.31%``
    when the actual position loss was -57% (mark=2.61, entry=6.10,
    qty=1 contract of an option — true per-share P/L -$3.49, true
    per-contract P/L -$349). The bug was that the old code computed
    ``loss_pct = broker_unrealized_pl / (entry * qty)`` and the
    broker's field is in total dollars (1 contract = 100 shares)
    while ``entry * qty`` is per-share, so the ratio was 100x too
    large.

    The fix: ``loss_pct = (mark - entry) / entry``. Both numerator
    and denominator are per-share, so the result is unit-consistent
    regardless of contract multiplier. The broker's ``unrealized_pl``
    is still preserved as the ``unrealized_pnl`` field for operator
    display.
    """
    pos = {
        "symbol": "COIN260911P00175000",  # OCC option
        "qty": 1,
        "avg_entry_price": 6.10,
        "current_price": 2.61,  # mid-move; true per-share loss = -3.49
        "unrealized_pl": -349.00,  # broker's field — total dollars, 100x off
    }
    out = pm.positions_to_close([pos], pct=-0.30)
    # The position is past -30% so it should be returned.
    assert len(out) == 1
    rec = out[0]
    # loss_pct must be the per-share ratio, ~ -57.2%, NOT -5721%.
    assert rec["loss_pct"] == pytest.approx((2.61 - 6.10) / 6.10)
    assert -0.60 < rec["loss_pct"] < -0.55, (
        f"loss_pct={rec['loss_pct']!r} is not in the expected -57% range; "
        "the broker's unrealized_pl unit mismatch is back."
    )
    # The dollar P/L field still shows the broker's actual value for
    # operator display (-$349 total, not -$3.49 per share).
    assert rec["unrealized_pnl"] == -349.00


def test_loss_pct_works_for_equities_with_consistent_broker_field():
    """Equities don't have the 100x contract multiplier, so the broker's
    ``unrealized_pl`` and the per-share entry*qty denominator are
    in the same units. The new math still gives the right answer.
    """
    pos = {
        "symbol": "AAPL",  # equity, not OCC
        "qty": 10,
        "avg_entry_price": 150.0,
        "current_price": 145.0,  # -$5/share, -$50 total
        "unrealized_pl": -50.0,  # broker's field matches the math
    }
    out = pm.positions_to_close([pos], pct=-0.02)
    assert len(out) == 1
    rec = out[0]
    assert rec["loss_pct"] == pytest.approx((145.0 - 150.0) / 150.0)
    assert rec["unrealized_pnl"] == -50.0


def test_loss_pct_works_without_broker_unrealized_pl_field():
    """When the broker doesn't return ``unrealized_pl``, the fallback
    uses the per-share recompute (×100 for options, ×1 for equities)."""
    pos = {
        "symbol": "SPY260304P00450000",  # OCC option
        "qty": 2,  # 2 contracts
        "avg_entry_price": 5.00,
        "current_price": 4.00,  # -$1/share, -$100/contract, -$200 total
        # No unrealized_pl / unrealized_pnl field.
    }
    out = pm.positions_to_close([pos], pct=-0.10)
    assert len(out) == 1
    rec = out[0]
    assert rec["loss_pct"] == pytest.approx(-0.20)
    # 2 contracts × 100 shares × -$1 = -$200 total.
    assert rec["unrealized_pnl"] == pytest.approx(-200.0)


def test_early_warning_loss_pct_uses_unit_consistent_math():
    """The early-warning path also fixed: an option with broker
    unrealized_pl that's 100x off should still produce a sane
    loss_pct, not a 100x-blown-up one."""
    pos = {
        "symbol": "META260909P00592500",  # OCC option
        "qty": 1,
        "avg_entry_price": 2.16,
        "current_price": 1.20,  # -$0.96/share, ~ -44% loss
        "unrealized_pl": -96.0,  # broker 100x off
    }
    out = pm.early_warning_positions([pos], pct=-0.30, stop_loss_pct=-0.50)
    assert len(out) == 1
    rec = out[0]
    # ~ -44% loss, in the early-warning zone (between -30% and -50%).
    assert rec["loss_pct"] == pytest.approx((1.20 - 2.16) / 2.16)
    assert -0.50 < rec["loss_pct"] < -0.40
    # The dollar P/L field preserves the broker's actual value.
    assert rec["unrealized_pnl"] == -96.0


def test_profit_take_loss_pct_uses_unit_consistent_math():
    """The profit-take path also fixed: an option with broker
    unrealized_pl that's 100x off should still produce a sane
    gain_pct, not a 100x-blown-up one."""
    pos = {
        "symbol": "AAPL260919C00200000",  # OCC option
        "qty": 1,
        "avg_entry_price": 2.00,
        "current_price": 3.50,  # +$1.50/share, +75% gain
        "unrealized_pl": 150.0,  # broker 100x off (but consistent sign)
    }
    out = pm.profit_take_positions([pos], pct=0.50)
    assert len(out) == 1
    rec = out[0]
    assert rec["gain_pct"] == pytest.approx((3.50 - 2.00) / 2.00)
    assert 0.70 < rec["gain_pct"] < 0.80
    assert rec["unrealized_pnl"] == 150.0


def test_get_blocked_symbols_uses_unit_consistent_loss_pct():
    """The open-loss-block path also fixed: a -$1 per-share loss on
    an option should still trip the -1% block, even when the broker's
    ``unrealized_pl`` is the total-dollar value (which is 100x larger)."""
    pos = {
        "symbol": "NVDA260919C00100000",  # OCC option
        "qty": 1,
        "avg_entry_price": 5.00,
        "current_price": 4.90,  # -2% per-share loss
        "unrealized_pl": -10.0,  # broker says -$10 total
    }
    blocked = pm.get_blocked_symbols(
        positions=[pos], conn=None,
        cooldown_seconds=0, open_loss_block_pct=-0.01,
    )
    # NVDA is the underlying of the OCC symbol and should be blocked
    # because the position is in -2% loss (past the -1% threshold).
    assert "NVDA" in blocked


def test_is_option_symbol_classification():
    """OCC option symbols are detected by length and embedded digits.
    Equities (1-5 uppercase letters) are NOT classified as options."""
    assert pm._is_option_symbol("AAPL") is False
    assert pm._is_option_symbol("SPY") is False
    assert pm._is_option_symbol("GOOGL") is False
    assert pm._is_option_symbol("COIN260911P00175000") is True
    assert pm._is_option_symbol("SPY260304C00450000") is True
    assert pm._is_option_symbol("AAPL260919C00200000") is True
    # Edge case: 6+ letter equity-like tickers (rare) still classified
    # as options, which is the safe default — the only consequence is
    # an extra *100 in the dollar-P/L fallback, which is a no-op when
    # the broker's field is present.
    assert pm._is_option_symbol("abcdef") is True
    # Empty / None.
    assert pm._is_option_symbol("") is False
    assert pm._is_option_symbol(None) is False


# ---------------------------------------------------------------------------
# MCP server wiring (execution node)
# ---------------------------------------------------------------------------
def test_execution_calls_mcp_get_positions(monkeypatch, tmp_path):
    """When the execution node is built with a SkillRegistry, it
    must call get_positions through the registry before submitting
    the order. The pre-flight call exercises the MCP wiring without
    changing the submit path."""
    from src.agents.mcp import AlpacaMCPServer
    from src.agents.mcp.skills import build_skill_registry
    from src.agents.nodes import execution as exec_node
    from src.agents.protocol import (
        DecisionState, OrderIntent, RiskDecision, RiskAction, Side,
        StrategyProposal, Leg, AgentObservation, MessageType,
    )
    from datetime import datetime, timezone
    import sqlite3
    from src.db import init_db

    # Stub AlpacaTradingClient inside execution so the test never
    # tries to reach the broker.
    monkeypatch.setenv("AIZEN_LLM_PROVIDER", "mock")
    monkeypatch.setenv("RUN_MODE", "paper")
    with patch("src.agents.nodes.execution.AlpacaTradingClient") as MockClient:
        instance = MagicMock()
        instance.submit_order.return_value = {
            "id": "broker-1", "status": "submitted",
            "symbol": "AAPL", "qty": 1, "side": "buy",
        }
        MockClient.return_value = instance

        server = AlpacaMCPServer(run_mode="paper")
        skills = build_skill_registry("execution", server)
        # Patch the MCP server's trading client too so get_positions
        # returns a stub list without hitting the broker.
        with patch.object(server, "_get_trading_client") as mock_get_tc:
            mock_get_tc.return_value.list_positions.return_value = [
                {"symbol": "AAPL", "qty": 1, "avg_entry_price": 150.0,
                 "current_price": 155.0},
            ]
            node = exec_node.build_node(
                llm=MagicMock(), config={"run_mode": "paper"},
                risk_limits=None, skills=skills,
            )
            intent = OrderIntent(
                strategy_id="test", underlying="AAPL",
                legs=[Leg(asset_class="equity", contract_symbol="AAPL",
                          side=Side.BUY, quantity=1)],
                quantity=1,
            )
            state = DecisionState(
                decision_id="dec-1",
                cycle_started_at=datetime.now(timezone.utc).isoformat(),
                order_intent=intent,
                risk_decision=RiskDecision(
                    decision=RiskAction.APPROVE, approved_quantity=1,
                    max_loss=100.0, reasons=[],
                ),
            )
            out = node(state)
            # MCP pre-flight was called, broker submit happened.
            assert out["mcp_preflight"]["called"] is True
            assert out["mcp_preflight"]["n_positions"] == 1
            instance.submit_order.assert_called_once()
