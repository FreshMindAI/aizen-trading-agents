"""Tests for position_management pre-flight (a)+(b)+(c).

Three independent pre-flight checks, each covered by a focused test:

  (a) Daily loss kill-switch (test_daily_loss_kill_switch_breached_*
      and test_daily_loss_kill_switch_safe_*): today's realized +
      open-position unrealized P/L is compared to capital_usd *
      daily_loss_kill_switch_pct. If the total is below the cap, the
      function reports breached=True.

  (b) Per-position stop-loss (test_positions_to_close_*): positions
      whose loss_pct is worse than the threshold are returned; the
      rest are skipped. ``auto_close_stop_loss`` is also covered but
      with a fake client so the test does not call the broker.

  (c) Per-symbol cooldown (test_get_blocked_symbols_*): symbols with
      an open position in loss, OR a recent journal row with
      realized_pnl<0, are returned. The journal row's age is bounded
      by cooldown_seconds. Underlyings are derived from OCC symbols.

The orchestrator-level wiring (state.supplementary, supervisor
filter, kill-switch short-circuit) is covered by a separate
integration test using the public ``Orchestrator.run_cycle`` API
with a stub broker.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents import position_management as pm
from src.agents.protocol import PortfolioPosition
from src.db import connect, init_db


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


# ---------------------------------------------------------------------------
# (a) Daily loss kill-switch
# ---------------------------------------------------------------------------
def test_daily_loss_kill_switch_breached_when_total_below_cap():
    res = pm.check_daily_loss_kill_switch(
        capital_usd=100_000.0,
        positions=[_make_pos("AAPL", 10, 200, 195),  # -50
                   _make_pos("MSFT", 5, 400, 380)],  # -100
        realized_pnl_today=-2_000.0,  # -2150 total vs -2000 cap -> breached
        pct=-0.02,
    )
    assert res.breached is True
    assert res.total_pnl == pytest.approx(-2_150.0)
    assert res.threshold_usd == pytest.approx(-2_000.0)


def test_daily_loss_kill_switch_safe_when_total_above_cap():
    res = pm.check_daily_loss_kill_switch(
        capital_usd=100_000.0,
        positions=[_make_pos("AAPL", 10, 200, 198)],  # -20
        realized_pnl_today=-500.0,  # -520 total, above -2000 cap
        pct=-0.02,
    )
    assert res.breached is False
    assert res.total_pnl == pytest.approx(-520.0)


def test_daily_loss_kill_switch_breach_uses_underlying_value_not_pct():
    # Even with zero realized P/L, a single big unrealized loss breaches
    # the cap. Locks in the "open losses count" semantics.
    res = pm.check_daily_loss_kill_switch(
        capital_usd=50_000.0,
        positions=[_make_pos("NVDA", 100, 500, 470)],  # -3000
        realized_pnl_today=0.0,
        pct=-0.02,
    )
    assert res.breached is True
    assert res.threshold_usd == pytest.approx(-1_000.0)


def test_daily_loss_kill_switch_handles_empty_positions():
    res = pm.check_daily_loss_kill_switch(
        capital_usd=100_000.0,
        positions=[],
        realized_pnl_today=0.0,
        pct=-0.02,
    )
    assert res.breached is False
    assert res.total_pnl == 0.0


# ---------------------------------------------------------------------------
# (b) Per-position stop-loss
# ---------------------------------------------------------------------------
def test_positions_to_close_returns_only_losers_past_threshold():
    positions = [
        _make_pos("AAPL", 10, 200, 195),  # -2.5%
        _make_pos("MSFT", 5, 400, 100),   # -75%  -> should close
        _make_pos("TSLA", 3, 300, 290),   # -3.3%
        _make_pos("AMD", 1, 100, 30),     # -70%  -> should close
    ]
    to_close = pm.positions_to_close(positions, pct=-0.50)
    syms = [p["symbol"] for p in to_close]
    assert syms == ["MSFT", "AMD"]
    for p in to_close:
        assert p["loss_pct"] <= -0.50
        assert p["quantity"] >= 1


def test_positions_to_close_includes_equity_winners_in_profit():
    to_close = pm.positions_to_close(
        [_make_pos("SPY", 5, 500, 520)],  # +4% — winner
        pct=-0.50,
    )
    assert to_close == []


def test_positions_to_close_handles_zero_or_missing_qty():
    positions = [
        {"symbol": "BAD", "qty": 0, "avg_entry_price": 100, "current_price": 50},
        {"symbol": "EMPTY", "avg_entry_price": 100},  # missing qty
    ]
    assert pm.positions_to_close(positions, pct=-0.10) == []


def test_auto_close_stop_loss_uses_fake_client():
    class _FakeClient:
        def __init__(self):
            self.closed: list[str] = []
        def close_position(self, sym: str):
            self.closed.append(sym)
            return {"id": f"order_{sym}"}
    fake = _FakeClient()
    positions = [
        _make_pos("AAPL", 10, 200, 50),  # -75%  -> close
        _make_pos("MSFT", 5, 400, 100),  # -75%  -> close
        _make_pos("TSLA", 3, 300, 290),  # -3.3% -> skip
    ]
    out = pm.auto_close_stop_loss(fake, positions, pct=-0.50)
    assert sorted(c["symbol"] for c in out) == ["AAPL", "MSFT"]
    assert all(c["status"] == "closed" for c in out)
    assert sorted(fake.closed) == ["AAPL", "MSFT"]


def test_auto_close_stop_loss_records_broker_error():
    class _BoomClient:
        def close_position(self, sym: str):
            raise RuntimeError("broker timeout")
    out = pm.auto_close_stop_loss(
        _BoomClient(),
        [_make_pos("AAPL", 10, 200, 50)],
        pct=-0.50,
    )
    assert len(out) == 1
    assert out[0]["status"] == "error"
    assert "broker timeout" in out[0]["error"]


# ---------------------------------------------------------------------------
# (c) Per-symbol cooldown
# ---------------------------------------------------------------------------
def test_underlying_of_position_handles_equity_and_option_symbols():
    assert pm.underlying_of_position({"symbol": "AAPL"}) == "AAPL"
    assert pm.underlying_of_position({"symbol": "AAPL260919C00200000"}) == "AAPL"
    assert pm.underlying_of_position({"symbol": "SPY260919P00400000"}) == "SPY"
    assert pm.underlying_of_position({"symbol": ""}) is None


def test_get_blocked_symbols_open_losers_only():
    positions = [
        _make_pos("AAPL", 10, 200, 195),   # -2.5%  -> not blocked (above -5%)
        _make_pos("MSFT", 5, 400, 100),    # -75%   -> blocked
        _make_pos("TSLA", 3, 300, 290),    # -3.3%  -> not blocked
    ]
    blocked = pm.get_blocked_symbols(
        positions=positions,
        conn=None,
        cooldown_seconds=0,  # disable journal lookup
    )
    assert blocked == {"MSFT"}


def test_get_blocked_symbols_winners_not_blocked():
    positions = [_make_pos("NVDA", 5, 500, 510)]  # +2%
    blocked = pm.get_blocked_symbols(
        positions=positions, conn=None, cooldown_seconds=0,
    )
    assert blocked == set()


def test_get_blocked_symbols_journal_recent_loss_blocks_underlying(tmp_path):
    db = tmp_path / "j.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE decision_journal ("
        "decision_id TEXT PRIMARY KEY, completed_at TEXT, "
        "underlying_focus TEXT, realized_pnl REAL)"
    )
    now = datetime.now(timezone.utc)
    # Recent loss on AAPL — should be blocked
    recent = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Old loss on AMD — outside 1h cooldown, should NOT be blocked
    old = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.executemany(
        "INSERT INTO decision_journal VALUES (?, ?, ?, ?)",
        [
            ("a", recent, "AAPL", -50.0),
            ("b", old, "AMD", -200.0),
            ("c", recent, "TSLA", 100.0),  # profit, not blocked
        ],
    )
    conn.commit()
    blocked = pm.get_blocked_symbols(
        positions=[],
        conn=conn,
        cooldown_seconds=3600,
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    assert "AAPL" in blocked
    assert "AMD" not in blocked
    assert "TSLA" not in blocked


def test_get_blocked_symbols_combines_open_loss_and_recent_loss(tmp_path):
    db = tmp_path / "j.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE decision_journal ("
        "decision_id TEXT PRIMARY KEY, completed_at TEXT, "
        "underlying_focus TEXT, realized_pnl REAL)"
    )
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO decision_journal VALUES (?, ?, ?, ?)",
        ("a", recent, "NVDA", -100.0),
    )
    conn.commit()
    positions = [
        _make_pos("AAPL", 10, 200, 50),    # -75% open loss
        _make_pos("NVDA", 5, 500, 510),    # +2% (not in open loss)
    ]
    blocked = pm.get_blocked_symbols(
        positions=positions,
        conn=conn,
        cooldown_seconds=3600,
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    # AAPL: open loss. NVDA: recent journal loss.
    assert blocked == {"AAPL", "NVDA"}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def test_load_thresholds_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("AIZEN_DAILY_LOSS_KILL_SWITCH_PCT", "-0.05")
    monkeypatch.setenv("AIZEN_STOP_LOSS_PCT", "-0.30")
    monkeypatch.setenv("AIZEN_COOLDOWN_SECONDS", "7200")
    t = pm._load_thresholds({})  # empty config -> all from env
    assert t["daily_loss_kill_switch_pct"] == pytest.approx(-0.05)
    assert t["stop_loss_pct"] == pytest.approx(-0.30)
    assert t["cooldown_seconds"] == pytest.approx(7200.0)


def test_load_thresholds_falls_back_to_yaml_when_no_env(monkeypatch):
    monkeypatch.delenv("AIZEN_DAILY_LOSS_KILL_SWITCH_PCT", raising=False)
    monkeypatch.delenv("AIZEN_STOP_LOSS_PCT", raising=False)
    t = pm._load_thresholds({
        "position_management": {
            "daily_loss_kill_switch_pct": -0.10,
            "stop_loss_pct": -0.25,
        }
    })
    assert t["daily_loss_kill_switch_pct"] == pytest.approx(-0.10)
    assert t["stop_loss_pct"] == pytest.approx(-0.25)
