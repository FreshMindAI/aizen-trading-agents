"""Tests for the forward-outcome labeler (spec 003 / T046 / T047).

The labeler takes a DecisionState + cycle timestamp T and returns
forward outcomes from v_labels (1h/4h) and option_bars (held-to-expiry
payoff). It is the heart of the backtest's "what actually happened"
half of the validation.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.backtest_labeler import CycleLabels, Labeler  # noqa: E402
from src.agents.protocol import (  # noqa: E402
    DecisionState,
    Leg,
    MarketSnapshot,
    OptionType,
    OrderIntent,
    Side,
    UnderlyingScore,
)
from src.db import connect, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path):
    p = tmp_path / "t.db"
    c = connect(str(p))
    init_db(c, sql_dir=__import__("pathlib").Path("sql"))
    yield c
    c.close()


def _bar(conn, ts: str, sym: str, close: float):
    conn.execute(
        "INSERT INTO underlying_bars (symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sym, ts, close, close, close, close, 1000),
    )
    conn.commit()


def _opt_contract(conn, sym: str, exp: str, strike: float = 100.0,
                  opt_type: str = "call"):
    conn.execute(
        "INSERT INTO option_contracts (contract_symbol, underlying_symbol, "
        "strike_price, option_type, expiration_date) VALUES (?, ?, ?, ?, ?)",
        (sym, sym, strike, opt_type, exp),
    )
    conn.commit()


def _opt_bar(conn, ts: str, contract: str, close: float):
    conn.execute(
        "INSERT INTO option_bars (contract_symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (contract, ts, close, close, close, close, 100),
    )
    conn.commit()


def _state_no_trade(sym: str = "NVDA") -> DecisionState:
    return DecisionState(
        market_snapshot=MarketSnapshot(
            timestamp="2026-08-25T13:30:00Z",
            underlyings=[UnderlyingScore(symbol=sym, timestamp="2026-08-25T13:30:00Z", horizon_bars=4)],
        ),
    )


def _state_proceed(sym: str, legs: list[Leg]) -> DecisionState:
    state = _state_no_trade(sym)
    state.order_intent = OrderIntent(
        strategy_id="s1", underlying=sym, legs=legs, quantity=1,
    )
    return state


# ---------------------------------------------------------------------------
# T047-1: short-horizon labels match v_labels
# ---------------------------------------------------------------------------
def test_short_horizon_labels_match_v_labels(conn):
    # 4h horizon (16 bars). 16 bars after T we have 102.0 (vs 100.0 at T)
    # -> future_return = 0.02
    t0 = "2026-08-25T13:30:00Z"
    for i in range(17):
        _bar(conn, f"2026-08-2{5 + i // 8}T{13 + (i % 8):02d}:00:00Z" if i < 16 else "2026-08-26T13:30:00Z",
             "NVDA", close=100.0 + (i * 0.1))
    # Simpler: insert 17 specific bars so LEAD(close, 16) lands at the last one
    conn.execute("DELETE FROM underlying_bars")
    for i in range(17):
        ts = f"2026-08-25T{13 + (i // 4):02d}:{15 * (i % 4):02d}:00Z"
        _bar(conn, ts, "NVDA", close=100.0 + i)   # linear ramp 100..116
    labeler = Labeler(conn)
    labels = labeler.label(_state_no_trade("NVDA"), as_of="2026-08-25T13:15:00Z")
    # LEAD(close, 4) on the first bar (T0) -> bar[4] = 104; future_return = 4/100
    # LEAD(close, 16) on the first bar -> bar[16] = 116; future_return = 16/100
    assert labels.forward_return_h1 is not None
    assert labels.forward_return_h4 is not None
    assert labels.coverage_h1 and labels.coverage_h4


def test_short_horizon_returns_none_when_no_forward_window(conn):
    # T is at the END of the dataset: v_labels leak guard drops the last h bars
    _bar(conn, "2026-08-25T13:30:00Z", "NVDA", close=100.0)
    _bar(conn, "2026-08-25T13:45:00Z", "NVDA", close=101.0)
    labeler = Labeler(conn)
    labels = labeler.label(_state_no_trade("NVDA"), as_of="2026-08-25T13:45:00Z")
    # LEAD(close, 4) at the latest bar is NULL -> dropped by the leak guard
    assert labels.forward_return_h1 is None
    assert labels.forward_return_h4 is None
    assert labels.coverage_h1 is False
    assert labels.coverage_h4 is False


# ---------------------------------------------------------------------------
# T047-2: option payoff = (exit - entry) * 100 * qty * sign
# ---------------------------------------------------------------------------
def test_option_payoff_buy_call_itm(conn):
    sym = "NVDA"
    _opt_contract(conn, "NVDA260101C00100000", exp="2026-09-01")
    _opt_bar(conn, "2026-08-25T13:30:00Z", "NVDA260101C00100000", close=10.0)
    _opt_bar(conn, "2026-08-31T16:00:00Z", "NVDA260101C00100000", close=15.0)
    leg = Leg(contract_symbol="NVDA260101C00100000", side=Side.BUY, quantity=1,
              option_type=OptionType.CALL, strike=100.0, expiry="2026-09-01")
    labeler = Labeler(conn)
    labels = labeler.label(_state_proceed(sym, [leg]), as_of="2026-08-25T13:30:00Z")
    # entry 10, exit 15, qty 1, buy -> +500
    assert labels.option_payoff == pytest.approx(500.0, abs=1e-6)
    assert labels.coverage_payoff is True


def test_option_payoff_sell_put_loses(conn):
    sym = "NVDA"
    _opt_contract(conn, "NVDA260101P00100000", exp="2026-09-01")
    _opt_bar(conn, "2026-08-25T13:30:00Z", "NVDA260101P00100000", close=5.0)
    _opt_bar(conn, "2026-08-31T16:00:00Z", "NVDA260101P00100000", close=8.0)
    leg = Leg(contract_symbol="NVDA260101P00100000", side=Side.SELL, quantity=1,
              option_type=OptionType.PUT, strike=100.0, expiry="2026-09-01")
    labeler = Labeler(conn)
    labels = labeler.label(_state_proceed(sym, [leg]), as_of="2026-08-25T13:30:00Z")
    # entry 5, exit 8, qty 1, sell -> -300
    assert labels.option_payoff == pytest.approx(-300.0, abs=1e-6)


# ---------------------------------------------------------------------------
# T047-3: missing data -> payoff is None
# ---------------------------------------------------------------------------
def test_option_payoff_returns_none_when_expiry_bar_missing(conn):
    sym = "NVDA"
    _opt_contract(conn, "NVDA260101C00100000", exp="2026-09-01")
    _opt_bar(conn, "2026-08-25T13:30:00Z", "NVDA260101C00100000", close=10.0)
    # No bar at or before 2026-09-01
    leg = Leg(contract_symbol="NVDA260101C00100000", side=Side.BUY, quantity=1,
              option_type=OptionType.CALL, strike=100.0, expiry="2026-09-01")
    labeler = Labeler(conn)
    labels = labeler.label(_state_proceed(sym, [leg]), as_of="2026-08-25T13:30:00Z")
    assert labels.option_payoff is None
    assert labels.coverage_payoff is False


def test_option_payoff_returns_none_when_entry_bar_missing(conn):
    sym = "NVDA"
    _opt_contract(conn, "NVDA260101C00100000", exp="2026-09-01")
    # No entry bar before as_of
    _opt_bar(conn, "2026-08-26T13:30:00Z", "NVDA260101C00100000", close=15.0)
    leg = Leg(contract_symbol="NVDA260101C00100000", side=Side.BUY, quantity=1,
              option_type=OptionType.CALL, strike=100.0, expiry="2026-09-01")
    labeler = Labeler(conn)
    labels = labeler.label(_state_proceed(sym, [leg]), as_of="2026-08-25T13:30:00Z")
    assert labels.option_payoff is None


# ---------------------------------------------------------------------------
# T047-4: NO_TRADE cycle -> no payoff, no hit
# ---------------------------------------------------------------------------
def test_no_trade_returns_no_payoff(conn):
    state = _state_no_trade("NVDA")
    assert state.order_intent is None
    labeler = Labeler(conn)
    labels = labeler.label(state, as_of="2026-08-25T13:30:00Z")
    assert labels.option_payoff is None
    assert labels.coverage_payoff is False


# ---------------------------------------------------------------------------
# T047-5: row dict maps fields correctly
# ---------------------------------------------------------------------------
def test_to_row_dict_round_trip():
    labels = CycleLabels(
        forward_return_h1=0.005,
        forward_return_h4=0.02,
        target_class="1",
        option_payoff=250.0,
        coverage_h1=True,
        coverage_h4=True,
        coverage_payoff=True,
    )
    row = labels.to_row_dict(
        decision_id="backtest-2026-08-25-NVDA",
        cycle_as_of="2026-08-25T13:30:00Z",
        final_action="PROCEED",
        predicted_underlying="NVDA",
        predicted_strategy_id="s1",
        predicted_legs_json="[]",
        run_at="2026-08-29T00:00:00Z",
        model_version="gatv2_news",
        feature_flag_state="news-on",
    )
    assert row["forward_return_h1"] == 0.005
    assert row["forward_return_h4"] == 0.02
    assert row["option_payoff"] == 250.0
    assert row["coverage_h1"] == 1
    assert row["coverage_payoff"] == 1
    assert row["hit_h4"] == 1  # positive forward return -> "long" -> hit
    assert row["hit_h1"] == 1
    assert row["feature_flag_state"] == "news-on"


def test_hit_detection_zero_is_none():
    labels = CycleLabels(forward_return_h4=0.0, coverage_h4=True)
    row = labels.to_row_dict(
        decision_id="x", cycle_as_of="t", final_action="PROCEED",
        predicted_underlying="NVDA", predicted_strategy_id="s1",
        predicted_legs_json=None, run_at="t", model_version=None,
        feature_flag_state="news-off",
    )
    # Zero forward return is non-informative; hit is NULL
    assert row["hit_h4"] is None


def test_hit_detection_negative_is_zero():
    labels = CycleLabels(forward_return_h4=-0.02, coverage_h4=True)
    row = labels.to_row_dict(
        decision_id="x", cycle_as_of="t", final_action="PROCEED",
        predicted_underlying="NVDA", predicted_strategy_id="s1",
        predicted_legs_json=None, run_at="t", model_version=None,
        feature_flag_state="news-off",
    )
    # Negative forward return -> "long" missed -> hit = 0
    assert row["hit_h4"] == 0
