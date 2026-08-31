"""Tests for the equity-path backtest (parallel options+stocks).

Covers:
  * ``Labeler._equity_payoff`` math (long-only, no 100x multiplier).
  * ``BacktestCycleResult`` carries the new equity fields.
  * ``_aggregate`` computes ``hit_rate_equity_h4``, ``mean_equity_pnl_h4``,
    ``equity_coverage``, ``n_proceed_equity``.
  * ``_persist_result`` survives a pre-migration DB (no equity columns).
  * End-to-end: a runner replays an equity-leg order intent and writes
    the new columns.
"""
from __future__ import annotations

import os
import sys
from dataclasses import fields
from unittest.mock import MagicMock

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.backtest import (  # noqa: E402
    BacktestCycleResult,
    BacktestRunner,
    _aggregate,
    _persist_result,
)
from src.agents.backtest_labeler import Labeler  # noqa: E402
from src.agents.protocol import (  # noqa: E402
    DecisionState,
    Leg,
    MarketSnapshot,
    OptionType,
    OrderIntent,
    Side,
    StrategyProposal,
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


def _equity_state(action: str = "PROCEED") -> DecisionState:
    """DecisionState with an equity-leg OrderIntent (long NVDA, qty=10)."""
    leg = Leg(
        contract_symbol="NVDA",
        side=Side.BUY,
        quantity=10,
        asset_class="equity",
        limit_price=100.0,
    )
    state = DecisionState(
        market_snapshot=MarketSnapshot(
            timestamp="2026-08-25T13:30:00Z",
            underlyings=[
                UnderlyingScore(
                    symbol="NVDA",
                    timestamp="2026-08-25T13:30:00Z",
                    horizon_bars=4,
                )
            ],
        ),
        final_action=action,
    )
    state.selected_strategy = StrategyProposal(
        strategy_id="eq1", underlying="NVDA", legs=[leg],
        thesis="long-only equity", expected_return=0.02,
        probability_profit=0.6, confidence=0.7, max_loss=200.0,
        expiry="2026-08-25",
    )
    state.order_intent = OrderIntent(
        strategy_id="eq1", underlying="NVDA", legs=[leg], quantity=10,
    )
    return state


def _seed_underlying(conn, *, symbol: str, ts: str, close: float) -> None:
    conn.execute(
        "INSERT INTO underlying_bars (symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (symbol, ts, close, close, close, close, 1000),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------
def test_backtest_cycle_result_has_equity_fields():
    names = {f.name for f in fields(BacktestCycleResult)}
    for required in (
        "equity_payoff_h1", "equity_payoff_h4",
        "equity_hit_h1", "equity_hit_h4",
        "coverage_equity_h1", "coverage_equity_h4",
    ):
        assert required in names, f"BacktestCycleResult missing {required}"


# ---------------------------------------------------------------------------
# Labeler: equity payoff math
# ---------------------------------------------------------------------------
def test_labeler_equity_payoff_long_profit(conn):
    # Buy 10 shares of NVDA at 100; sell at 101 four hours later. PnL = +10.
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T13:30:00Z", close=100.0)
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T14:00:00Z", close=100.5)
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T17:30:00Z", close=101.0)

    labeler = Labeler(conn)
    state = _equity_state()
    labels = labeler.label(state, "2026-08-25T13:30:00Z")

    # 1h horizon = 4 bars = 60 min, target = 14:30Z; 4h = 17:30Z.
    assert labels.equity_payoff_h1 is not None
    assert labels.equity_payoff_h4 is not None
    # 4h payoff: (101 - 100) * 10 * 1 = +10.0
    assert labels.equity_payoff_h4 == pytest.approx(10.0, abs=0.01)
    # Coverage: we have bars at both horizons
    assert labels.coverage_equity_h1 is True
    assert labels.coverage_equity_h4 is True


def test_labeler_equity_payoff_long_loss(conn):
    # Buy 10 at 100; sell at 99 four hours later. PnL = -10.
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T13:30:00Z", close=100.0)
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T17:30:00Z", close=99.0)

    labeler = Labeler(conn)
    labels = labeler.label(_equity_state(), "2026-08-25T13:30:00Z")
    assert labels.equity_payoff_h4 == pytest.approx(-10.0, abs=0.01)
    # The labeler itself returns the float; the row dict turns it into 0/1
    # via _hit. We just check coverage is on.
    assert labels.coverage_equity_h4 is True


def test_labeler_equity_payoff_no_exit_bar_returns_none(conn):
    # Entry at T, but no bar at T+1h or T+4h.
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T13:30:00Z", close=100.0)

    labeler = Labeler(conn)
    labels = labeler.label(_equity_state(), "2026-08-25T13:30:00Z")
    # No exit bar in either horizon → None, no coverage.
    assert labels.equity_payoff_h1 is None
    assert labels.equity_payoff_h4 is None
    assert labels.coverage_equity_h1 is False
    assert labels.coverage_equity_h4 is False


def test_labeler_skips_equity_for_option_only_intent(conn):
    # Order intent has no equity leg → equity fields are untouched.
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T13:30:00Z", close=100.0)
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T17:30:00Z", close=101.0)

    # Build an option-only state (the existing _state pattern in the test suite).
    leg = Leg(
        contract_symbol="NVDA260101C00100000", side=Side.BUY, quantity=1,
        option_type=OptionType.CALL, strike=100.0, expiry="2026-09-01",
    )
    state = DecisionState(
        market_snapshot=MarketSnapshot(
            timestamp="2026-08-25T13:30:00Z",
            underlyings=[UnderlyingScore(
                symbol="NVDA", timestamp="2026-08-25T13:30:00Z",
                horizon_bars=4)],
        ),
        final_action="PROCEED",
    )
    state.order_intent = OrderIntent(
        strategy_id="s1", underlying="NVDA", legs=[leg], quantity=1,
    )

    labeler = Labeler(conn)
    labels = labeler.label(state, "2026-08-25T13:30:00Z")
    assert labels.equity_payoff_h1 is None
    assert labels.equity_payoff_h4 is None
    assert labels.coverage_equity_h1 is False
    assert labels.coverage_equity_h4 is False


# ---------------------------------------------------------------------------
# Aggregate: equity metrics
# ---------------------------------------------------------------------------
def test_aggregate_equity_metrics():
    cycles = [
        # Equity win (long NVDA +$5)
        BacktestCycleResult(
            cycle_as_of="t1", decision_id="d1", final_action="PROCEED",
            predicted_underlying="NVDA", predicted_strategy_id="eq1",
            predicted_legs_json=None,
            forward_return_h1=0.01, forward_return_h4=0.02, option_payoff=None,
            target_class="1", hit_h4=1, hit_h1=1,
            coverage_h1=1, coverage_h4=1, coverage_payoff=0,
            equity_payoff_h1=2.0, equity_payoff_h4=5.0,
            equity_hit_h4=1, equity_hit_h1=1,
            coverage_equity_h1=1, coverage_equity_h4=1,
            run_at="t", model_version="v", feature_flag_state="news-on",
        ),
        # Equity loss (long NVDA -$3)
        BacktestCycleResult(
            cycle_as_of="t2", decision_id="d2", final_action="PROCEED",
            predicted_underlying="NVDA", predicted_strategy_id="eq1",
            predicted_legs_json=None,
            forward_return_h1=-0.01, forward_return_h4=-0.02, option_payoff=None,
            target_class="-1", hit_h4=0, hit_h1=0,
            coverage_h1=1, coverage_h4=1, coverage_payoff=0,
            equity_payoff_h1=-1.0, equity_payoff_h4=-3.0,
            equity_hit_h4=0, equity_hit_h1=0,
            coverage_equity_h1=1, coverage_equity_h4=1,
            run_at="t", model_version="v", feature_flag_state="news-on",
        ),
        # NO_TRADE: no equity, contributes nothing
        BacktestCycleResult(
            cycle_as_of="t3", decision_id="d3", final_action="NO_TRADE",
            predicted_underlying=None, predicted_strategy_id=None,
            predicted_legs_json=None,
            forward_return_h1=0.005, forward_return_h4=0.01, option_payoff=None,
            target_class="1", hit_h4=None, hit_h1=None,
            coverage_h1=1, coverage_h4=1, coverage_payoff=0,
            run_at="t", model_version="v", feature_flag_state="news-off",
        ),
    ]
    agg = _aggregate(cycles)
    assert agg["n_cycles"] == 3
    # 2 PROCEED with equity coverage
    assert agg["n_proceed_equity"] == 2
    # 1 hit out of 2
    assert agg["hit_rate_equity_h4"] == 0.5
    # mean equity pnl: (5 + -3) / 2 = 1.0
    assert agg["mean_equity_pnl_h4"] == 1.0
    # equity coverage: 2/3
    assert agg["equity_coverage"] == pytest.approx(2 / 3)


def test_aggregate_equity_empty_returns_none():
    cycles = [
        BacktestCycleResult(
            cycle_as_of="t1", decision_id="d1", final_action="NO_TRADE",
            predicted_underlying=None, predicted_strategy_id=None,
            predicted_legs_json=None,
            forward_return_h1=None, forward_return_h4=None, option_payoff=None,
            target_class=None, hit_h4=None, hit_h1=None,
            coverage_h1=0, coverage_h4=0, coverage_payoff=0,
            run_at="t", model_version=None, feature_flag_state="news-off",
        ),
    ]
    agg = _aggregate(cycles)
    assert agg["n_proceed_equity"] == 0
    assert agg["hit_rate_equity_h4"] is None
    assert agg["mean_equity_pnl_h4"] is None
    assert agg["equity_coverage"] == 0.0  # 0/1 = 0.0


# ---------------------------------------------------------------------------
# Persistence: handles pre-migration DB (no equity columns)
# ---------------------------------------------------------------------------
def test_persist_result_handles_legacy_db(tmp_path, caplog):
    """If backtest_cycles is missing the equity columns, the row still gets
    written (legacy shape) instead of crashing the runner."""
    import sqlite3
    legacy_path = tmp_path / "legacy.db"
    lc = sqlite3.connect(str(legacy_path))
    lc.execute(
        "CREATE TABLE backtest_cycles ("
        " decision_id TEXT PRIMARY KEY,"
        " cycle_as_of TEXT NOT NULL,"
        " final_action TEXT NOT NULL,"
        " predicted_underlying TEXT,"
        " predicted_strategy_id TEXT,"
        " predicted_legs_json TEXT,"
        " forward_return_h1 REAL,"
        " forward_return_h4 REAL,"
        " target_class TEXT,"
        " option_payoff REAL,"
        " hit_h4 INTEGER,"
        " hit_h1 INTEGER,"
        " coverage_h1 INTEGER NOT NULL DEFAULT 0,"
        " coverage_h4 INTEGER NOT NULL DEFAULT 0,"
        " coverage_payoff INTEGER NOT NULL DEFAULT 0,"
        " run_at TEXT NOT NULL,"
        " model_version TEXT,"
        " feature_flag_state TEXT NOT NULL,"
        " notes TEXT"
        ")"
    )
    lc.commit()
    lc.row_factory = sqlite3.Row

    result = BacktestCycleResult(
        cycle_as_of="2026-08-25T13:30:00Z", decision_id="d1",
        final_action="PROCEED", predicted_underlying="NVDA",
        predicted_strategy_id="eq1", predicted_legs_json=None,
        forward_return_h1=0.01, forward_return_h4=0.02,
        option_payoff=None, target_class="1", hit_h4=1, hit_h1=1,
        coverage_h1=1, coverage_h4=1, coverage_payoff=0,
        equity_payoff_h1=1.0, equity_payoff_h4=5.0,
        equity_hit_h4=1, equity_hit_h1=1,
        coverage_equity_h1=1, coverage_equity_h4=1,
        run_at="t", model_version="v", feature_flag_state="news-on",
    )
    _persist_result(lc, result)
    row = lc.execute(
        "SELECT * FROM backtest_cycles WHERE decision_id = ?", ("d1",)
    ).fetchone()
    assert row is not None
    assert row["final_action"] == "PROCEED"
    # Equity columns are absent in the legacy schema → not present in row.
    assert "equity_payoff_h1" not in row.keys()
    lc.close()


# ---------------------------------------------------------------------------
# End-to-end: runner with an equity-leg order intent
# ---------------------------------------------------------------------------
def test_runner_writes_equity_fields_for_equity_intent(conn, monkeypatch):
    """A 1-cycle backtest with an equity-leg order intent writes the new
    equity_payoff_h4 and coverage_equity_h4 columns."""
    from src.agents import backtest as bt_mod
    # Seed an underlying bar at the cycle time and 4h later.
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T13:30:00Z", close=100.0)
    _seed_underlying(conn, symbol="NVDA", ts="2026-08-25T17:30:00Z", close=102.0)

    fake_orch = MagicMock()
    fake_orch.run_cycle.return_value = _equity_state("PROCEED")
    monkeypatch.setattr(bt_mod, "Orchestrator", MagicMock(return_value=fake_orch))

    db_file = conn.execute("PRAGMA database_list").fetchone()["file"]
    runner = BacktestRunner(db_path=db_file)
    report = runner.run(["2026-08-25T13:30:00Z"])

    assert len(report.cycles) == 1
    cycle = report.cycles[0]
    # Equity leg, +2$ * 10 shares = +20$ at 4h
    assert cycle.equity_payoff_h4 == pytest.approx(20.0, abs=0.01)
    assert cycle.coverage_equity_h4 == 1
    assert cycle.equity_hit_h4 == 1  # positive payoff → hit

    # The aggregate metrics include the new equity fields.
    agg = report.aggregate
    assert agg["n_proceed_equity"] == 1
    assert agg["hit_rate_equity_h4"] == 1.0
    assert agg["mean_equity_pnl_h4"] == pytest.approx(20.0, abs=0.01)
    assert agg["equity_coverage"] == 1.0
