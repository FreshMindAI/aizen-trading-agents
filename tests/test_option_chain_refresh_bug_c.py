"""Regression tests for the Bug C fix: ``refresh_option_chains``
picks BOTH calls and puts (the previous version filtered on
``option_type = 'call'`` only, leaving the put side of the chain
without live bars).

Without this fix, a long-put decision would land on a contract
whose ``option_bars`` table was last updated on the day the call
side was downloaded, and the option_h4 regressor would score it
on stale data - producing an artificially low expected_return and
making the supervisor consistently pick calls over puts.

The fix: ``_pick_atm_contracts`` now includes
``option_type IN ('call', 'put')`` AND sorts by ATM distance
first so the ``LIMIT max_contracts`` budget balances calls and
puts at the nearest strikes, instead of all-calls on the earliest
expiry.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)


@pytest.fixture
def option_chain_db(tmp_path):
    """A sqlite db with option_contracts populated for one symbol
    across multiple strikes, expiries, and option types."""
    import sqlite3
    db = tmp_path / "trading.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE option_contracts (
            contract_symbol   TEXT PRIMARY KEY,
            underlying_symbol TEXT NOT NULL,
            expiration_date   TEXT NOT NULL,
            strike_price      REAL NOT NULL,
            option_type       TEXT NOT NULL,
            tradable          INTEGER NOT NULL DEFAULT 1,
            status            TEXT NOT NULL DEFAULT 'active'
        );
        CREATE TABLE underlying_bars (
            symbol    TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            close     REAL NOT NULL
        );
    """)
    # Spot = $100 for the synthetic symbol
    conn.execute(
        "INSERT INTO underlying_bars (symbol, timestamp, close) VALUES (?, ?, ?)",
        ("FOO", "2026-09-01T19:00:00Z", 100.0),
    )
    # 3 expiries x 3 strikes (90/100/110) x 2 types (call/put) = 18 contracts
    import datetime
    base = datetime.date(2026, 9, 4)   # 3 DTE
    for dte_off, days in [(0, 3), (1, 5), (2, 10)]:
        exp = base + datetime.timedelta(days=days)
        exp_str = exp.strftime("%Y-%m-%d")
        for strike in (90.0, 100.0, 110.0):
            for otype in ("call", "put"):
                sym = (
                    f"FOO{exp.strftime('%y%m%d')}"
                    f"{'C' if otype == 'call' else 'P'}"
                    f"{int(strike * 1000):08d}"
                )
                conn.execute(
                    "INSERT INTO option_contracts "
                    "(contract_symbol, underlying_symbol, expiration_date, "
                    " strike_price, option_type, tradable, status) "
                    "VALUES (?, ?, ?, ?, ?, 1, 'active')",
                    (sym, "FOO", exp_str, strike, otype),
                )
    conn.commit()
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_pick_atm_includes_both_calls_and_puts(option_chain_db):
    """Before the fix this returned 6 calls (alphabetical) and
    0 puts. After the fix it returns a balanced mix at the
    nearest strikes."""
    from src.agents.data_refresh import _pick_atm_contracts
    contracts = _pick_atm_contracts(
        option_chain_db, "FOO", spot=100.0, max_contracts=6, max_dte=45,
    )
    assert len(contracts) == 6
    types = []
    for cs in contracts:
        r = option_chain_db.execute(
            "SELECT option_type FROM option_contracts WHERE contract_symbol = ?",
            (cs,),
        ).fetchone()
        types.append(r["option_type"])
    c = Counter(types)
    assert c["call"] >= 1, f"expected at least 1 call, got {dict(c)}"
    assert c["put"] >= 1, f"expected at least 1 put, got {dict(c)}"


def test_pick_atm_prefers_nearest_strike_to_spot(option_chain_db):
    """Ranking: ABS(strike - spot) is the primary sort key. With
    spot=$100 and max_contracts=6, all 6 should be the 100-strike
    contracts (3 expiries x 2 types)."""
    from src.agents.data_refresh import _pick_atm_contracts
    contracts = _pick_atm_contracts(
        option_chain_db, "FOO", spot=100.0, max_contracts=6, max_dte=45,
    )
    strikes = []
    for cs in contracts:
        r = option_chain_db.execute(
            "SELECT strike_price FROM option_contracts WHERE contract_symbol = ?",
            (cs,),
        ).fetchone()
        strikes.append(float(r["strike_price"]))
    assert all(s == 100.0 for s in strikes), (
        f"expected all strikes == 100 (closest to spot), got {strikes}"
    )


def test_pick_atm_dte_filter_excludes_expired(option_chain_db):
    """Contracts whose DTE is outside [1, max_dte] must be skipped.
    Insert one expired and one too-far-out contract and assert
    neither comes back."""
    import datetime
    expired = datetime.date(2026, 9, 1)   # 1 day ago, DTE = 0
    far_out = datetime.date(2027, 9, 1)   # DTE > 365
    for exp, otype in (
        (expired, "call"),
        (expired, "put"),
        (far_out, "call"),
        (far_out, "put"),
    ):
        exp_str = exp.strftime("%Y-%m-%d")
        sym = (
            f"FOO{exp.strftime('%y%m%d')}"
            f"{'C' if otype == 'call' else 'P'}"
            f"{int(100 * 1000):08d}"
        )
        option_chain_db.execute(
            "INSERT INTO option_contracts "
            "(contract_symbol, underlying_symbol, expiration_date, "
            " strike_price, option_type, tradable, status) "
            "VALUES (?, 'FOO', ?, 100, ?, 1, 'active')",
            (sym, exp_str, otype),
        )
    option_chain_db.commit()

    from src.agents.data_refresh import _pick_atm_contracts
    contracts = _pick_atm_contracts(
        option_chain_db, "FOO", spot=100.0, max_contracts=20, max_dte=45,
    )
    # None of the 4 boundary contracts should be in the result.
    for cs in contracts:
        r = option_chain_db.execute(
            "SELECT expiration_date FROM option_contracts WHERE contract_symbol = ?",
            (cs,),
        ).fetchone()
        exp = datetime.date.fromisoformat(r["expiration_date"])
        dte = (exp - datetime.date(2026, 9, 2)).days   # today is 2026-09-02
        assert 1 <= dte <= 45, (
            f"contract {cs} has dte={dte} - out of [1, 45] filter"
        )


def test_pick_atm_excludes_untradable(option_chain_db):
    """Contracts with tradable=0 (or status != 'active') must be
    skipped so the function doesn't waste HTTP calls on contracts
    the broker will reject."""
    option_chain_db.execute(
        "UPDATE option_contracts SET tradable = 0 "
        "WHERE strike_price = 100.0 AND option_type = 'call'"
    )
    option_chain_db.commit()
    from src.agents.data_refresh import _pick_atm_contracts
    contracts = _pick_atm_contracts(
        option_chain_db, "FOO", spot=100.0, max_contracts=6, max_dte=45,
    )
    # At 100-strike, only the puts are now tradable. The result
    # should not include any 100-strike call.
    for cs in contracts:
        r = option_chain_db.execute(
            "SELECT option_type, tradable FROM option_contracts "
            "WHERE contract_symbol = ?", (cs,),
        ).fetchone()
        assert not (r["option_type"] == "call"
                    and r["tradable"] == 0), (
            f"untradable call {cs} leaked into the result"
        )
