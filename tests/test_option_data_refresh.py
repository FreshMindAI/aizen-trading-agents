"""Tests for ``refresh_option_chains`` (live-tick option bar refresh).

Background
----------
The cron tick's ``refresh_one_bar`` only writes to ``underlying_bars``
- it never touches ``option_contracts`` or ``option_bars``. As a
result, the option_h4 XGBoost model in the inference layer has no
per-contract bar history to score. ``refresh_option_chains`` is the
new live-tick path that picks the ATM subset of contracts already in
``option_contracts`` and pulls the last 4 hours of delayed-feed option
bars from Alpaca.

These tests pin:
  - the ATM-selection SQL (correct DTE band, correct ranking);
  - the broker-error swallow (a broker outage must not break the cron
    tick).
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.agents.data_refresh import (  # noqa: E402
    _iter_option_bars,
    _pick_atm_contracts,
    refresh_option_chains,
)
from src.db import connect, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def conn(tmp_path):
    p = tmp_path / "test.db"
    c = connect(str(p))
    init_db(c, sql_dir=Path("sql"))
    yield c
    c.close()


def _seed_chain(conn, *, sym: str, spot: float, n_strikes: int = 8,
                expiry_days: int = 21, strike_step: float = 5.0,
                center_strike: float | None = None) -> list[str]:
    """Populate ``option_contracts`` with ``n_strikes`` ATM-ish call
    contracts at one expiry. Returns the contract_symbols."""
    expiry = (date.today() + timedelta(days=expiry_days)).isoformat()
    if center_strike is None:
        # Round to the nearest strike_step around spot.
        center_strike = round(spot / strike_step) * strike_step
    symbols: list[str] = []
    for i in range(n_strikes):
        strike = center_strike - (n_strikes // 2) * strike_step + i * strike_step
        if strike <= 0:
            continue
        # Construct an OCC-ish symbol. The tests don't depend on the
        # format being valid for Alpaca, just that it is a unique
        # string and matches the schema.
        sym_str = f"{sym}260116C{int(strike):08d}"
        conn.execute(
            "INSERT INTO option_contracts "
            "(contract_symbol, contract_id, underlying_symbol, expiration_date, "
            " strike_price, option_type, style, status, tradable, root_symbol, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sym_str, f"uuid-{sym_str}", sym, expiry,
             strike, "call", "american", "active", 1, sym, expiry),
        )
        symbols.append(sym_str)
    conn.commit()
    return symbols


# ---------------------------------------------------------------------------
# _pick_atm_contracts
# ---------------------------------------------------------------------------
def test_pick_atm_contracts_runs_against_real_db(conn):
    """Spot=100, 8 strikes around it; we should pick the 4 closest
    strikes to spot when max_contracts=4."""
    _seed_chain(conn, sym="NVDA", spot=100.0, n_strikes=8,
                strike_step=5.0, center_strike=100.0)
    picked = _pick_atm_contracts(conn, "NVDA", spot=100.0,
                                 max_contracts=4, max_dte=45)
    assert len(picked) == 4
    # The SQL orders by abs(strike - spot), so the 4 closest contracts
    # to spot=100 are 95, 100, 105, 90 (tied with 110 at 10 distance,
    # SQL returns the smaller strike first).
    rows = conn.execute(
        "SELECT strike_price FROM option_contracts "
        "WHERE contract_symbol IN ({})".format(",".join("?" * len(picked))),
        picked,
    ).fetchall()
    strikes = sorted(r["strike_price"] for r in rows)
    assert strikes == [90.0, 95.0, 100.0, 105.0]


def test_pick_atm_contracts_dte_filter_excludes_far_expiries(conn):
    """A contract with DTE > max_dte must be skipped, even if its
    strike is closest to spot."""
    _seed_chain(conn, sym="NVDA", spot=100.0, n_strikes=3,
                strike_step=5.0, center_strike=100.0, expiry_days=21)
    # Now insert a far-dated contract at the spot strike.
    far_expiry = (date.today() + timedelta(days=90)).isoformat()
    far_sym = "NVDA260116C00100000"
    conn.execute(
        "INSERT INTO option_contracts "
        "(contract_symbol, contract_id, underlying_symbol, expiration_date, "
        " strike_price, option_type, style, status, tradable, root_symbol, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (far_sym, f"uuid-{far_sym}", "NVDA", far_expiry,
         100.0, "call", "american", "active", 1, "NVDA", far_expiry),
    )
    conn.commit()
    picked = _pick_atm_contracts(conn, "NVDA", spot=100.0,
                                 max_contracts=10, max_dte=45)
    # The far-dated contract must NOT be in the picks.
    assert far_sym not in picked


def test_pick_atm_contracts_returns_empty_when_no_chain(conn):
    """A symbol with no option_contracts rows must return [] (and the
    caller can then skip the broker call entirely)."""
    picked = _pick_atm_contracts(conn, "XYZ", spot=100.0,
                                 max_contracts=6, max_dte=45)
    assert picked == []


# ---------------------------------------------------------------------------
# refresh_option_chains: end-to-end with a stubbed broker
# ---------------------------------------------------------------------------
def test_refresh_option_chains_writes_one_row_per_bar(tmp_path, monkeypatch):
    """When the broker returns one bar per contract, ``refresh_option_chains``
    must upsert exactly that many rows into ``option_bars`` and return
    ``{contract: 1}`` for each."""
    db_path = tmp_path / "test.db"
    seed = connect(str(db_path))
    init_db(seed, sql_dir=Path("sql"))
    _seed_chain(seed, sym="NVDA", spot=100.0, n_strikes=4,
                strike_step=5.0, center_strike=100.0)
    seed.execute(
        "INSERT INTO underlying_bars (symbol, timestamp, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("NVDA", "2026-08-25T13:30:00Z", 100.0, 100.0, 100.0, 100.0, 1000),
    )
    seed.commit()
    contracts = _pick_atm_contracts(seed, "NVDA", spot=100.0,
                                    max_contracts=6, max_dte=45)
    seed.close()
    # Build a payload shaped like Alpaca's options-bars response.
    payload = {
        "bars": {
            csym: [{
                "t": "2026-08-25T13:30:00Z",
                "o": 2.50, "h": 2.55, "l": 2.45, "c": 2.52,
                "v": 100, "vw": 2.50, "n": 50,
            }]
            for csym in contracts
        },
    }
    # Stub the broker client.
    fake_client = mock.MagicMock()
    fake_client.get.return_value = payload
    monkeypatch.setattr(
        "src.agents.data_refresh._alpaca_data_client",
        lambda: fake_client,
    )
    written = refresh_option_chains(["NVDA"], db_path=db_path,
                                    lookback_minutes=240,
                                    max_contracts_per_symbol=6)
    # One row per contract written.
    assert len(written) == len(contracts)
    assert all(v == 1 for v in written.values())
    # And the rows are in option_bars.
    check = connect(str(db_path))
    n_rows = check.execute(
        "SELECT COUNT(*) AS n FROM option_bars"
    ).fetchone()["n"]
    assert n_rows == len(contracts)
    # Each row is tagged with feed='indicative' (the default).
    rows = check.execute(
        "SELECT DISTINCT feed FROM option_bars"
    ).fetchall()
    assert {r["feed"] for r in rows} == {"indicative"}
    check.close()


def test_refresh_option_chains_swallows_broker_errors(tmp_path, monkeypatch):
    """A broker outage must not raise into the cron tick. The function
    returns ``{}`` and the caller (``_refresh_data`` in run_loop.py) can
    fall through to the underlying refresh path."""
    db_path = tmp_path / "test.db"
    fresh = connect(str(db_path))
    init_db(fresh, sql_dir=Path("sql"))
    fresh.close()
    # Stub the broker to raise on .get()
    fake_client = mock.MagicMock()
    fake_client.get.side_effect = RuntimeError("alpaca unreachable")
    monkeypatch.setattr(
        "src.agents.data_refresh._alpaca_data_client",
        lambda: fake_client,
    )
    # Must not raise.
    written = refresh_option_chains(["NVDA"], db_path=db_path)
    assert written == {}


# ---------------------------------------------------------------------------
# _iter_option_bars shape handling
# ---------------------------------------------------------------------------
def test_iter_option_bars_handles_dict_payload():
    """Alpaca returns the bars dict keyed by contract_symbol."""
    payload = {
        "bars": {
            "AAA260116C00100000": [
                {"t": "2026-08-25T13:30:00Z", "c": 2.5},
            ],
            "BBB260116P00050000": [
                {"t": "2026-08-25T13:30:00Z", "c": 0.5},
            ],
        },
    }
    out = list(_iter_option_bars(payload))
    assert len(out) == 2
    assert {csym for csym, _ in out} == {
        "AAA260116C00100000", "BBB260116P00050000",
    }


def test_iter_option_bars_handles_list_payload():
    """The endpoint may return a flat list of bars each carrying S/symbol."""
    payload = {
        "bars": [
            {"S": "AAA260116C00100000", "t": "2026-08-25T13:30:00Z", "c": 2.5},
            {"S": "BBB260116P00050000", "t": "2026-08-25T13:30:00Z", "c": 0.5},
        ],
    }
    out = list(_iter_option_bars(payload))
    assert len(out) == 2
    csyms = [csym for csym, _ in out]
    assert "AAA260116C00100000" in csyms
    assert "BBB260116P00050000" in csyms
