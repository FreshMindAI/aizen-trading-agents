"""Smoke tests for the backtest CLI (spec 003 / T049).

Verifies the CLI is callable, parses arguments, runs end-to-end
against a synthetic SQLite DB, and writes a BacktestReport JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def seeded_db(tmp_path):
    """Seed a tiny DB with 3 weeks of NVDA bars."""
    import sqlite3
    import datetime

    db_path = tmp_path / "bt.db"
    sys.path.insert(0, str(REPO))
    from src.db import connect, init_db

    c = connect(str(db_path))
    init_db(c)
    start = datetime.date(2026, 8, 4)
    for i in range(15):
        d = start + datetime.timedelta(days=i)
        if d.weekday() >= 5:
            continue
        ts = d.isoformat() + "T13:30:00Z"
        close = 100.0 + i * 0.5
        c.execute(
            "INSERT INTO underlying_bars "
            "(symbol, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("NVDA", ts, close, close + 0.2, close - 0.2, close, 1000),
        )
    c.commit()
    c.close()
    return db_path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.agents.cli.backtest", *args],
        cwd=str(REPO),
        env={**os.environ, "AIZEN_LLM_PROVIDER": "mock"},
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_help_runs():
    proc = _run_cli(["--help"])
    assert proc.returncode == 0
    assert "--start" in proc.stdout
    assert "--interval" in proc.stdout


def test_cli_daily_end_to_end(seeded_db, tmp_path):
    out = tmp_path / "report.json"
    proc = _run_cli([
        "--start", "2026-08-04",
        "--end", "2026-08-22",
        "--interval", "daily",
        "--universe", "NVDA",
        "--out", str(out),
        "--db-path", str(seeded_db),
    ])
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["aggregate"]["n_cycles"] >= 1
    assert "n_proceed" in data["aggregate"]
    assert "hit_rate_h4" in data["aggregate"]


def test_cli_weekly_mondays_only(seeded_db, tmp_path):
    out = tmp_path / "report.json"
    proc = _run_cli([
        "--start", "2026-08-04",
        "--end", "2026-08-22",
        "--interval", "weekly",
        "--universe", "NVDA",
        "--out", str(out),
        "--db-path", str(seeded_db),
    ])
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    data = json.loads(out.read_text(encoding="utf-8"))
    cycles = [c["cycle_as_of"] for c in data["cycles"]]
    # Aug 4 is Tue, so Mondays in range are Aug 10, Aug 17.
    assert "2026-08-10T13:30:00Z" in cycles
    assert "2026-08-17T13:30:00Z" in cycles
    # Non-Mondays must be absent.
    assert "2026-08-11T13:30:00Z" not in cycles


def test_cli_end_before_start_errors(tmp_path):
    proc = _run_cli([
        "--start", "2026-08-22",
        "--end", "2026-08-04",
        "--interval", "daily",
        "--universe", "NVDA",
        "--out", str(tmp_path / "r.json"),
    ])
    assert proc.returncode != 0
    assert "error" in proc.stderr.lower() or "after" in proc.stderr.lower()


def test_cli_unknown_interval_errors(tmp_path):
    proc = _run_cli([
        "--start", "2026-08-04",
        "--end", "2026-08-22",
        "--interval", "monthly",
        "--universe", "NVDA",
        "--out", str(tmp_path / "r.json"),
    ])
    # argparse rejects the choice
    assert proc.returncode != 0
