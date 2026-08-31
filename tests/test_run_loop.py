"""Smoke tests for the Render Cron Job entry point (run_loop).

The test pins three invariants Render will rely on:
  1. ``--once`` runs a single cycle and exits 0 (Render kills any job
     that exceeds maxRunDuration; we want a clean exit code, not 1).
  2. AIZEN_DB_PATH env var is honored (the persistent disk path).
  3. The summary block prints even on a NO_TRADE cycle (Render's
     log-grep workflows need a consistent marker).
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_run_once_runs_one_cycle_and_exits(tmp_path, monkeypatch):
    """End-to-end smoke: run_loop --once should print the summary block
    and exit 0. Uses mock LLM + dry-run to avoid any broker calls."""
    db = tmp_path / "trading.db"
    env = {
        **os.environ,
        "AIZEN_LLM_PROVIDER": "mock",
        "RUN_MODE": "dry-run",
        "AIZEN_DB_PATH": str(db),
        "AIZEN_TRACE": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "src.agents.cli.run_loop", "--once"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"run_loop failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    # The summary block is the contract with Render's log scrapers.
    assert "== run_loop summary ==" in proc.stdout
    assert "decision_id" in proc.stdout
    assert "final_action" in proc.stdout
    # The DB file should have been created and the decision journal
    # should have a row.
    assert db.exists()
    import sqlite3
    conn = sqlite3.connect(str(db))
    n_journal = conn.execute("SELECT COUNT(*) FROM decision_journal").fetchone()[0]
    assert n_journal >= 1


def test_aizen_db_path_is_honored(tmp_path, monkeypatch):
    """AIZEN_DB_PATH must override the default location."""
    custom = tmp_path / "custom-path.db"
    env = {
        **os.environ,
        "AIZEN_LLM_PROVIDER": "mock",
        "RUN_MODE": "dry-run",
        "AIZEN_DB_PATH": str(custom),
        "AIZEN_TRACE": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "src.agents.cli.run_loop", "--once"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0
    assert custom.exists(), "AIZEN_DB_PATH was not honored"


def test_summary_block_includes_all_required_fields():
    """The summary block contract: every Render log scrape depends on
    these exact field names. Lock them in."""
    from src.agents.cli.run_loop import _print_summary
    import time as _t

    class _FakeState:
        decision_id = "abc"
        final_action = "NO_TRADE"
        market_snapshot = None
        gnn_output = None
        cycle_started_at = "2026-08-30T10:00:00Z"
        selected_strategy = None
        candidate_strategies = []
        order_intent = None
        execution_result = None
        ml_predictions = []
        agent_observations = []
        agent_messages = []

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_summary(_FakeState(), n_refreshed={}, started_at="2026-08-30T10:00:00Z", started_at_epoch=_t.time())
    out = buf.getvalue()
    for field in (
        "started_at", "duration_s", "refreshed", "decision_id",
        "final_action", "underlying", "strategy_id", "score",
        "candidates", "order_legs", "exec_status", "cycle_started",
    ):
        assert field in out, f"summary block missing field {field!r}"
