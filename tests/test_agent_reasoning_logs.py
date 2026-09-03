"""Tests for the per-cycle agent-reasoning log line.

The cycle trace JSONL (models/cycle_traces/cycle_*.jsonl) captures the
full per-agent observation payload, but the operator doesn't see that
file in the GH Actions log. The 2026-09-03 cron run logged a final
``action=NO_TRADE`` with no agent-level reasoning visible — the user
asked why no agent reasoning showed up. The fix is :func:`_log_agent_observation`
in :mod:`src.agents.nodes._common` which emits one INFO line per agent
per cycle on the cron stdout.

These tests pin the behavior:
  * One log line is emitted per agent with confidence + signal + evidence.
  * The signal dict is rendered compactly (no full-payload dump).
  * The evidence list is truncated to 3 items.
  * The render is exception-safe (a broken signal still emits a warn,
    never breaks the cycle).
  * The startup data-state line in run_loop emits the expected keys
    (llm provider, table counts, news freshness, max DTE).
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.nodes._common import _log_agent_observation, _compact
from src.agents.protocol import AgentObservation, MessageType


# ---------------------------------------------------------------------------
# _log_agent_observation
# ---------------------------------------------------------------------------
def _obs(confidence: float, signal: dict, evidence=None, risks=None) -> AgentObservation:
    return AgentObservation(
        agent_id="test_agent",
        message_type=MessageType.REGIME_VIEW,
        confidence=confidence,
        signal=signal,
        evidence=list(evidence or []),
        risks=list(risks or []),
    )


def test_log_emits_one_info_line_with_confidence_signal_evidence(caplog):
    """The happy path: an observation with confidence, signal dict, and
    2 evidence strings produces one INFO line containing all three."""
    obs = _obs(
        0.45,
        {"candidates_returned": 0, "dte_fallback": "no_dte_data"},
        evidence=["no option chains or equity signals met the min-score filter",
                  "strict [5,10] empty; widened to [18,28]"],
    )
    with caplog.at_level(logging.INFO, logger="aizen.agent"):
        _log_agent_observation("options_structure_agent", obs)
    # caplog.records holds every record emitted during the with-block.
    matching = [r for r in caplog.records if r.name == "aizen.agent"]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "options_structure_agent" in msg
    assert "confidence=0.45" in msg
    assert "candidates_returned=0" in msg
    assert "dte_fallback=no_dte_data" in msg
    # The first evidence string should appear (truncation is at 3 items,
    # so both fit).
    assert "min-score filter" in msg


def test_log_truncates_evidence_to_three_items(caplog):
    """5 evidence strings -> only the first 3 make it into the log line.
    Keeps the cron stdout readable when an agent dumps a long evidence
    list (the JSONL trace still has all 5)."""
    obs = _obs(0.5, {}, evidence=[f"evidence #{i}" for i in range(5)])
    with caplog.at_level(logging.INFO, logger="aizen.agent"):
        _log_agent_observation("regime_agent", obs)
    msg = [r for r in caplog.records if r.name == "aizen.agent"][0].getMessage()
    assert "evidence #0" in msg
    assert "evidence #1" in msg
    assert "evidence #2" in msg
    assert "evidence #3" not in msg
    assert "evidence #4" not in msg


def test_log_renders_empty_signal_gracefully(caplog):
    """An observation with no signal dict still emits a line."""
    obs = _obs(0.2, {})
    with caplog.at_level(logging.INFO, logger="aizen.agent"):
        _log_agent_observation("supervisor", obs)
    msg = [r for r in caplog.records if r.name == "aizen.agent"][0].getMessage()
    assert "supervisor" in msg
    assert "confidence=0.20" in msg
    # The "(no signal)" sentinel appears when signal is empty.
    assert "(no signal)" in msg or "signal=" in msg


def test_log_does_not_break_on_pathological_signal(caplog):
    """A signal that contains an unrenderable object must not crash the
    cycle. The fallback path emits a WARN with the exception class."""
    class _Boom:
        def __repr__(self):  # raises on repr
            raise RuntimeError("repr explosion")

    obs = _obs(0.1, {"x": _Boom()})
    with caplog.at_level(logging.INFO, logger="aizen.agent"):
        # Should not raise.
        _log_agent_observation("test_agent", obs)
    # We expect either a successful INFO line or a fallback WARN. Both
    # are acceptable outcomes — the test only asserts that we logged
    # *something* and did not raise.
    matching = [r for r in caplog.records if r.name == "aizen.agent"]
    assert len(matching) >= 1


def test_log_uses_aizen_agent_logger_name(caplog):
    """The logger name is fixed at ``aizen.agent`` so an operator can
    ``grep '^.* aizen.agent '`` in the cron stdout to extract just the
    per-agent reasoning without the LLM/HTTP noise."""
    obs = _obs(0.5, {"k": 1})
    with caplog.at_level(logging.INFO, logger="aizen.agent"):
        _log_agent_observation("regime_agent", obs)
    names = {r.name for r in caplog.records}
    assert "aizen.agent" in names


# ---------------------------------------------------------------------------
# _compact (the renderer the log helper uses)
# ---------------------------------------------------------------------------
def test_compact_renders_floats_with_2_decimals():
    assert _compact(0.456) == "0.46"
    assert _compact(-0.57213) == "-0.57"


def test_compact_renders_bools():
    assert _compact(True) == "true"
    assert _compact(False) == "false"


def test_compact_renders_short_dicts_inline():
    out = _compact({"a": 1, "b": "x"})
    assert "a=1" in out
    assert "b=x" in out


def test_compact_renders_list_preview():
    out = _compact([1, 2, 3, 4, 5])
    # 3-item preview with trailing ellipsis.
    assert "1" in out
    assert "..." in out


def test_compact_truncates_long_strings():
    long = "x" * 200
    out = _compact(long)
    # 80-char cap, then "..."
    assert out.endswith("...")
    assert len(out) < 100


# ---------------------------------------------------------------------------
# Startup data-state log in run_loop
# ---------------------------------------------------------------------------
def test_log_startup_data_state_emits_required_keys(tmp_path, caplog):
    """The run_loop startup log line must include: llm provider, table
    counts for the four key tables, news last timestamp, news age in
    minutes, and max option DTE. The line is the operator's at-a-glance
    answer to "what did the cycle have to work with?"."""
    from src.agents.cli.run_loop import _log_startup_data_state
    from src.db import init_db

    db_path = tmp_path / "trading.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    # Seed a few rows so the counts are non-zero.
    conn.executemany(
        "INSERT INTO news_snapshot (timestamp, symbol, sentiment, article_count, "
        "topics_json, raw_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2026-09-03T12:00:00Z", "AAPL", 0.1, 1, "[]", "{}", "2026-09-03T12:00:00Z"),
            ("2026-09-03T12:00:00Z", "NVDA", -0.2, 2, "[]", "{}", "2026-09-03T12:00:00Z"),
        ],
    )
    conn.executemany(
        "INSERT INTO option_contracts (underlying_symbol, contract_symbol, "
        "expiration_date, strike_price, option_type) VALUES (?, ?, ?, ?, ?)",
        [
            ("AAPL", "AAPL260911C00200000", "2026-09-11", 200.0, "C"),
            ("NVDA", "NVDA260925P00100000", "2026-09-25", 100.0, "P"),
        ],
    )
    conn.commit()
    with caplog.at_level(logging.INFO, logger="run_loop"):
        _log_startup_data_state(conn)
    # The relevant record is the one whose message starts with "data_state".
    matching = [r for r in caplog.records
                if r.name == "run_loop" and "data_state" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    # All seven required keys must appear.
    for key in ("llm=", "underlying_bars=", "option_contracts=",
                "option_bars=", "news_snapshot=", "news_last=",
                "news_age_min=", "option_dte_max="):
        assert key in msg, f"missing key {key!r} in: {msg}"
    # The seeded contracts gave us a non-zero news count.
    assert "news_snapshot=2" in msg
    assert "option_contracts=2" in msg
    # Default label appears in the leading key.
    assert msg.startswith("data_state ")
    conn.close()


def test_log_startup_data_state_accepts_label_for_before_after_diff(caplog):
    """The post-refresh log line uses label='data_state_after_refresh' so
    the operator can grep the BEFORE/AFTER pair with a single regex."""
    from src.agents.cli.run_loop import _log_startup_data_state
    from src.db import init_db
    import tempfile, os

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    with caplog.at_level(logging.INFO, logger="run_loop"):
        _log_startup_data_state(conn, label="data_state_after_refresh")
    matching = [r for r in caplog.records
                if r.name == "run_loop" and "data_state_after_refresh" in r.getMessage()]
    assert len(matching) == 1
    conn.close()
    os.unlink(path)


def test_log_startup_data_state_handles_empty_db(caplog):
    """A brand-new DB with no rows must still log the line (with 0s
    and '?'). The line is the operator's first signal that a fresh
    cron tick has no data to work with."""
    from src.agents.cli.run_loop import _log_startup_data_state
    from src.db import init_db
    import tempfile, os

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    with caplog.at_level(logging.INFO, logger="run_loop"):
        _log_startup_data_state(conn)
    matching = [r for r in caplog.records
                if r.name == "run_loop" and "data_state" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "underlying_bars=0" in msg
    assert "option_contracts=0" in msg
    # No news rows -> "(none)" sentinel.
    assert "news_last=(none)" in msg
    # The "no tradable options" WARNING should also fire for an empty DB.
    warning_lines = [r for r in caplog.records
                     if r.name == "run_loop" and r.levelname == "WARNING"
                     and "no tradable options" in r.getMessage()]
    assert len(warning_lines) == 1
    conn.close()
    os.unlink(path)
