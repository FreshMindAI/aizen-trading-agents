"""Tests for the US equity market-hours gate."""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the default-on behavior unless a test overrides it.
    monkeypatch.setenv("AIZEN_MARKET_HOURS_ONLY", "1")


def test_mid_market_weekday_is_open() -> None:
    from src.agents.market_hours import evaluate
    # Tuesday 10:30 ET, 2026-09-01 (regular session).
    now = datetime(2026, 9, 1, 10, 30, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is True
    assert g.reason.startswith("market open")


def test_pre_market_is_closed() -> None:
    from src.agents.market_hours import evaluate
    now = datetime(2026, 9, 1, 8, 0, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert "pre-market" in g.reason
    assert g.next_open_et is not None
    assert g.next_open_et.hour == 9
    assert g.next_open_et.minute == 30


def test_post_close_is_closed_with_buffer() -> None:
    from src.agents.market_hours import evaluate
    # 16:05 ET - 5 minute buffer has passed
    now = datetime(2026, 9, 1, 16, 5, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert "post-close" in g.reason


def test_lunchtime_is_open() -> None:
    from src.agents.market_hours import evaluate
    now = datetime(2026, 9, 1, 12, 0, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is True


def test_saturday_is_closed() -> None:
    from src.agents.market_hours import evaluate
    # 2026-09-05 is a Saturday
    now = datetime(2026, 9, 5, 10, 30, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert "weekend" in g.reason
    # Next open should be Monday 2026-09-08
    assert g.next_open_et is not None
    assert g.next_open_et.date().isoformat() == "2026-09-08"


def test_sunday_is_closed() -> None:
    from src.agents.market_hours import evaluate
    now = datetime(2026, 9, 6, 10, 30, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert "weekend" in g.reason


def test_labor_day_is_closed() -> None:
    from src.agents.market_hours import evaluate
    # 2026-09-07 is Labor Day
    now = datetime(2026, 9, 7, 10, 30, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert "holiday" in g.reason
    # Next open is Tuesday 2026-09-08
    assert g.next_open_et is not None
    assert g.next_open_et.date().isoformat() == "2026-09-08"


def test_christmas_is_closed() -> None:
    from src.agents.market_hours import evaluate
    now = datetime(2026, 12, 25, 11, 0, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert "holiday" in g.reason


def test_friday_post_close_rolls_to_monday() -> None:
    from src.agents.market_hours import evaluate
    # 2026-09-04 is a Friday
    now = datetime(2026, 9, 4, 17, 0, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is False
    assert g.next_open_et is not None
    # Should be Monday 2026-09-08 (skip weekend)
    assert g.next_open_et.date().isoformat() == "2026-09-08"


def test_opt_out_via_kwarg() -> None:
    from src.agents.market_hours import evaluate
    now = datetime(2026, 9, 5, 10, 30, tzinfo=ET)  # Saturday
    g = evaluate(now, market_hours_only=False)
    assert g.should_run is True
    assert "off" in g.reason.lower()


def test_opt_out_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.market_hours import evaluate
    monkeypatch.setenv("AIZEN_MARKET_HOURS_ONLY", "0")
    now = datetime(2026, 9, 5, 10, 30, tzinfo=ET)  # Saturday
    g = evaluate(now)
    assert g.should_run is True


def test_should_run_helper() -> None:
    from src.agents.market_hours import should_run
    now_open = datetime(2026, 9, 1, 11, 0, tzinfo=ET)
    now_closed = datetime(2026, 9, 5, 11, 0, tzinfo=ET)
    assert should_run(now_et=now_open) is True
    assert should_run(now_et=now_closed) is False


def test_exactly_at_open_is_open() -> None:
    from src.agents.market_hours import evaluate
    now = datetime(2026, 9, 1, 9, 30, 0, tzinfo=ET)
    g = evaluate(now)
    assert g.should_run is True


def test_dst_transition_spring_forward() -> None:
    """March 8 2026 is the spring-forward DST transition in the US.
    The gate should still treat it as a regular trading day."""
    from src.agents.market_hours import evaluate
    now = datetime(2026, 3, 9, 10, 30, tzinfo=ET)  # Monday after
    g = evaluate(now)
    assert g.should_run is True
    # 10:30 ET on March 9 should be 14:30 or 15:30 UTC depending on
    # whether DST has kicked in - but the wall clock time is what
    # matters for the market gate.
    assert g.now_et.hour == 10
