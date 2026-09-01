"""US equity market-hours gate.

Lightweight, dependency-free check for whether the cron tick should
actually run. US equity market hours are:

  * Monday - Friday
  * 09:30 - 16:00 America/New_York
  * Closed on NYSE holidays (handled best-effort; if the calendar
    is unavailable we fall back to weekday + time, which is correct
    for ~99% of business days).

This exists so the cron tick doesn't:
  1. Waste minutes hammering Alpaca's data-feed API for bars it
     won't have outside market hours (Alpaca returns 403 on some
     paper accounts when the SIP feed is off-hours).
  2. Burn a Claude call on GMI when there's no fresh market data.
  3. Flood the operator with NO_TRADE noise overnight.

The gate is opt-out: pass ``market_hours_only=False`` to bypass it
entirely (useful for backfills and local dev).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US/Eastern (handles EST/EDT automatically)
ET = ZoneInfo("America/New_York")

# Market open / close in ET
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# 5-minute buffer so the tick that fires at 16:00:00 doesn't try to
# grab a 16:00 bar that's still being aggregated.
POST_CLOSE_BUFFER_MIN = 5

# Known 2026 NYSE holidays (best-effort list - source: NYSE official).
# This is intentionally a small list (Jan-Dec 2026) so we don't pull
# a holiday-calendar dependency. If a date is in this set, the gate
# returns ``is_open=False`` regardless of weekday.
NYSE_HOLIDAYS_2026: frozenset[str] = frozenset({
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving Day
    "2026-12-25",  # Christmas Day
})


@dataclass
class MarketHoursGate:
    """Decision object for whether a tick should run.

    ``should_run`` is the only field the cron job needs. The other
    fields are useful for logging.
    """

    should_run: bool
    reason: str
    now_et: datetime
    next_open_et: datetime | None = None
    next_close_et: datetime | None = None

    def __str__(self) -> str:
        return (
            f"MarketHoursGate(should_run={self.should_run}, "
            f"now_et={self.now_et.isoformat()}, reason={self.reason!r})"
        )


def _now_et() -> datetime:
    return datetime.now(tz=ET)


def _next_weekday_open(now_et: datetime) -> datetime:
    """Find the next weekday 09:30 ET strictly after ``now_et``."""
    d = now_et.date()
    while True:
        from datetime import timedelta
        d = d + timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS_2026:
            return datetime.combine(d, MARKET_OPEN, tzinfo=ET)


def evaluate(
    now_et: datetime | None = None,
    *,
    market_hours_only: bool | None = None,
) -> MarketHoursGate:
    """Return a MarketHoursGate describing whether a tick should run.

    Args:
        now_et: override the current time (used in tests).
        market_hours_only: if False, always return should_run=True
            (escape hatch for backfills, smoke tests, etc.). If None,
            read from env var AIZEN_MARKET_HOURS_ONLY (default True).
    """
    if market_hours_only is None:
        raw = os.getenv("AIZEN_MARKET_HOURS_ONLY", "1").strip().lower()
        market_hours_only = raw not in ("0", "false", "no", "off", "")

    if not market_hours_only:
        now = now_et or _now_et()
        return MarketHoursGate(
            should_run=True, reason="AIZEN_MARKET_HOURS_ONLY=off",
            now_et=now,
        )

    now = now_et or _now_et()
    today_iso = now.date().isoformat()

    # 1. Holiday check
    if today_iso in NYSE_HOLIDAYS_2026:
        nxt = _next_weekday_open(now)
        return MarketHoursGate(
            should_run=False,
            reason=f"holiday ({today_iso}); next open {nxt.isoformat()}",
            now_et=now, next_open_et=nxt,
        )

    # 2. Weekend check
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        nxt = _next_weekday_open(now)
        return MarketHoursGate(
            should_run=False,
            reason=f"weekend; next open {nxt.isoformat()}",
            now_et=now, next_open_et=nxt,
        )

    # 3. Time-of-day check
    t = now.time()
    close_with_buffer = time(
        MARKET_CLOSE.hour,
        MARKET_CLOSE.minute + POST_CLOSE_BUFFER_MIN,
    )
    if t < MARKET_OPEN:
        nxt = datetime.combine(now.date(), MARKET_OPEN, tzinfo=ET)
        return MarketHoursGate(
            should_run=False,
            reason=f"pre-market ({t.isoformat()}); opens {nxt.isoformat()}",
            now_et=now, next_open_et=nxt,
        )
    if t >= close_with_buffer:
        # If today is Friday, next open is Monday (next weekday).
        # If today is a weekday, next open is the next weekday's open.
        from datetime import timedelta
        if now.weekday() == 4:  # Friday
            nxt = _next_weekday_open(now)  # jumps Sat -> Mon
        else:
            nxt = _next_weekday_open(now)
        return MarketHoursGate(
            should_run=False,
            reason=f"post-close ({t.isoformat()}); next open {nxt.isoformat()}",
            now_et=now, next_open_et=nxt,
        )

    # 4. Market is open
    return MarketHoursGate(
        should_run=True,
        reason=f"market open ({t.isoformat()})",
        now_et=now,
        next_close_et=datetime.combine(now.date(), MARKET_CLOSE, tzinfo=ET),
    )


def should_run(**kwargs: object) -> bool:
    """Boolean shorthand for the cron entry point."""
    return evaluate(**kwargs).should_run  # type: ignore[arg-type]
