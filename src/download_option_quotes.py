"""Placeholder for live-agent data pulls (doc sections 4.4-4.6).

Historical option TRADES are deliberately deferred: volume is far larger than
aggregated bars and adds nothing to Phase-1 training. Snapshots / latest quotes
are LIVE-agent concerns; the option_quotes and option_snapshots tables already
exist in the schema as landing zones.

This module intentionally does nothing beyond explaining that - run it any time
you want a reminder.
"""

from __future__ import annotations

import argparse

DEFERRED_NOTE = (
    "Option trades/quotes/snapshots are deferred per spec:\n"
    "  4.4 historical option trades  - OPTIONAL, too heavy initially\n"
    "  4.5 snapshots / chain         - live agent phase\n"
    "  4.6 latest option quotes      - live agent phase\n"
    "Tables option_quotes / option_snapshots exist and stay empty until then.\n"
    "Training must never mix current snapshots into past observations (doc 4.6)."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deferred live-data collectors (no-op)")
    parser.parse_args(argv)
    print(DEFERRED_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
