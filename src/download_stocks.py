"""Download historical equity bars into underlying_bars (doc section 4.1).

Usage:
  python -m src.download_stocks --symbols AAPL,SPY --start 2026-08-17T00:00:00Z \
      --end 2026-08-22T04:00:00Z
  python -m src.download_stocks --universe --days-back 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from .alpaca_client import AlpacaClient, normalize_ts, setup_logging
from .config import DATA_HOST, get_settings
from .db import connect, insert_rows, record_run, utc_now_iso


def parse_when(text: str | None, *, end_default: bool = False) -> str | None:
    """Accept YYYY-MM-DD or full RFC3339; return RFC3339 UTC."""
    if text is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if end_default else None
    if len(text) == 10:  # bare date
        text += "T00:00:00Z" if not end_default else "T23:59:59Z"
    return normalize_ts(text)


def bars_for_symbol(
    client: AlpacaClient,
    symbol: str,
    *,
    timeframe: str,
    start: str,
    end: str,
    feed: str,
    adjustment: str,
) -> Iterator[tuple[Any, ...]]:
    """Yield one insert-ready tuple per bar across all pages."""
    url = f"{DATA_HOST}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "feed": feed,
        "adjustment": adjustment,
        "sort": "asc",
        "limit": 10000,
    }
    for payload in client.paginate(url, params):
        for bar in payload.get("bars") or []:
            yield (
                symbol,
                normalize_ts(bar["t"]),
                bar["o"],
                bar["h"],
                bar["l"],
                bar["c"],
                bar.get("v"),
                bar.get("vw"),
                bar.get("n"),
                feed,
                adjustment,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download equity bars from Alpaca")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--symbols", help="comma-separated symbols (default: AAPL,SPY)")
    targets.add_argument("--universe", action="store_true", help="use the configured full universe")
    parser.add_argument("--start", help="YYYY-MM-DD or RFC3339")
    parser.add_argument("--end", help="YYYY-MM-DD or RFC3339 (default: now)")
    parser.add_argument("--days-back", type=int, help="alternative to --start")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    symbols = (
        settings.universe if args.universe
        else [s.strip().upper() for s in (args.symbols or "AAPL,SPY").split(",") if s.strip()]
    )
    start = parse_when(args.start)
    if args.days_back is not None:
        start = (datetime.now(timezone.utc) - timedelta(days=args.days_back)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    if start is None:
        parser.error("provide --start or --days-back")
    end = parse_when(args.end, end_default=True)

    inserted_total = skipped_total = 0
    with connect() as conn, record_run(
        conn,
        dataset_type="underlying_bars",
        symbols=",".join(symbols),
        timeframe=settings.timeframe,
        start_time=start,
        end_time=end,
        feed=settings.feed,
        adjustment=settings.adjustment,
        api_endpoint=f"{DATA_HOST}/v2/stocks/{{symbol}}/bars",
    ) as run, AlpacaClient() as client:
        for symbol in symbols:
            rows = list(
                bars_for_symbol(
                    client,
                    symbol,
                    timeframe=settings.timeframe,
                    start=start,
                    end=end,
                    feed=settings.feed,
                    adjustment=settings.adjustment,
                )
            )
            inserted, skipped = insert_rows(conn, "underlying_bars", rows)
            inserted_total += inserted
            skipped_total += skipped
            print(f"{symbol}: {inserted} inserted, {skipped} skipped")
            run.rows_inserted, run.rows_skipped = inserted_total, skipped_total

        # Keep the symbols metadata table populated alongside the bars.
        conn.executemany(
            "INSERT OR IGNORE INTO symbols (symbol, active, updated_at) VALUES (?, 1, ?)",
            [(s, utc_now_iso()) for s in symbols],
        )
        conn.commit()

    print(f"TOTAL: {inserted_total} inserted, {skipped_total} skipped "
          f"({start} .. {end}, feed={settings.feed}, adjustment={settings.adjustment})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
