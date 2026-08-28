"""Download historical option bars for selected contracts (doc section 4.3).

Constraints honored: max 100 contract symbols per request, next_page_token
pagination, free 'indicative' feed recorded on every row/run.

Usage:
  python -m src.download_option_bars --all-selected --days-back 7
  python -m src.download_option_bars --symbols AAPL --cap-per-symbol 8
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from .alpaca_client import AlpacaClient, normalize_ts, setup_logging
from .config import DATA_HOST, get_settings
from .db import connect, insert_rows, record_run

_BARS_URL = f"{DATA_HOST}/v1beta1/options/bars"
MAX_SYMBOLS_PER_REQUEST = 100


def _field(bar: dict, *names: str, default: Any = None) -> Any:
    """Option-bar JSON field naming is confirmed empirically at first use;
    accept short ('t','o','h','l','c','v','vw','n') and long spellings."""
    for name in names:
        if name in bar:
            return bar[name]
    return default


def _iter_bar_items(payload: dict) -> Iterator[tuple[str | None, dict]]:
    """Yield (contract_symbol, bar) whether bars arrive grouped by symbol
    (dict of lists) or flat (list whose items carry S/symbol)."""
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if isinstance(bars, dict):
        for contract_symbol, items in bars.items():
            for bar in items:
                yield contract_symbol, bar
    elif isinstance(bars, list):
        for bar in bars:
            if isinstance(bar, dict):
                yield bar.get("S") or bar.get("symbol"), bar


def selected_contract_symbols(conn, underlying: list[str] | None, cap_per_symbol: int) -> list[str]:
    if underlying:
        placeholders = ",".join("?" * len(underlying))
        rows = conn.execute(
            f"SELECT contract_symbol FROM contract_selection WHERE underlying_symbol IN ({placeholders}) "
            "ORDER BY underlying_symbol, rank",
            underlying,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT contract_symbol FROM contract_selection ORDER BY underlying_symbol, rank"
        ).fetchall()
    symbols = [r["contract_symbol"] for r in rows]
    if cap_per_symbol:
        counts: dict[str, int] = {}
        limited: list[str] = []
        underlying_of = dict(
            conn.execute("SELECT contract_symbol, underlying_symbol FROM contract_selection").fetchall()
        )
        for csym in symbols:
            u = underlying_of.get(csym, "?")
            counts[u] = counts.get(u, 0)
            if counts[u] < cap_per_symbol:
                counts[u] += 1
                limited.append(csym)
        symbols = limited
    return symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download historical option bars")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--symbols", help="underlying symbols; use their contract_selection picks")
    source.add_argument("--all-selected", action="store_true", help="every row in contract_selection")
    parser.add_argument("--contract-symbols", help="explicit comma-separated OCC contract symbols")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=MAX_SYMBOLS_PER_REQUEST)
    parser.add_argument("--cap-per-symbol", type=int, help="limit contracts per underlying")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    # IMPORTANT: default `end` to the start of today (UTC), never `now`. Ranges
    # touching the live session require a signed OPRA agreement -> HTTP 403
    # "OPRA agreement is not signed". Fully-past ranges use the free historical
    # feed. Pass --end explicitly only if you know the account has OPRA access.
    end = normalize_ts(args.end) if args.end else datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    start = (
        normalize_ts(args.start) if args.start
        else (datetime.now(timezone.utc) - timedelta(days=args.days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    conn = connect()
    if args.contract_symbols:
        contract_symbols = [s.strip().upper() for s in args.contract_symbols.split(",") if s.strip()]
    else:
        underlyings = (
            [s.strip().upper() for s in args.symbols.split(",")] if args.symbols
            else None if args.all_selected else ["AAPL", "SPY"]
        )
        contract_symbols = selected_contract_symbols(conn, underlyings, args.cap_per_symbol or 0)

    known = {r["contract_symbol"] for r in conn.execute("SELECT contract_symbol FROM option_contracts")}
    unknown = [c for c in contract_symbols if c not in known]
    if unknown:
        print(f"WARN: {len(unknown)} contract symbols have no option_contracts row "
              f"(e.g. {unknown[0][:24]}...) - they will read as orphans in validation")

    chunks = [
        contract_symbols[i : i + args.chunk_size]
        for i in range(0, len(contract_symbols), args.chunk_size)
    ]
    print(f"pulling bars for {len(contract_symbols)} contracts in {len(chunks)} chunk(s), "
          f"{start} .. {end}")

    inserted_total = skipped_total = 0
    shape_logged = False
    with record_run(
        conn,
        dataset_type="option_bars",
        symbols=",".join(contract_symbols)[:2000],
        timeframe=settings.timeframe,
        start_time=start,
        end_time=end,
        feed="indicative",
        api_endpoint=_BARS_URL,
    ) as run, AlpacaClient() as client:
        for chunk_no, chunk in enumerate(chunks, 1):
            params = {
                "symbols": ",".join(chunk),
                "timeframe": settings.timeframe,
                "start": start,
                "end": end,
                "sort": "asc",
                "limit": 10000,
            }
            for payload in client.paginate(_BARS_URL, params):
                if not shape_logged:
                    sample = next(iter(_iter_bar_items(payload)), (None, {}))[1]
                    kind = "short" if sample and "t" in sample else "long" if sample else "empty"
                    print(f"response shape probe: bars keyed-by-symbol="
                          f"{isinstance(payload.get('bars'), dict)}, field style={kind}")
                    shape_logged = True
                rows = []
                for contract_symbol, bar in _iter_bar_items(payload):
                    ts_raw = _field(bar, "t", "timestamp")
                    if not ts_raw or not contract_symbol:
                        continue
                    rows.append((
                        contract_symbol,
                        normalize_ts(ts_raw),
                        _field(bar, "o", "open"),
                        _field(bar, "h", "high"),
                        _field(bar, "l", "low"),
                        _field(bar, "c", "close"),
                        _field(bar, "v", "volume"),
                        _field(bar, "vw", "vwap"),
                        _field(bar, "n", "trade_count"),
                        "indicative",
                    ))
                inserted, skipped = insert_rows(conn, "option_bars", rows)
                inserted_total += inserted
                skipped_total += skipped
                run.rows_inserted, run.rows_skipped = inserted_total, skipped_total
            print(f"chunk {chunk_no}/{len(chunks)} done: total {inserted_total} inserted, "
                  f"{skipped_total} skipped")

    print(f"TOTAL: {inserted_total} option bars inserted, {skipped_total} skipped")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
