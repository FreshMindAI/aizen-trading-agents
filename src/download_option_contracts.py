"""Download active option contracts (doc section 4.2) and optionally select a
deterministic near-the-money subset for bar pulls (--select-atm).

Note the host: contracts are served from the paper/trading API host, not the
data host - see doc sections 4.2/4.3.

Usage:
  python -m src.download_option_contracts --symbols AAPL,SPY
  python -m src.download_option_contracts --symbols AAPL,SPY --select-atm --cap 12
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from .alpaca_client import AlpacaClient, setup_logging
from .config import PAPER_HOST, get_settings
from .db import connect, insert_rows, record_run, utc_now_iso

# Flip to True only if the type-coverage probe shows one-sided results
# (i.e. the endpoint needs explicit type= filters to return calls AND puts).
TWO_PASS_TYPES = False

_CONTRACTS_URL = f"{PAPER_HOST}/v2/options/contracts"


def spot_from_db(conn, symbol: str) -> float | None:
    """Latest stored close for the symbol; stocks are downloaded before contracts."""
    row = conn.execute(
        "SELECT close FROM underlying_bars WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return row["close"] if row else None


def _extract_contracts(payload: Any) -> list[dict]:
    """Defensive: accept either documented response wrapper or a bare list."""
    if isinstance(payload, list):
        return payload
    for key in ("option_contracts", "contracts"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Alpaca option contracts")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--symbols", help="comma-separated symbols (default: AAPL,SPY)")
    targets.add_argument("--universe", action="store_true", help="use the configured full universe")
    parser.add_argument("--min-dte", type=int)
    parser.add_argument("--max-dte", type=int)
    parser.add_argument("--band-pct", type=float, help="strike band around spot (default 0.10)")
    parser.add_argument("--select-atm", action="store_true", help="pick ATM +/- offset subset")
    parser.add_argument("--cap", type=int, help="max selected contracts per symbol")
    parser.add_argument("--offset", type=int, help="strikes kept either side of ATM (default 5)")
    parser.add_argument("--far", action="store_true",
                        help="prefer FURTHEST expiries (deepest bar history) via round-robin; "
                             "default prefers nearest expiries")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    symbols = (
        settings.universe if args.universe
        else [s.strip().upper() for s in (args.symbols or "AAPL,SPY").split(",") if s.strip()]
    )
    cap = args.cap or settings.pilot_contract_cap
    offset = args.offset or settings.strike_offset
    band_pct = args.band_pct if args.band_pct is not None else settings.strike_band_pct
    min_exp = (date.today() + timedelta(days=args.min_dte or settings.expiry_min_dte)).isoformat()
    max_exp = (date.today() + timedelta(days=args.max_dte or settings.expiry_max_dte)).isoformat()

    conn = connect()
    inserted_total = skipped_total = 0
    with record_run(
        conn,
        dataset_type="option_contracts",
        symbols=",".join(symbols),
        start_time=min_exp,
        end_time=max_exp,
        api_endpoint=_CONTRACTS_URL,
    ) as run, AlpacaClient() as client:
        for symbol in symbols:
            spot = spot_from_db(conn, symbol)
            if spot is None:
                print(f"{symbol}: SKIP - no underlying_bars row yet; download stocks first")
                continue
            lo = round(spot * (1 - band_pct), 2)
            hi = round(spot * (1 + band_pct), 2)
            params: dict[str, Any] = {
                "underlying_symbols": symbol,
                "status": "active",
                "expiration_date_gte": min_exp,
                "expiration_date_lte": max_exp,
                "strike_price_gte": lo,
                "strike_price_lte": hi,
                "limit": 10000,
            }
            fetched_at = utc_now_iso()
            rows: list[tuple[Any, ...]] = []
            seen: set[str] = set()
            for pass_type in (["call", "put"] if TWO_PASS_TYPES else [None]):
                page_params = dict(params)
                if pass_type:
                    page_params["type"] = pass_type
                for payload in client.paginate(_CONTRACTS_URL, page_params):
                    for c in _extract_contracts(payload):
                        if c.get("symbol") in seen:
                            continue
                        seen.add(c.get("symbol"))
                        rows.append((
                            c.get("symbol"),
                            c.get("id"),
                            c.get("underlying_symbol") or symbol,
                            c.get("expiration_date"),
                            c.get("strike_price"),
                            c.get("type"),
                            c.get("style"),
                            c.get("status"),
                            1 if c.get("tradable") else 0,
                            c.get("root_symbol"),
                            fetched_at,
                        ))
            inserted, skipped = insert_rows(conn, "option_contracts", rows)
            inserted_total += inserted
            skipped_total += skipped
            run.rows_inserted, run.rows_skipped = inserted_total, skipped_total
            print(f"{symbol}: spot={spot:.2f} band=[{lo}, {hi}] "
                  f"-> {inserted} contracts inserted, {skipped} skipped")

        # Type-coverage probe (drives TWO_PASS_TYPES if ever needed).
        counts = conn.execute(
            f"SELECT option_type, COUNT(*) n FROM option_contracts "
            f"WHERE underlying_symbol IN ({','.join('?' * len(symbols))}) GROUP BY option_type",
            symbols,
        ).fetchall()
        print("contract coverage by type:", {r["option_type"]: r["n"] for r in counts})

        if args.select_atm:
            select_atm(conn, symbols, run_id=run.run_id, cap=cap, offset=offset, far=args.far)

    conn.close()
    return 0


def select_atm(conn, symbols: list[str], *, run_id: str, cap: int, offset: int,
               far: bool = False) -> None:
    """Deterministic pick: per (symbol, expiry) keep +/-offset strikes around ATM.

    far=False (pilot default): rank survivors by (nearest expiry, distance from
    spot) and take the top `cap`.

    far=True (training-depth mode): a contract's bar history starts at its
    LISTING date (~12 months before expiry), so the deepest obtainable history
    lives in the furthest-dated active expiries. Round-robin across expiries,
    furthest first, closest-strike first within each, until `cap` is reached.
    """
    for symbol in symbols:
        spot_row = spot_from_db(conn, symbol)
        if spot_row is None:
            continue
        spot: float = spot_row
        contracts = conn.execute(
            "SELECT contract_symbol, expiration_date, strike_price FROM option_contracts "
            "WHERE underlying_symbol=? AND status='active'",
            (symbol,),
        ).fetchall()
        by_expiry: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in contracts:
            by_expiry[row["expiration_date"]].append((row["contract_symbol"], row["strike_price"]))

        ranked_by_expiry: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for expiry, items in by_expiry.items():
            strikes = sorted({s for _, s in items})
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
            keep = set(strikes[max(0, atm_idx - offset): atm_idx + offset + 1])
            ranked_by_expiry[expiry].extend(
                (abs(strike - spot), csym) for csym, strike in items if strike in keep
            )
        for candidates in ranked_by_expiry.values():
            candidates.sort()  # closest strike first

        if far:
            picks: list[str] = []
            depth = 0
            while len(picks) < cap:
                took_any = False
                for expiry in sorted(ranked_by_expiry, reverse=True):  # furthest first
                    candidates = ranked_by_expiry[expiry]
                    if depth < len(candidates):
                        picks.append(candidates[depth][1])
                        took_any = True
                        if len(picks) >= cap:
                            break
                if not took_any:
                    break
                depth += 1
        else:
            flat = [(expiry, dist, csym)
                    for expiry, candidates in ranked_by_expiry.items()
                    for dist, csym in candidates]
            flat.sort(key=lambda t: (t[0], t[1]))
            picks = [csym for _, _, csym in flat[:cap]]

        conn.execute("DELETE FROM contract_selection WHERE underlying_symbol=?", (symbol,))
        conn.executemany(
            "INSERT OR REPLACE INTO contract_selection "
            "(run_id, contract_symbol, underlying_symbol, rank, spot_at_selection) VALUES (?,?,?,?,?)",
            [(run_id, csym, symbol, i + 1, spot) for i, csym in enumerate(picks)],
        )
        conn.commit()
        kinds = dict(
            conn.execute(
                f"SELECT option_type, COUNT(*) n FROM option_contracts WHERE contract_symbol IN "
                f"({','.join('?' * len(picks))}) GROUP BY option_type",
                picks,
            ).fetchall()
        ) if picks else {}
        chosen_expiries = {exp for exp, cands in ranked_by_expiry.items()
                           for _, csym in cands if csym in picks}
        print(f"{symbol}: selected {len(picks)} contracts across {len(chosen_expiries)} expiries "
              f"({min(chosen_expiries)} .. {max(chosen_expiries)})" if chosen_expiries
              else f"{symbol}: nothing to select",
              f"(types={kinds})")


if __name__ == "__main__":
    raise SystemExit(main())
