"""Data validation gates (spec milestone step 6).

Usage:
  python -m src.validate_data --stage stocks        # after stock download
  python -m src.validate_data --stage contracts     # after contract download
  python -m src.validate_data --stage option_bars   # after option-bar download
  python -m src.validate_data --stage all           # everything incl. views

Exit code is nonzero when any check FAILs. WARNs explain known data-quality
reality (IEX gaps, half sessions) without blocking.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

TS_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z"


class Check:
    def __init__(self, name: str, status: str, detail: str) -> None:
        self.name, self.status, self.detail = name, status, detail


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> object:
    return conn.execute(sql, params).fetchone()[0]


def check_stocks(conn) -> list[Check]:
    checks: list[Check] = []

    bad_ohlc = _scalar(
        conn,
        "SELECT COUNT(*) FROM underlying_bars WHERE high < MAX(open, close) "
        "OR low > MIN(open, close) OR MIN(open, low, close) <= 0 OR high <= 0",
    )
    checks.append(Check("underlying OHLC sanity", "PASS" if bad_ohlc == 0 else "FAIL",
                        f"{bad_ohlc} bad rows"))

    neg_vol = _scalar(conn, "SELECT COUNT(*) FROM underlying_bars WHERE volume < 0")
    checks.append(Check("underlying volume >= 0", "PASS" if neg_vol == 0 else "FAIL",
                        f"{neg_vol} negative"))

    null_vwap = _scalar(conn, "SELECT COUNT(*) FROM underlying_bars WHERE vwap IS NULL")
    pct = 100.0 * null_vwap / max(_scalar(conn, "SELECT COUNT(*) FROM underlying_bars"), 1)
    checks.append(Check("underlying vwap presence", "WARN" if pct > 20 else "PASS",
                        f"{null_vwap} NULL vwap ({pct:.1f}%) - common on free IEX feed"))

    bad_ts = _scalar(conn,
                     f"SELECT COUNT(*) FROM underlying_bars WHERE timestamp NOT GLOB '{TS_GLOB}'")
    checks.append(Check("underlying timestamp format", "PASS" if bad_ts == 0 else "FAIL",
                        f"{bad_ts} malformed"))

    thin = conn.execute(
        "SELECT symbol, date(timestamp) d, COUNT(*) n FROM underlying_bars "
        "GROUP BY symbol, d HAVING n NOT BETWEEN 20 AND 40 ORDER BY symbol, d"
    ).fetchall()
    detail = ", ".join(f"{r['symbol']} {r['d']}={r['n']}" for r in thin[:6])
    # Band rationale: 26 RTH bars is the norm; IEX extended-hours prints can push
    # a full session to ~40. Below 20 means a genuinely thin/partial session.
    checks.append(Check("bars/day grid [20,40]", "WARN" if thin else "PASS",
                        f"{len(thin)} off-grid sessions"
                        + (f" (half-days / thin IEX slots): {detail}" if thin else "")))
    return checks


def check_contracts(conn) -> list[Check]:
    checks: list[Check] = []

    total = _scalar(conn, "SELECT COUNT(*) FROM option_contracts")
    checks.append(Check("contracts stored", "PASS" if total else "FAIL", f"{total} rows"))

    types = dict(
        conn.execute("SELECT option_type, COUNT(*) FROM option_contracts GROUP BY option_type").fetchall()
    )
    ok = len(types) >= 2
    checks.append(Check("call+put coverage", "PASS" if ok else "WARN",
                        f"{types}"
                        + ("" if ok else " - set TWO_PASS_TYPES=True in download_option_contracts")))

    expiries = _scalar(conn, "SELECT COUNT(DISTINCT expiration_date) FROM option_contracts")
    checks.append(Check("distinct expiries", "INFO", f"{expiries}"))

    selected = _scalar(conn, "SELECT COUNT(*) FROM contract_selection")
    orphans_sel = _scalar(
        conn,
        "SELECT COUNT(*) FROM contract_selection cs LEFT JOIN option_contracts oc USING (contract_symbol) "
        "WHERE oc.contract_symbol IS NULL",
    )
    checks.append(Check("ATM selection referential integrity",
                        "PASS" if orphans_sel == 0 else "FAIL",
                        f"{selected} selected, {orphans_sel} dangling"))
    return checks


def check_option_bars(conn) -> list[Check]:
    checks: list[Check] = []

    bad_ohlc = _scalar(
        conn,
        "SELECT COUNT(*) FROM option_bars WHERE high IS NOT NULL AND low IS NOT NULL "
        "AND (high < low OR low <= 0)",
    )
    checks.append(Check("option OHLC sanity", "PASS" if bad_ohlc == 0 else "FAIL",
                        f"{bad_ohlc} bad rows"))

    orphans = _scalar(
        conn,
        "SELECT COUNT(*) FROM option_bars b LEFT JOIN option_contracts oc USING (contract_symbol) "
        "WHERE oc.contract_symbol IS NULL",
    )
    checks.append(Check("option bars referential integrity", "PASS" if orphans == 0 else "FAIL",
                        f"{orphans} orphan bars"))

    bad_ts = _scalar(conn, f"SELECT COUNT(*) FROM option_bars WHERE timestamp NOT GLOB '{TS_GLOB}'")
    checks.append(Check("option timestamp format", "PASS" if bad_ts == 0 else "FAIL",
                        f"{bad_ts} malformed"))

    feeds = dict(conn.execute("SELECT feed, COUNT(*) FROM option_bars GROUP BY feed").fetchall())
    checks.append(Check("feed recorded", "PASS" if feeds else "FAIL", f"{feeds}"))

    low_cov = conn.execute(
        """
        SELECT contract_symbol, COUNT(DISTINCT date(timestamp)) sessions, COUNT(*) bars
        FROM option_bars GROUP BY contract_symbol ORDER BY sessions ASC LIMIT 3
        """
    ).fetchall()
    detail = "; ".join(f"{r['contract_symbol'][:20]}.. {r['bars']}b/{r['sessions']}d" for r in low_cov)
    checks.append(Check("per-contract coverage (thinnest)", "INFO", detail or "no option bars yet"))
    return checks


def check_views(conn) -> list[Check]:
    checks: list[Check] = []

    try:
        balance = dict(
            ((r["horizon_bars"], r["target_class"]), r["n"])
            for r in conn.execute(
                "SELECT horizon_bars, target_class, COUNT(*) n FROM v_labels GROUP BY 1, 2"
            )
        )
    except sqlite3.Error as exc:
        return [Check("views queryable", "FAIL", str(exc))]

    total_h4 = sum(n for (h, _), n in balance.items() if h == 4)
    checks.append(Check("labels produced", "PASS" if total_h4 else "FAIL",
                        f"h4={total_h4}, h16={sum(n for (h, _), n in balance.items() if h == 16)}"))

    worst_share = 0.0
    for h in (4, 16):
        h_total = sum(n for (hh, _), n in balance.items() if hh == h)
        if h_total:
            worst_share = max(worst_share,
                              max(n for (hh, _), n in balance.items() if hh == h) / h_total)
    checks.append(Check("class balance non-degenerate",
                        "WARN" if worst_share > 0.99 else "PASS",
                        f"largest single-class share {worst_share:.1%}"))

    leaks = _scalar(
        conn,
        # Aggregate each side per symbol FIRST - joining raw rows multiplies
        # 500k labels x 25k bars/symbol into billions of combinations.
        "SELECT COUNT(*) FROM "
        "(SELECT symbol, MAX(timestamp) AS max_label FROM v_labels GROUP BY symbol) l "
        "JOIN (SELECT symbol, MAX(timestamp) AS max_bar FROM underlying_bars GROUP BY symbol) b "
        "USING (symbol) WHERE l.max_label >= b.max_bar",
    )
    checks.append(Check("leak guard (label ts < last bar)", "PASS" if leaks == 0 else "FAIL",
                        f"{leaks} symbols leaking"))

    train_rows = _scalar(conn, "SELECT COUNT(*) FROM v_ml_training_dataset")
    ctx_join = _scalar(conn, "SELECT COUNT(*) FROM v_ml_training_dataset WHERE n_contracts IS NOT NULL")
    checks.append(Check("training set size", "INFO",
                        f"{train_rows} rows, {ctx_join} with option context"))

    bad_class = _scalar(conn,
                        "SELECT COUNT(*) FROM v_ml_training_dataset WHERE target_class NOT IN (-1, 0, 1)")
    checks.append(Check("target_class domain", "PASS" if bad_class == 0 else "FAIL",
                        f"{bad_class} invalid"))
    return checks


STAGES = {
    "stocks": check_stocks,
    "contracts": check_contracts,
    "option_bars": check_option_bars,
    "views": check_views,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate stored data quality")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = parser.parse_args(argv)

    from .db import connect  # local import keeps --help fast

    conn = connect()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    # views depend on everything else; run them last automatically
    ordered = [s for s in ("stocks", "contracts", "option_bars") if s in stages]
    if "views" in stages:
        ordered.append("views")

    results: list[Check] = []
    for stage in ordered:
        print(f"\n== {stage} ==")
        stage_results = STAGES[stage](conn)
        for c in stage_results:
            print(f"  [{c.status:<4}] {c.name}: {c.detail}")
        results.extend(stage_results)

    failed = [c for c in results if c.status == "FAIL"]
    warned = [c for c in results if c.status == "WARN"]
    print(f"\n{len(results)} checks: {len(failed)} FAILED, {len(warned)} warnings")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
