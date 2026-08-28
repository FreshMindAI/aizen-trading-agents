"""SQLite helpers: connections, schema init, bulk inserts, data-run bookkeeping."""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "sql"


def utc_now_iso() -> str:
    """UTC timestamp in the canonical storage format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else PROJECT_ROOT / "data" / "trading.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Single-process writer; WAL keeps readers (notebooks) unblocked.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection, sql_dir: Path | None = None) -> list[str]:
    """Apply every sql/*.sql script in sorted filename order. Idempotent."""
    directory = sql_dir or SQL_DIR
    applied: list[str] = []
    for script in sorted(directory.glob("*.sql")):
        conn.executescript(script.read_text(encoding="utf-8"))
        applied.append(script.name)
    conn.commit()
    return applied


def insert_rows(
    conn: sqlite3.Connection, table: str, rows: Iterable[Sequence[Any]], chunk_size: int = 1000
) -> tuple[int, int]:
    """INSERT OR IGNORE rows into `table`; returns (inserted, skipped).

    The skip count is our only duplicate signal possible under PK constraints -
    validate_data uses it to prove immutability on reruns. `table` is always a
    code-level constant, never user input.
    """
    rows = list(rows)
    if not rows:
        return 0, 0
    placeholders = ",".join("?" * len(rows[0]))
    sql = f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})"
    before = conn.total_changes
    for i in range(0, len(rows), chunk_size):
        conn.executemany(sql, rows[i : i + chunk_size])
    conn.commit()
    inserted = conn.total_changes - before
    return inserted, len(rows) - inserted


class RunHandle:
    """Mutable counters a downloader fills in while its run is open."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.rows_inserted = 0
        self.rows_skipped = 0
        self.fields: dict[str, Any] = {}  # extra data_runs columns (symbols, endpoint, ...)


_RUN_COLUMNS = (
    "run_id", "started_at", "completed_at", "dataset_type", "symbols", "timeframe",
    "start_time", "end_time", "feed", "adjustment", "api_endpoint",
    "rows_inserted", "rows_skipped", "status", "error_message",
)


@contextmanager
def record_run(conn: sqlite3.Connection, dataset_type: str, **fields: Any) -> Iterator[RunHandle]:
    """Open/close a data_runs row around one download (or materialization).

    Usage:
        with record_run(conn, "underlying_bars", symbols="AAPL,SPY") as run:
            ...
            run.rows_inserted, run.rows_skipped = insert_rows(conn, ...)
    Errors mark the run 'error' with a truncated message and re-raise.
    """
    run_id = str(uuid.uuid4())
    handle = RunHandle(run_id)
    handle.fields.update(fields)
    values = {col: None for col in _RUN_COLUMNS}
    values.update(run_id=run_id, started_at=utc_now_iso(), dataset_type=dataset_type, **handle.fields)
    cols = ",".join(_RUN_COLUMNS)
    marks = ",".join("?" * len(_RUN_COLUMNS))
    conn.execute(f"INSERT INTO data_runs ({cols}) VALUES ({marks})", [values[c] for c in _RUN_COLUMNS])
    conn.commit()

    try:
        yield handle
    except BaseException as exc:  # noqa: BLE001 - record everything, then re-raise
        conn.execute(
            "UPDATE data_runs SET completed_at=?, status='error', error_message=? WHERE run_id=?",
            (utc_now_iso(), f"{type(exc).__name__}: {exc}"[:1000], run_id),
        )
        conn.commit()
        raise
    values = {
        "completed_at": utc_now_iso(),
        "status": "completed",
        "rows_inserted": handle.rows_inserted,
        "rows_skipped": handle.rows_skipped,
        **{k: v for k, v in handle.fields.items() if k in _RUN_COLUMNS},
    }
    assigns = ",".join(f"{k}=?" for k in values)
    conn.execute(
        f"UPDATE data_runs SET {assigns} WHERE run_id=?",
        [*values.values(), run_id],
    )
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQLite database utilities")
    parser.add_argument("command", choices=["init"], help="init: apply all sql/*.sql scripts")
    args = parser.parse_args(argv)

    if args.command == "init":
        conn = connect()
        applied = init_db(conn)
        print("objects now in database:")
        rows = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        for row in rows:
            print(f"  {row['type']:<6} {row['name']}")
        print(f"\napplied scripts: {', '.join(applied)}")
        print(f"total objects: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
