"""Materialize sql/ views into physical tables so training reads don't
recompute window functions over raw bars every time.

Two strategies, chosen per target:
  * Declared tables (features / labels / asset_correlations) keep their doc
    schema and PKs: rows are DELETEd then re-INSERTed by explicit column names
    inside one transaction.
  * ml_training_dataset is a pure cache: DROP + CREATE TABLE AS SELECT.

Raw observation tables are never touched - views remain recomputable at any
time; these caches are disposable.

Usage:
  python -m src.materialize                                  # features+labels+training set
  python -m src.materialize --tables labels                  # subset
  python -m src.materialize --tables asset_correlations      # heavy; bounded ranges only
"""

from __future__ import annotations

import argparse

from .db import connect, record_run

# table -> view it materializes from
VIEW_OF = {
    "features": "v_features_underlying",   # equity feature set ('u15m_v1')
    "labels": "v_labels",
    "asset_correlations": "v_asset_correlations",
    "ml_training_dataset": "v_ml_training_dataset",
}
DECLARED = {"features", "labels", "asset_correlations"}  # PK-carrying tables from 01_schema.sql
DEFAULT_TABLES = ["features", "labels", "ml_training_dataset"]


def _table_columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _view_columns(conn, view: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({view})")]


def materialize(conn, table: str) -> int:
    """Refresh one cache table from its view. Returns row count written."""
    view = VIEW_OF[table]
    if table in DECLARED:
        target_cols = _table_columns(conn, table)
        view_cols = set(_view_columns(conn, view))
        missing = [c for c in target_cols if c not in view_cols]
        if missing:
            raise RuntimeError(f"view {view} cannot fill {table}; missing columns: {missing}")
        col_list = ",".join(target_cols)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"DELETE FROM {table}")
            conn.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {view}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM {view}")
            if table == "ml_training_dataset":
                conn.execute(
                    "CREATE INDEX idx_mtd_horizon ON ml_training_dataset(horizon_bars)"
                )
                conn.execute(
                    "CREATE INDEX idx_mtd_sym_ts ON ml_training_dataset(symbol, timestamp)"
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize SQL views into cache tables")
    parser.add_argument("--tables", help=f"comma list of: {', '.join(VIEW_OF)}",
                        default=",".join(DEFAULT_TABLES))
    args = parser.parse_args(argv)

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = [t for t in tables if t not in VIEW_OF]
    if unknown:
        parser.error(f"unknown tables: {unknown}; choose from {sorted(VIEW_OF)}")

    conn = connect()
    with record_run(conn, dataset_type="materialize", symbols=",".join(tables)) as run:
        for table in tables:
            n = materialize(conn, table)
            print(f"{table:<22} <- {VIEW_OF[table]:<26} {n:>8} rows")
            run.rows_inserted += n
    print("done")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
