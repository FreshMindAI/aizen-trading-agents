"""Register existing GNN model artifacts in `gnn_model_artifacts`.

Why this exists
---------------
``InferenceService.gnn_output`` (src/agents/inference.py:335) looks up
the newest ``gnn_model_artifacts`` row and instantiates a
:class:`src.gnn.service.GNNService` from it. When the table is empty
the service falls back to :class:`src.gnn.stub.StubGNNService`, whose
contract is to return ``bias=0.0`` and ``centrality=0.0`` for every
symbol (src/gnn/stub.py:38-42).

The 2026-09-03 live cron run showed every cycle logging
``gnn: success=False, model_version=stub-1, n_nodes=10, n_edges=0`` in
``models/cycle_traces/`` even though ``models/gnn-20260829-0004.pt``
and ``models/gatv2-news-20260829-0002.pt`` exist on disk. The DB has
zero rows in ``gnn_model_artifacts`` so the loader cannot find them.

This script backfills the table by reading the ``.pt`` + ``.meta.json``
file pairs in ``models/`` and INSERTing one row per artifact. The
training pipeline (``src/gnn/train.py:_persist_artifact_row``) does
the same insert on every successful training run; this script is the
"register without retraining" path for artifacts that were produced
in an earlier session, on a different DB, or on a different machine.

Usage
-----
::

    # default: aizen.db in repo root
    python -m scripts.register_gnn_artifacts

    # explicit DB
    python -m scripts.register_gnn_artifacts --db /path/to/aizen.db

    # explicit models dir
    python -m scripts.register_gnn_artifacts --models-dir /path/to/models

Idempotent
----------
The insert is ``INSERT OR REPLACE`` keyed on ``model_version``, so
running this script twice in a row is a no-op the second time. Safe
to run on every cron tick.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Repo-relative imports (so this works under ``python -m scripts.register_gnn_artifacts``).
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger("register_gnn_artifacts")

# Match the schemas declared in sql/50_gnn_snapshots.sql and sql/53_gnn_news_columns.sql.
_INSERT_SQL = (
    "INSERT OR REPLACE INTO gnn_model_artifacts "
    "(model_version, path, architecture, topology_version, "
    " feature_names, impute_medians, split_bounds, test_metrics, "
    " created_at, used_news, ablation_fold_id) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _resolve_db_path(explicit: str | None) -> Path:
    """Pick the DB path. Priority: --db arg > AIZEN_DB_PATH env > aizen.db."""
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("AIZEN_DB_PATH")
    if env:
        return Path(env).resolve()
    return (REPO / "aizen.db").resolve()


def _resolve_models_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (REPO / "models").resolve()


def _list_artifact_pairs(models_dir: Path) -> list[tuple[Path, Path]]:
    """Return ``[(meta_path, pt_path), ...]`` for every registered artifact
    in the models directory. The pair must share the same stem
    (``.meta.json`` and ``.pt``)."""
    pairs: list[tuple[Path, Path]] = []
    for meta in sorted(models_dir.glob("*.meta.json")):
        stem = meta.name[: -len(".meta.json")]
        pt = models_dir / f"{stem}.pt"
        if not pt.exists():
            logger.warning("meta found but .pt missing: %s (skipped)", meta.name)
            continue
        pairs.append((meta, pt))
    return pairs


def _row_for_artifact(meta_path: Path, pt_path: Path, default_created_at: str) -> tuple:
    """Read the meta JSON and project it into the row tuple for the
    INSERT. Missing optional fields are filled with empty JSON
    objects so the schema's NOT NULL constraints (where present) hold."""
    meta = json.loads(meta_path.read_text())
    model_version = meta.get("model_version") or pt_path.stem
    architecture = meta.get("architecture") or "unknown"
    topology_version = meta.get("topology_version") or "fixed-1"
    feature_names = meta.get("feature_names") or []
    impute_medians = meta.get("impute_medians") or {}
    split_bounds = meta.get("split_bounds") or {}
    test_metrics = meta.get("test_metrics") or {}
    created_at = meta.get("created_at") or default_created_at
    # 53_gnn_news_columns.sql: news-augmented artifacts set used_news=1.
    # Detect from the model name OR the topology_version — different
    # checkpoints carry the signal in different fields (e.g. the
    # GATv2-news model has topology_version="fixed-1" but
    # model_version="gatv2-news-..."). Both checks together cover the
    # legacy + new naming.
    used_news = 1 if (
        "news" in model_version.lower()
        or "news" in topology_version.lower()
    ) else 0
    return (
        model_version,
        str(pt_path),
        architecture,
        topology_version,
        json.dumps(list(feature_names)),
        json.dumps(impute_medians),
        json.dumps(split_bounds),
        json.dumps(test_metrics),
        created_at,
        used_news,
        # ablation_fold_id is a later migration; default NULL.
        None,
    )


def register_artifacts(db_path: Path, models_dir: Path) -> int:
    """Backfill ``gnn_model_artifacts``. Returns the number of rows
    upserted. The function is idempotent — running it twice upserts
    the same rows because the insert is ``INSERT OR REPLACE`` keyed
    on ``model_version``."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"DB not found: {db_path}. Pass --db to point at a different file."
        )
    if not models_dir.exists():
        raise FileNotFoundError(
            f"models dir not found: {models_dir}. Pass --models-dir."
        )
    pairs = _list_artifact_pairs(models_dir)
    if not pairs:
        logger.warning("no .meta.json+.pt pairs found under %s", models_dir)
        return 0
    # Fallback created_at for legacy meta files that don't carry one.
    default_created_at = pairs[0][0].stat().st_mtime and _iso_mtime(pairs[0][0]) or "1970-01-01T00:00:00Z"
    conn = sqlite3.connect(str(db_path))
    try:
        # Confirm the table exists; an early abort is friendlier than a
        # sqlite "no such table" error after a 0-row loop.
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='gnn_model_artifacts'"
        )
        if cur.fetchone() is None:
            raise RuntimeError(
                f"gnn_model_artifacts table not found in {db_path}. "
                "Run sql/50_gnn_snapshots.sql first."
            )
        n_upserted = 0
        for meta_path, pt_path in pairs:
            row = _row_for_artifact(meta_path, pt_path, default_created_at)
            conn.execute(_INSERT_SQL, row)
            n_upserted += 1
            logger.info("registered: %s (architecture=%s, topology=%s)",
                        row[0], row[2], row[3])
        conn.commit()
    finally:
        conn.close()
    return n_upserted


def _iso_mtime(path: Path) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Register on-disk GNN artifacts into gnn_model_artifacts."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path. Defaults to $AIZEN_DB_PATH or aizen.db in repo root.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing .pt + .meta.json pairs. Defaults to models/.",
    )
    args = parser.parse_args(argv)
    db_path = _resolve_db_path(args.db)
    models_dir = _resolve_models_dir(args.models_dir)
    logger.info("DB: %s", db_path)
    logger.info("models dir: %s", models_dir)
    n = register_artifacts(db_path, models_dir)
    logger.info("upserted %d artifact(s) into gnn_model_artifacts", n)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
