"""GNN training loop (T022 / US2).

Chronological TRAIN -> VAL -> TEST split per ``config/gnn.yaml``;
early-stops on VAL ROC-AUC. All three seeds (torch / numpy / python
hash) are pinned before any tensor is allocated and recorded in the
artifact's meta sidecar so a re-run is byte-deterministic (spec SC1).

The CLI exposes ``--config``, ``--timestamps`` (optional explicit
list) and ``--epochs`` (overrides the YAML for the smoke test).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config import get_yaml
from ..db import connect
from .constants import UNIVERSE
from .dataset import GNNGraphDataset, build_graph_data
from .model import MODEL_REGISTRY, StockGNN
from .protocol import GNNArtifactMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_everywhere(torch_seed: int, numpy_seed: int, pyhash_seed: str) -> None:
    os.environ["PYTHONHASHSEED"] = str(pyhash_seed)
    random.seed(int(pyhash_seed))
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)


def _select_timestamps(conn: sqlite3.Connection, n: int) -> list[str]:
    """Return ``n`` most-recent distinct timestamps that have at least
    6 universe symbols with a non-flat target_class in v_labels (h=16).

    Filtering on non-flat targets keeps the per-snapshot mask wide
    enough to train; bar-timestamps past market close have no valid
    lookahead, so we naturally skip them.
    """
    placeholders = ",".join("?" * len(UNIVERSE))
    sql = (
        f"SELECT timestamp, COUNT(DISTINCT symbol) as n_sym "
        f"FROM v_labels WHERE horizon_bars=16 AND symbol IN ({placeholders}) "
        f"AND target_class IS NOT NULL AND target_class != 0 "
        f"GROUP BY timestamp HAVING n_sym >= 6 "
        f"ORDER BY timestamp DESC LIMIT ?"
    )
    rows = conn.execute(sql, [*UNIVERSE, n]).fetchall()
    return sorted({r["timestamp"] for r in rows})


def _chronological_split(
    timestamps: Sequence[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[str], list[str], list[str]]:
    """Split an ascending-sorted list into TRAIN/VAL/TEST.

    The split is strict chronological: newest held out for TEST.
    For tiny fixtures (n < 5) we still give every split at least 1
    element so the training loop has a validation signal.
    """
    ts = sorted(timestamps)
    n = len(ts)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        # Collapse: train = all but last, val = last, test = last.
        return ts[:-1], ts[-1:], ts[-1:]
    train = ts[:n_train]
    val = ts[n_train : n_train + n_val]
    test = ts[n_train + n_val :]
    return train, val, test


def _binary_targets(y: torch.Tensor) -> torch.Tensor:
    """Map target_class in {-1, 0, 1} to {0.0, 0.5, 1.0} for BCE."""
    out = torch.zeros_like(y, dtype=torch.float32)
    out[y > 0] = 1.0
    out[y < 0] = 0.0
    return out


def _roc_auc_pair(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Hand-rolled ROC-AUC that doesn't require sklearn at import time.
    Returns 0.5 for the degenerate case (all same label)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-y_score)
    ranks = np.empty_like(y_score, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average-precision PR-AUC, also hand-rolled."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    yt = y_true[order]
    if yt.sum() == 0:
        return 0.0
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    P = int(yt.sum())
    for v in yt:
        if v == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / P)
    # Average precision via the step function (precision at each recall step).
    ap = 0.0
    prev_r = 0.0
    prev_p = 1.0
    for p, r in zip(precisions, recalls):
        ap += (r - prev_r) * prev_p
        prev_r = r
        prev_p = p
    return float(ap)


def _log_loss(y_true: np.ndarray, y_score: np.ndarray, eps: float = 1e-7) -> float:
    y_score = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score)).mean())


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(((np.asarray(y_true) - np.asarray(y_score)) ** 2).mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    conn: sqlite3.Connection,
    *,
    timestamps: Sequence[str] | None = None,
    n_snapshots: int = 50,
    config_path: str | None = None,
    architecture: str = "gcn-32-16-1",
    epochs: int | None = None,
    out_dir: str | Path = "models",
    out_prefix: str = "gnn",
    horizon_bars: int = 16,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train a StockGNN and write ``{prefix}-{ts}-{n}.pt`` + ``.meta.json``.

    Returns a dict with the meta payload + the saved paths so the
    caller (training CLI or test) can assert on it.
    """
    cfg = get_yaml("gnn") or {}
    training_cfg = cfg.get("training", {})
    seeds = training_cfg.get("seeds", {}) or {}
    torch_seed = int(seeds.get("torch", 1337))
    numpy_seed = int(seeds.get("numpy", 1337))
    pyhash_seed = str(seeds.get("python_hash", "1337"))
    early_cfg = training_cfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", True))
    early_metric = str(early_cfg.get("metric", "val_roc_auc"))
    early_patience = int(early_cfg.get("patience", 10))
    if epochs is None:
        epochs = int(training_cfg.get("epochs", 50))
    lr = float(training_cfg.get("learning_rate", 0.005))
    wd = float(training_cfg.get("weight_decay", 0.0005))

    _seed_everywhere(torch_seed, numpy_seed, pyhash_seed)

    if timestamps is None:
        timestamps = _select_timestamps(conn, n_snapshots)
    if len(timestamps) < 3:
        raise RuntimeError(
            f"need at least 3 snapshot timestamps; got {len(timestamps)}"
        )
    train_ts, val_ts, test_ts = _chronological_split(timestamps)

    train_ds = GNNGraphDataset(conn, train_ts, horizon_bars=horizon_bars)
    val_ds = GNNGraphDataset(conn, val_ts, horizon_bars=horizon_bars)
    test_ds = GNNGraphDataset(conn, test_ts, horizon_bars=horizon_bars)

    in_dim = train_ds[0].x.shape[1]
    model = StockGNN(in_dim=in_dim, architecture=architecture)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = early_patience if early_enabled else epochs
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        for i in range(len(train_ds)):
            d = train_ds[i]
            mask = d.mask
            if not mask.any():
                continue
            opt.zero_grad()
            logit, _ = model.forward_with_centrality(d.x, d.edge_index)
            tgt = _binary_targets(d.y)[mask]
            loss = F.binary_cross_entropy_with_logits(logit[mask], tgt)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss = train_loss / max(1, n_batches)
        # ---- VAL ----
        model.eval()
        all_y: list[int] = []
        all_p: list[float] = []
        with torch.no_grad():
            for i in range(len(val_ds)):
                d = val_ds[i]
                mask = d.mask
                if not mask.any():
                    continue
                logit, _ = model.forward_with_centrality(d.x, d.edge_index)
                p = torch.sigmoid(logit[mask]).cpu().numpy()
                y = _binary_targets(d.y[mask]).cpu().numpy().astype(int)
                all_p.extend(p.tolist())
                all_y.extend(y.tolist())
        if all_y:
            val_auc = _roc_auc_pair(np.array(all_y), np.array(all_p))
        else:
            val_auc = 0.5
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = early_patience
        else:
            patience_left -= 1
        if early_enabled and patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- TEST ----
    test_metrics = _evaluate_dataset(model, test_ds)
    elapsed = time.time() - started

    # ---- Persist ----
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    existing = sorted(out_dir_p.glob(f"{out_prefix}-{ts_tag}-*.pt"))
    n_tag = len(existing) + 1
    artifact_path = out_dir_p / f"{out_prefix}-{ts_tag}-{n_tag:04d}.pt"
    meta_path = artifact_path.with_suffix(".meta.json")
    model_version = artifact_path.stem

    # Impute medians from the training set so the service can fill
    # cold-start nodes.
    medians = _impute_medians(train_ds, feature_names)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "in_dim": in_dim,
            "out_dim": model.out_dim,
        },
        artifact_path,
    )

    fnames = list(feature_names) if feature_names else _default_feature_names()
    meta = GNNArtifactMeta(
        model_version=model_version,
        architecture=architecture,  # type: ignore[arg-type]
        topology_version="fixed-1",
        feature_names=fnames,
        impute_medians=medians,
        split_bounds={
            "train_end": train_ts[-1],
            "val_end": val_ts[-1] if val_ts else train_ts[-1],
            "test_end": test_ts[-1] if test_ts else (val_ts[-1] if val_ts else train_ts[-1]),
        },
        test_metrics=test_metrics,
        created_at=_utcnow(),
        torch_seed=torch_seed,
        numpy_seed=numpy_seed,
        python_hash_seed=pyhash_seed,
    )
    meta_path.write_text(json.dumps(meta.model_dump(mode="json"), indent=2), encoding="utf-8")

    # ---- Insert into gnn_model_artifacts + gnn_model_evaluations ----
    _persist_artifact_row(conn, meta, str(artifact_path), fnames, medians)
    _persist_evaluation_row(conn, meta.model_version, "test", test_metrics)

    return {
        "model_version": model.model_version if hasattr(model, "model_version") else model_version,
        "artifact_path": str(artifact_path),
        "meta_path": str(meta_path),
        "meta": meta.model_dump(mode="json"),
        "test_metrics": test_metrics,
        "val_auc": best_val,
        "history": history,
        "elapsed_s": elapsed,
    }


def _default_feature_names() -> list[str]:
    from .build_node_features import feature_columns
    return list(feature_columns())


def _impute_medians(
    ds: GNNGraphDataset, feature_names: Sequence[str] | None = None
) -> dict[str, float]:
    """Compute the per-column median over the training tensors, used
    to fill cold-start nodes at inference time."""
    cols = list(feature_names) if feature_names else _default_feature_names()
    all_x: list[torch.Tensor] = []
    for i in range(len(ds)):
        all_x.append(ds[i].x)
    if not all_x:
        return {c: 0.0 for c in cols}
    cat = torch.cat(all_x, dim=0)
    # Tensor.median over each column.
    med = cat.median(dim=0).values.cpu().tolist()
    out: dict[str, float] = {}
    for i, c in enumerate(cols):
        out[c] = float(med[i]) if i < len(med) else 0.0
    return out


def _evaluate_dataset(
    model: StockGNN, ds: GNNGraphDataset
) -> dict[str, float]:
    """Return {roc_auc, pr_auc, log_loss, brier} on a dataset."""
    model.eval()
    all_y: list[int] = []
    all_p: list[float] = []
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            mask = d.mask
            if not mask.any():
                continue
            logit, _ = model.forward_with_centrality(d.x, d.edge_index)
            p = torch.sigmoid(logit[mask]).cpu().numpy()
            y = _binary_targets(d.y[mask]).cpu().numpy().astype(int)
            all_p.extend(p.tolist())
            all_y.extend(y.tolist())
    if not all_y:
        return {"roc_auc": 0.5, "pr_auc": 0.0, "log_loss": 0.0, "brier": 0.0}
    y_arr = np.array(all_y)
    p_arr = np.array(all_p)
    return {
        "roc_auc": _roc_auc_pair(y_arr, p_arr),
        "pr_auc": _pr_auc(y_arr, p_arr),
        "log_loss": _log_loss(y_arr, p_arr),
        "brier": _brier(y_arr, p_arr),
    }


def _persist_artifact_row(
    conn: sqlite3.Connection,
    meta: GNNArtifactMeta,
    path: str,
    feature_names: Sequence[str],
    medians: dict[str, float],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO gnn_model_artifacts "
        "(model_version, path, architecture, topology_version, "
        " feature_names, impute_medians, split_bounds, test_metrics, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            meta.model_version,
            path,
            meta.architecture,
            meta.topology_version,
            json.dumps(list(feature_names)),
            json.dumps(medians),
            json.dumps(meta.split_bounds),
            json.dumps(meta.test_metrics),
            meta.created_at,
        ),
    )
    conn.commit()


def _persist_evaluation_row(
    conn: sqlite3.Connection,
    model_version: str,
    split: str,
    metrics: dict[str, float],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO gnn_model_evaluations "
        "(model_version, split, metrics_json, created_at) "
        "VALUES (?, ?, ?, ?)",
        (model_version, split, json.dumps(metrics), _utcnow()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Phase 2 GNN")
    ap.add_argument("--config", default="config/gnn.yaml")
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--n-snapshots", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--architecture", default="gcn-32-16-1")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--out-prefix", default="gnn")
    args = ap.parse_args(argv)

    conn = connect(args.db_path) if args.db_path else connect()
    try:
        result = train_model(
            conn,
            n_snapshots=args.n_snapshots,
            config_path=args.config,
            architecture=args.architecture,
            epochs=args.epochs,
            out_dir=args.out_dir,
            out_prefix=args.out_prefix,
        )
    finally:
        conn.close()
    print(json.dumps(
        {
            "model_version": result["model_version"],
            "artifact_path": result["artifact_path"],
            "meta_path": result["meta_path"],
            "test_metrics": result["test_metrics"],
            "val_auc": result["val_auc"],
            "elapsed_s": result["elapsed_s"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
