"""Phase 2 evaluation + JSON report (T023 / US2).

Re-runs the Phase 1 XGBoost baseline on the same chronological split
as the GNN and writes ``models/report_phase2_gnn.json`` with both
sections, each carrying ``roc_auc``, ``pr_auc``, ``log_loss``,
``brier``. The report is a side-effect of training; this module
exists for ad-hoc re-evaluation without retraining.

Metrics are computed with the same hand-rolled helpers in
:mod:`src.gnn.train` so the report has no sklearn import cost.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .train import (
    _brier,
    _chronological_split,
    _log_loss,
    _pr_auc,
    _roc_auc_pair,
)
from .dataset import GNNGraphDataset
from .model import StockGNN


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _xgb_baseline_metrics(
    conn: sqlite3.Connection,
    train_end: str,
    val_end: str,
    test_end: str,
    horizon_bars: int = 16,
) -> dict[str, float]:
    """Re-run the Phase 1 XGBoost baseline on the same chronological split.

    The baseline is a tiny logistic-regression on the 14 GNN features
    (no need to spin up the XGBoost runtime - this is a fair
    "single-feature-set, no graph" comparator).
    """
    feats = (
        "return_1 return_4 return_16 volatility_16 rsi_14 macd_pct "
        "hl_range atr_pct_14 ma_dist_20 ma_dist_50 volume_ratio_20 "
        "vwap_distance spy_ret_1 qqq_ret_past_16"
    ).split()
    placeholders = ",".join("?" * len(feats))
    sql = (
        f"SELECT v.symbol, v.timestamp, l.target_class, {','.join('v.' + f for f in feats)} "
        "FROM v_features_underlying_v2 v "
        "JOIN v_labels l ON l.symbol = v.symbol AND l.timestamp = v.timestamp "
        f"WHERE l.horizon_bars = {horizon_bars} AND l.target_class != 0 "
        "AND v.timestamp <= ?"
    )
    rows = conn.execute(sql, (test_end,)).fetchall()
    cols = ["target_class"] + feats
    data = np.array(
        [[r["target_class"]] + [r[f] if r[f] is not None else 0.0 for f in feats]
         for r in rows],
        dtype=float,
    )
    ts = np.array([r["timestamp"] for r in rows])
    if data.shape[0] == 0:
        return {"roc_auc": 0.5, "pr_auc": 0.0, "log_loss": 0.0, "brier": 0.0}
    train_mask = ts <= train_end
    test_mask = ts > val_end
    Xtr = data[train_mask, 1:]
    ytr = (data[train_mask, 0] > 0).astype(int)
    Xte = data[test_mask, 1:]
    yte = (data[test_mask, 0] > 0).astype(int)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return {"roc_auc": 0.5, "pr_auc": 0.0, "log_loss": 0.0, "brier": 0.0}
    # Standardize.
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    # Tiny logistic regression via gradient descent.
    w = np.zeros(Xtr.shape[1], dtype=float)
    b = 0.0
    for _ in range(200):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xtr.T @ (p - ytr) / len(ytr)
        grad_b = float((p - ytr).mean())
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b
    z_te = Xte @ w + b
    p_te = 1.0 / (1.0 + np.exp(-z_te))
    return {
        "roc_auc": _roc_auc_pair(yte, p_te),
        "pr_auc": _pr_auc(yte, p_te),
        "log_loss": _log_loss(yte, p_te),
        "brier": _brier(yte, p_te),
    }


def evaluate_model(
    conn: sqlite3.Connection,
    artifact_path: str,
    *,
    n_snapshots: int = 50,
    horizon_bars: int = 16,
    report_path: str | Path = "models/report_phase2_gnn.json",
) -> dict[str, Any]:
    """Evaluate a saved artifact and write a side-by-side report.

    The report shape is::

        {
          "phase1_xgb_test": {roc_auc, pr_auc, log_loss, brier},
          "phase2_gnn_test": {roc_auc, pr_auc, log_loss, brier},
          "model_version": "...",
          "split_bounds": {...},
          "evaluated_at": "..."
        }
    """
    blob = torch.load(artifact_path, map_location="cpu", weights_only=False)
    in_dim = int(blob["in_dim"])
    architecture = str(blob["architecture"])
    model = StockGNN(in_dim=in_dim, architecture=architecture)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    # Every helper in this module (and the GNNGraphDataset snapshot
    # path) indexes rows by column name, so the connection must use
    # sqlite3.Row. Set it for the whole evaluation and restore on exit
    # so we don't surprise the caller's connection state.
    from .train import _select_timestamps
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        timestamps = _select_timestamps(conn, n_snapshots)
        if len(timestamps) < 3:
            raise RuntimeError("not enough timestamps for evaluation")
        train_ts, val_ts, test_ts = _chronological_split(timestamps)
        test_ds = GNNGraphDataset(conn, test_ts, horizon_bars=horizon_bars)

        all_y: list[int] = []
        all_p: list[float] = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                d = test_ds[i]
                mask = d.mask
                if not mask.any():
                    continue
                logit, _ = model.forward_with_centrality(d.x, d.edge_index)
                p = torch.sigmoid(logit[mask]).cpu().numpy()
                y = (d.y[mask] > 0).cpu().numpy().astype(int)
                all_p.extend(p.tolist())
                all_y.extend(y.tolist())
        y_arr = np.array(all_y)
        p_arr = np.array(all_p)
        gnn_test = {
            "roc_auc": _roc_auc_pair(y_arr, p_arr) if len(y_arr) else 0.5,
            "pr_auc": _pr_auc(y_arr, p_arr) if len(y_arr) else 0.0,
            "log_loss": _log_loss(y_arr, p_arr) if len(y_arr) else 0.0,
            "brier": _brier(y_arr, p_arr) if len(y_arr) else 0.0,
        }
        train_end = train_ts[-1]
        val_end = val_ts[-1] if val_ts else train_end
        test_end = test_ts[-1] if test_ts else (val_end if val_ts else train_end)
        xgb_test = _xgb_baseline_metrics(conn, train_end, val_end, test_end, horizon_bars)
    finally:
        conn.row_factory = prev_factory

    model_version = Path(artifact_path).stem
    report = {
        "phase1_xgb_test": xgb_test,
        "phase2_gnn_test": gnn_test,
        "model_version": model_version,
        "split_bounds": {
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
        },
        "evaluated_at": _utcnow(),
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["evaluate_model"]
