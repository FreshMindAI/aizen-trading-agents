"""Evaluation metrics per training doc section 12.

All functions return plain dicts of JSON-serialisable numbers so run metadata can
embed them verbatim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # scipy ships with scikit-learn


# -- classification (direction / option_profit) ------------------------------
def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, prob)),
        "precision@0.5": float(precision_score(y_true, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, pred, zero_division=0)),
        "base_rate": float(y_true.mean()),
    }

    # calibration curve: predicted vs observed in 10 probability bins
    bins = np.clip((prob * 10).astype(int), 0, 9)
    calib = []
    for b in range(10):
        m = bins == b
        if m.sum():
            calib.append({"bin_mid": (b + 0.5) / 10,
                          "mean_pred": float(prob[m].mean()),
                          "observed": float(y_true[m].mean()),
                          "n": int(m.sum())})
    out["calibration"] = calib
    return out


def majority_baseline_metrics(y_true) -> dict:
    """Always predict the train base rate - the doc's 'naive/majority baseline'."""
    p = float(np.mean(y_true))
    return classification_metrics(y_true, np.full(len(y_true), p))


# -- regression (future RV / option expected return) -------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    rho = stats.spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else float("nan")
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spearman_rho": float(rho),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }

    # error by volatility regime: terciles of the TRUE value (doc section 12)
    terciles = pd.qcut(pd.Series(y_true), 3, labels=["low", "mid", "high"])
    out["error_by_regime"] = {
        str(level): {"mae": float(np.abs(err[terciles == level]).mean()),
                     "n": int((terciles == level).sum())}
        for level in ("low", "mid", "high")
    }
    return out


# -- option-specific ----------------------------------------------------------
def ranking_quality(df: pd.DataFrame, score_col: str, true_col: str,
                    group_col: str = "timestamp", min_group: int = 5) -> dict:
    """Mean per-timestamp Spearman rank corr across contracts (doc: 'ranking
    quality across contracts') - the metric that matters when an agent picks
    among same-moment candidates."""
    rhos = []
    for _, g in df.groupby(group_col):
        if len(g) >= min_group and g[true_col].std() > 0:
            rho = stats.spearmanr(g[score_col], g[true_col]).statistic
            if not np.isnan(rho):
                rhos.append(rho)
    return {"groups": len(rhos),
            "mean_rho": float(np.mean(rhos)) if rhos else float("nan"),
            "median_rho": float(np.median(rhos)) if rhos else float("nan")}


def topk_backtest(df: pd.DataFrame, score_col: str, ret_col: str,
                  cost: float, k: int = 3) -> dict:
    """Net-of-cost selection backtest on per-trade returns (doc section 12).

    Each timestamp buys the k highest-scored contracts, holds H bars, books
    realized y_option_return minus round-trip cost. Equal weight per trade;
    per-trade compounding, no leverage.
    """
    picks = (df.sort_values([score_col], ascending=False)
               .groupby("timestamp", sort=True)
               .head(k))
    r = picks.sort_values("timestamp")[ret_col].to_numpy() - cost
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    wins = r > 0
    gross_win, gross_loss = r[wins].sum(), -r[~wins].sum()
    return {
        "trades": int(len(r)),
        "total_return": float(eq[-1] - 1.0) if len(r) else 0.0,
        "avg_trade_return": float(r.mean()) if len(r) else 0.0,
        "sharpe_per_trade": float(r.mean() / r.std()) if len(r) > 1 and r.std() > 0 else 0.0,
        "max_drawdown": float(dd.min()) if len(r) else 0.0,
        "win_rate": float(wins.mean()) if len(r) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "cost_per_trade": cost,
        "top_k": k,
    }
