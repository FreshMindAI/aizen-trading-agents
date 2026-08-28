"""Phase-1 model trainers (training doc sections 3, 8, 9, 12).

Usage:
  python -m src.ml.train --task direction --dataset underlying --horizon 16
  python -m src.ml.train --task rv        --dataset underlying --horizon 16
  python -m src.ml.train --task option    --dataset option    --horizon 16
  python -m src.ml.train --task ... --walk-forward          # doc section 8 protocol

Protocol enforced here:
  * chronological TRAIN -> VALIDATION -> TEST split (oldest first, TEST newest);
  * imputation medians / scaler fitted on TRAIN only (doc section 6);
  * thresholds (tau, round-trip cost) frozen before touching TEST;
  * naive/majority baselines always reported next to LogReg/XGBoost;
  * artifacts persisted under models/<version>.{pkl,meta.json} with full
    provenance, and every run logged into data_runs.

Hyperparameters are FROZEN defaults chosen before any test evaluation - tune on
validation only if you must, then bump MODEL_VERSION_TAG.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ..db import connect, record_run
from . import dataset as ds
from . import metrics as mx
from .dataset import _param
from .features import OPTION_FEATURES, UNDERLYING_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

SEED = 42

# Frozen hyperparameters (doc section 6: freeze before touching final test period).
XGB_COMMON = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=25,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)
XGB_CLF_PARAMS = {**XGB_COMMON, "eval_metric": "logloss", "early_stopping_rounds": 50}
XGB_REG_PARAMS = {**XGB_COMMON, "eval_metric": "rmse", "early_stopping_rounds": 50}
RIDGE_PARAMS = {"alpha": 1.0}
LOGREG_PARAMS = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced",
                 "random_state": SEED}

LABEL_SOURCE_NOTE = (
    "Option labels use indicative-feed BAR CLOSE (historical quotes unavailable "
    "on free tier) - documented per training doc section 5."
)


# ---------------------------------------------------------------------------
# preprocessing: train-fitted imputation + scaling (never refit on val/test)
# ---------------------------------------------------------------------------
def fit_imputer(train_df: pd.DataFrame, feats: list[str]) -> pd.Series:
    medians = train_df[feats].median(numeric_only=True)
    return medians.fillna(0.0)   # column all-NaN in train -> constant 0


def design_matrix(df: pd.DataFrame, feats: list[str], medians: pd.Series):
    return df[feats].fillna(medians).to_numpy(dtype=np.float64)


class ScaledLinear:
    """Linear model wrapped with a TRAIN-fitted StandardScaler (doc section 6).

    Classifiers expose predict_proba -> POSITIVE-class 1-D vector.
    """

    def __init__(self, est):
        self.scaler = StandardScaler()
        self.est = est

    def fit(self, X, y):
        self.est.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.est.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        proba = self.est.predict_proba(self.scaler.transform(X))
        return proba[:, 1] if proba.ndim == 2 else proba


def _xgb(params, kind: str):
    return XGBClassifier(**params) if kind == "clf" else XGBRegressor(**params)


def _fit_with_early_stop(model, kind, X_tr, y_tr, X_val, y_val):
    """XGBoost early-stops on the validation slice; linear models ignore it."""
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model


def _predict(model, X, kind: str) -> np.ndarray:
    if kind == "clf":
        proba = np.asarray(model.predict_proba(X), dtype=float)
        # xgboost >=3 returns 1-D positive-class probs; older / sklearn return (n, 2)
        return proba[:, 1] if proba.ndim == 2 else proba
    return np.asarray(model.predict(X), dtype=float)


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def run_task(task: str, horizon: int, conn, args) -> None:
    t0 = time.time()
    tau = args.tau if args.tau is not None else _param(conn, "direction_threshold")
    cost = args.cost if args.cost is not None else _param(conn, "option_cost_roundtrip")

    if task == "direction":
        feats = UNDERLYING_FEATURES
        warmup_ds = "underlying"
        df = ds.attach_underlying_targets(ds.load_underlying(conn, horizon), tau)
        clf_target = "y_direction"
    elif task == "rv":
        feats = UNDERLYING_FEATURES
        warmup_ds = "underlying"
        df = ds.load_underlying(conn, horizon)
        clf_target = None
    elif task == "option":
        feats = [*OPTION_FEATURES, "option_log_volume"]
        warmup_ds = "option"
        df = ds.load_option(conn, horizon)
        # recompute profit against the ACTIVE cost assumption so overrides apply
        df["y_option_profit"] = (df["y_option_return"] > cost).astype(int)
        df["cost"] = cost
        clf_target = "y_option_profit"
    else:
        raise SystemExit(f"unknown task {task!r}")

    df = ds.apply_warmup_gate(df, warmup_ds)
    parts, bounds = ds.chronological_split(df, args.split[0], args.split[1])
    tr, va, te = parts["train"], parts["val"], parts["test"]
    print(f"rows: train={len(tr)}  val={len(va)}  test={len(te)}  "
          f"({bounds['train_end']} | {bounds['val_end']} | -> {bounds['test_end']})")
    print(f"frozen params: tau={tau:.4f} cost={cost:.4f} horizon={horizon}")

    medians = fit_imputer(tr, feats)
    X_tr, X_va, X_te = (design_matrix(d, feats, medians) for d in (tr, va, te))

    results: dict = {}
    saved_models: list[str] = []

    # ---------------- classification heads ----------------
    if clf_target:
        y_tr, y_va, y_te = tr[clf_target].to_numpy(), va[clf_target].to_numpy(), te[clf_target].to_numpy()
        print(f"base rate up/profitable: train={y_tr.mean():.3f} "
              f"val={y_va.mean():.3f} test={y_te.mean():.3f}")

        results["majority"] = {
            "val": mx.majority_baseline_metrics(y_va),
            "test": mx.majority_baseline_metrics(y_te),
        }

        models = {}
        lr = ScaledLinear(LogisticRegression(**LOGREG_PARAMS))
        lr.fit(X_tr, y_tr)
        models["logreg"] = lr
        xgb = _xgb(XGB_CLF_PARAMS, "clf")
        _fit_with_early_stop(xgb, "clf", X_tr, y_tr, X_va, y_va)
        models["xgb"] = xgb

        for name, model in models.items():
            p_va, p_te = _predict(model, X_va, "clf"), _predict(model, X_te, "clf")
            results[name] = {"val": mx.classification_metrics(y_va, p_va),
                             "test": mx.classification_metrics(y_te, p_te)}
        save_model(models["xgb"], f"{task}_h{horizon}_xgb_clf", args, task, horizon,
                   feats, medians, bounds, {"tau": tau, "cost": cost}, results["xgb"])
        saved_models.append(f"{task}_h{horizon}_xgb_clf")

        if task == "option":
            _option_extras(results, models["xgb"], te, X_te, cost)

    # ---------------- regression heads ----------------
    if task == "rv":
        target = "future_realized_vol"
        y_tr, y_va, y_te = (d[target].to_numpy() for d in (tr, va, te))
        # naive baselines: history mean, and persistence (today's realized vol)
        results["mean_naive"] = {
            "val": mx.regression_metrics(y_va, np.full(len(y_va), y_tr.mean())),
            "test": mx.regression_metrics(y_te, np.full(len(y_te), y_tr.mean())),
        }
        persist_cols = ["volatility_16"]
        p_med = fit_imputer(tr, persist_cols)
        persist_te = design_matrix(te, persist_cols, p_med).ravel()
        results["persistence_naive"] = {
            "val": mx.regression_metrics(y_va, design_matrix(va, persist_cols, p_med).ravel()),
            "test": mx.regression_metrics(y_te, persist_te),
        }
        ridge = ScaledLinear(Ridge(**RIDGE_PARAMS))
        ridge.fit(X_tr, y_tr)
        xgbr = _xgb(XGB_REG_PARAMS, "reg")
        _fit_with_early_stop(xgbr, "reg", X_tr, y_tr, X_va, y_va)
        results["ridge"] = {"val": mx.regression_metrics(y_va, ridge.predict(X_va)),
                            "test": mx.regression_metrics(y_te, ridge.predict(X_te))}
        results["xgb"] = {"val": mx.regression_metrics(y_va, xgbr.predict(X_va)),
                          "test": mx.regression_metrics(y_te, xgbr.predict(X_te))}
        save_model(xgbr, f"{task}_h{horizon}_xgb_reg", args, task, horizon, feats,
                   medians, bounds, {"tau": tau, "cost": cost}, results["xgb"])
        saved_models.append(f"{task}_h{horizon}_xgb_reg")

    if task == "option":
        target = "y_option_return"
        y_tr, y_va, y_te = (d[target].to_numpy() for d in (tr, va, te))
        results["zero_naive"] = {
            "val": mx.regression_metrics(y_va, np.zeros(len(y_va))),
            "test": mx.regression_metrics(y_te, np.zeros(len(y_te))),
        }
        ridge = ScaledLinear(Ridge(**RIDGE_PARAMS))
        ridge.fit(X_tr, y_tr)
        xgbr = _xgb(XGB_REG_PARAMS, "reg")
        _fit_with_early_stop(xgbr, "reg", X_tr, y_tr, X_va, y_va)
        results["ret_ridge"] = {"val": mx.regression_metrics(y_va, ridge.predict(X_va)),
                                "test": mx.regression_metrics(y_te, ridge.predict(X_te))}
        results["ret_xgb"] = {"val": mx.regression_metrics(y_va, xgbr.predict(X_va)),
                              "test": mx.regression_metrics(y_te, xgbr.predict(X_te))}
        save_model(xgbr, f"{task}_h{horizon}_xgb_reg", args, task, horizon, feats,
                   medians, bounds, {"tau": tau, "cost": cost}, results["ret_xgb"])
        saved_models.append(f"{task}_h{horizon}_xgb_reg")

    _report(task, horizon, results)

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_run = {
        "task": task, "dataset": args.dataset, "horizon_bars": horizon,
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "split_bounds": bounds, "frozen_params": {"tau": tau, "roundtrip_cost": cost},
        "label_source_note": LABEL_SOURCE_NOTE if task == "option" else None,
        "results": results, "saved_models": saved_models,
        "hyperparams": {"xgb_clf": XGB_CLF_PARAMS, "xgb_reg": XGB_REG_PARAMS,
                        "logreg": LOGREG_PARAMS, "ridge": RIDGE_PARAMS},
        "completed_at_utc": stamp, "elapsed_s": round(time.time() - t0, 1),
    }
    out = MODEL_DIR / f"report_{task}_h{horizon}.json"
    MODEL_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(meta_run, indent=2, default=str))
    print(f"metrics JSON -> {out}")
    if args.walk_forward:
        walk_forward(task, horizon, conn, args)


def _option_extras(results, clf_model, te: pd.DataFrame, X_te, cost: float) -> None:
    """Ranking quality + net-of-cost selection backtest on TEST (doc section 12)."""
    prob = _predict(clf_model, X_te, "clf")
    scoped = te.assign(prob=prob)
    results["ranking_by_prob"] = mx.ranking_quality(scoped, "prob", "y_option_return")
    results["topk_backtest_prob"] = mx.topk_backtest(scoped, "prob", "y_option_return", cost)


def walk_forward(task: str, horizon: int, conn, args) -> None:
    """Doc section 8: expanding-window folds after the single-split baseline."""
    print(f"\n== walk-forward ({args.folds} folds, primary model) ==")
    fold_rows = []
    if task == "direction":
        tau = args.tau if args.tau is not None else _param(conn, "direction_threshold")
        full = ds.attach_underlying_targets(ds.load_underlying(conn, horizon), tau)
        target, kind, feats, warm = "y_direction", "clf", UNDERLYING_FEATURES, "underlying"

        def score(y_true, pred):
            m = mx.classification_metrics(y_true, pred)
            return {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]}
    elif task == "rv":
        full, target, kind, feats, warm = (
            ds.load_underlying(conn, horizon), "future_realized_vol", "reg",
            UNDERLYING_FEATURES, "underlying")

        def score(y_true, pred):
            m = mx.regression_metrics(y_true, pred)
            return {"mae": m["mae"], "spearman_rho": m["spearman_rho"]}
    else:
        cost = args.cost if args.cost is not None else _param(conn, "option_cost_roundtrip")
        full = ds.load_option(conn, horizon)
        full["y_option_profit"] = (full["y_option_return"] > cost).astype(int)
        target, kind, feats, warm = ("y_option_profit", "clf",
                                     [*OPTION_FEATURES, "option_log_volume"], "option")

        def score(y_true, pred):
            m = mx.classification_metrics(y_true, pred)
            return {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]}

    full = ds.apply_warmup_gate(full, warm)
    for fold in ds.walk_forward_folds(full, args.folds):
        tr, te = fold["train"], fold["test"]
        med = fit_imputer(tr, feats)
        X_tr, X_te = design_matrix(tr, feats, med), design_matrix(te, feats, med)
        # internal validation slice for early stopping: last 15% of fold-train time
        stamps = np.sort(tr["timestamp"].unique())
        cut = stamps[int(len(stamps) * 0.85)]
        va_mask = tr["timestamp"] > cut
        X_va, y_va = X_tr[np.asarray(va_mask)], tr.loc[va_mask, target].to_numpy()
        X_tr_f = X_tr[~np.asarray(va_mask)]
        y_tr_f = tr.loc[~va_mask, target].to_numpy()

        model = _xgb(XGB_CLF_PARAMS if kind == "clf" else XGB_REG_PARAMS, kind)
        _fit_with_early_stop(model, kind, X_tr_f, y_tr_f, X_va, y_va)
        s = score(te[target].to_numpy(), _predict(model, X_te, kind))
        fold_rows.append({"fold": fold["fold"], "train_rows": len(tr),
                          "test_rows": len(te), "train_end": str(cut), **s})
        print(f"  fold {fold['fold']}: {s}")

    agg = {k: {"mean": float(np.mean([r[k] for r in fold_rows])),
               "std": float(np.std([r[k] for r in fold_rows]))}
           for k in fold_rows[0] if k.startswith(("roc", "pr_", "mae", "spearman"))}
    out = MODEL_DIR / f"walkforward_{task}_h{horizon}.json"
    out.write_text(json.dumps({"folds": fold_rows, "aggregate": agg}, indent=2))
    print("  aggregate:", json.dumps(agg))
    print(f"  -> {out}")


def save_model(model, basename: str, args, task: str, horizon: int, feats,
               medians, bounds, frozen: dict, metrics: dict) -> None:
    import joblib

    MODEL_DIR.mkdir(exist_ok=True)
    version = f"{basename}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = MODEL_DIR / f"{version}.pkl"
    joblib.dump({"model": model, "feature_names": feats, "impute_medians": medians.to_dict(),
                 "task": task, "kind": "clf" if task != "rv" and basename.endswith("_clf") else "reg"},
                path)
    meta = {
        "model_version": version, "path": str(path), "task": task,
        "dataset": args.dataset, "horizon_bars": horizon, "features": feats,
        "impute_medians": medians.to_dict(), "split_bounds": bounds,
        "frozen_params": frozen, "test_metrics": metrics,
        "label_source_note": LABEL_SOURCE_NOTE if task == "option" else None,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (MODEL_DIR / f"{version}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"model -> {path}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _report(task: str, horizon: int, results: dict) -> None:
    print(f"\n== {task} h={horizon}: TEST metrics ==")
    for name, res in results.items():
        if name == "topk_backtest_prob":
            bt = res
            print(f"  {name:<18} total={bt['total_return']:.3f} win={bt['win_rate']:.3f} "
                  f"sharpe={bt['sharpe_per_trade']:.3f} trades={bt['trades']} "
                  f"maxDD={bt['max_drawdown']:.3f} PF={bt['profit_factor']:.2f}")
            continue
        if name == "ranking_by_prob":
            print(f"  {name:<18} mean_rho={res['mean_rho']:.4f} "
                  f"median_rho={res['median_rho']:.4f} groups={res['groups']}")
            continue
        if "test" not in res:
            continue
        head = ["roc_auc", "pr_auc", "log_loss", "brier", "mae", "rmse", "spearman_rho"]
        cells = "  ".join(f"{k}={_fmt(res['test'][k])}" for k in head if k in res["test"])
        print(f"  {name:<18} {cells}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train Phase-1 models")
    p.add_argument("--task", required=True, choices=["direction", "rv", "option"])
    p.add_argument("--dataset", choices=["underlying", "option"], default=None)
    p.add_argument("--horizon", type=int, choices=[4, 16], default=16)
    p.add_argument("--split", type=float, nargs=2, default=[0.70, 0.15],
                   metavar=("TRAIN", "VAL"))
    p.add_argument("--tau", type=float, default=None,
                   help="direction threshold (default: v_params.direction_threshold)")
    p.add_argument("--cost", type=float, default=None,
                   help="option round-trip cost (default: v_params.option_cost_roundtrip)")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--folds", type=int, default=4)
    args = p.parse_args(argv)
    if args.dataset is None:
        args.dataset = "underlying" if args.task in ("direction", "rv") else "option"

    conn = connect()
    with record_run(conn, dataset_type=f"ml_train_{args.task}",
                    timeframe=f"{args.horizon}bars",
                    feed="sqlite_views") as run:
        run.rows_inserted = 0
        run_task(args.task, args.horizon, conn, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
