"""Score the LATEST market state with saved Phase-1 models.

Emits the training doc section 18 output contract as JSON Lines:

    {"timestamp": "...", "underlying": "NVDA", "contract": null,
     "direction_probability": 0.74, "predicted_future_return": null,
     "predicted_future_realized_vol": 0.46, "expected_option_return": null,
     "probability_option_profitable": null, "model_version": "..."}

Usage:
  python -m src.ml.predict                          # auto-load newest artifacts
  python -m src.ml.predict --underlyings NVDA,AAPL --out predictions.jsonl
  python -m src.ml.predict --models models/direction_h4_xgb_clf-....pkl

Leakage note: inference rows are built from the most recent COMPLETED bars only -
the option query deliberately does NOT reuse v_option_training (whose rows passed
a forward-window leak guard, i.e. they are historical by construction).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..db import connect
from .dataset import _param
from .features import OPTION_FEATURES, UNDERLYING_FEATURES

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# Latest completed bar per symbol, with the full v2 feature row.
_UNDERLYING_LATEST_SQL = """
SELECT {cols}
FROM (SELECT f.*, ROW_NUMBER() OVER (PARTITION BY f.symbol ORDER BY f.timestamp DESC) AS rn
      FROM v_features_underlying_v2 f) x
WHERE x.rn = 1
"""

# Latest bar per SELECTED contract, features rebuilt exactly like
# sql/22_view_option_training.sql's `ob` CTE but WITHOUT any forward filter.
# NB: windows are computed over FULL history first; only afterwards do we keep
# each contract's newest bar (filtering earlier would collapse the vol window).
_OPTION_LATEST_SQL = """
WITH bars AS (
    SELECT b.contract_symbol,
           b.timestamp,
           b.close                                        AS opt_close,
           b.volume                                       AS option_volume,
           oc.underlying_symbol                           AS symbol,
           oc.strike_price,
           CASE WHEN oc.option_type = 'call' THEN 1 ELSE 0 END AS is_call,
           julianday(oc.expiration_date)
               - julianday(substr(b.timestamp, 1, 10))    AS days_to_expiry,
           b.close * 1.0 / NULLIF(LAG(b.close) OVER wo, 0) - 1 AS opt_r1,
           ROW_NUMBER() OVER wb                           AS rn
    FROM option_bars b
    JOIN option_contracts oc USING (contract_symbol)
    WINDOW wo  AS (PARTITION BY b.contract_symbol ORDER BY b.timestamp),
           wb  AS (PARTITION BY b.contract_symbol ORDER BY b.timestamp DESC)
),
agg AS (
    SELECT *,
           COUNT(opt_r1) OVER v16                         AS n16,
           SUM(opt_r1 * opt_r1) OVER v16                  AS s2,
           SUM(opt_r1) OVER v16                           AS s1
    FROM bars
    WINDOW v16 AS (PARTITION BY contract_symbol ORDER BY timestamp ROWS BETWEEN 15 PRECEDING AND CURRENT ROW)
)
SELECT g.contract_symbol, g.symbol, g.timestamp,
       g.opt_r1                                             AS option_return_1,
       CASE WHEN g.n16 = 16 THEN SQRT(MAX(g.s2 - g.s1 * g.s1 / 16.0, 0) / 16.0) END
                                                            AS option_volatility_16,
       g.opt_close                                          AS option_close_t,
       g.is_call, g.days_to_expiry,
       g.strike_price * 1.0 / NULLIF(ub.close, 0)           AS moneyness,
       LN(g.strike_price * 1.0 / NULLIF(ub.close, 0))       AS log_moneyness,
       -- 27th feature the trained option_h4 artifact expects. Pulled
       -- from the option bar's raw print volume; LN(0+1) is well-defined
       -- for zero-volume contracts (matches the impute_median ~1.10
       -- the artifact carries for this column).
       LN(NULLIF(g.option_volume, 0) + 1)                   AS option_log_volume,
       fu.{u_cols}
FROM agg g
JOIN underlying_bars ub ON ub.symbol = g.symbol AND ub.timestamp = g.timestamp
LEFT JOIN v_features_underlying_v2 fu
     ON fu.symbol = g.symbol AND fu.timestamp = g.timestamp
WHERE g.rn = 1
"""


def _latest_artifacts(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    # newest artifact per (task, horizon, head-kind) - a task's clf AND reg heads
    # must BOTH survive because they answer different halves of the contract.
    picks: dict[tuple[str, int, str], Path] = {}
    for p in sorted(MODEL_DIR.glob("*.pkl")):
        stem = p.name                       # e.g. direction_h4_xgb_clf-20260826-...
        head = stem.split("-")[0]           # direction_h4_xgb_clf
        parts = head.split("_")             # -> [task, hNN, model, clf|reg]
        if len(parts) < 4:
            continue
        task, hz, kind = parts[0], int(parts[1][1:]), parts[3]
        key = (task, hz, kind)
        if key not in picks or p.name > picks[key].name:
            picks[key] = p
    return sorted(picks.values())


def _design(df: pd.DataFrame, feats: list[str], medians: dict) -> np.ndarray:
    med = pd.Series(medians)
    return df[feats].fillna(med).to_numpy(dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score latest state with Phase-1 models")
    ap.add_argument("--models", nargs="*", default=None, help="explicit .pkl paths")
    ap.add_argument("--underlyings", default=None,
                    help="comma list to restrict underlying rows (default: all)")
    ap.add_argument("--out", default=None, help="JSONL output path (default: stdout)")
    args = ap.parse_args(argv)

    conn = connect()
    tau = _param(conn, "direction_threshold")
    cost = _param(conn, "option_cost_roundtrip")

    artifacts = []
    for path in _latest_artifacts(args.models or []):
        blob = joblib.load(path)
        artifacts.append({"path": path, "blob": blob, "version": path.stem})
        # status -> stderr: stdout is pure JSONL (the doc section 18 contract)
        print(f"loaded {path.name} ({blob['task']}, kind={blob['kind']})", file=sys.stderr)

    under_df = None
    opt_df = None
    # one record per (task-group, timestamp, underlying, contract); option clf+reg
    # heads share a record, different horizons do NOT (they answer different Qs).
    groups: dict[tuple, dict] = {}

    for art in artifacts:
        blob = art["blob"]
        task = blob["task"]
        head = art["path"].name.split("-")[0]
        horizon = int(head.split("_")[1][1:])
        feats, medians = blob["feature_names"], blob["impute_medians"]

        if task in ("direction", "rv") and under_df is None:
            cols = ", ".join(f"x.{c}" for c in ["symbol", "timestamp", *UNDERLYING_FEATURES])
            under_df = pd.read_sql_query(
                _UNDERLYING_LATEST_SQL.format(cols=cols), conn)
        if task == "option" and opt_df is None:
            u_cols = ", ".join(UNDERLYING_FEATURES)   # template adds the fu. prefix
            opt_df = pd.read_sql_query(_OPTION_LATEST_SQL.format(u_cols=u_cols), conn)
            opt_df["option_log_volume"] = 0.0   # raw print volume is not a live feature

        frame = under_df if task in ("direction", "rv") else opt_df
        if args.underlyings:
            want = {s.strip().upper() for s in args.underlyings.split(",")}
            frame = frame[frame["symbol"].isin(want)].reset_index(drop=True)
        X = _design(frame, feats, medians)
        preds = blob["model"].predict_proba(X) if blob["kind"] == "clf" \
            else blob["model"].predict(X)
        preds = np.asarray(preds, dtype=float)
        if blob["kind"] == "clf" and preds.ndim == 2:
            preds = preds[:, 1]

        gkey = (task, horizon)
        for i, (_, row) in enumerate(frame.iterrows()):
            key = (gkey, row["timestamp"], row["symbol"], row.get("contract_symbol"))
            rec = groups.setdefault(key, {
                "timestamp": row["timestamp"], "underlying": row["symbol"],
                "contract": row.get("contract_symbol"), "horizon_bars": horizon,
                "direction_probability": None,
                "predicted_future_return": None,
                "predicted_future_realized_vol": None,
                "expected_option_return": None,
                "probability_option_profitable": None,
                "tau": tau, "assumed_roundtrip_cost": cost,
                "model_version": art["version"],
            })
            if task == "direction":
                rec["direction_probability"] = round(float(preds[i]), 4)
            elif task == "rv":
                rec["predicted_future_realized_vol"] = round(float(preds[i]), 6)
            elif task == "option" and blob["kind"] == "reg":
                rec["expected_option_return"] = round(float(preds[i]), 4)
            else:
                rec["probability_option_profitable"] = round(float(preds[i]), 4)

    lines = list(groups.values())
    lines.sort(key=lambda r: (r["underlying"], r["contract"] or "", r["horizon_bars"]))
    payload = "".join(json.dumps(r) + "\n" for r in lines)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"{len(lines)} predictions -> {args.out}", file=sys.stderr)
    else:
        print(payload.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
