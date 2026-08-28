"""Dataset assembly: SQLite views -> pandas frames with targets attached.

Leakage posture (training doc section 6):
  * features come from trailing-only SQL windows;
  * the ONLY forward columns are label columns read from v_labels /
    v_option_training (which leak-guard rows without a full forward window);
  * chronological splits are cut on unique timestamps - never shuffled;
  * scalers/imputers are fitted on the TRAIN slice only (train.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    OPTION_FEATURES,
    OPTION_KEYS,
    OPTION_WARMUP_COLS,
    UNDERLYING_FEATURES,
    UNDERLYING_KEYS,
    UNDERLYING_WARMUP_COLS,
)

_UNDERLYING_SQL = """
    SELECT fu.symbol, fu.timestamp, {feat_cols},
           l.future_return, l.future_realized_vol
    FROM v_features_underlying_v2 fu
    JOIN v_labels l USING (symbol, timestamp)
    WHERE l.horizon_bars = ?
"""

_OPTION_SQL = """
    SELECT o.contract_symbol, o.symbol, o.timestamp, {feat_cols},
           o.y_option_return, o.y_option_profit, o.underlying_close,
           o.option_close_t
    FROM v_option_training o
    WHERE o.horizon_bars = ?
"""

# log1p tames the heavy tail of raw option print volume (indicative feed).
OPTION_DERIVED = {
    "option_log_volume": lambda df: np.log1p(df["option_volume"].clip(lower=0)),
}


def _param(conn, key: str) -> float:
    row = conn.execute("SELECT value FROM v_params WHERE key = ?", (key,)).fetchone()
    if row is None:
        raise KeyError(f"v_params has no key {key!r} - run `python -m src.db init`")
    return float(row[0])


def load_underlying(conn, horizon: int) -> pd.DataFrame:
    """Rows keyed (symbol, timestamp); labels at the requested horizon."""
    cols = ", ".join(f"fu.{c}" for c in UNDERLYING_FEATURES)
    df = pd.read_sql_query(_UNDERLYING_SQL.format(feat_cols=cols), conn, params=(horizon,))
    df["horizon_bars"] = horizon
    return df


def load_option(conn, horizon: int) -> pd.DataFrame:
    """Rows keyed (contract_symbol, symbol, timestamp); contract-level labels."""
    # raw sources for derived features / diagnostics ride along untrained
    extra = ["option_volume", "option_trade_count", "underlying_close", "option_close_t"]
    seen: dict[str, None] = {}
    for c in [*OPTION_FEATURES, *extra]:
        seen.setdefault(c, None)
    cols = ", ".join(f"o.{c}" for c in seen)
    df = pd.read_sql_query(_OPTION_SQL.format(feat_cols=cols), conn, params=(horizon,))
    for name, fn in OPTION_DERIVED.items():
        df[name] = fn(df)
    df["horizon_bars"] = horizon
    return df


def attach_underlying_targets(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """y_direction per training doc section 2A: 1[future_return > tau]."""
    out = df.copy()
    out["y_direction"] = (out["future_return"] > tau).astype(int)
    out["tau"] = tau
    return out


def apply_warmup_gate(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Drop rows whose indicator windows have not fully warmed up."""
    warmup = UNDERLYING_WARMUP_COLS if dataset == "underlying" else OPTION_WARMUP_COLS
    keep = df["volatility_16"].notna() & df[warmup].notna().all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  warmup gate: dropping {dropped} pre-warmup rows "
              f"({100.0 * dropped / len(df):.1f}%)")
    return df.loc[keep].reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Cut on unique sorted timestamps (doc section 8: oldest->newest).

    Returns dict of key-frames + X/y slices are assembled by callers; boundaries
    are returned so they can be persisted into model metadata.
    """
    stamps = np.sort(df["timestamp"].unique())
    n = len(stamps)
    i_train = int(n * train_frac)
    i_val = i_train + int(n * val_frac)
    bounds = {
        "train_end": stamps[i_train - 1],
        "val_start": stamps[i_train],
        "val_end": stamps[i_val - 1],
        "test_start": stamps[i_val],
        "test_end": stamps[-1],
    }
    parts = {
        "train": df[df["timestamp"] <= bounds["train_end"]].reset_index(drop=True),
        "val": df[(df["timestamp"] >= bounds["val_start"])
                  & (df["timestamp"] <= bounds["val_end"])].reset_index(drop=True),
        "test": df[df["timestamp"] >= bounds["test_start"]].reset_index(drop=True),
    }
    return parts, bounds


def walk_forward_folds(df: pd.DataFrame, folds: int = 4):
    """Expanding-window folds over unique timestamps (doc section 8).

    Fold i trains on chunks [0..i) and evaluates chunk i; fold 0 is skipped so
    every training window spans at least half the history.
    """
    stamps = np.sort(df["timestamp"].unique())
    edges = np.linspace(0, len(stamps), folds + 1).astype(int)
    for i in range(1, folds):
        train_end, test_end = stamps[edges[i] - 1], stamps[edges[i + 1] - 1]
        train = df[df["timestamp"] <= train_end]
        test = df[(df["timestamp"] > train_end) & (df["timestamp"] <= test_end)]
        yield {"fold": i, "train": train.reset_index(drop=True),
               "test": test.reset_index(drop=True)}
