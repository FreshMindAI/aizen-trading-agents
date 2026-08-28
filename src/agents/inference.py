"""Inference service - builds MarketSnapshot from ML predictions + portfolio.

This is the ONLY module that reads from the ML artifact directory. Agents
consume the resulting snapshot; they never load .pkl files themselves.

Why this design?
- The ML layer is a numerical inference component (doc section 3). Treating
  it as a service keeps agents language-only and keeps the artifact-format
  concern in one place.
- The GNN hook is a stub today (returns {}), so agents already work against
  the documented shape; the real GNN can be dropped in without touching
  agent code.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .protocol import (
    MarketSnapshot,
    OptionScore,
    PortfolioPosition,
    Side,
    UnderlyingScore,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# SQL: latest underlying features joined to the latest prediction rows. Mirrors
# src.ml.predict but lives here so agents do not depend on the training-side
# module's import path.
_LATEST_UNDERLYING_SQL = """
SELECT f.symbol,
       f.timestamp,
       COALESCE(d.direction_probability, 0.5)   AS direction_probability,
       COALESCE(d.predicted_future_realized_vol, NULL) AS predicted_future_realized_vol,
       COALESCE(d.gnn_directional_bias, 0.0)    AS gnn_directional_bias,
       COALESCE(d.gnn_centrality, 0.0)          AS gnn_centrality,
       d.model_version
FROM   v_features_underlying_v2 f
LEFT JOIN (
    SELECT u.symbol,
           u.timestamp,
           u.direction_probability,
           u.predicted_future_realized_vol,
           NULL AS gnn_directional_bias,
           NULL AS gnn_centrality,
           u.model_version,
           ROW_NUMBER() OVER (PARTITION BY u.symbol ORDER BY u.timestamp DESC) rn
    FROM   ml_training_dataset u
) d ON d.symbol = f.symbol AND d.rn = 1
ORDER BY f.timestamp DESC
"""

_LATEST_OPTION_SQL = """
SELECT ob.contract_symbol,
       oc.underlying_symbol AS symbol,
       ob.timestamp,
       ob.close              AS opt_close,
       oc.strike_price,
       oc.option_type,
       oc.expiration_date,
       CAST(julianday(oc.expiration_date) - julianday(substr(ob.timestamp, 1, 10)) AS INTEGER) AS dte,
       COALESCE(p.probability_profitable, NULL) AS probability_profitable,
       COALESCE(p.expected_return,         NULL) AS expected_return,
       p.model_version
FROM   option_bars ob
JOIN   option_contracts oc USING (contract_symbol)
LEFT JOIN (
    SELECT contract_symbol, probability_profitable, expected_return, model_version
    FROM   (SELECT *, ROW_NUMBER() OVER (PARTITION BY contract_symbol
                                         ORDER BY timestamp DESC) rn
            FROM   ml_training_dataset
            WHERE  contract_symbol IS NOT NULL)
    WHERE rn = 1
) p USING (contract_symbol)
WHERE  ob.timestamp = (SELECT MAX(timestamp) FROM option_bars)
ORDER BY ob.contract_symbol
"""


@dataclass
class InferenceService:
    """Builds MarketSnapshot from the SQLite system of record."""

    conn: sqlite3.Connection
    universe: list[str]
    model_dir: Path = MODEL_DIR
    topology_version: str = "stub-1"

    # ---- public API ----
    def build_snapshot(self) -> MarketSnapshot:
        underlying_rows = self._load_underlying_predictions()
        option_rows = self._load_option_predictions()
        portfolio = self._load_portfolio()
        return MarketSnapshot(
            timestamp=underlying_rows[0].timestamp if underlying_rows else _now_iso(),
            underlyings=underlying_rows,
            options=option_rows,
            portfolio=portfolio,
            account_equity=self._account_equity(),
            account_cash=self._account_cash(),
        )

    def gnn_output(self) -> dict[str, Any]:
        """Returns Phase-2 graph output. Stub until GNN ships."""
        path = self.model_dir / "gnn_snapshot.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                pass
        return {"version": self.topology_version, "edges": [], "node_features": {}}

    # ---- helpers ----
    def _load_underlying_predictions(self) -> list[UnderlyingScore]:
        try:
            df = pd.read_sql_query(_LATEST_UNDERLYING_SQL, self.conn)
        except Exception:
            # Tables/views not yet created (e.g. test DB) - return a stub
            # universe so agents still run.
            return [UnderlyingScore(
                symbol=s, timestamp=_now_iso(), horizon_bars=4,
                direction_probability=0.5, predicted_future_realized_vol=0.20,
                model_version="stub",
            ) for s in self.universe]
        if df.empty:
            return [UnderlyingScore(
                symbol=s, timestamp=_now_iso(), horizon_bars=4,
                direction_probability=0.5, predicted_future_realized_vol=0.20,
                model_version="stub",
            ) for s in self.universe]
        df = df[df["symbol"].isin(self.universe)]
        rows: list[UnderlyingScore] = []
        for _, r in df.iterrows():
            rows.append(UnderlyingScore(
                symbol=str(r["symbol"]),
                timestamp=str(r["timestamp"]),
                horizon_bars=4,
                direction_probability=_safe_float(r.get("direction_probability")),
                predicted_future_realized_vol=_safe_float(r.get("predicted_future_realized_vol")),
                predicted_iv=None,           # not yet available
                gnn_directional_bias=_safe_float(r.get("gnn_directional_bias")),
                gnn_centrality=_safe_float(r.get("gnn_centrality")),
                model_version=str(r.get("model_version") or "n/a"),
            ))
        return rows

    def _load_option_predictions(self) -> list[OptionScore]:
        try:
            df = pd.read_sql_query(_LATEST_OPTION_SQL, self.conn)
        except Exception:
            return []
        if df.empty:
            return []
        rows: list[OptionScore] = []
        for _, r in df.iterrows():
            if r["symbol"] not in self.universe:
                continue
            try:
                dte = int(r["dte"]) if r["dte"] is not None else None
            except (TypeError, ValueError):
                dte = None
            rows.append(OptionScore(
                contract_symbol=str(r["contract_symbol"]),
                underlying=str(r["symbol"]),
                timestamp=str(r["timestamp"]),
                horizon_bars=4,
                probability_profitable=_safe_float(r.get("probability_profitable")),
                expected_return=_safe_float(r.get("expected_return")),
                moneyness=_safe_float(r.get("strike_price")) or 0.0,
                days_to_expiry=dte,
                option_volatility_16=None,
                model_version=str(r.get("model_version") or "n/a"),
            ))
        return rows

    def _load_portfolio(self) -> list[PortfolioPosition]:
        """Returns an empty list when the user has no Alpaca account wired in.
        Real positions come from the trading API in a follow-up."""
        return []

    def _account_equity(self) -> float | None:
        return None

    def _account_cash(self) -> float | None:
        return None


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
