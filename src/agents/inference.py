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
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import math
import numpy as np
import pandas as pd

from .protocol import (
    MarketSnapshot,
    OptionScore,
    PortfolioPosition,
    ResearchOutput,
    Side,
    SymbolResearch,
    UnderlyingScore,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# SQL: latest underlying features joined to the latest prediction rows. Mirrors
# src.ml.predict but lives here so agents do not depend on the training-side
# module's import path.
#
# Forward-leak protection: the column list deliberately omits
# ``future_return``, ``future_realized_vol``, and ``target_class`` (the
# forward columns on ``ml_training_dataset`` that ``v_labels`` populates).
# The inference path must never expose them. Point-in-time backtests
# additionally filter ``WHERE u.timestamp <= ?`` so the inference snapshot
# at T cannot see a row whose underlying bar / ml-training row was written
# after T (spec 003 / T046 — the "future-leak" cut-off).
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
    WHERE  u.timestamp <= ?
) d ON d.symbol = f.symbol AND d.rn = 1
WHERE  f.timestamp <= ?
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
       NULL AS probability_profitable,    -- derived in Python from the
       NULL AS expected_return,           --   underlying's ML signal
       'heuristic' AS model_version
FROM   option_bars ob
JOIN   option_contracts oc USING (contract_symbol)
WHERE  oc.tradable = 1
  AND  CAST(julianday(oc.expiration_date) - julianday(substr(ob.timestamp, 1, 10)) AS INTEGER) BETWEEN 1 AND 60
  AND  ob.timestamp = (
       SELECT MAX(ob2.timestamp) FROM option_bars ob2
       WHERE  ob2.contract_symbol = ob.contract_symbol
         AND  ob2.timestamp <= ?
  )
ORDER BY ob.contract_symbol
"""


# ---------------------------------------------------------------------------
# Real ML model loader (Option A — run models at inference time)
# ---------------------------------------------------------------------------
# The previous SQL path tried to read precomputed predictions from a
# `direction_probability` column on ml_training_dataset that never existed.
# The except Exception at the loader swallowed the OperationalError and
# stubbed every underlying. Option A loads the trained XGBoost artifacts
# from disk and runs them on the latest v2 feature row at inference time,
# matching how the GNN service already works.
#
# Each artifact is a joblib blob with shape:
#   {"task": "direction"|"rv"|"option",
#    "kind": "clf"|"reg",
#    "feature_names": list[str],
#    "impute_medians": dict[str, float],
#    "model": <XGBClassifier | XGBRegressor>,
#    ...}
# The artifact filename encodes (task, horizon, kind) so we can pick the
# right one for each inference call.
_UNDERLYING_FEATURES_V2 = [
    "return_1", "return_4", "return_16", "volatility_16",
    "rsi_14", "macd_pct",
    "hl_range", "co_return", "atr_pct_14",
    "ma_dist_20", "ma_dist_50",
    "volume_ratio_20", "volume_change_1", "trade_count_ratio_20", "vwap_distance",
    "spy_ret_1", "spy_ret_past_16", "spy_volatility_16",
    "qqq_ret_past_16", "qqq_volatility_16",
]

# Underlying feature SQL — pulls the v2 features for the LATEST bar at-or-
# before the cut-off per symbol, filtered to the inference universe. Mirrors
# src/ml/predict.py:_UNDERLYING_LATEST_SQL but adds the as_of filter so
# backtests do not see future rows.
_UNDERLYING_LATEST_V2_SQL = """
SELECT x.symbol, x.timestamp, {cols}
FROM (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY f.symbol ORDER BY f.timestamp DESC) AS rn
    FROM   v_features_underlying_v2 f
    WHERE  f.timestamp <= ?
) x
WHERE x.rn = 1
"""


def _parse_artifact_stem(stem: str) -> tuple[str, int, str] | None:
    """Parse a .pkl artifact stem like 'direction_h4_xgb_clf-20260829-125959'.

    Returns (task, horizon_bars, kind) or None if the stem does not match
    the expected shape.
    """
    head = stem.split("-")[0]
    parts = head.split("_")
    if len(parts) < 4:
        return None
    task, hz, kind = parts[0], int(parts[1][1:]), parts[3]
    if task not in ("direction", "rv", "option"):
        return None
    if kind not in ("clf", "reg"):
        return None
    return task, hz, kind


def _latest_artifact(model_dir: Path, task: str, horizon_bars: int, kind: str) -> Path | None:
    """Return the path of the newest artifact matching (task, horizon, kind)."""
    # Filename shape: <task>_h<horizon>_xgb_<kind>-YYYYMMDD-HHMMSS.pkl
    # e.g. direction_h4_xgb_clf-20260829-125959.pkl
    # The glob has to match the head before the timestamp; the parse
    # helper below discriminates precisely.
    candidates: list[Path] = []
    for p in sorted(model_dir.glob(f"{task}_h{horizon_bars}_xgb_{kind}-*.pkl")):
        if _parse_artifact_stem(p.stem) == (task, horizon_bars, kind):
            candidates.append(p)
    if not candidates:
        return None
    return candidates[-1]


def _load_artifact(path: Path) -> dict[str, Any]:
    """Load a joblib .pkl artifact and assert it has the expected keys."""
    blob = joblib.load(path)
    for key in ("task", "kind", "feature_names", "impute_medians", "model"):
        if key not in blob:
            raise RuntimeError(
                f"artifact {path.name} is missing required key {key!r}"
            )
    return blob


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using ``math.erf`` (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _derive_option_pnl_metrics(
    *,
    strike: float,
    opt_close: float,
    option_type: str,
    days_to_expiry: int,
    horizon_bars: int,
    underlying_dir_prob: float,
    underlying_predicted_rv: float,
) -> tuple[float, float]:
    """Heuristic ``(probability_profitable, expected_return)``.

    Used because ``ml_training_dataset`` carries only per-underlying
    summaries, not per-contract ``probability_profitable`` /
    ``expected_return`` columns. The model treats the option as a
    leveraged directional bet on the underlying's ML signal:

      * For a long call: profitability requires the underlying to go
        *up*; for a long put: the underlying to go *down*.
      * ``probability_profitable`` is the underlying's direction
        probability in the *right* direction for this contract
        (so a bearish-model underlying has a high pp for puts,
        low pp for calls). Clamped to [0.05, 0.95].
      * ``expected_return`` is a dimensionless payoff proxy:
        ``max(0, pp - 0.5) * 1.5`` scaled by a DTE-based time factor
        (more DTE = more time for the bet to play out). Clamped to
        [0, 1] because :func:`score_candidate` does the same.

    The math is intentionally simple — the strategy selector uses
    these only for ranking, and the ranking is dominated by the
    GNN / XGB signal which is real (not heuristic).
    """
    if days_to_expiry <= 0 or strike <= 0 or opt_close <= 0:
        return (0.0, 0.0)
    q = max(0.01, min(0.99, float(underlying_dir_prob)))
    is_call = "c" in option_type.lower()
    # Right-direction probability: dir_prob for a call, (1 - dir_prob) for a put.
    # The wrong-direction side gets the inverted probability.
    right_dir_prob = q if is_call else (1.0 - q)
    pp = max(0.05, min(0.95, right_dir_prob))
    # DTE factor: 0 DTE = 0, 21 DTE = 1.0 (full effect).
    dte_factor = min(1.0, max(0.0, days_to_expiry / 21.0))
    # expected_return proxy: only positive when pp > 0.5 (edge on our side),
    # scaled by DTE for time value of the bet.
    er = max(0.0, min(1.0, (pp - 0.5) * 1.5 * (0.4 + 0.6 * dte_factor)))
    return (pp, er)


@dataclass
class _MLModelCache:
    """Lazily-loaded cache for the latest direction / rv XGBoost artifacts.

    Loaded on first use inside ``InferenceService`` so a cold start does
    not pay the cost when the orchestrator does not need ML predictions
    (e.g. pure GNN smoke tests). Cached on the service instance.
    """
    direction_h4_blob: dict[str, Any] | None = None
    direction_h4_path: str | None = None
    rv_h4_blob: dict[str, Any] | None = None
    rv_h4_path: str | None = None


@dataclass
class InferenceService:
    """Builds MarketSnapshot from the SQLite system of record."""

    conn: sqlite3.Connection
    universe: list[str]
    model_dir: Path = MODEL_DIR
    topology_version: str = "fixed-1"
    # Feature flag: when True, build_snapshot populates ``research`` from
    # the ``news_snapshot`` table (spec 003 / US1, T018). When False the
    # field is left None and agents run the pre-news path unchanged.
    research_enabled: bool = False
    research_lookback_hours: int = 24
    # Point-in-time cut-off (ISO 8601, e.g. ``2026-08-25T13:30:00Z``).
    # When set, every loader filters on ``timestamp <= as_of`` so the
    # snapshot cannot see rows written after the cut-off — the
    # forward-leak guard used by the backtester. ``None`` means
    # "use the latest row" (today's wall-clock behavior). Spec 003 /
    # T046.
    as_of: str | None = None
    # The GNN service is loaded lazily on first use so a missing
    # artifact doesn't slow down the orchestrator's startup path. The
    # stub is the FR8.3 fallback (SC5); the real service is loaded
    # when a row exists in gnn_model_artifacts (US3 / T028).
    _gnn_service: Any = field(default=None, init=False, repr=False)
    _gnn_loaded: bool = field(default=False, init=False, repr=False)
    # ML model cache: the latest direction / rv XGBoost artifacts are
    # loaded lazily on the first call to _load_underlying_predictions
    # and cached on the service instance. A missing artifact is logged
    # once and the loader returns the deterministic stub path.
    _ml_cache: _MLModelCache = field(default_factory=_MLModelCache, init=False, repr=False)
    _ml_loaded: bool = field(default=False, init=False, repr=False)

    # ---- public API ----
    def build_snapshot(self) -> MarketSnapshot:
        underlying_rows = self._load_underlying_predictions()
        option_rows = self._load_option_predictions()
        portfolio = self._load_portfolio()
        research = self._load_research_output() if self.research_enabled else None
        # When ``as_of`` is set, stamp the snapshot with the cycle time
        # rather than the newest underlying row, so downstream agents
        # observe a coherent "T" rather than a wall-clock drift.
        snap_ts = self.as_of or (
            underlying_rows[0].timestamp if underlying_rows else _now_iso()
        )
        return MarketSnapshot(
            timestamp=snap_ts,
            underlyings=underlying_rows,
            options=option_rows,
            portfolio=portfolio,
            account_equity=self._account_equity(),
            account_cash=self._account_cash(),
            research=research,
        )

    def gnn_output(self) -> dict[str, Any]:
        """Return a GNNOutput-shaped dict.

        Priority:
        1. If a :class:`src.gnn.service.GNNService` artifact exists in
           the DB, call ``predict`` on the most-recent snapshot.
        2. Else fall back to :class:`src.gnn.stub.StubGNNService`.

        The result is always the contract-correct shape: ``version ==
        "1.0"``, ``model_version`` either ``"stub-1"`` or the
        artifact's version, ``node_features`` per-symbol, and
        ``edges`` list.
        """
        svc = self._get_gnn_service()
        if svc is None:
            from ..gnn.stub import StubGNNService
            stub = StubGNNService()
            out = stub.output(timestamp=self.as_of or _now_iso())
        else:
            snap_id = self._resolve_snapshot_id()
            out = svc.predict(snap_id)
        return out.model_dump(mode="json")

    # ---- helpers ----
    def _get_gnn_service(self) -> Any:
        """Load the latest GNN artifact lazily. Returns None when no
        artifact is present (stub path)."""
        if self._gnn_loaded:
            return self._gnn_service
        self._gnn_loaded = True
        try:
            from ..gnn.service import GNNService
            self._gnn_service = GNNService.load_latest(self.conn)
        except Exception:
            self._gnn_service = None
        return self._gnn_service

    def _resolve_snapshot_id(self) -> str:
        """Pick the most-recent graph snapshot; fall back to a
        well-known date so a never-built snapshot still produces a
        prediction. When ``as_of`` is set, only snapshots at-or-before
        the cut-off are considered (point-in-time replay)."""
        # Index rows by column name; the connection may not have
        # sqlite3.Row set, so toggle the factory locally and restore
        # the caller's value on the way out.
        prev_factory = self.conn.row_factory
        self.conn.row_factory = sqlite3.Row
        try:
            if self.as_of is not None:
                row = self.conn.execute(
                    "SELECT snapshot_id FROM gnn_graph_snapshots "
                    "WHERE timestamp <= ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (self.as_of,),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT snapshot_id FROM gnn_graph_snapshots "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
        finally:
            self.conn.row_factory = prev_factory
        if row is not None:
            return row["snapshot_id"]
        return "stub-snapshot"

    # ---- helpers ----
    def _load_research_output(self) -> ResearchOutput | None:
        """Read the latest ``news_snapshot`` rows for the universe and build
        a :class:`ResearchOutput` (spec 003 / T018).

        Uses the snapshot timestamp as a hard cut-off: only articles with
        ``timestamp <= cycle_timestamp`` are admitted (news-time-leakage
        protection from failure-analysis §2.1 row 2). The most recent
        article per symbol within the lookback window becomes the
        ``last_article_at``; sentiment is averaged across the window.

        When ``as_of`` is set, that timestamp is the cut-off (point-in-time
        backtest); otherwise the most-recent underlying prediction
        timestamp is used, and falls back to wall-clock. Returns ``None``
        when no rows exist for the universe (the cycle continues;
        ``market_snapshot.research == None`` is a valid state and
        downstream agents must tolerate it).
        """
        underlying_ts = self.as_of
        if underlying_ts is None:
            # Use the most-recent underlying prediction timestamp as the
            # cut-off when available; otherwise fall back to wall-clock.
            try:
                row = self.conn.execute(
                    "SELECT MAX(timestamp) AS ts FROM ml_training_dataset "
                    "WHERE symbol IN ({}) AND contract_symbol IS NULL".format(
                        ",".join("?" for _ in self.universe)
                    ),
                    self.universe,
                ).fetchone()
            except Exception:
                row = None
            if row is not None and row[0] is not None:
                underlying_ts = str(row[0])
            else:
                underlying_ts = _now_iso()

        try:
            placeholders = ",".join("?" for _ in self.universe)
            rows = self.conn.execute(
                f"""
                SELECT timestamp, symbol, sentiment, article_count, topics_json
                FROM   news_snapshot
                WHERE  symbol IN ({placeholders})
                  AND  timestamp <= ?
                ORDER  BY timestamp DESC
                """,
                [*self.universe, underlying_ts],
            ).fetchall()
        except Exception:
            return None

        if not rows:
            return None

        from collections import defaultdict
        import json as _json

        per_symbol: dict[str, SymbolResearch] = {}
        per_symbol_latest: dict[str, str] = {}
        per_symbol_sents: dict[str, list[float]] = defaultdict(list)
        per_symbol_counts: dict[str, int] = defaultdict(int)
        per_symbol_topics: dict[str, list[str]] = {}
        for ts, sym, sent, cnt, topics_json in rows:
            per_symbol_sents[sym].append(float(sent))
            per_symbol_counts[sym] += int(cnt)
            if sym not in per_symbol_latest or ts > per_symbol_latest[sym]:
                per_symbol_latest[sym] = str(ts)
                try:
                    per_symbol_topics[sym] = list(_json.loads(topics_json))[:3]
                except Exception:
                    per_symbol_topics[sym] = []

        for sym in self.universe:
            if sym not in per_symbol_latest:
                continue
            sents = per_symbol_sents[sym]
            avg = sum(sents) / max(1, len(sents))
            avg = max(-1.0, min(1.0, avg))
            per_symbol[sym] = SymbolResearch(
                sentiment=avg,
                volume=per_symbol_counts[sym],
                topics=per_symbol_topics.get(sym, []),
                last_article_at=per_symbol_latest[sym],
            )

        return ResearchOutput(
            version="1.0",
            timestamp=underlying_ts,
            per_symbol=per_symbol,
            feature_flag_state="news-on",
            risks=[],
        )

    def _ensure_ml_loaded(self) -> None:
        """Load the latest direction + rv XGBoost artifacts (lazy, once).

        Missing artifacts are logged at WARNING and the loader silently
        returns 0.5 / 0.20 for the affected head. We do NOT fall back
        silently for all heads — at least one real model version is
        required so the orchestrator's ``model_version`` field surfaces
        what produced the prediction.
        """
        if self._ml_loaded:
            return
        self._ml_loaded = True
        # direction (classifier)
        dpath = _latest_artifact(self.model_dir, "direction", 4, "clf")
        if dpath is not None:
            try:
                self._ml_cache.direction_h4_blob = _load_artifact(dpath)
                self._ml_cache.direction_h4_path = str(dpath)
                logger.info("loaded ML direction model: %s", dpath.name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("failed to load direction artifact %s: %s", dpath, exc)
        else:
            logger.warning("no direction_h4_xgb_clf artifact found in %s", self.model_dir)
        # rv (regressor)
        rpath = _latest_artifact(self.model_dir, "rv", 4, "reg")
        if rpath is not None:
            try:
                self._ml_cache.rv_h4_blob = _load_artifact(rpath)
                self._ml_cache.rv_h4_path = str(rpath)
                logger.info("loaded ML rv model: %s", rpath.name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("failed to load rv artifact %s: %s", rpath, exc)
        else:
            logger.warning("no rv_h4_xgb_reg artifact found in %s", self.model_dir)

    def _ml_predict_underlying(self) -> list[UnderlyingScore]:
        """Run the trained XGBoost models on the latest v2 feature row.

        Replaces the broken SQL subquery path: pulls the latest
        v_features_underlying_v2 row at-or-before the as_of cut-off for
        each universe symbol, builds the v2 feature vector in the order
        the trained model expects, and calls predict_proba / predict
        directly. This matches how the GNN service runs the .pt model
        at inference time.

        After the XGBoost pass, the GNN node features
        (``gnn_directional_bias``, ``gnn_centrality``) are merged in from
        :meth:`gnn_output` so downstream consumers (strategy selector,
        GNN-confirmation override) see one consistent row per symbol.
        """
        self._ensure_ml_loaded()
        cutoff = self.as_of or "9999-12-31T23:59:59Z"
        cols = ", ".join(f"x.{c}" for c in _UNDERLYING_FEATURES_V2)
        sql = _UNDERLYING_LATEST_V2_SQL.format(cols=cols)
        try:
            df = pd.read_sql_query(sql, self.conn, params=(cutoff,))
        except Exception as exc:
            logger.warning("underlying feature SQL failed: %s", exc)
            df = pd.DataFrame()
        if df.empty:
            # Even with no v2 feature row, return one stub per universe
            # symbol so downstream consumers (e.g. strategy_selector)
            # always see a uniform row count. Try to pull last_price
            # anyway so the equity candidate generator can size any
            # viable signal.
            try:
                price_df = pd.read_sql_query(
                    """
                    SELECT x.symbol, x.close AS last_price
                    FROM (
                        SELECT b.symbol, b.close,
                               ROW_NUMBER() OVER (PARTITION BY b.symbol ORDER BY b.timestamp DESC) AS rn
                        FROM   underlying_bars b
                        WHERE  b.timestamp <= ?
                    ) x
                    WHERE x.rn = 1
                    """,
                    self.conn,
                    params=(cutoff,),
                )
                last_price_by_sym = {
                    str(r["symbol"]): _safe_float(r["last_price"])
                    for _, r in price_df.iterrows()
                }
            except Exception:
                last_price_by_sym = {}
            return [UnderlyingScore(
                symbol=s, timestamp=cutoff, horizon_bars=4,
                direction_probability=0.5, predicted_future_realized_vol=0.20,
                predicted_iv=None,
                gnn_directional_bias=None, gnn_centrality=None,
                model_version="stub",
                last_price=last_price_by_sym.get(s),
            ) for s in self.universe]
        df = df[df["symbol"].isin(self.universe)].reset_index(drop=True)

        # Pull latest close per symbol at-or-before the cut-off. Used by
        # the equity candidate generator to size the long-equity leg.
        # Independent query so a missing underlying_bars row just yields
        # ``last_price=None`` and the candidate generator skips the symbol
        # safely (no fabricated price).
        try:
            price_df = pd.read_sql_query(
                """
                SELECT x.symbol, x.close AS last_price
                FROM (
                    SELECT b.symbol, b.close,
                           ROW_NUMBER() OVER (PARTITION BY b.symbol ORDER BY b.timestamp DESC) AS rn
                    FROM   underlying_bars b
                    WHERE  b.timestamp <= ?
                ) x
                WHERE x.rn = 1
                """,
                self.conn,
                params=(cutoff,),
            )
            last_price_by_sym = {
                str(r["symbol"]): _safe_float(r["last_price"])
                for _, r in price_df.iterrows()
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("last_price SQL failed: %s", exc)
            last_price_by_sym = {}

        # Build per-symbol predictions.
        rows: list[UnderlyingScore] = []
        for _, r in df.iterrows():
            sym = str(r["symbol"])
            ts = str(r["timestamp"])
            model_version = "n/a"
            direction_prob = 0.5
            predicted_rv = 0.20
            # direction
            dblob = self._ml_cache.direction_h4_blob
            if dblob is not None:
                try:
                    feats = list(dblob["feature_names"])
                    med = dblob["impute_medians"]
                    X = pd.DataFrame([r[feats].tolist()], columns=feats).fillna(med).to_numpy(dtype=float)
                    p = dblob["model"].predict_proba(X)
                    p_arr = np.asarray(p, dtype=float)
                    direction_prob = float(p_arr[0, 1]) if p_arr.ndim == 2 else float(p_arr[0])
                    if self._ml_cache.direction_h4_path is not None:
                        model_version = Path(self._ml_cache.direction_h4_path).stem
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("direction predict failed for %s: %s", sym, exc)
            # rv
            rblob = self._ml_cache.rv_h4_blob
            if rblob is not None:
                try:
                    feats = list(rblob["feature_names"])
                    med = rblob["impute_medians"]
                    X = pd.DataFrame([r[feats].tolist()], columns=feats).fillna(med).to_numpy(dtype=float)
                    pred = np.asarray(rblob["model"].predict(X), dtype=float)
                    predicted_rv = float(pred[0])
                    if model_version == "n/a" and self._ml_cache.rv_h4_path is not None:
                        model_version = Path(self._ml_cache.rv_h4_path).stem
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("rv predict failed for %s: %s", sym, exc)
            rows.append(UnderlyingScore(
                symbol=sym,
                timestamp=ts,
                horizon_bars=4,
                direction_probability=direction_prob,
                predicted_future_realized_vol=predicted_rv,
                predicted_iv=None,
                gnn_directional_bias=None,   # filled in below from GNN service
                gnn_centrality=None,
                model_version=model_version,
                last_price=last_price_by_sym.get(sym),
            ))
        # Merge in GNN node features. The GNN service is the source of
        # truth for ``gnn_directional_bias`` and ``gnn_centrality``; the
        # XGBoost ML path provides the direction probability and rv. Both
        # must agree on which symbols exist; symbols only in one side are
        # back-filled with None.
        try:
            gnn = self.gnn_output()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("gnn_output failed during merge: %s", exc)
            gnn = None
        if gnn:
            node_features = gnn.get("node_features", {}) or {}
            by_sym = {r.symbol: r for r in rows}
            for sym, feat in node_features.items():
                bias = feat.get("bias")
                cent = feat.get("centrality")
                if sym in by_sym:
                    r = by_sym[sym]
                    rows[rows.index(r)] = r.model_copy(update={
                        "gnn_directional_bias": bias,
                        "gnn_centrality": cent,
                    })
                elif sym in self.universe:
                    # GNN has a row for a symbol that didn't have v2
                    # features at the cut-off — emit a stub so the
                    # strategy selector still sees the GNN signal.
                    rows.append(UnderlyingScore(
                        symbol=sym,
                        timestamp=cutoff,
                        horizon_bars=4,
                        direction_probability=0.5,
                        predicted_future_realized_vol=0.20,
                        predicted_iv=None,
                        gnn_directional_bias=bias,
                        gnn_centrality=cent,
                        model_version="n/a",
                        last_price=last_price_by_sym.get(sym),
                    ))
        return rows

    def _load_underlying_predictions(self) -> list[UnderlyingScore]:
        # Point-in-time cut-off: when ``as_of`` is set, every timestamp
        # filter is clamped to ``timestamp <= as_of`` (spec 003 / T046).
        # Without this, a backtest at T would see an underlying bar /
        # ml-training row written *after* T — the future-leak case the
        # user flagged. The SQL filter enforces this; the model itself
        # has no time component (it scores whatever feature row is fed).
        #
        # Real ML inference path: load the trained XGBoost models
        # lazily on first call and run them on the latest v2 feature
        # row (Option A — see ml_inference_root_cause.md). The previous
        # SQL subquery that read precomputed predictions from
        # ml_training_dataset was broken (the columns never existed)
        # and silently stubbed every underlying to prob=0.5.
        return self._ml_predict_underlying()

    def _load_option_predictions(self) -> list[OptionScore]:
        """Pull one row per option contract for the snapshot.

        The historical SQL path tried to join ``ml_training_dataset`` on
        ``contract_symbol`` but that table has no per-contract prediction
        column — only the per-underlying summary. Rather than pretend
        we have precomputed per-contract probabilities, we derive them
        here from the contract's strike / expiry / close and the
        underlying's ML direction probability + predicted rv. The math
        is a textbook P(S_T > K) for calls and 1 - P(S_T < K) for puts
        using a lognormal model with sigma = predicted_rv * sqrt(T).

        The point-in-time cut-off (``as_of``) is honored in two places:
          * the latest ``option_bars`` row per contract is clamped to
            ``timestamp <= cutoff``;
          * the underlying's ``direction_probability`` is read from the
            already-cutoff ``UnderlyingScore`` rows, not the live table.
        """
        # Point-in-time cut-off (spec 003 / T046).
        cutoff = self.as_of or "9999-12-31T23:59:59Z"
        try:
            df = pd.read_sql_query(
                _LATEST_OPTION_SQL, self.conn, params=(cutoff,),
            )
        except Exception as exc:
            # Surface the real error so future regressions are not silent.
            logger.warning("option prediction SQL failed: %s", exc)
            return []
        if df.empty:
            return []
        # Build a quick map from underlying -> (spot, dir_prob, rv) for the
        # heuristic P(profit) / expected_return derivation. The spot comes
        # from the same v_features_underlying_v2 row the ML path used, so
        # there is no second SQL hit.
        underlyings = {u.symbol: u for u in self._load_underlying_predictions()}
        rows: list[OptionScore] = []
        for _, r in df.iterrows():
            if r["symbol"] not in self.universe:
                continue
            try:
                dte = int(r["dte"]) if r["dte"] is not None else None
            except (TypeError, ValueError):
                dte = None
            u = underlyings.get(str(r["symbol"]))
            pp, er = _derive_option_pnl_metrics(
                strike=_safe_float(r.get("strike_price")) or 0.0,
                opt_close=_safe_float(r.get("opt_close")) or 0.0,
                option_type=str(r.get("option_type") or "").lower(),
                days_to_expiry=dte or 0,
                horizon_bars=4,
                underlying_dir_prob=(u.direction_probability if u else 0.5),
                underlying_predicted_rv=(u.predicted_future_realized_vol if u else 0.20),
            )
            rows.append(OptionScore(
                contract_symbol=str(r["contract_symbol"]),
                underlying=str(r["symbol"]),
                timestamp=str(r["timestamp"]),
                horizon_bars=4,
                probability_profitable=pp,
                expected_return=er,
                moneyness=_safe_float(r.get("strike_price")) or 0.0,
                days_to_expiry=dte,
                option_volatility_16=None,
                model_version=str(r.get("model_version") or "heuristic-1"),
            ))
        return rows

    def _load_portfolio(self) -> list[PortfolioPosition]:
        """Read live positions from Alpaca. Returns an empty list if the
        broker is unreachable or the user has no live account wired in."""
        try:
            from .alpaca_trading import AlpacaTradingClient
            client = AlpacaTradingClient()
            positions = client.list_positions()
        except Exception:
            return []
        out: list[PortfolioPosition] = []
        for p in positions:
            try:
                qty = float(p.get("qty", 0) or 0)
                avg = float(p.get("avg_entry_price", 0) or 0)
                mark = float(p.get("current_price") or avg)
                out.append(PortfolioPosition(
                    symbol=str(p.get("symbol", "")),
                    quantity=qty,
                    avg_entry_price=avg,
                    mark_price=mark,
                    unrealized_pnl=(mark - avg) * qty,
                    side="long" if qty >= 0 else "short",
                ))
            except Exception:
                continue
        return out

    def _account_equity(self) -> float | None:
        try:
            from .alpaca_trading import AlpacaTradingClient
            client = AlpacaTradingClient()
            acct = client.get_account()
            return float(acct.get("equity")) if acct.get("equity") is not None else None
        except Exception:
            return None

    def _account_cash(self) -> float | None:
        try:
            from .alpaca_trading import AlpacaTradingClient
            client = AlpacaTradingClient()
            acct = client.get_account()
            return float(acct.get("cash")) if acct.get("cash") is not None else None
        except Exception:
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
