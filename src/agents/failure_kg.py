"""Failure-knowledge-graph writer (T168 / T169).

Writes structured failure events into the ``failure_nodes`` /
``failure_edges`` tables declared in ``sql/70_failure_kg.sql`` so
the multi-agent pipeline can:
  1. Persist every observable failure mode as a typed node
     (symbol / agent / cycle) instead of letting it disappear into
     a stack trace.
  2. Connect failure nodes with weighted edges
     (``caused_by`` / ``co_occurs``) so the GNN can treat the
     failure channel as a first-class signal in the topology.
  3. Expose a per-symbol ``failure_count`` so the strategy
     selector / risk node can down-weight a symbol that has been
     failing all morning.

The writer is deliberately small and side-effect free. All public
methods take a ``sqlite3.Connection`` so tests can run against an
in-memory DB; nothing in here opens its own connection.

Public surface
--------------
::

    from src.agents.failure_kg import FailureKG, classify_error

    kg = FailureKG(conn)
    kg.record_symbol_failure("AAPL", "2026-08-25T13:30:00Z",
                             exc=TimeoutError("option chain fetch timed out"))
    kg.record_agent_failure("direction", "2026-08-25T13:30:00Z", "AAPL",
                            exc=ValueError("ml output was None"))
    kg.record_cycle_failure("decision-abc", "2026-08-25T13:30:00Z",
                            final_action="NO_TRADE", exc=RuntimeError("alpaca 503"))
    # Per-symbol aggregate, ready to splice into the GNN's
    # node feature vector.
    counts = kg.symbol_failure_counts("2026-08-25T13:30:00Z")
    # {"AAPL": 2, "MSFT": 0, ...}
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# Map exception class names to a coarse "kind" so the GNN can learn
# each channel separately. Names are matched as exact strings (so
# subclasses are still classified under their parent class name).
_LLM_ERROR_NAMES: frozenset[str] = frozenset({
    "LLMError", "LLMTransportError",
    "AnthropicError", "OpenAIError", "GMIError",
})
_BROKER_ERROR_NAMES: frozenset[str] = frozenset({
    "AlpacaAPIError", "BrokerError", "OrderRejectedError",
    "InsufficientFundsError", "MarketClosedError",
})
_REQUEST_ERROR_NAMES: frozenset[str] = frozenset({
    "ConnectionError", "Timeout", "TimeoutError", "ReadTimeout",
    "RequestException", "HTTPError", "MaxRetryError",
})


def classify_error(exc: BaseException | None) -> tuple[str, str]:
    """Classify an exception into a (kind, severity) pair.

    Args:
        exc: the exception object. May be None (treated as info).

    Returns:
        A (kind, severity) tuple. ``kind`` is one of
        ``"llm" | "broker" | "request" | "data" | "agent" | "unknown"``;
        ``severity`` is one of ``"info" | "warn" | "error" | "critical"``.
    """
    if exc is None:
        return "info", "info"
    cls = type(exc).__name__
    if cls in _LLM_ERROR_NAMES:
        kind = "llm"
        sev = "error"
    elif cls in _BROKER_ERROR_NAMES:
        kind = "broker"
        sev = "error"
    elif cls in _REQUEST_ERROR_NAMES:
        kind = "request"
        sev = "warn"
    elif cls in ("KeyError", "ValueError", "TypeError", "AttributeError"):
        kind = "agent"
        sev = "error"
    elif cls in ("RuntimeError",):
        # RuntimeError is the generic cycle-failure exception; treat
        # as data-class unless the message contains broker / llm
        # hints. The graph writer only has the string to go on; this
        # heuristic keeps the classification deterministic without
        # re-running the cycle.
        msg = str(exc).lower()
        if "alpaca" in msg or "broker" in msg or "order" in msg:
            kind, sev = "broker", "error"
        elif "llm" in msg or "anthropic" in msg or "openai" in msg:
            kind, sev = "llm", "error"
        elif "data" in msg or "sql" in msg or "schema" in msg:
            kind, sev = "data", "error"
        else:
            kind, sev = "agent", "error"
    else:
        kind, sev = "unknown", "warn"
    return kind, sev


def _severity_for_kind(kind: str, default: str) -> str:
    """Map a kind to a baseline severity. Always returns one of
    info|warn|error|critical. The writer can override via the
    `severity` kwarg."""
    return {
        "llm": "error",
        "broker": "error",
        "request": "warn",
        "data": "error",
        "agent": "error",
        "unknown": default,
    }.get(kind, default)


def _node_id(kind: str, symbol: str | None, agent_id: str | None,
             decision_id: str | None, occurred_at: str) -> str:
    """Stable node id. Same inputs => same id, so the writer can
    dedup via the unique constraint instead of computing the id.

    The symbol is *not* included for ``agent_failure`` rows: the
    same agent failing on two symbols in the same minute is the
    same failure event, and we want to dedup them. The symbol is
    still stored on the row's ``symbol`` column for query
    purposes.
    """
    sym_part = "-" if kind == "agent_failure" else (symbol or "-")
    return f"{kind}:{sym_part}:{agent_id or '-'}:{decision_id or '-'}:{occurred_at}"


def _day(iso_ts: str) -> str:
    """Return the YYYY-MM-DD day key for an ISO timestamp."""
    if "T" in iso_ts:
        return iso_ts.split("T", 1)[0]
    return iso_ts[:10]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# FailureKG
# ---------------------------------------------------------------------------
class FailureKG:
    """Thin writer over the failure_nodes / failure_edges tables.

    All methods are idempotent: re-recording the same (kind, symbol,
    agent_id, decision_id, occurred_at) tuple increments
    ``error_count`` rather than inserting a duplicate row.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---- public API ----------------------------------------------------

    def record_symbol_failure(
        self,
        symbol: str,
        occurred_at: str,
        *,
        exc: BaseException | None = None,
        message: str | None = None,
        severity: str | None = None,
        metadata: dict[str, Any] | None = None,
        decision_id: str | None = None,
    ) -> str:
        """Record a failure that touched ``symbol`` at ``occurred_at``.

        The optional ``decision_id`` ties this symbol-level failure
        to a particular orchestrator cycle, so a subsequent
        :meth:`record_cycle_failure` (with the same ``decision_id``)
        can emit ``co_occurs`` edges from the cycle-failure node to
        every per-symbol failure that contributed to it.

        Returns the resulting ``node_id`` (or the existing one if the
        row was deduped).
        """
        kind, default_sev = classify_error(exc)
        sev = severity or _severity_for_kind(kind, default_sev)
        err_class = type(exc).__name__ if exc is not None else None
        err_msg = message if message is not None else (str(exc) if exc is not None else None)
        meta = {"kind": kind, **(metadata or {})}
        return self._upsert_node(
            kind="symbol_failure",
            symbol=symbol,
            agent_id=None,
            decision_id=decision_id,
            occurred_at=occurred_at,
            severity=sev,
            error_class=err_class,
            error_message=err_msg,
            metadata=meta,
        )

    def record_agent_failure(
        self,
        agent_id: str,
        occurred_at: str,
        symbol: str | None = None,
        *,
        exc: BaseException | None = None,
        message: str | None = None,
        severity: str | None = None,
        metadata: dict[str, Any] | None = None,
        decision_id: str | None = None,
    ) -> str:
        """Record a failure of a named agent at ``occurred_at``.

        The optional ``symbol`` is the underlying the agent was
        processing; it lets the cycle's edge-writer connect the
        agent's failure to the symbol's failure node. The
        agent-failure node id is keyed on (agent_id, occurred_at)
        only — the same agent failing on two symbols at the same
        minute is the same failure event, not two; the symbol is
        stored in the row's `symbol` column.
        """
        kind, default_sev = classify_error(exc)
        sev = severity or _severity_for_kind(kind, default_sev)
        err_class = type(exc).__name__ if exc is not None else None
        err_msg = message if message is not None else (str(exc) if exc is not None else None)
        meta = {"kind": kind, **(metadata or {})}
        return self._upsert_node(
            kind="agent_failure",
            symbol=symbol,
            agent_id=agent_id,
            decision_id=decision_id,
            occurred_at=occurred_at,
            severity=sev,
            error_class=err_class,
            error_message=err_msg,
            metadata=meta,
        )

    def record_cycle_failure(
        self,
        decision_id: str,
        occurred_at: str,
        *,
        final_action: str | None = None,
        exc: BaseException | None = None,
        message: str | None = None,
        severity: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a failure of the cycle as a whole (the orchestrator
        caught an exception that prevented the cycle from completing).

        The writer also emits ``co_occurs`` edges from this node to
        every per-symbol failure row for the same decision_id, so
        the GNN sees the cycle-level event as the cause of all
        per-symbol failures it triggered.
        """
        kind, default_sev = classify_error(exc)
        sev = severity or "critical" if exc is not None else (severity or "warn")
        err_class = type(exc).__name__ if exc is not None else None
        err_msg = message if message is not None else (str(exc) if exc is not None else None)
        meta: dict[str, Any] = {"kind": kind, "final_action": final_action}
        meta.update(metadata or {})
        node_id = self._upsert_node(
            kind="cycle_failure",
            symbol=None,
            agent_id=None,
            decision_id=decision_id,
            occurred_at=occurred_at,
            severity=sev,
            error_class=err_class,
            error_message=err_msg,
            metadata=meta,
        )
        # Emit co_occurs edges to every per-symbol failure for this
        # cycle. The edge weight is 1.0 (we KNOW the cycle failure
        # coincided with the symbol failures).
        day = _day(occurred_at)
        for row in self.conn.execute(
            "SELECT node_id FROM failure_nodes "
            "WHERE decision_id = ? AND kind = 'symbol_failure' AND day = ?",
            (decision_id, day),
        ).fetchall():
            self._upsert_edge(
                source_node_id=node_id,
                target_node_id=row["node_id"],
                relation="co_occurs",
                weight=1.0,
                day=day,
            )
        return node_id

    def link_agent_to_symbol(
        self,
        agent_id: str,
        symbol: str,
        occurred_at: str,
        *,
        weight: float = 1.0,
        relation: str = "caused_by",
    ) -> None:
        """Connect the most-recent agent-failure node for
        ``(agent_id, occurred_at)`` to the most-recent
        symbol-failure node for ``(symbol, occurred_at)`` with a
        weighted edge.

        The relation defaults to ``caused_by``: the symbol's failure
        was *caused by* the agent's failure. Callers that want the
        reverse (agent was caused by the symbol's data) can pass
        relation="similar_to" instead.
        """
        day = _day(occurred_at)
        agent_row = self.conn.execute(
            "SELECT node_id FROM failure_nodes "
            "WHERE kind = 'agent_failure' AND agent_id = ? "
            "AND day = ? ORDER BY occurred_at DESC LIMIT 1",
            (agent_id, day),
        ).fetchone()
        sym_row = self.conn.execute(
            "SELECT node_id FROM failure_nodes "
            "WHERE kind = 'symbol_failure' AND symbol = ? "
            "AND day = ? ORDER BY occurred_at DESC LIMIT 1",
            (symbol, day),
        ).fetchone()
        if agent_row is None or sym_row is None:
            return
        self._upsert_edge(
            source_node_id=agent_row["node_id"],
            target_node_id=sym_row["node_id"],
            relation=relation,
            weight=max(0.0, min(1.0, float(weight))),
            day=day,
        )

    # ---- queries --------------------------------------------------------

    def symbol_failure_counts(
        self,
        as_of: str,
        *,
        window_days: int = 7,
    ) -> dict[str, int]:
        """Return {symbol: failure_count} for the rolling ``window_days``
        ending at ``as_of``. Zero-failure symbols are omitted (callers
        that need a uniform dict should fill with the universe)."""
        day = _day(as_of)
        rows = self.conn.execute(
            """
            SELECT symbol, COALESCE(SUM(error_count), 0) AS n
            FROM   failure_nodes
            WHERE  kind = 'symbol_failure'
              AND  day >= date(?, ?)
            GROUP BY symbol
            """,
            (day, f"-{int(window_days) - 1} days"),
        ).fetchall()
        return {r["symbol"]: int(r["n"]) for r in rows}

    def symbol_failure_features(
        self,
        as_of: str,
        *,
        window_days: int = 7,
    ) -> dict[str, dict[str, float]]:
        """Return {symbol: feature_dict} for the rolling window. The
        feature dict has the same keys as
        ``v_symbol_failure_features`` and is ready to be merged
        into a GNN node-feature vector.
        """
        day = _day(as_of)
        rows = self.conn.execute(
            f"""
            SELECT symbol,
                   COALESCE(SUM(error_count), 0) AS failure_count_{window_days}d,
                   COALESCE(SUM(CASE WHEN json_extract(metadata_json, '$.kind') = 'llm'
                                    THEN error_count ELSE 0 END), 0) AS llm_failure_count_{window_days}d,
                   COALESCE(SUM(CASE WHEN json_extract(metadata_json, '$.kind') = 'broker'
                                    THEN error_count ELSE 0 END), 0) AS broker_failure_count_{window_days}d,
                   COALESCE(SUM(error_count *
                                CASE severity
                                    WHEN 'critical' THEN 1.0
                                    WHEN 'error'    THEN 0.7
                                    WHEN 'warn'     THEN 0.3
                                    ELSE 0.1
                                END), 0.0) AS weighted_failure_score_{window_days}d
            FROM   failure_nodes
            WHERE  kind = 'symbol_failure'
              AND  day >= date(?, ?)
            GROUP BY symbol
            """,
            (day, f"-{int(window_days) - 1} days"),
        ).fetchall()
        return {
            r["symbol"]: {
                f"failure_count_{window_days}d": float(r[f"failure_count_{window_days}d"]),
                f"llm_failure_count_{window_days}d": float(r[f"llm_failure_count_{window_days}d"]),
                f"broker_failure_count_{window_days}d": float(r[f"broker_failure_count_{window_days}d"]),
                f"weighted_failure_score_{window_days}d": float(r[f"weighted_failure_score_{window_days}d"]),
            }
            for r in rows
        }

    # ---- internals -----------------------------------------------------

    def _upsert_node(
        self,
        *,
        kind: str,
        symbol: str | None,
        agent_id: str | None,
        decision_id: str | None,
        occurred_at: str,
        severity: str,
        error_class: str | None,
        error_message: str | None,
        metadata: dict[str, Any],
    ) -> str:
        node_id = _node_id(kind, symbol, agent_id, decision_id, occurred_at)
        day = _day(occurred_at)
        meta_json = json.dumps(metadata, sort_keys=True, default=str)
        # Two-step: SELECT for existing, then INSERT or UPDATE.
        # Wrapping the whole thing in a single UPSERT would also work,
        # but the explicit dedup makes the increment semantics
        # auditable.
        existing = self.conn.execute(
            "SELECT node_id FROM failure_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if existing is not None:
            self.conn.execute(
                """
                UPDATE failure_nodes
                SET    error_count = error_count + 1,
                       error_message = COALESCE(?, error_message),
                       error_class   = COALESCE(?, error_class)
                WHERE  node_id = ?
                """,
                (error_message, error_class, node_id),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO failure_nodes
                  (node_id, kind, symbol, agent_id, decision_id,
                   occurred_at, day, severity, error_class, error_message,
                   error_count, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (node_id, kind, symbol, agent_id, decision_id,
                 occurred_at, day, severity, error_class, error_message,
                 meta_json, _now_iso()),
            )
        self.conn.commit()
        return node_id

    def _upsert_edge(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        relation: str,
        weight: float,
        day: str,
    ) -> None:
        if source_node_id == target_node_id:
            return
        # The UNIQUE constraint on (source, target, relation, day)
        # makes the UPSERT safe to re-run.
        self.conn.execute(
            """
            INSERT INTO failure_edges
              (source_node_id, target_node_id, relation, weight, day, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_node_id, target_node_id, relation, day) DO UPDATE
              SET weight = excluded.weight
            """,
            (source_node_id, target_node_id, relation, float(weight), day, _now_iso()),
        )
        self.conn.commit()


__all__ = ["FailureKG", "classify_error"]
