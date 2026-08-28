"""Decision journal: persist and retrieve complete DecisionState cycles.

The journal is the only place an LLM's reasoning outlives the process. Every
field on DecisionState has a corresponding column or JSON blob - serialize
once, never reconstruct from logs.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .protocol import DecisionState


class DecisionJournal:
    """Thin wrapper over decision_journal. One process -> one instance."""

    def __init__(self, conn: sqlite3.Connection, table: str = "decision_journal",
                 run_mode: str = "paper") -> None:
        self.conn = conn
        self.table = table
        self.run_mode = run_mode

    def upsert(self, state: DecisionState) -> None:
        """Insert the cycle. If it already exists, overwrite the mutable
        fields (completion time, risk, order, execution, final_action)."""
        underlying = None
        if state.selected_strategy is not None:
            underlying = state.selected_strategy.underlying
        elif state.candidate_strategies:
            underlying = state.candidate_strategies[0].underlying
        elif state.market_snapshot and state.market_snapshot.underlyings:
            underlying = state.market_snapshot.underlyings[0].symbol

        # Pydantic v2: model_dump -> dict; mode='json' to force JSON-native types.
        ms = state.market_snapshot.model_dump(mode="json") if state.market_snapshot else {}
        ml = [u.model_dump(mode="json") for u in state.ml_predictions]
        observations = [o.model_dump(mode="json") for o in state.agent_observations]
        messages = [m.model_dump(mode="json") for m in state.agent_messages]

        row = {
            "decision_id": state.decision_id,
            "timestamp": state.cycle_started_at,
            "completed_at": state.cycle_completed_at,
            "market_state_hash": state.market_state_hash,
            "schema_version": state.schema_version,
            "run_mode": self.run_mode,
            "underlying_focus": underlying,
            "final_action": state.final_action,
            "outcome_label": state.outcome_label,
            "realized_pnl": state.realized_pnl,
            "market_snapshot_json": json.dumps(ms, default=str),
            "ml_prediction_json": json.dumps(ml, default=str),
            "gnn_output_json": json.dumps(state.gnn_output, default=str),
            "topology_version": state.topology_version,
            "agent_messages_json": json.dumps(messages, default=str),
            "agent_observations_json": json.dumps(observations, default=str),
            "strategy_proposal_json": json.dumps(
                state.selected_strategy.model_dump(mode="json") if state.selected_strategy else None,
                default=str,
            ),
            "selected_strategy_json": json.dumps(
                state.selected_strategy.model_dump(mode="json") if state.selected_strategy else None,
                default=str,
            ),
            "risk_decision_json": json.dumps(
                state.risk_decision.model_dump(mode="json") if state.risk_decision else None,
                default=str,
            ),
            "order_intent_json": json.dumps(
                state.order_intent.model_dump(mode="json") if state.order_intent else None,
                default=str,
            ),
            "execution_result_json": json.dumps(state.execution_result or {}, default=str),
            "model_versions": json.dumps(
                sorted({u.model_version for u in state.ml_predictions if u.model_version}),
                default=str,
            ),
        }

        cols = list(row.keys())
        placeholders = ",".join("?" * len(cols))
        update_assigns = ",".join(f"{c}=excluded.{c}" for c in cols if c != "decision_id")
        sql = (
            f"INSERT INTO {self.table} ({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(decision_id) DO UPDATE SET {update_assigns}"
        )
        self.conn.execute(sql, [row[c] for c in cols])
        self.conn.commit()

    def get(self, decision_id: str) -> dict[str, Any] | None:
        cur = self.conn.execute(
            f"SELECT * FROM {self.table} WHERE decision_id = ?", (decision_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_recent(self, limit: int = 20, underlying: str | None = None) -> list[dict[str, Any]]:
        if underlying:
            cur = self.conn.execute(
                f"SELECT * FROM {self.table} WHERE underlying_focus = ? "
                f"ORDER BY timestamp DESC LIMIT ?",
                (underlying, limit),
            )
        else:
            cur = self.conn.execute(
                f"SELECT * FROM {self.table} ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        # Accept both sqlite3.Row and tuple rows; prefer .keys() if present.
        if hasattr(row, "keys"):
            d = {k: row[k] for k in row.keys()}
        else:
            d = dict(row)
        for col in (
            "market_snapshot_json", "ml_prediction_json", "gnn_output_json",
            "agent_messages_json", "agent_observations_json",
            "strategy_proposal_json", "selected_strategy_json",
            "risk_decision_json", "order_intent_json",
            "execution_result_json", "model_versions",
        ):
            try:
                d[col] = json.loads(d[col]) if d[col] else ({} if col.endswith("json") and col != "agent_messages_json" and col != "agent_observations_json" else [])
            except (TypeError, ValueError):
                pass
        return d
