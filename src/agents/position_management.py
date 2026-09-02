"""Pre-cycle position management.

Three pre-flight checks added 2026-09-02 in response to the loss-position
audit (the system previously had NO stop-loss, NO trailing stop, NO exit
logic, and a per-symbol cooldown that was declared in config but never
read). These three functions are pure, broker-agnostic where possible,
and are wired into :meth:`src.agents.graph.Orchestrator.run_cycle` so
every cycle benefits regardless of entry point (CLI, Render, GH Actions,
tests).

The three checks:

  (a) Daily loss kill-switch
        If today's realized + open-position unrealized P/L is worse than
        ``daily_loss_kill_switch_pct`` of capital, return NO_TRADE and
        skip the cycle. Configurable via env
        ``AIZEN_DAILY_LOSS_KILL_SWITCH_PCT`` (default ``-0.02``).

  (b) Per-position stop-loss
        For any open position whose ``unrealized_pnl / (entry * qty)``
        is worse than ``stop_loss_pct`` (default ``-0.50``), submit a
        market-close via :meth:`AlpacaTradingClient.close_position`.
        Side effect, idempotent — a position closed on a prior cycle
        is no longer in the live portfolio and is skipped.

  (c) Per-symbol cooldown after a loss
        Build a set of symbols that are blocked from new entries. A
        symbol is blocked if (i) it currently has an open position
        in loss, OR (ii) its most recent ``decision_journal`` row
        for an ``underlying_focus`` trade recorded
        ``realized_pnl < 0`` within the last
        ``cooldown_seconds``. The supervisor filters these symbols
        out of ``candidate_strategies`` before the PROCEED gate.

Thresholds live in :mod:`config.agents.yaml` under ``position_management``
with env-var overrides for cron-time tweaks without editing the file.

The functions are intentionally independent of LangGraph, the LLM
provider, and the decision state — the orchestrator wires them in
at cycle start and the supervisor reads ``state.supplementary``
after the wire-up.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold config — env override > config/agents.yaml > safe default.
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a float; using default %.4f", name, raw, default)
        return default


def _load_thresholds(agents_cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve position-management thresholds. Order: env > agents.yaml > default.

    ``agents_cfg`` is the parsed ``config/agents.yaml`` mapping (or None,
    in which case we re-read it here).
    """
    cfg_pm: dict[str, Any] = {}
    if agents_cfg is None:
        try:
            from ..config import get_yaml
            cfg_pm = ((get_yaml("agents") or {}).get("position_management") or {})
        except Exception:  # pragma: no cover - defensive
            cfg_pm = {}
    else:
        cfg_pm = (agents_cfg.get("position_management") or {})
    return {
        "daily_loss_kill_switch_pct": _env_float(
            "AIZEN_DAILY_LOSS_KILL_SWITCH_PCT",
            float(cfg_pm.get("daily_loss_kill_switch_pct", -0.02)),
        ),
        "stop_loss_pct": _env_float(
            "AIZEN_STOP_LOSS_PCT",
            float(cfg_pm.get("stop_loss_pct", -0.50)),
        ),
        "cooldown_seconds": _env_float(
            "AIZEN_COOLDOWN_SECONDS",
            float(cfg_pm.get("cooldown_seconds", 3600.0)),
        ),
        # Block threshold for "currently in loss" — slightly tighter than
        # the stop-loss so we prevent re-entry before a stop fires.
        "open_loss_block_pct": _env_float(
            "AIZEN_OPEN_LOSS_BLOCK_PCT",
            float(cfg_pm.get("open_loss_block_pct", -0.05)),
        ),
    }


# ---------------------------------------------------------------------------
# (a) Daily loss kill-switch
# ---------------------------------------------------------------------------
@dataclass
class KillSwitchResult:
    """Result of the daily-loss kill-switch probe.

    ``breached`` is True iff today's combined realized + open-position
    unrealized P/L is worse than ``threshold_usd`` (which is
    ``capital_usd * daily_loss_kill_switch_pct``; negative).
    """
    breached: bool
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    threshold_usd: float
    capital_usd: float
    pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "breached": self.breached,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "threshold_usd": self.threshold_usd,
            "capital_usd": self.capital_usd,
            "pct": self.pct,
        }


def check_daily_loss_kill_switch(
    *,
    capital_usd: float,
    positions: Iterable[Any],
    realized_pnl_today: float = 0.0,
    pct: float = -0.02,
) -> KillSwitchResult:
    """Return a :class:`KillSwitchResult` describing today's P/L vs the
    daily-loss cap.

    Args:
        capital_usd: paper or live account equity used to size the cap.
        positions: any iterable of objects with ``unrealized_pnl`` (we
            accept both :class:`PortfolioPosition` and the raw dicts
            returned by :meth:`AlpacaTradingClient.list_positions`).
        realized_pnl_today: sum of closed trade P/L since 00:00 UTC today.
            The caller is expected to compute this from the
            ``decision_journal.realized_pnl`` rows for today.
        pct: loss threshold as a negative fraction of capital. Default
            ``-0.02`` = -2% of capital.
    """
    unrealized = 0.0
    for p in positions:
        u = getattr(p, "unrealized_pnl", None)
        if u is None and isinstance(p, dict):
            u = p.get("unrealized_pl") or p.get("unrealized_pnl")
        if isinstance(u, (int, float)):
            unrealized += float(u)
    total = float(realized_pnl_today) + float(unrealized)
    threshold_usd = float(capital_usd) * float(pct)
    return KillSwitchResult(
        breached=total < threshold_usd,
        realized_pnl=float(realized_pnl_today),
        unrealized_pnl=float(unrealized),
        total_pnl=total,
        threshold_usd=threshold_usd,
        capital_usd=float(capital_usd),
        pct=float(pct),
    )


# ---------------------------------------------------------------------------
# (b) Per-position stop-loss
# ---------------------------------------------------------------------------
def positions_to_close(
    positions: Iterable[Any],
    *,
    pct: float = -0.50,
) -> list[dict[str, Any]]:
    """Return the subset of positions whose unrealized loss exceeds ``pct``
    of cost basis (i.e. entry_price * quantity).

    Each returned dict has ``symbol``, ``quantity``, ``entry_price``,
    ``current_price``, ``unrealized_pnl``, and ``loss_pct`` so the caller
    can either log them or pass to :meth:`AlpacaTradingClient.close_position`.
    """
    out: list[dict[str, Any]] = []
    for p in positions:
        if isinstance(p, dict):
            symbol = p.get("symbol")
            qty = float(p.get("qty") or p.get("quantity") or 0)
            entry = float(p.get("avg_entry_price") or p.get("entry_price") or 0)
            mark = float(p.get("current_price") or p.get("mark_price") or entry)
            upl = float(p.get("unrealized_pl") or p.get("unrealized_pnl") or
                        ((mark - entry) * qty))
        else:
            symbol = getattr(p, "symbol", None)
            qty = float(getattr(p, "quantity", 0) or 0)
            entry = float(getattr(p, "entry_price", 0) or 0)
            mark = float(getattr(p, "current_price", None) or
                         getattr(p, "mark_price", 0) or entry)
            upl = float(getattr(p, "unrealized_pnl", None) or
                        ((mark - entry) * qty))
        if not symbol or qty == 0 or entry <= 0:
            continue
        loss_pct = upl / (entry * qty)
        if loss_pct <= pct:  # pct is negative; -0.50 means "loss >= 50%"
            out.append({
                "symbol": symbol,
                "quantity": int(abs(qty)),
                "entry_price": entry,
                "current_price": mark,
                "unrealized_pnl": upl,
                "loss_pct": loss_pct,
            })
    return out


def auto_close_stop_loss(client, positions: Iterable[Any], *,
                        pct: float = -0.50) -> list[dict[str, Any]]:
    """Market-close every position whose loss exceeds ``pct`` of cost basis.

    Returns a list of close results, one per position. Each result is
    ``{"symbol": ..., "status": "closed"|"error", "broker_order_id": ...,
    "error": ...}``. The caller is expected to log this and write it
    into ``state.execution_result`` for the journal.
    """
    to_close = positions_to_close(positions, pct=pct)
    out: list[dict[str, Any]] = []
    for pos in to_close:
        sym = pos["symbol"]
        try:
            res = client.close_position(sym)
            out.append({
                "symbol": sym,
                "status": "closed",
                "broker_order_id": (res or {}).get("id"),
                "loss_pct": pos["loss_pct"],
                "unrealized_pnl": pos["unrealized_pnl"],
            })
            logger.warning(
                "stop-loss closed %s @ loss_pct=%.2f%% unrealized_pnl=%.2f",
                sym, pos["loss_pct"] * 100, pos["unrealized_pnl"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("stop-loss close failed for %s: %s: %s",
                           sym, type(exc).__name__, exc)
            out.append({
                "symbol": sym,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return out


# ---------------------------------------------------------------------------
# (c) Per-symbol cooldown
# ---------------------------------------------------------------------------
def underlying_of_position(pos: Any) -> str | None:
    """Return the OCC underlying (e.g. ``AAPL``) of an option position.

    Long option OCC symbols are ``TICKER + YYMMDD + C/P + 8-digit-strike*1000``
    (e.g. ``AAPL260919C00200000``). For equity positions the symbol IS
    the underlying. We return the first 1-6 letter prefix matching
    ``[A-Z]+`` before the first digit.
    """
    sym = pos.symbol if hasattr(pos, "symbol") else pos.get("symbol") if isinstance(pos, dict) else None
    if not sym:
        return None
    out = []
    for ch in sym:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out) or None


def get_blocked_symbols(
    *,
    positions: Iterable[Any],
    conn: sqlite3.Connection | None,
    cooldown_seconds: float,
    open_loss_block_pct: float = -0.05,
    now_iso: str | None = None,
) -> set[str]:
    """Return the set of symbols blocked from new entries right now.

    A symbol is blocked if either:

      (i)  it currently has an open position in loss (entry vs current
           price ratio worse than ``open_loss_block_pct``). This is the
           "don't double-down on a loser" check.
      (ii) its most recent ``decision_journal`` row that recorded
           ``realized_pnl < 0`` is within the last ``cooldown_seconds``.
           This is the "stay out after a loss" cooldown.

    ``now_iso`` defaults to "now" in UTC ISO-8601. Tests override it for
    determinism.
    """
    blocked: set[str] = set()
    # (i) Open positions in loss.
    for pos in positions:
        if isinstance(pos, dict):
            qty = float(pos.get("qty") or pos.get("quantity") or 0)
            entry = float(pos.get("avg_entry_price") or pos.get("entry_price") or 0)
            mark = float(pos.get("current_price") or pos.get("mark_price") or entry)
            upl = float(pos.get("unrealized_pl") or pos.get("unrealized_pnl") or
                        ((mark - entry) * qty))
        else:
            qty = float(getattr(pos, "quantity", 0) or 0)
            entry = float(getattr(pos, "entry_price", 0) or 0)
            mark = float(getattr(pos, "current_price", None) or
                         getattr(pos, "mark_price", 0) or entry)
            upl = float(getattr(pos, "unrealized_pnl", None) or
                        ((mark - entry) * qty))
        if qty == 0 or entry <= 0:
            continue
        loss_pct = upl / (entry * abs(qty))
        if loss_pct <= open_loss_block_pct:
            ul = underlying_of_position(pos)
            if ul:
                blocked.add(ul)
    # (ii) Recent realized losses in the journal.
    if conn is not None and cooldown_seconds > 0:
        try:
            from datetime import datetime, timedelta, timezone
            if now_iso is None:
                now_dt = datetime.now(timezone.utc)
            else:
                now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            cutoff = (now_dt - timedelta(seconds=cooldown_seconds)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            # The most recent PROCEED row per symbol with a negative
            # realized_pnl, plus the most recent decision_id that recorded
            # a realized loss. We read it as "all symbols with a
            # realized_pnl<0 row newer than the cutoff".
            rows = conn.execute(
                "SELECT DISTINCT underlying_focus FROM decision_journal "
                "WHERE realized_pnl IS NOT NULL AND realized_pnl < 0 "
                "AND completed_at IS NOT NULL AND completed_at > ? "
                "AND underlying_focus IS NOT NULL",
                (cutoff,),
            ).fetchall()
            for (sym,) in rows:
                if sym:
                    blocked.add(sym)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("cooldown journal query failed: %s: %s",
                           type(exc).__name__, exc)
    return blocked


# ---------------------------------------------------------------------------
# Convenience: pre-flight summary written into state.execution_result
# ---------------------------------------------------------------------------
@dataclass
class PreFlightResult:
    """Aggregate of the three pre-flight checks. Stored in
    ``state.supplementary['position_management']`` for the trace."""
    kill_switch: KillSwitchResult | None = None
    stop_loss_closes: list[dict[str, Any]] = field(default_factory=list)
    blocked_symbols: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kill_switch": self.kill_switch.as_dict() if self.kill_switch else None,
            "stop_loss_closes": self.stop_loss_closes,
            "blocked_symbols": sorted(self.blocked_symbols),
        }
