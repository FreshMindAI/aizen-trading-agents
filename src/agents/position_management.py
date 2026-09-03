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
from datetime import datetime, timezone
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
            float(cfg_pm.get("daily_loss_kill_switch_pct", -0.03)),
        ),
        "stop_loss_pct": _env_float(
            "AIZEN_STOP_LOSS_PCT",
            float(cfg_pm.get("stop_loss_pct", -0.30)),
        ),
        "profit_take_pct": _env_float(
            "AIZEN_PROFIT_TAKE_PCT",
            float(cfg_pm.get("profit_take_pct", 0.50)),
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
        # Early-warning threshold: log a WARNING (no auto-close) when a
        # position crosses this loss fraction, halfway to the stop-loss
        # default. Gives the operator a visible signal before disaster.
        "early_warning_pct": _env_float(
            "AIZEN_EARLY_WARNING_PCT",
            float(cfg_pm.get("early_warning_pct", -0.15)),
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
# Symbol classification + loss-pct math
# ---------------------------------------------------------------------------
def _is_option_symbol(symbol: str | None) -> bool:
    """Return True iff ``symbol`` looks like an OCC option contract.

    Equities are 1-5 uppercase letters (e.g. ``AAPL``, ``SPY``, ``GOOGL``).
    OCC option symbols embed a 6-digit date and 8-digit strike multiplier
    after the underlying ticker (e.g. ``COIN260911P00175000``,
    ``SPY260304C00450000``), so any symbol with a digit, or length >= 6,
    is treated as an option.

    Why this matters: for an option, 1 contract represents 100 shares.
    The broker's ``unrealized_pl`` field reports the total dollar P/L for
    the position, but our per-share ``entry`` value times ``qty`` (contracts)
    is NOT the total dollar cost basis. Mixing those two scales was the
    2026-09-03 production bug that produced ``loss_pct=-5721.31%`` in
    the cron log (see ``tests/test_position_management_hardening.py``).
    """
    if not symbol:
        return False
    if len(symbol) >= 6:
        return True
    return any(c.isdigit() for c in symbol)


def _compute_loss_pct(mark: float, entry: float) -> float:
    """Return the unit-consistent loss_pct as a fraction.

    The formula is ``(mark - entry) / entry``: both numerator and
    denominator are per-share (or per-contract when the symbol is a
    contract and entry/mark are contract-sized), so the result is
    always a fraction in the rough range ``[-1.0, +inf)``.

    The broker's ``unrealized_pl`` field is NOT used here because for
    options it is reported in total dollars (1 contract = 100 shares)
    while our ``entry`` and ``qty`` are per-share / per-contract —
    dividing one by the other gave a 100x-blown-up ratio.
    """
    if entry <= 0:
        return 0.0
    return (mark - entry) / entry


def _recompute_unrealized_pnl(
    mark: float, entry: float, qty: float, symbol: str | None,
) -> float:
    """Return the per-position dollar P/L recomputed from (mark, entry).

    For an option, ``qty`` is the number of contracts and 1 contract
    represents 100 shares, so the total dollar P/L is multiplied by
    100. For an equity, ``qty`` is the number of shares (no multiplier).
    This is the fallback when the broker's ``unrealized_pl`` field is
    missing; the broker's value is preferred when present.
    """
    per_unit = (mark - entry) * qty
    if _is_option_symbol(symbol):
        return per_unit * 100
    return per_unit


def _extract_position_fields(p: Any) -> dict[str, Any] | None:
    """Return ``{symbol, qty, entry, mark, broker_upl}`` for either a
    dict or an object position. Returns None when required fields are
    missing or invalid so the caller can skip the position.
    """
    if isinstance(p, dict):
        symbol = p.get("symbol")
        qty = float(p.get("qty") or p.get("quantity") or 0)
        entry = float(p.get("avg_entry_price") or p.get("entry_price") or 0)
        mark = float(p.get("current_price") or p.get("mark_price") or entry)
        broker_upl = p.get("unrealized_pl")
        if broker_upl is None:
            broker_upl = p.get("unrealized_pnl")
    else:
        symbol = getattr(p, "symbol", None)
        qty = float(getattr(p, "quantity", 0) or 0)
        entry = float(getattr(p, "entry_price", 0) or 0)
        mark = float(getattr(p, "current_price", None) or
                     getattr(p, "mark_price", 0) or entry)
        broker_upl = getattr(p, "unrealized_pnl", None)
    if not symbol or qty == 0 or entry <= 0:
        return None
    return {
        "symbol": symbol,
        "qty": qty,
        "entry": entry,
        "mark": mark,
        "broker_upl": float(broker_upl) if broker_upl is not None else None,
    }


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

    ``loss_pct`` is computed from ``(mark - entry) / entry`` — see
    :func:`_compute_loss_pct` for the rationale (the broker's
    ``unrealized_pl`` field is in total dollars while our cost basis
    is per-share, so using the broker's field for the ratio produced
    100x-blown-up values in production for option contracts).
    """
    out: list[dict[str, Any]] = []
    for p in positions:
        fields = _extract_position_fields(p)
        if fields is None:
            continue
        loss_pct = _compute_loss_pct(fields["mark"], fields["entry"])
        if loss_pct > pct:  # pct is negative; -0.50 means "loss >= 50%"
            continue
        upl = (fields["broker_upl"]
               if fields["broker_upl"] is not None
               else _recompute_unrealized_pnl(
                   fields["mark"], fields["entry"],
                   fields["qty"], fields["symbol"]))
        out.append({
            "symbol": fields["symbol"],
            "quantity": int(abs(fields["qty"])),
            "entry_price": fields["entry"],
            "current_price": fields["mark"],
            "unrealized_pnl": upl,
            "loss_pct": loss_pct,
        })
    return out


def early_warning_positions(
    positions: Iterable[Any],
    *,
    pct: float = -0.15,
    stop_loss_pct: float = -0.30,
) -> list[dict[str, Any]]:
    """Return positions whose loss is worse than ``pct`` of cost basis but
    has not yet crossed the ``stop_loss_pct`` threshold. These are pure
    warnings — no auto-close fires — but they give the operator a
    visible signal ~halfway to disaster so they can intervene before
    the stop-loss cutoff is reached.

    The same dict shape as :func:`positions_to_close` is returned so
    the caller can log uniformly. A position that would also be
    returned by :func:`positions_to_close` is *excluded* here (the
    stop-loss auto-close is the louder signal; warning is for the
    "we still have time to act" zone).
    """
    out: list[dict[str, Any]] = []
    for p in positions:
        fields = _extract_position_fields(p)
        if fields is None:
            continue
        loss_pct = _compute_loss_pct(fields["mark"], fields["entry"])
        # Only the "early" zone: between warning and stop-loss.
        if not (loss_pct <= pct and loss_pct > stop_loss_pct):
            continue
        upl = (fields["broker_upl"]
               if fields["broker_upl"] is not None
               else _recompute_unrealized_pnl(
                   fields["mark"], fields["entry"],
                   fields["qty"], fields["symbol"]))
        out.append({
            "symbol": fields["symbol"],
            "quantity": int(abs(fields["qty"])),
            "entry_price": fields["entry"],
            "current_price": fields["mark"],
            "unrealized_pnl": upl,
            "loss_pct": loss_pct,
        })
    return out


def profit_take_positions(
    positions: Iterable[Any],
    *,
    pct: float = 0.50,
) -> list[dict[str, Any]]:
    """Return positions whose unrealized GAIN exceeds ``pct`` of cost
    basis. The intent is to lock in option premium before it decays
    or the move reverses.

    Symmetric to :func:`positions_to_close` but for the +X% side. The
    default ``+0.50`` (+50%) is the user-tuned value; the previous
    behavior had no profit-take, so winners could round-trip back to
    breakeven before the operator noticed. The same idempotency
    rules apply (a 404 on the close is success, not error).
    """
    out: list[dict[str, Any]] = []
    for p in positions:
        fields = _extract_position_fields(p)
        if fields is None:
            continue
        gain_pct = _compute_loss_pct(fields["mark"], fields["entry"])
        if gain_pct < pct:  # pct is positive; 0.50 means "gain >= 50%"
            continue
        upl = (fields["broker_upl"]
               if fields["broker_upl"] is not None
               else _recompute_unrealized_pnl(
                   fields["mark"], fields["entry"],
                   fields["qty"], fields["symbol"]))
        out.append({
            "symbol": fields["symbol"],
            "quantity": int(abs(fields["qty"])),
            "entry_price": fields["entry"],
            "current_price": fields["mark"],
            "unrealized_pnl": upl,
            "gain_pct": gain_pct,
        })
    return out


def auto_close_profit_take(client, positions: Iterable[Any], *,
                          pct: float = 0.50) -> list[dict[str, Any]]:
    """Market-close every position whose gain exceeds ``pct`` of cost
    basis. Symmetric to :func:`auto_close_stop_loss`. Same 404-as-
    already-closed idempotency rule applies.
    """
    to_close = profit_take_positions(positions, pct=pct)
    out: list[dict[str, Any]] = []
    for pos in to_close:
        sym = pos["symbol"]
        try:
            res = client.close_position(sym)
            out.append({
                "symbol": sym,
                "status": "closed",
                "broker_order_id": (res or {}).get("id"),
                "gain_pct": pos["gain_pct"],
                "unrealized_pnl": pos["unrealized_pnl"],
            })
            logger.info(
                "profit-take closed %s @ gain_pct=%.2f%% unrealized_pnl=%.2f",
                sym, pos["gain_pct"] * 100, pos["unrealized_pnl"],
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            if "404" in msg or "position does not exist" in msg.lower() \
                    or "not found" in msg.lower():
                logger.info(
                    "profit-take: %s already closed by another path (%s)",
                    sym, msg,
                )
                out.append({
                    "symbol": sym,
                    "status": "already_closed",
                    "gain_pct": pos["gain_pct"],
                    "unrealized_pnl": pos["unrealized_pnl"],
                })
                continue
            logger.warning("profit-take close failed for %s: %s", sym, msg)
            out.append({
                "symbol": sym,
                "status": "error",
                "error": msg,
            })
    return out


def auto_close_stop_loss(client, positions: Iterable[Any], *,
                        pct: float = -0.50) -> list[dict[str, Any]]:
    """Market-close every position whose loss exceeds ``pct`` of cost basis.

    Returns a list of close results, one per position. Each result is
    ``{"symbol": ..., "status": "closed"|"already_closed"|"error",
    "broker_order_id": ..., "error": ...}``. The caller is expected to
    log this and write it into ``state.execution_result`` for the journal.

    Idempotency: a 404 from the broker (the position was closed by
    another path between our snapshot and this call, or a stale local
    position list) is logged as ``already_closed`` not ``error``. The
    operator's intent (position is closed) is achieved, so the
    WARNING-level log that would fire on a real failure is suppressed
    in favor of an INFO line.
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
            # 404 / "no position" => already closed by another path. Treat
            # as success for cycle purposes (the desired state is achieved);
            # the next refresh will drop the position from the live list.
            msg = f"{type(exc).__name__}: {exc}"
            if "404" in msg or "position does not exist" in msg.lower() \
                    or "not found" in msg.lower():
                logger.info(
                    "stop-loss: %s already closed by another path (%s)",
                    sym, msg,
                )
                out.append({
                    "symbol": sym,
                    "status": "already_closed",
                    "loss_pct": pos["loss_pct"],
                    "unrealized_pnl": pos["unrealized_pnl"],
                })
                continue
            logger.warning("stop-loss close failed for %s: %s", sym, msg)
            out.append({
                "symbol": sym,
                "status": "error",
                "error": msg,
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
    # Defense in depth: if the broker returned an empty list, the
    # operator cannot tell "I really have no positions" from "broker
    # silently failed". The 2026-09-03 cloud cron run was the
    # canonical example: list_positions() returned [] with no
    # exception, the open-loss check found nothing, and the
    # supervisor re-entered AVGO while the live portfolio still
    # held two losing AVGO puts. Emit a WARNING so the operator
    # sees the position-management layer is operating blind.
    # The orchestrator's pre-flight already short-circuits to
    # NO_TRADE/BLOCKED on a real exception (graph.py fail-closed
    # path); this is the equivalent signal for the silent-empty
    # case.
    if not list(positions or []):
        logger.warning(
            "get_blocked_symbols called with empty positions list - "
            "broker may have silently failed. Open-loss and stop-loss "
            "checks are operating on an empty portfolio; do not trust "
            "a NO_TRADE-on-cooldown result until the broker is "
            "verified reachable."
        )
    # (i) Open positions in loss.
    for pos in positions:
        fields = _extract_position_fields(pos)
        if fields is None:
            continue
        loss_pct = _compute_loss_pct(fields["mark"], fields["entry"])
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
    profit_take_closes: list[dict[str, Any]] = field(default_factory=list)
    blocked_symbols: set[str] = field(default_factory=set)
    early_warnings: list[dict[str, Any]] = field(default_factory=list)
    kill_switch_latched: bool = False
    broker_unreachable: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kill_switch": self.kill_switch.as_dict() if self.kill_switch else None,
            "stop_loss_closes": self.stop_loss_closes,
            "profit_take_closes": self.profit_take_closes,
            "blocked_symbols": sorted(self.blocked_symbols),
            "early_warnings": self.early_warnings,
            "kill_switch_latched": self.kill_switch_latched,
            "broker_unreachable": self.broker_unreachable,
        }


# ---------------------------------------------------------------------------
# (d) Daily kill-switch latch
#
# The kill-switch is per-tick, not per-day. If it trips at 11:00 ET,
# realized P/L stops being negative at 15:00 ET (because the stop-loss
# closed the worst positions), the next tick would see total P/L back
# above the cap and the kill-switch would NOT re-fire. But the operator
# intent was "no new entries for the rest of today". Persist the trip
# in the ``kill_switch_latch`` table for the UTC day; the next tick
# reads it and short-circuits to NO_TRADE without re-running the
# math. Cleared implicitly by the date-based primary key (a new
# calendar day = a new row).
# ---------------------------------------------------------------------------
def _latch_table_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff the ``kill_switch_latch`` table exists.

    Older DBs may predate sql/71_kill_switch_latch.sql; tests that
    build a minimal schema without running the full migration set
    will hit this path. A fresh ``init_db`` on a normal DB will
    always have it.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kill_switch_latch'"
        ).fetchone()
        return row is not None
    except Exception:  # pragma: no cover
        return False


def is_kill_switch_latched_today(
    conn: sqlite3.Connection | None,
    *,
    now: datetime | None = None,
) -> bool:
    """Return True iff a kill-switch trip has been recorded for today's
    UTC date. ``conn=None`` returns False (no DB = no latch); the
    caller is expected to handle that as "proceed normally"."""
    if conn is None or not _latch_table_ready(conn):
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    day_prefix = now.strftime("%Y-%m-%d")
    try:
        row = conn.execute(
            "SELECT 1 FROM kill_switch_latch WHERE day_utc = ?",
            (day_prefix,),
        ).fetchone()
        return row is not None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("kill_switch_latch read failed: %s: %s",
                       type(exc).__name__, exc)
        return False


def record_kill_switch_latch(
    conn: sqlite3.Connection | None,
    *,
    total_pnl: float,
    threshold_usd: float,
    pct: float,
    now: datetime | None = None,
) -> None:
    """Persist a kill-switch trip for the current UTC day. Idempotent
    (re-recording overwrites the snapshot). No-op if conn is None or
    the table is missing."""
    if conn is None or not _latch_table_ready(conn):
        return
    if now is None:
        now = datetime.now(timezone.utc)
    day_prefix = now.strftime("%Y-%m-%d")
    iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        conn.execute(
            "INSERT OR REPLACE INTO kill_switch_latch "
            "(day_utc, breached_at, total_pnl, threshold_usd, pct) "
            "VALUES (?, ?, ?, ?, ?)",
            (day_prefix, iso, float(total_pnl), float(threshold_usd), float(pct)),
        )
        conn.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("kill_switch_latch write failed: %s: %s",
                       type(exc).__name__, exc)
