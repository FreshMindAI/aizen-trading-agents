"""Pipeline configuration: Alpaca credentials + tunables.

Credentials live only in the project-root .env (gitignored). Everything else
that drives the multi-agent system is in `config/*.yaml` so the same code
runs against paper, dry-run, or (with explicit override) live endpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load eagerly at import time so every entry point sees the same environment.
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_UNIVERSE = "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD"

# Alpaca base hosts. Defaults match the paper environment; YAML overrides.
DATA_HOST = "https://data.alpaca.markets"
PAPER_HOST = "https://paper-api.alpaca.markets"

CONFIG_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def _load_yaml_bundle() -> dict[str, dict[str, Any]]:
    """Load every config/*.yaml once per process and merge the small subset
    that the rest of the codebase expects as flat fields."""
    bundle: dict[str, dict[str, Any]] = {}
    for name in ("settings", "alpaca", "agents", "risk", "gnn"):
        bundle[name] = _read_yaml(CONFIG_DIR / f"{name}.yaml")
    return bundle


def get_yaml(name: str) -> dict[str, Any]:
    """Return one top-level YAML section (e.g. 'risk', 'agents')."""
    return _load_yaml_bundle().get(name, {})


# ---------------------------------------------------------------------------
# Settings dataclass (preserves the original public API for back-compat)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    api_key_id: str
    api_secret: str
    universe: list[str]

    timeframe: str = "15Min"
    feed: str = "iex"                # free equities feed; recorded on every row/run
    adjustment: str = "split"        # split-adjusted equity bars

    # Option contract selection
    expiry_min_dte: int = 7          # expiration_date_gte = today + N days
    expiry_max_dte: int = 45         # expiration_date_lte = today + N days
    strike_offset: int = 5           # keep +/-N strikes around ATM per expiry
    strike_band_pct: float = 0.10    # server-side strike band = spot * (1 +/- band)
    pilot_contract_cap: int = 12     # deterministic top-N nearest ATM per symbol

    # Request pacing / retries
    rate_limit_per_min: int = 190    # budget under the ~200/min free tier
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0
    timeout_s: tuple[float, float] = (5.0, 30.0)

    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "AIZEN_DB_PATH",
                str(PROJECT_ROOT / "data" / "trading.db"),
            )
        )
    )

    # --- Phase-3 multi-agent fields ------------------------------------------
    run_mode: str = "paper"          # paper | dry-run | live
    data_base_url: str = DATA_HOST
    trading_base_url: str = PAPER_HOST
    decision_journal_table: str = "decision_journal"

    # Account capital (USD). Drives the risk engine's per-trade / per-
    # symbol / gross caps when ``risk.yaml`` is set to "scale from
    # capital" mode (see ``RiskLimits.scaled_from_capital``). Hackathon
    # default is $100k so the multi-agent pipeline produces realistic
    # position sizes; an explicit AIZEN_CAPITAL env var wins.
    capital_usd: float = 100_000.0
    agent_config_path: Path = CONFIG_DIR / "agents.yaml"
    risk_config_path: Path = CONFIG_DIR / "risk.yaml"
    alpaca_config_path: Path = CONFIG_DIR / "alpaca.yaml"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and add your Alpaca paper credentials."
        )
    return value


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build Settings once per process. Never repr() the result: it holds secrets."""
    alpaca_cfg = get_yaml("alpaca")
    settings_cfg = get_yaml("settings")

    data_cfg = alpaca_cfg.get("data", {})
    trading_cfg = alpaca_cfg.get("trading", {})

    return Settings(
        api_key_id=_require("ALPACA_API_KEY_ID"),
        api_secret=_require("ALPACA_API_SECRET_KEY"),
        universe=[s.strip() for s in os.getenv("UNIVERSE", DEFAULT_UNIVERSE).split(",") if s.strip()],
        timeframe=os.getenv("TIMEFRAME", Settings.timeframe),
        feed=os.getenv("FEED", data_cfg.get("feed", Settings.feed)),
        adjustment=os.getenv("ADJUSTMENT", data_cfg.get("adjustment", Settings.adjustment)),
        expiry_min_dte=_env_int("EXPIRY_MIN_DTE", Settings.expiry_min_dte),
        expiry_max_dte=_env_int("EXPIRY_MAX_DTE", Settings.expiry_max_dte),
        strike_offset=_env_int("STRIKE_OFFSET", Settings.strike_offset),
        strike_band_pct=_env_float("STRIKE_BAND_PCT", Settings.strike_band_pct),
        pilot_contract_cap=_env_int("PILOT_CONTRACT_CAP", Settings.pilot_contract_cap),
        rate_limit_per_min=_env_int(
            "RATE_LIMIT_PER_MIN", data_cfg.get("rate_limit_per_min", Settings.rate_limit_per_min)
        ),
        max_retries=_env_int("MAX_RETRIES", data_cfg.get("max_retries", Settings.max_retries)),
        backoff_base_s=_env_float("BACKOFF_BASE_S", Settings.backoff_base_s),
        backoff_cap_s=_env_float("BACKOFF_CAP_S", Settings.backoff_cap_s),
        # Phase-3 additions
        run_mode=os.getenv("RUN_MODE", alpaca_cfg.get("run_mode", Settings.run_mode)),
        data_base_url=os.getenv("ALPACA_DATA_URL", data_cfg.get("base_url", Settings.data_base_url)),
        trading_base_url=os.getenv(
            "ALPACA_TRADING_URL", trading_cfg.get("base_url", Settings.trading_base_url)
        ),
        decision_journal_table=settings_cfg.get(
            "project", {}
        ).get("decision_journal_table", Settings.decision_journal_table),
        capital_usd=_env_float("AIZEN_CAPITAL", Settings.capital_usd),
    )
