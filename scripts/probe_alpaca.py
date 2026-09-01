"""Probe multiple Alpaca endpoints to diagnose 401s.

Reads keys from .env (no command-line exposure), tries a list of
endpoints, prints just the status code + a short body preview. Used
to figure out which Alpaca host the paper account is actually
served from (some accounts moved off paper-api.alpaca.markets in
2025/2026).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

key = os.environ.get("ALPACA_API_KEY_ID", "")
secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
if not key or not secret:
    print("no ALPACA keys in env", file=sys.stderr)
    sys.exit(2)

endpoints = [
    # Trading
    ("paper", "GET",  "https://paper-api.alpaca.markets/v2/account"),
    ("paper", "GET",  "https://paper-api.alpaca.markets/v2/orders?status=all&limit=1"),
    ("paper", "GET",  "https://paper-api.alpaca.markets/v2/positions"),
    # Data (free IEX vs paid SIP)
    ("data",  "GET",  "https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=1Day&start=2026-08-30T00:00:00Z&limit=1"),
    ("data",  "GET",  "https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=1Day&start=2026-08-30T00:00:00Z&limit=1", {"Apca-Data-Source": "iex"}),
    # Some accounts are routed to a different host
    ("alt",   "GET",  "https://broker-api.alpaca.markets/v2/account"),
]

for entry in endpoints:
    if len(entry) == 3:
        label, method, url = entry
        headers = {}
    else:
        label, method, url, headers = entry
    r = requests.request(method, url, auth=(key, secret), headers=headers, timeout=10)
    body_preview = (r.text or "")[:160].replace("\n", " ")
    print(f"[{r.status_code}] {label:5s} {url}  -> {body_preview}")
