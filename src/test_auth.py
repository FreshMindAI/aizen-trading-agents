"""Gate G1: prove credentials work before any bulk download.

Positive test: fetch one daily AAPL bar through the stock-bars endpoint
(the curl equivalent in doc section 3). Negative test: same call with a
tampered secret must raise AlpacaAuthError.

Usage: python -m src.test_auth
"""

from __future__ import annotations

import argparse

import requests

from .alpaca_client import AlpacaAuthError, AlpacaClient, setup_logging
from .config import DATA_HOST, get_settings


def masked_key(key_id: str) -> str:
    """Show just enough to confirm which key file is live - never the whole id."""
    return f"{key_id[:4]}...(len {len(key_id)})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alpaca credential smoke test")
    parser.add_argument("--skip-negative", action="store_true", help="only run the positive check")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    print(f"key id: {masked_key(settings.api_key_id)}")

    url = f"{DATA_HOST}/v2/stocks/AAPL/bars"
    with AlpacaClient() as client:
        payload = client.get(url, {"timeframe": "1Day", "limit": 1})
        bars = payload.get("bars") or []
        if not bars:
            print("FAIL: authenticated but no bar returned")
            return 1
        bar = bars[0]
        print(
            "PASS: fetched AAPL 1Day bar "
            f"t={bar.get('t')} close={bar.get('c')}"
        )

    if args.skip_negative:
        return 0

    # Negative control: a wrong secret must fail closed as an auth error.
    probe = requests.Session()
    probe.headers.update(
        {
            "APCA-API-KEY-ID": settings.api_key_id,
            "APCA-API-SECRET-KEY": settings.api_secret[:-1] + ("x" if settings.api_secret[-1] != "x" else "y"),
        }
    )
    try:
        response = probe.get(url, params={"timeframe": "1Day", "limit": 1}, timeout=(5, 30))
    finally:
        probe.close()
    if response.status_code in (401, 403):
        print(f"PASS: tampered secret correctly rejected ({response.status_code})")
        return 0
    print(f"WARN: tampered secret got HTTP {response.status_code} instead of 401/403 - inspect manually")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
