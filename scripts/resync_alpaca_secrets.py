"""Re-upload the local .env Alpaca keys to GitHub Secrets.

Use when the cron job starts getting 403s from Alpaca, which usually
means the encrypted secret has drifted from the live .env (e.g. the
account was recreated, or the secret was set from a stale copy).
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import requests
from nacl import encoding, public


def main() -> int:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        print(f"no .env at {env_path}", file=sys.stderr)
        return 2
    env_dict: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        env_dict[k.strip()] = v.strip()

    required = ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
    for r in required:
        if r not in env_dict or not env_dict[r]:
            print(f"missing {r} in .env", file=sys.stderr)
            return 2

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        print("GH_TOKEN env var required", file=sys.stderr)
        return 2

    owner_repo = "FreshMindAI/aizen-trading-agents"
    headers = {"Authorization": f"token {gh_token}"}

    pk_resp = requests.get(
        f"https://api.github.com/repos/{owner_repo}/actions/secrets/public-key",
        headers=headers,
    )
    if pk_resp.status_code != 200:
        print(f"public-key failed: {pk_resp.status_code} {pk_resp.text[:200]}", file=sys.stderr)
        return 2
    pk = pk_resp.json()
    key_id = pk["key"]
    pub_b64 = pk["key"]

    def encrypt(value: str) -> str:
        pub = public.PublicKey(pub_b64.encode("utf-8"), encoding.Base64Encoder())
        sealed = public.SealedBox(pub)
        return base64.b64encode(sealed.encrypt(value.encode("utf-8"))).decode("utf-8")

    for name in required:
        encrypted = encrypt(env_dict[name])
        r = requests.put(
            f"https://api.github.com/repos/{owner_repo}/actions/secrets/{name}",
            headers=headers,
            json={"encrypted_value": encrypted, "key_id": key_id},
        )
        print(f"PUT {name}: {r.status_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
