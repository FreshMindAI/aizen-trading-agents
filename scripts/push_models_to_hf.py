"""Upload the most recent version of each trained model to the
HuggingFace dataset `FreshMindAI/aizen-models`.

We upload:
  * the binary (.pkl / .pt)
  * the corresponding .meta.json (so the orchestrator can locate
    the artifact via the same lookup it uses locally)

Historical versions are intentionally skipped — the orchestrator
only ever loads the *most recent* per family, so uploading
intermediate versions would waste storage and slow downloads.

HF token is read from the HF_TOKEN env var; it must NOT be echoed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami


# REPO_ID is derived at runtime from the authenticated user's namespace
# so that any maintainer can run this script without needing
# FreshMindAI namespace write access.
REPO_NAME = "aizen-models"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def latest_per_family() -> list[tuple[str, Path, Path]]:
    """For each model family (the substring before the timestamp),
    return the (name, .pkl|.pt, .meta.json) tuple for the most recent
    version based on the meta.json's `created_at*` field."""
    metas: list[Path] = sorted(MODELS_DIR.glob("*.meta.json"))
    by_family: dict[str, list[Path]] = {}
    for m in metas:
        stem = m.name.replace(".meta.json", "")
        # family = prefix before the timestamp
        # e.g. direction_h4_xgb_clf-20260830-011107
        idx = stem.rfind("-20")
        if idx == -1:
            continue
        family = stem[:idx]
        by_family.setdefault(family, []).append(m)

    out: list[tuple[str, Path, Path]] = []
    for family, items in by_family.items():
        def key(p: Path) -> str:
            try:
                meta = json.loads(p.read_text())
                return (
                    meta.get("created_at_utc")
                    or meta.get("created_at")
                    or p.stem
                )
            except Exception:
                return p.stem

        items.sort(key=key)
        chosen = items[-1]  # latest
        stem = chosen.name.replace(".meta.json", "")
        binary = MODELS_DIR / f"{stem}.pkl"
        if not binary.exists():
            binary = MODELS_DIR / f"{stem}.pt"
        if not binary.exists():
            print(f"!! missing binary for {stem}, skipping", file=sys.stderr)
            continue
        out.append((stem, binary, chosen))
    return out


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var is required", file=sys.stderr)
        return 2
    if not token.startswith("hf_"):
        print("HF_TOKEN does not look like a valid HF token", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    try:
        me = whoami(token=token)
    except Exception as exc:
        print(f"whoami failed: {exc}", file=sys.stderr)
        return 2
    user = me.get("name") or ""
    repo_id = f"{user}/{REPO_NAME}"
    print(f"authenticated as: {user}")
    print(f"target repo: https://huggingface.co/datasets/{repo_id}")

    # Create the dataset repo if it doesn't exist.
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        print(f"repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        print(f"create_repo failed: {exc}", file=sys.stderr)
        return 2

    pairs = latest_per_family()
    print(f"uploading {len(pairs)} model artifacts (binary + meta) ...")
    for stem, binary, meta in pairs:
        rel_bin = f"models/{binary.name}"
        rel_meta = f"models/{meta.name}"
        try:
            api.upload_file(
                path_or_fileobj=str(binary),
                path_in_repo=rel_bin,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"upload {binary.name}",
            )
            api.upload_file(
                path_or_fileobj=str(meta),
                path_in_repo=rel_meta,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"upload {meta.name}",
            )
            size_kb = binary.stat().st_size / 1024
            print(f"  + {binary.name} ({size_kb:.1f} KB) + {meta.name}")
        except Exception as exc:
            print(f"  ! upload failed for {stem}: {exc}", file=sys.stderr)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
