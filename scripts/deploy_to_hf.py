"""Deploy the latest direction-classifier (XGBoost) to Hugging Face.

The hackathon deployment model is "keep Render lite": the dashboard
and the multi-agent orchestrator run on Render, while the trained
ML artifacts live on Hugging Face. This script:

  1. Finds the most recent ``models/direction_h4_xgb_clf-*.pkl`` (or
     accepts a path via --model).
  2. Builds a model card (README.md) summarising the task, features,
     and test metrics, plus a usage snippet.
  3. Creates a Hugging Face repo if missing.
  4. Uploads the .pkl, the .meta.json, and the README.

Security
  The HF token is read from ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``).
  It is NEVER printed, logged, or written to disk. The token is set
  as the ``HUGGING_FACE_HUB_TOKEN`` env var that ``huggingface_hub``
  reads internally; the call surface here only references it through
  ``HfApi(token=...)`` and never calls ``print(token)`` or
  ``repr(token)``.

Usage
  HF_TOKEN=hf_xxx python scripts/deploy_to_hf.py \\
      --namespace AdithyaByri --repo direction-h4-clf
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# IMPORTANT: huggingface_hub reads the token from any of these env vars.
# We never echo the value; we only assert it is set.
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

logger = logging.getLogger("deploy_to_hf")


def _resolve_token() -> str:
    """Read the HF token from env. Never prints it; raises if missing."""
    for var in HF_TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            # Sanity: token must look like an HF token (start with hf_).
            if not val.startswith("hf_"):
                raise ValueError(
                    f"{var} is set but does not start with 'hf_' — refusing to "
                    "use an obviously-wrong token. Re-check the value."
                )
            return val
    raise SystemExit(
        f"error: none of {HF_TOKEN_ENV_VARS} are set. Export the HF token "
        "and re-run. (Never paste the token into the shell history or a "
        "committable file.)"
    )


def _find_latest_model(models_dir: Path) -> tuple[Path, Path]:
    """Return ``(pkl_path, meta_path)`` for the newest direction_h4 model.

    The newest is determined by the timestamp encoded in the filename
    (``direction_h4_xgb_clf-YYYYMMDD-HHMMSS.pkl``); ties broken by
    lexicographic order. Raises FileNotFoundError if none exist.
    """
    pkls = sorted(models_dir.glob("direction_h4_xgb_clf-*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"no direction_h4_xgb_clf-*.pkl under {models_dir}"
        )
    pkl = pkls[-1]
    meta = pkl.with_name(pkl.stem + ".meta.json")
    if not meta.exists():
        raise FileNotFoundError(f"missing sidecar meta: {meta}")
    return pkl, meta


def _build_model_card(meta: dict, pkl: Path) -> str:
    """Render a HF model card (README.md) for this artifact."""
    metrics = (meta.get("test_metrics") or {}).get("test") or {}
    val_metrics = (meta.get("test_metrics") or {}).get("val") or {}
    created = meta.get("created_at_utc", "?")
    version = meta.get("model_version", pkl.stem)
    features = meta.get("features") or []
    split = meta.get("split_bounds") or {}
    frozen = meta.get("frozen_params") or {}

    feature_table = "\n".join(f"- `{f}`" for f in features)

    metrics_lines = []
    for k, v in (val_metrics or {}).items():
        metrics_lines.append(f"  - val **{k}** = {v}")
    for k, v in metrics.items():
        metrics_lines.append(f"  - test **{k}** = {v}")

    metrics_block = "\n".join(metrics_lines) if metrics_lines else "  - (no metrics recorded)"

    return f"""---
language: en
license: mit
tags:
  - trading
  - xgboost
  - finance
  - alpaca
  - direction-classifier
  - hackathon
---

# Aizen Trading — Direction Classifier (h=4 bars)

Trained XGBoost model that predicts the **directional probability** of an
underlying's next-4-bars return. The 4-bar horizon is approximately
1 hour on a 15-minute bar grid. The model is a building block of the
multi-agent trading system described in
[`aizentrading/Aizen-Trading`](https://github.com/aizentrading/Aizen-Trading).

## Model

- **Task**: binary classification (1 = up over the next 4 bars)
- **Version**: `{version}`
- **Created**: `{created}`
- **Frozen params**: tau = {frozen.get("tau")}, cost = {frozen.get("cost")}

## Features ({len(features)})

{feature_table}

## Data splits

- train: ends `{split.get("train_end", "?")}`
- val:   `{split.get("val_start", "?")}` to `{split.get("val_end", "?")}`
- test:  starts `{split.get("test_start", "?")}`

## Metrics

{metrics_block}

## Usage

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

pkl_path = hf_hub_download(
    repo_id="AdithyaByri/direction-h4-clf",
    filename="direction_h4_xgb_clf-*.pkl",
)
clf = joblib.load(pkl_path)
# `clf` is a sklearn-style XGBClassifier with .predict_proba(X)[:, 1]
proba = clf.predict_proba(X)[:, 1]
```

## Provenance

Trained by the orchestrator's nightly retrain step
(`src/agents/train_direction.py`). Deployed via
`scripts/deploy_to_hf.py`. The model is re-trained daily on a
walk-forward split and the latest version replaces the previous one
on this hub.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--namespace", required=True,
                   help="HF namespace (user or org), e.g. 'AdithyaByri'")
    p.add_argument("--repo", default="direction-h4-clf",
                   help="HF repo name (default: direction-h4-clf)")
    p.add_argument("--model", default=None,
                   help="Path to a specific .pkl (default: newest in models/)")
    p.add_argument("--private", action="store_true",
                   help="Make the repo private (default: public)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    token = _resolve_token()  # raises if missing; never logs the value

    # Lazy import so the token is not imported until the user runs us.
    from huggingface_hub import HfApi, whoami

    api = HfApi(token=token)

    # Sanity: confirm the token resolves to a real user. whoami() does
    # NOT return the token; it returns the user/org name + role. We
    # log only the username, not the token.
    me = whoami(token=token)
    user = (me.get("name") or me.get("fullname") or "").strip()
    logger.info("authenticated as %s", user or "<unknown>")

    if args.model:
        pkl = Path(args.model)
        meta = pkl.with_name(pkl.stem + ".meta.json")
    else:
        pkl, meta = _find_latest_model(REPO / "models")
    logger.info("deploying %s (%d bytes)", pkl.name, pkl.stat().st_size)

    repo_id = f"{args.namespace}/{args.repo}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    logger.info("repo ready: https://huggingface.co/%s", repo_id)

    # Build the model card in a temp path so we don't litter the repo.
    card_path = REPO / "models" / f"README_HF.md"
    card_path.write_text(_build_model_card(json.loads(meta.read_text()), pkl),
                         encoding="utf-8")

    # Upload the three files. upload_file is the simple, transactional
    # one-file-at-a-time call; for a 1.3MB model + 1KB card + 1KB meta,
    # batching is unnecessary.
    for src, dst in [
        (pkl, pkl.name),
        (meta, meta.name),
        (card_path, "README.md"),
    ]:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"deploy {pkl.name}",
        )
        logger.info("uploaded %s -> %s/%s", src.name, repo_id, dst)

    # Clean up the local README so we don't ship a stale card to git.
    card_path.unlink(missing_ok=True)
    print(f"[deploy_to_hf] ok: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
