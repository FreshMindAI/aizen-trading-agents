"""Push README.md to the aizen-models HF dataset."""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi

REPO_NAME = "aizen-models"

README = """---
license: mit
---

# Aizen Trading Models

Most-recent trained artifacts for the Aizen multi-agent trading
system. Each model family is uploaded with its `.pkl`/`.pt` binary
+ a `.meta.json` sidecar (test metrics, hyper-params, feature list).
Only the latest version per family is kept; the orchestrator
selects on the `created_at_utc` / `created_at` field.

## Models

- `direction_h4_xgb_clf` - XGBoost classifier, 4-bar (1h) forward
  return direction (underlying).
- `direction_h16_xgb_clf` - XGBoost classifier, 16-bar (4h)
  forward return direction (underlying).
- `option_h4_xgb_clf` - XGBoost classifier, 4-bar option
  opportunity flag.
- `option_h4_xgb_reg` - XGBoost regressor, 4-bar option payoff.
- `option_h16_xgb_clf` - XGBoost classifier, 16-bar option
  opportunity flag.
- `option_h16_xgb_reg` - XGBoost regressor, 16-bar option payoff.
- `rv_h4_xgb_reg` - XGBoost regressor, 4-bar realized volatility.
- `rv_h16_xgb_reg` - XGBoost regressor, 16-bar realized
  volatility.
- `gnn` - GCN baseline, 32 hidden / 16 out, fixed topology.
- `gatv2-news` - GATv2 with news-driven dynamic topology,
  32 hidden / 1 head.

## Usage

```python
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="<user>/aizen-models",
    repo_type="dataset",
    allow_patterns=["models/*.pkl", "models/*.pt", "models/*.meta.json"],
)
# path/ contains the latest binaries; the orchestrator picks
# the freshest meta per family.
```
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token or not token.startswith("hf_"):
        print("HF_TOKEN env var required (must start with hf_)", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=f"AdithyaByri/{REPO_NAME}",
        repo_type="dataset",
        commit_message="add README",
    )
    print("README uploaded to AdithyaByri/aizen-models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
