"""Push the local decision_journal to a HuggingFace dataset.

The GitHub Actions cron runs on an ephemeral runner - its
``data/trading.db`` is wiped on every invocation, so all PROCEED
decisions the cloud makes are lost the moment the job exits. The
Alpaca paper account keeps a record of what was actually filled,
but the orchestrator's journal (with thesis, model_versions, agent
observations) lives only in the runner's SQLite file.

This script snapshots the local ``decision_journal`` table to a
JSONL file and uploads it to a Hugging Face dataset
(``AdithyaByri/aizen-journal``) where it survives the runner
shutdown. Local dev can pull the dataset back to reconstruct the
journal after the cloud has run for a while.

Wire-up: the cron workflow's last step is

    python scripts/push_journal_to_hf.py

so every tick leaves a journal row in the HF dataset. The script
is no-op if no new decisions have been recorded since the last
upload (it tracks the last-uploaded timestamp in
``data/journal_upload_state.json``).

Security
  The HF token is read from ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``,
  ``HUGGING_FACE_HUB_TOKEN``). It is NEVER printed, logged, or
  written to disk. ``huggingface_hub`` reads it directly from the
  env; we only assert it is set.

Usage (one-shot, from the cron tick):
    python scripts/push_journal_to_hf.py

Usage (back-fill a local DB to the dataset):
    python scripts/push_journal_to_hf.py --full
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
DEFAULT_DATASET = "AdithyaByri/aizen-journal"
DEFAULT_DB = "data/trading.db"
DEFAULT_STATE = "data/journal_upload_state.json"

logger = logging.getLogger("push_journal_to_hf")


def _resolve_token() -> str:
    """Read the HF token from env. Never prints it; raises if missing
    or malformed. Mirrors the contract used in scripts/deploy_to_hf.py
    so a wrong token surfaces the same error in both scripts."""
    for var in HF_TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            if not val.startswith("hf_"):
                raise ValueError(
                    f"{var} is set but does not start with 'hf_' - "
                    "refusing to use an obviously-wrong token."
                )
            return val
    raise SystemExit(
        f"error: none of {HF_TOKEN_ENV_VARS} are set. Export the HF "
        "token in the workflow env (HF_TOKEN: ${{ secrets.HF_TOKEN }})."
    )


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not parse state file %s: %s", path, exc)
    return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# Columns that contain LLM session content (envelopes, observations,
# free-text messages) and are redacted from the default upload.
# The cloud journal exists to reconstruct the *trading signal* (what
# we decided, what we filled, what the ML said) - not the LLM
# prompt/response transcripts. Those stay local. Override with
# ``--include-llm-transcripts`` to upload the raw content.
_LLM_TRANSCRIPT_COLUMNS = (
    "agent_observations_json",  # list of AgentObservation dicts
    "agent_messages_json",      # list of AgentMessage envelopes
)

# Columns that are the trading signal itself (kept by default).
_SIGNAL_JSON_COLUMNS = (
    "strategy_proposal_json", "selected_strategy_json",
    "market_snapshot_json", "ml_prediction_json",
    "gnn_output_json", "research_output_json",
    "execution_result_json", "model_versions",
)


def _new_decisions(
    conn: sqlite3.Connection,
    since: str | None,
    *,
    include_llm_transcripts: bool = False,
) -> list[dict]:
    """Return all decision_journal rows newer than ``since`` (ISO ts).
    A None ``since`` means full backfill.

    By default, LLM transcript columns (``agent_observations_json``,
    ``agent_messages_json``) are redacted to ``[]`` with a marker
    so the row still round-trips but does not leak prompt/response
    content to a public dataset. Pass
    ``include_llm_transcripts=True`` to keep the raw values.
    """
    sql = "SELECT * FROM decision_journal"
    params: tuple = ()
    if since:
        sql += " WHERE timestamp > ?"
        params = (since,)
    sql += " ORDER BY timestamp ASC"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    out: list[dict] = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        # JSON columns come back as strings; keep them as strings in
        # the JSONL output so a re-import is a straight round-trip.
        for jk in _SIGNAL_JSON_COLUMNS:
            if jk in d and d[jk] is not None and not isinstance(d[jk], str):
                d[jk] = json.dumps(d[jk], default=str)
        # LLM transcript columns: redact by default.
        if include_llm_transcripts:
            for jk in _LLM_TRANSCRIPT_COLUMNS:
                if jk in d and d[jk] is not None and not isinstance(d[jk], str):
                    d[jk] = json.dumps(d[jk], default=str)
        else:
            for jk in _LLM_TRANSCRIPT_COLUMNS:
                d[jk] = json.dumps(
                    [{"_redacted": "llm_transcript", "column": jk}],
                    default=str,
                )
        out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=os.getenv("AIZEN_DB_PATH", DEFAULT_DB),
                    help=f"path to trading.db (default: {DEFAULT_DB})")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"HF dataset repo id (default: {DEFAULT_DATASET})")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help="local state file tracking the last-uploaded ts")
    ap.add_argument("--full", action="store_true",
                    help="ignore state and upload the full journal")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the JSONL in /tmp but skip the upload")
    ap.add_argument("--include-llm-transcripts", action="store_true",
                    help="upload agent_observations_json and "
                         "agent_messages_json verbatim (default: "
                         "redact to a marker so prompt/response "
                         "content stays local)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db_path = Path(args.db)
    if not db_path.exists():
        logger.warning("no %s - nothing to upload; exiting 0", db_path)
        return 0
    state_path = Path(args.state)
    state = _load_state(state_path)
    since = None if args.full else state.get("last_uploaded_ts")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        decisions = _new_decisions(
            conn, since,
            include_llm_transcripts=args.include_llm_transcripts,
        )
    finally:
        conn.close()
    if not decisions:
        logger.info("no new decisions since %s; nothing to upload", since)
        return 0

    out_path = Path("/tmp/aizen_journal_snapshot.jsonl")
    with out_path.open("w") as fh:
        for d in decisions:
            fh.write(json.dumps(d, default=str) + "\n")
    logger.info(
        "snapshot built: %d new decisions, last_ts=%s -> %s",
        len(decisions), decisions[0].get("timestamp"),
        decisions[-1].get("timestamp"),
    )

    if args.dry_run:
        logger.info("--dry-run set, skipping upload")
        return 0

    token = _resolve_token()
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    # Commit each snapshot as a single file upload. The dataset is
    # a JSON Lines time-series: each upload overwrites the prior
    # ``snapshot.jsonl`` with the *merged* set. To avoid losing
    # older rows on subsequent uploads, we merge with the existing
    # remote snapshot if present.
    remote_path = "snapshot.jsonl"
    remote_lines: list[str] = []
    try:
        from huggingface_hub import hf_hub_download
        local_remote = hf_hub_download(
            repo_id=args.dataset, repo_type="dataset",
            filename=remote_path, token=token,
        )
        with open(local_remote) as fh:
            remote_lines = fh.readlines()
        logger.info("merged with existing %d lines from remote", len(remote_lines))
    except Exception as exc:  # noqa: BLE001
        # 404 / file not found is the expected first-upload case.
        logger.info("no prior remote snapshot (%s) - uploading fresh", exc)

    local_keys = {d.get("decision_id") for d in decisions if d.get("decision_id")}
    # Keep remote lines whose decision_id is not in the new batch.
    keep_remote = [
        line for line in remote_lines
        if line.strip()
        and not _line_decision_id(line) in local_keys
    ]
    merged = keep_remote + [json.dumps(d, default=str) + "\n" for d in decisions]
    # Stable order by timestamp.
    merged.sort(key=_line_sort_key)
    final_path = Path("/tmp/aizen_journal_merged.jsonl")
    final_path.write_text("".join(merged))
    api.upload_file(
        path_or_fileobj=str(final_path),
        path_in_repo=remote_path,
        repo_id=args.dataset,
        repo_type="dataset",
        token=token,
        commit_message=(
            f"journal snapshot: {len(decisions)} new, "
            f"total {len(merged)}"
        ),
    )
    new_last_ts = decisions[-1].get("timestamp")
    if new_last_ts:
        state["last_uploaded_ts"] = new_last_ts
        _save_state(state_path, state)
    logger.info(
        "uploaded %d decisions (merged total %d) to https://huggingface.co/datasets/%s",
        len(decisions), len(merged), args.dataset,
    )
    return 0


def _line_decision_id(line: str) -> str:
    try:
        return json.loads(line).get("decision_id", "")
    except Exception:
        return ""


def _line_sort_key(line: str) -> str:
    try:
        return json.loads(line).get("timestamp", "")
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
