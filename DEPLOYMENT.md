# Deploying to Render (Cron Job)

The hackathon window is **2026-08-28 → 2026-09-04** (5 days from today).
This guide deploys the multi-agent pipeline as a Render Cron Job that
runs **one cycle every 15 minutes** during market hours (and around the
clock — the cycle is cheap, ~10-30s, and uses the mock LLM unless you
override).

## What gets deployed

- **Service type**: Render Cron Job (not Background Worker — each
  invocation is a fresh process, easier to debug).
- **Schedule**: `*/15 * * * *` (every 15 min).
- **Max run duration**: 5 minutes (safety net; cycles complete in 10-30s).
- **Persistent disk**: 1 GB at `/var/data/aizen` (SQLite DB + trace files).
- **Image**: built from the `Dockerfile` in this repo.
- **Entry point**: `python -m src.agents.cli.run_loop --once`.

## Setup steps (one time)

1. **Push the repo to GitHub** (or GitLab). The blueprint reads from a
   connected Git branch.

2. **In the Render dashboard**:
   - "New +" → "Blueprint".
   - Point it at the repo. Render will detect `render.yaml`.
   - Review the plan. The Cron Job is **Starter** ($7/mo).
   - Click "Apply".

3. **Set secrets** in Dashboard → `aizen-trading-loop` → Environment:
   - `ALPACA_API_KEY_ID` — paper account key.
   - `ALPACA_API_SECRET_KEY` — paper account secret.
   - (Optional) `ANTHROPIC_AUTH_TOKEN` — only if you want LLM-backed
     reasoning instead of the mock provider.

4. **Manual first run** — trigger a tick to validate:
   - Dashboard → `aizen-trading-loop` → "Manual Deploy" → "Run cron job".
   - Watch the live logs. You should see the `_print_summary` block
     within 30s. If you see a stack trace, the most common cause is
     missing Alpaca keys.

5. **Confirm the disk mounted**:
   - Click the running container → "Shell".
   - `ls -la /var/data/aizen/` should show `trading.db` after the first
     successful tick.

## Per-tick lifecycle

1. **Init**: open SQLite at `/var/data/aizen/trading.db`, run
   `init_db()` (idempotent schema).
2. **Data refresh**: pull the latest 1-bar from Alpaca for the
   10-name universe. Failures log a warning and continue on stale data.
3. **Cycle**: `Orchestrator.run_cycle()` builds the snapshot, runs the
   7-agent graph, persists to `decision_journal`, writes
   `models/cycle_traces/cycle_{decision_id}.jsonl` and appends to
   `models/cycle_traces.jsonl`.
4. **Execution** (if PROCEED): submit paper order to Alpaca.
5. **Exit**: process exits 0 (or 1 on cycle error — Render marks
   the run as failed and you get a notification).

## Observability

- **Logs**: Render dashboard → Logs tab. The `_print_summary` block is
  designed to be greppable. `grep final_action` to count decisions.
- **Decision journal**: query the live `decision_journal` table on the
  persistent disk. Recent rows are the most recent decisions.
- **Traces**: each cycle writes one JSONL file. Use
  `python scripts/print_cycle_trace.py` locally to inspect (download
  the file first).
- **Run history**: Render dashboard → "Events" shows past cron runs
  with exit codes.

## Local development

The Dockerfile is the source of truth, so `docker build` matches
Render exactly. For quick iteration:

```bash
# Local single-tick smoke test (mock LLM, no broker)
docker build -t aizen:dev .
docker run --rm -e AIZEN_LLM_PROVIDER=mock -e RUN_MODE=dry-run \
    -v $(pwd)/data:/app/data aizen:dev

# Local persistent-disk simulation
mkdir -p /tmp/aizen-data
docker run --rm -e AIZEN_LLM_PROVIDER=mock -e AIZEN_DB_PATH=/var/data/aizen/trading.db \
    -v /tmp/aizen-data:/var/data/aizen aizen:dev
```

## Rollback / disable

- Pause: Dashboard → Cron Job → "Suspend".
- Delete: Dashboard → Cron Job → "Settings" → "Delete Service".

## Hackathon scoring checklist (operator)

The judges will likely look at:
- [ ] `decision_journal` is being populated every 15 min during market hours.
- [ ] `cycle_traces.jsonl` shows real ML/GNN signals (not stub defaults).
- [ ] At least one PROCEED → paper fill → `outcome_label` populated cycle
      (use the NVDA fill monitor — `python scripts/watch_nvda_paper_fill.py`).
- [ ] README.md points to the deployed cron logs URL.

## Cost

- Render Starter Cron Job: $7/mo.
- Persistent disk 1 GB: included.
- Alpaca paper account: free.
- Total: $7/mo, runs through the full hackathon + 2 weeks of buffer.
