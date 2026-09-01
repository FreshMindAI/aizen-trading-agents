# Aizen scheduler — Cloudflare Worker

Reliable cron trigger for the four Aizen GitHub Actions workflows.
Replaces GitHub's best-effort `on.schedule:` with Cloudflare's edge
cron triggers, which fire on time. Each tick calls the GitHub
Actions dispatch API, queuing a `workflow_dispatch` run on the
user-initiated queue (which is reliable — only the schedule queue
is flaky).

## What it fires

| Cron (UTC) | Workflow | Purpose |
|---|---|---|
| `0,15,30,45 * * * *` | `cron-loop.yml` | The 15-min trade tick |
| `5,20,35,50 * * * *` | `news-pre-tick.yml` | Pre-tick news + GNN topology |
| `7 * * * *` | `hourly-analysis.yml` | Hourly reflection report |
| `30 20 * * 1-5` | `daily-pnl.yml` | Daily P&L grading |

## One-time setup

1. **Create a Cloudflare account** if you don't have one
   (https://dash.cloudflare.com/sign-up).
2. **Subscribe to Workers Paid** ($5/mo) — required for cron
   triggers. Free tier does not include them.
3. **Get a GitHub PAT** with `repo` scope. The existing PAT used
   for the manual `git push` works fine; rotate if you want a
   dedicated token for the Worker.
4. **Install Wrangler** (Cloudflare's CLI):
   ```bash
   cd cloudflare-worker
   npm install
   npx wrangler login
   ```
5. **Set the GitHub token as a Cloudflare secret** (encrypted at
   rest, never in code):
   ```bash
   npx wrangler secret put GITHUB_TOKEN
   # paste the PAT when prompted (input is hidden)
   ```
6. **Deploy**:
   ```bash
   npx wrangler deploy
   ```

After deploy, the Worker is live at
`https://aizen-scheduler.<your-subdomain>.workers.dev`. The current
deployment is at
**`https://aizen-scheduler.skillcraftstudio999.workers.dev`**.

## Verifying

Tail the Worker's logs to see each dispatch in real time:
```bash
npx wrangler tail
```

You should see one log line per cron tick, like:
```json
{"level":"info","ts":"2026-09-01T16:30:00Z","msg":"dispatch_ok","cron":"0,15,30,45 * * * *","workflow":"cron-loop.yml","status":204}
```

If you see `dispatch_non_204` with a 401/403/404, the token is
wrong, the repo is wrong, or the workflow file doesn't exist. The
response body is included in the log (truncated to 500 chars).

## Local testing

Cloudflare exposes a local scheduled-handler endpoint on
`wrangler dev`:
```bash
npx wrangler dev
# in another shell, trigger a fake cron tick:
curl "http://localhost:8787/cdn-cgi/handler/scheduled?cron=0,15,30,45%20*%20*%20*%20*&time=$(date +%s%3N)"
```

This invokes the `scheduled` handler with the given `cron` string
without waiting for the actual schedule.

## Manual trigger (no waiting for cron)

The Worker also exposes a `fetch` handler so you can fire a
workflow on demand from the browser or curl:

```bash
curl -X POST "https://aizen-scheduler.<subdomain>.workers.dev/?workflow=cron-loop.yml"
```

Returns 204 if the dispatch was queued, or the GitHub error body
otherwise. Useful when you want to fire a tick immediately.

## Cost

- **Cloudflare Workers Paid: $5/mo flat** (includes 10M requests)
- 4 cron triggers firing 1×/15 min + 1×/hour + 1×/weekday = ~4,500
  requests/mo
- Total: **$5/mo**, well within the 10M-request free tier

## Why this is more reliable than `on.schedule:`

GitHub Actions scheduled workflows are documented as best-effort:
they can be delayed up to ~30 min and silently skipped during high
load. The `workflow_dispatch` queue, by contrast, is user-initiated
and runs immediately on the next available runner. By calling the
dispatch API from a Cloudflare cron (which is reliable), we get the
best of both: predictable scheduling + reliable execution.

## Caveats from the Cloudflare docs

- Cron triggers run on **UTC** (not your local timezone). All
  schedules in `wrangler.toml` are UTC.
- Trigger changes take up to **15 minutes** to propagate. If you
  add a new cron line and deploy, expect a one-time 15-min window
  where it doesn't fire.
- Weekday numbering is **1=Sunday through 7=Saturday** (different
  from Linux cron). `1-5` means Mon–Fri. Our `30 20 * * 1-5` means
  20:30 UTC every weekday = 4:30 PM ET on weekdays (DST-naive; ET
  flips between EDT and EST but 20:30 UTC is the same wall-clock
  time either way).

## Files

| File | Purpose |
|---|---|
| `wrangler.toml` | Cloudflare config: name, cron triggers, vars |
| `src/index.ts` | The Worker: cron handler + HTTP handler |
| `package.json` | Wrangler + TypeScript dev deps |
| `tsconfig.json` | TypeScript strict config |
| `.gitignore` | node_modules, .dev.vars, dist |
| `README.md` | This file |

## Security

- `GITHUB_TOKEN` is set via `wrangler secret put` — encrypted at
  rest by Cloudflare, never logged, never committed.
- `wrangler.toml` only contains the public repo name and the cron
  expressions; no secrets.
- The Worker's `fetch` handler is public but only POSTs; the
  unauthenticated surface just queues workflow_dispatch events
  (same surface as the `Run workflow` button in the GitHub UI).
  If you want to lock it down, add a `BASIC_AUTH` secret and check
  it in `fetch()`.
