/**
 * Aizen scheduler Worker.
 *
 * Receives cron triggers from Cloudflare's edge network and dispatches
 * the matching GitHub Actions workflow via the REST API. The dispatch
 * is a `workflow_dispatch` event, which goes onto GitHub's
 * user-initiated queue (NOT the best-effort schedule queue) and runs
 * reliably on the next available runner.
 *
 * Schedule mapping (matches .github/workflows/*.yml):
 *   "0,15,30,45 * * * *"  ->  cron-loop.yml
 *   "5,20,35,50 * * * *"  ->  news-pre-tick.yml
 *   "7 * * * *"           ->  hourly-analysis.yml
 *   "30 20 * * 1-5"       ->  daily-pnl.yml
 *
 * Required secrets (set with `npx wrangler secret put <NAME>`):
 *   GITHUB_TOKEN  - PAT with `repo` scope
 *
 * Optional vars (in wrangler.toml [vars]):
 *   GITHUB_REPO   - "owner/repo"  (default: FreshMindAI/aizen-trading-agents)
 *   LOG_LEVEL     - "info" | "debug"  (default: "info")
 *
 * Deploy:
 *   cd cloudflare-worker
 *   npm install
 *   npx wrangler secret put GITHUB_TOKEN
 *   npx wrangler deploy
 */

export interface Env {
  GITHUB_TOKEN: string;
  GITHUB_REPO?: string;
  LOG_LEVEL?: "info" | "debug";
}

// Map cron expressions to workflow filenames. Keys MUST match the
// cron strings in wrangler.toml [triggers].
const CRON_TO_WORKFLOW: Record<string, string> = {
  "0,15,30,45 * * * *": "cron-loop.yml",
  "5,20,35,50 * * * *": "news-pre-tick.yml",
  "7 * * * *": "hourly-analysis.yml",
  "30 20 * * 1-5": "daily-pnl.yml",
};

const DEFAULT_REPO = "FreshMindAI/aizen-trading-agents";
const GITHUB_API = "https://api.github.com";
const API_VERSION = "2022-11-28";

function log(level: "info" | "debug", env: Env, msg: string, extra?: unknown) {
  const configured = env.LOG_LEVEL ?? "info";
  if (level === "debug" && configured !== "debug") return;
  const payload = extra !== undefined ? { msg, extra } : { msg };
  // Workers' console.log goes to Logpush; structured fields help
  // filtering in the Cloudflare dashboard.
  console.log(JSON.stringify({ level, ts: new Date().toISOString(), ...payload }));
}

async function dispatchWorkflow(
  env: Env, workflow: string,
): Promise<{ status: number; body: string }> {
  const repo = env.GITHUB_REPO ?? DEFAULT_REPO;
  const url = `${GITHUB_API}/repos/${repo}/actions/workflows/${workflow}/dispatches`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": API_VERSION,
      "Content-Type": "application/json",
      "User-Agent": "aizen-cf-scheduler/1.0",
    },
    body: JSON.stringify({ ref: "master", inputs: {} }),
  });
  // Drain the body so the connection can be reused. GitHub returns
  // 204 No Content on success.
  const body = await resp.text();
  return { status: resp.status, body };
}

export default {
  /**
   * Cron trigger handler. Fires on each schedule line listed in
   * wrangler.toml [triggers]. The `cron` field is the literal cron
   * expression that fired (e.g. "0,15,30,45 * * * *"); we look it up
   * in CRON_TO_WORKFLOW to decide what to dispatch.
   *
   * Per the Cloudflare docs, cron triggers execute on UTC time and
   * weekday numbering is 1=Sunday through 7=Saturday.
   */
  async scheduled(
    controller: ScheduledController,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<void> {
    const cron = controller.cron;
    const workflow = CRON_TO_WORKFLOW[cron];

    if (!workflow) {
      // Should never happen if wrangler.toml is kept in sync, but
      // log and skip rather than fail the Worker.
      log("info", env, "unmapped_cron", { cron });
      return;
    }

    log("info", env, "dispatch_start", { cron, workflow });
    try {
      const { status, body } = await dispatchWorkflow(env, workflow);
      if (status === 204) {
        log("info", env, "dispatch_ok", { cron, workflow, status });
      } else {
        // 401/403/404/422 are config errors; log at info with the body
        // so the user can debug from the Cloudflare log dashboard.
        log("info", env, "dispatch_non_204", { cron, workflow, status, body: body.slice(0, 500) });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log("info", env, "dispatch_error", { cron, workflow, err: msg });
    }
  },

  /**
   * HTTP handler. Useful for manual trigger from the browser or curl
   * (e.g. `curl -X POST .../?workflow=news-pre-tick.yml`).
   *
   * Not strictly required — the cron handler does all the work — but
   * handy when you want to fire a workflow without waiting for the
   * next cron tick. Returns 204 on success, 400 on bad input, 500 on
   * GitHub error.
   */
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<Response> {
    const url = new URL(request.url);
    if (request.method !== "POST") {
      return new Response("Use POST /?workflow=<name>.yml\n", { status: 405 });
    }
    const workflow = url.searchParams.get("workflow");
    if (!workflow || !workflow.endsWith(".yml")) {
      return new Response("Missing or invalid ?workflow=...yml\n", { status: 400 });
    }
    try {
      const { status, body } = await dispatchWorkflow(env, workflow);
      return new Response(body, { status, headers: { "Content-Type": "application/json" } });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return new Response(JSON.stringify({ error: msg }), { status: 500 });
    }
  },
};
