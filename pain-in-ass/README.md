# pain-in-ass/ — phase-by-phase audit

This folder is the honest engineering diary. Every phase gets three files:

- **failures.md** — what broke, what surprised us, what we papered over.
- **features.md** — what we actually shipped, with the test / metric that proves it.
- **improvements.md** — concrete changes we'd make in a v2.

Each phase matches a numbered phase in the spec:
- `phase-0-data/` — ingestion + storage (bars, option contracts, option bars)
- `phase-1-ml/` — XGBoost direction / option / rv models
- `phase-2-gnn/` — graph snapshots, training, evaluation
- `phase-3-agents/` — orchestrator + 8 specialized agents
- `phase-4-broker/` — Alpaca paper trading integration
- `architecture-rating.md` — overall verdict
- `reference-architectures/` — what we patterned this on, what we diverged from
- `hackathon-week/` — the Mon 2026-08-31 → Fri 2026-09-04 plan + daily diary

Tone: candid, not promotional. The point is to capture what cost us the most
time, so the next iteration doesn't repeat it.

Last refreshed: 2026-08-29 (Sat, day 2 of the Alpaca AI Trading Agents Hackathon).
