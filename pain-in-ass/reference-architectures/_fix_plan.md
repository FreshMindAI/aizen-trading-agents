# Sub-4 fix plan (against the GATv2 doc)

> **Doc reference:** `Phase_2_GATv2_Dynamic_Financial_Topology_Design.docx` (extracted to `_phase2_gatv2_doc.txt`).
> **Method:** map every sub-4 rating to a section of the doc, then a concrete code change.

## What the doc says we should be doing

The doc is unambiguous on three things we got wrong in v1:

1. **§10** Run a *controlled ablation* A/B/C/D — we never did. We built a single model and reported one number. The doc requires A=XGB, B=GCN+current graph, C=GATv2+meaningful topology, D=hybrid XGB+GATv2.
2. **§12** *"Relational modeling is expected to be more useful for option-surface and contract-selection problems than for a simple underlying direction target."* We focused the GNN on direction, where XGBoost already wins. The GNN should be aimed at the **option opportunity** task.
3. **§16-17** Static → dynamic via a topology controller, with shadow-testing and rollback. We hand-coded `topology_version: "fixed-1"` everywhere.

The doc also has a strong meta-principle (§22): **don't optimize the GNN to beat XGBoost on one number; prove the graph has incremental information.** That's a healthier framing than the v1 "GNN must beat XGB" framing, which is why our standalone GNN looks bad.

## Sub-4 axes → fixes

| Axis | Current | Target | Doc section | Concrete fix |
|---|---|---|---|---|
| **ML accuracy (3)** | 0.6304 direction AUC, no calibration, no ensemble | 0.65 AUC + calibrated probs + 5-seed bagging | §14 (calibration, Brier) | Train signed-return regression (Huber), temperature-scale, 5-seed bagging |
| **GNN signal quality (2)** | Standalone 0.46 AUC, fixed topology, no edge features, focuses on direction | Option-opportunity model with GATv2 + multi-head + edge features; direction task stays XGB-led | §9, §10, §12 | Switch to GATv2 (multi-head), add edge features (correlation, expiry_distance, moneyness, iv_diff), focus on option-opportunity head, run the A/B/C/D ablation |
| **Agent reasoning (3)** | Mock LLM, no live portfolio state, no GNN-confirmation override, no research agent | Live positions + account wired in, GNN override, research agent, strategy-type selector | §13 (hybrid), §15 (regime-specific) | Wire Alpaca account/positions into `inference`, add `strategy_type` selector (long_stock / short_put_spread / protective_put / no_trade), add research agent (news), flip GNN hierarchy |
| **Live trading (3)** | 1 order, no P&L tracking, no kill switch, no stock leg, no retry | EOD P&L, kill switch, stock leg, retry+circuit breaker, daily digest | §16 (never bypass risk) | Add `Leg(asset_class="us_equity")` path, `eod_pnl.py`, kill switch on -5% drawdown, exponential-backoff retry, daily P&L digest |

## Top 8 fixes to land by Mon 2026-08-31 09:30 ET (in priority order)

1. **Train the GATv2 option-opportunity model** — the doc's §12 thesis is that the GNN is most useful for options, not direction. Build a new graph that includes **option contract nodes** (not just underlyings), with edges: `option→underlying`, `option→option_same_expiry`, `option→option_nearby_strike`, `option→option_neighboring_expiry`, `option→option_call_put_pair`. This is a fundamentally different graph than v1, and the doc calls it out specifically.
2. **Add edge features to GATv2** — `correlation`, `strike_distance`, `expiry_distance`, `moneyness_distance`, `iv_difference`, `return_correlation`, `liquidity_similarity`, `timestamp`. Use GATv2 (the v2 formulation) which supports edge features in the attention.
3. **Run the A/B/C/D ablation** — produce a side-by-side report on the option opportunity task: XGB / GCN+v1-graph / GATv2+new-graph / Hybrid.
4. **Switch direction task to signed-return regression + calibration + 5-seed bagging** — the doc's §14 calls for Brier/calibration. The current binary-up task wastes 38% of the data on the flat band.
5. **Wire live account + positions into the inference layer** — `inference._load_portfolio()` and `_account_equity()` should hit Alpaca.
6. **Add the strategy-type selector + GNN-confirmation override** — supervisor picks long_stock / short_put_spread / protective_put / no_trade based on `iv_rv_gap` + GNN bias + XGB.
7. **Add the research agent** — news-driven sentiment + the news edges (`news_cooccurrence`, `news_sentiment_correlation`).
8. **Add EOD P&L reconciliation + kill switch + stock leg to OrderIntent** — without these, the hackathon can't be evaluated.

## What we explicitly do NOT change (per the doc's "do not" guidance)

- We do **not** let the GNN freely create arbitrary edges in v2 (§16: "Do not allow the Phase-2 model to freely create arbitrary edges in the first experiment"). Edges remain curated and versioned.
- We do **not** bypass the risk layer with a "GNN says go" override (§17: "Never bypass deterministic risk controls"). The GNN-confirmation override is on the *signal* gate, not the *risk* gate.
- We do **not** chase "GNN must beat XGB on direction" (§22: "Do not optimize the GNN merely to beat XGBoost on one accuracy number"). The direction task stays XGB-led. The GNN's value is the option-opportunity surface and the dynamic news topology.
- We do **not** ship the Phase-6 adaptive topology in v2. The doc says shadow-test, then activate. We ship the *static* GATv2 with versioned topologies; the controller is a v3 feature.

## Time budget

| Day | Tasks | Deliverable |
|---|---|---|
| **Sat 29 Aug (today)** | This plan, doc extraction | `pain-in-ass/reference-architectures/_fix_plan.md` + `_phase2_gatv2_doc.txt` |
| **Sun 30 Aug** | Tasks 4, 5, 6 (ML signed-return + calibration + bagging, inference wiring, strategy selector) | Retrained Phase 1 + wired inference + strategy selector |
| **Mon 31 Aug (pre-open)** | Tasks 1, 2, 3 (option-graph GATv2 + edge features + A/B/C/D ablation) | New GNN + ablation report |
| **Mon 31 Aug (post-open)** | Task 7 (research agent) | Research agent live |
| **Tue 1 Sept** | Task 8 (EOD P&L + kill switch + stock leg) | Production-ready broker layer |
| **Wed-Thu 2-3 Sept** | Iterate on weights based on live P&L | Optimized scoring |
| **Fri 4 Sept** | Submit | Final P&L + write-up |
