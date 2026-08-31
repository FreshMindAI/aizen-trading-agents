"""Tests for the walk-forward A/B/C/D ablation trainer (spec 003 / T036-T040).

Covers:
- T036: sharpe_approx on a synthetic PnL stream (known input -> known output)
- T037: the promotion rule (FR-009) on synthetic metrics
- T038: seed reproducibility (same seed -> same 12 rows; different -> different)
- T039: metrics_json carries the new fields (sharpe, used_news, fold_id, flag)
- T040: every row validates against the JSON-Schema contract
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO)

from src.gnn.walk_forward_ablation import (  # noqa: E402
    ABLATION_CONFIGS,
    AblationReport,
    AblationRow,
    N_FOLDS,
    _compute_sharpe,
    _expanding_window_folds,
    _promote_default,
    _snapshots_synthetic,
    _train_one_config,
    run,
    write_report,
)


CONTRACT = json.loads(
    (Path(REPO) / "specs" / "003-news-driven-gnn-retrain" / "contracts" / "ablation-row.json").read_text(
        encoding="utf-8"
    )
)


# ---------------------------------------------------------------------------
# T036: sharpe_approx
# ---------------------------------------------------------------------------
def test_sharpe_known_stream():
    """10-element PnL stream of {+1, -1, ...} -> known sharpe value."""
    pnl = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
    s = _compute_sharpe(pnl)
    # mean = 0, std > 0 -> sharpe is 0
    assert abs(s) < 1e-9


def test_sharpe_known_positive_stream():
    pnl = [0.01] * 10  # constant positive -> 0 std -> 0 sharpe
    assert _compute_sharpe(pnl) == 0.0


def test_sharpe_degenerate_inputs():
    assert _compute_sharpe([]) == 0.0
    assert _compute_sharpe([0.05]) == 0.0


def test_sharpe_positive_drift():
    pnl = [0.001 * i for i in range(1, 21)]  # 20 days of growing returns
    s = _compute_sharpe(pnl)
    assert s > 5.0  # strong positive drift -> high sharpe


# ---------------------------------------------------------------------------
# T037: promotion rule
# ---------------------------------------------------------------------------
def test_promotion_rule_promotes_when_news_beats_all():
    rows = [
        AblationRow(2, "xgb", False, 0.55, 0.4, 0.25, 0.6, 0.5, 30, "news-off", 1337),
        AblationRow(2, "gcn_v1", False, 0.56, 0.4, 0.25, 0.6, 0.6, 30, "news-off", 1337),
        AblationRow(2, "gatv2_prenews", False, 0.58, 0.4, 0.25, 0.6, 0.7, 30, "news-off", 1337),
        AblationRow(2, "gatv2_news", True, 0.60, 0.4, 0.25, 0.6, 0.8, 30, "news-on", 1337),
    ]
    out = _promote_default(rows)
    assert out["promote"] is True
    assert out["promoted_model"] == "gatv2_news"


def test_promotion_rule_rejects_when_news_loses_on_sharpe():
    rows = [
        AblationRow(2, "xgb", False, 0.55, 0.4, 0.25, 0.6, 0.9, 30, "news-off", 1337),
        AblationRow(2, "gcn_v1", False, 0.56, 0.4, 0.25, 0.6, 0.9, 30, "news-off", 1337),
        AblationRow(2, "gatv2_prenews", False, 0.58, 0.4, 0.25, 0.6, 0.9, 30, "news-off", 1337),
        AblationRow(2, "gatv2_news", True, 0.60, 0.4, 0.25, 0.6, 0.5, 30, "news-on", 1337),  # lower sharpe
    ]
    out = _promote_default(rows)
    assert out["promote"] is False


def test_promotion_rule_rejects_when_news_loses_on_roc():
    rows = [
        AblationRow(2, "xgb", False, 0.90, 0.4, 0.25, 0.6, 0.5, 30, "news-off", 1337),
        AblationRow(2, "gcn_v1", False, 0.91, 0.4, 0.25, 0.6, 0.5, 30, "news-off", 1337),
        AblationRow(2, "gatv2_prenews", False, 0.92, 0.4, 0.25, 0.6, 0.5, 30, "news-off", 1337),
        AblationRow(2, "gatv2_news", True, 0.50, 0.4, 0.25, 0.6, 0.6, 30, "news-on", 1337),  # lower roc
    ]
    out = _promote_default(rows)
    assert out["promote"] is False


def test_promotion_rule_rejects_when_news_missing():
    rows = [
        AblationRow(2, "xgb", False, 0.55, 0.4, 0.25, 0.6, 0.5, 30, "news-off", 1337),
    ]
    out = _promote_default(rows)
    assert out["promote"] is False
    assert "gatv2_news" in out["reason"]


# ---------------------------------------------------------------------------
# T038: seed reproducibility
# ---------------------------------------------------------------------------
def test_seed_reproducibility_same_seed_same_rows():
    snaps = _snapshots_synthetic(30)
    a = run(snaps, seed=1337)
    b = run(snaps, seed=1337)
    assert len(a.rows) == 12 and len(b.rows) == 12
    for r1, r2 in zip(a.rows, b.rows):
        assert asdict(r1) == asdict(r2)
    assert a.promote_default == b.promote_default


def test_seed_reproducibility_different_seed_different_rows():
    snaps = _snapshots_synthetic(30)
    a = run(snaps, seed=1337)
    b = run(snaps, seed=2024)
    # At least one row must differ in at least one metric (hash salting).
    differs = False
    for r1, r2 in zip(a.rows, b.rows):
        if (r1.roc_auc, r1.pr_auc, r1.sharpe_approx) != (r2.roc_auc, r2.pr_auc, r2.sharpe_approx):
            differs = True
            break
    assert differs, "different seeds produced identical rows; hash is not seed-sensitive"


# ---------------------------------------------------------------------------
# T039: metrics shape
# ---------------------------------------------------------------------------
def test_metrics_shape_per_row():
    snaps = _snapshots_synthetic(30)
    report = run(snaps, seed=1337)
    assert len(report.rows) == len(ABLATION_CONFIGS) * N_FOLDS
    by_model: dict[str, list] = {m: [] for m in ABLATION_CONFIGS}
    for r in report.rows:
        # Field presence
        for f in ("fold_id", "model", "used_news", "roc_auc", "pr_auc",
                  "brier", "log_loss", "sharpe_approx", "n_snapshots",
                  "feature_flag_state", "seed"):
            assert hasattr(r, f), f"row missing field {f}"
        # Field ranges
        assert 0.0 <= r.roc_auc <= 1.0
        assert 0.0 <= r.pr_auc <= 1.0
        assert 0.0 <= r.brier
        assert 0.0 <= r.log_loss
        assert r.fold_id in (0, 1, 2)
        assert r.feature_flag_state in ("news-on", "news-off")
        assert r.used_news == (r.model == "gatv2_news")
        by_model[r.model].append(r)
    # 3 rows per config
    for m, rows in by_model.items():
        assert len(rows) == N_FOLDS, f"{m} has {len(rows)} rows, expected {N_FOLDS}"


# ---------------------------------------------------------------------------
# T040: JSON-Schema contract validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", list(ABLATION_CONFIGS))
@pytest.mark.parametrize("fold_id", [0, 1, 2])
def test_schema_validates_row(model, fold_id):
    snaps = _snapshots_synthetic(30)
    report = run(snaps, seed=1337)
    for r in report.rows:
        if r.model == model and r.fold_id == fold_id:
            jsonschema.validate(asdict(r), CONTRACT)
            return
    pytest.fail(f"no row found for model={model} fold_id={fold_id}")


# ---------------------------------------------------------------------------
# Fold structure
# ---------------------------------------------------------------------------
def test_folds_are_expanding_windows():
    snaps = _snapshots_synthetic(30)
    folds = _expanding_window_folds(snaps, n_folds=3)
    assert len(folds) == 3
    # Each subsequent fold has more training data
    assert len(folds[0].train_snapshots) < len(folds[1].train_snapshots) < len(folds[2].train_snapshots)
    # Fold 2 is the held-out validation slice
    assert folds[2].test_snapshots == [snaps[-1]]


def test_folds_too_few_snapshots_raises():
    with pytest.raises(ValueError):
        _expanding_window_folds(_snapshots_synthetic(2), n_folds=3)


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------
def test_write_report_round_trip(tmp_path):
    snaps = _snapshots_synthetic(30)
    report = run(snaps, seed=1337)
    out = write_report(report, tmp_path / "abl.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["seed"] == 1337
    assert len(data["rows"]) == 12
    assert data["schema_version"] == "1.0"


# Local helper to avoid importing dataclasses.asdict globally above
from dataclasses import asdict  # noqa: E402
