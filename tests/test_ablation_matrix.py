"""Tests for scripts/build_ablation_matrix.py (GRAPH_SAGE_IMPROVEMENT_PLAN.md
Phase J): pure aggregation over already-saved run artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ablation_matrix import _load_run, build_matrix, to_markdown  # noqa: E402


def _make_run_dir(tmp_path: Path, name: str, test_metrics: dict, onset_test: dict | None, variation_test: dict | None = None) -> str:
    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"by_split": {"test": test_metrics}}, f)
    if onset_test is not None:
        with open(run_dir / "onset_breakdown.json", "w") as f:
            json.dump({"test": onset_test}, f)
    if variation_test is not None:
        with open(run_dir / "temporal_variation.json", "w") as f:
            json.dump({"test": variation_test}, f)
    return str(run_dir)


def test_load_run_reads_overall_and_onset_metrics(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, "run_a",
        test_metrics={"pr_auc": 0.8, "roc_auc": 0.9},
        onset_test={"pr_auc_fresh_onset": 0.1, "roc_auc_fresh_onset": 0.4, "recall_fresh_onset": 0.0, "recall_already_ongoing": 0.9},
    )
    loaded = _load_run(run_dir)
    assert loaded["pr_auc"] == 0.8
    assert loaded["roc_auc"] == 0.9
    assert loaded["pr_auc_fresh_onset"] == 0.1
    assert loaded["recall_already_ongoing"] == 0.9


def test_load_run_handles_missing_onset_breakdown_file(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run_b", test_metrics={"pr_auc": 0.3, "roc_auc": 0.6}, onset_test=None)
    loaded = _load_run(run_dir)
    assert loaded["pr_auc"] == 0.3
    assert loaded["pr_auc_fresh_onset"] is None


def test_build_matrix_averages_across_multiple_run_dirs_per_row(tmp_path):
    run1 = _make_run_dir(tmp_path, "seed1", {"pr_auc": 0.8, "roc_auc": 0.9}, {"recall_fresh_onset": 0.0, "recall_already_ongoing": 1.0})
    run2 = _make_run_dir(tmp_path, "seed2", {"pr_auc": 0.6, "roc_auc": 0.7}, {"recall_fresh_onset": 0.2, "recall_already_ongoing": 0.8})
    df = build_matrix([("Full GraphSAGE", [run1, run2])])
    row = df.iloc[0]
    assert row["n_runs"] == 2
    assert row["Overall PR-AUC"] == pytest.approx(0.7)
    assert row["Overall ROC-AUC"] == pytest.approx(0.8)
    assert row["Fresh-Onset Recall"] == pytest.approx(0.1)
    assert row["Already-Ongoing Recall"] == pytest.approx(0.9)


def test_build_matrix_handles_none_values_without_crashing(tmp_path):
    run1 = _make_run_dir(tmp_path, "seed1", {"pr_auc": 0.5, "roc_auc": 0.6}, {"recall_fresh_onset": None, "recall_already_ongoing": 0.5})
    df = build_matrix([("X", [run1])])
    assert df.iloc[0]["Fresh-Onset Recall"] is None


def test_to_markdown_produces_a_valid_looking_table(tmp_path):
    run1 = _make_run_dir(tmp_path, "seed1", {"pr_auc": 0.5, "roc_auc": 0.6}, None)
    df = build_matrix([("Majority", [run1])])
    table = to_markdown(df)
    assert table.startswith("| Model/Input |")
    assert "Majority" in table
    assert "0.5000" in table
    assert "n/a" in table  # missing onset breakdown columns


# ---- fraction_time_varying / time-invariance warning ----
# Added after "static_plus_graph" scored higher than the full model with
# zero cross-seed variance -- every supplier's risk_probability turned out
# to be literally constant across time in that mode.


def test_load_run_reads_fraction_time_varying(tmp_path):
    run_dir = _make_run_dir(tmp_path, "run_a", {"pr_auc": 0.5, "roc_auc": 0.6}, None, variation_test={"fraction_time_varying": 0.9})
    loaded = _load_run(run_dir)
    assert loaded["fraction_time_varying"] == pytest.approx(0.9)


def test_low_fraction_time_varying_flags_the_row_label(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, "degenerate_run", {"pr_auc": 0.93, "roc_auc": 0.99}, None, variation_test={"fraction_time_varying": 0.0}
    )
    df = build_matrix([("Static + Graph Structure", [run_dir])])
    assert "WARNING" in df.iloc[0]["Model/Input"]
    assert df.iloc[0]["Fraction Time-Varying"] == pytest.approx(0.0)


def test_high_fraction_time_varying_does_not_flag_the_row_label(tmp_path):
    run_dir = _make_run_dir(
        tmp_path, "normal_run", {"pr_auc": 0.8, "roc_auc": 0.98}, None, variation_test={"fraction_time_varying": 0.95}
    )
    df = build_matrix([("Full GraphSAGE", [run_dir])])
    assert df.iloc[0]["Model/Input"] == "Full GraphSAGE"  # unchanged, no warning appended
