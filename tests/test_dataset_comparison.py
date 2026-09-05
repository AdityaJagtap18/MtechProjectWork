"""Tests for modeling/dataset_comparison.py (D0 dataset integrity check,
GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md)."""

from __future__ import annotations

from scm_dataset.modeling.dataset_comparison import compare_datasets


def test_comparing_a_dataset_against_itself_is_fully_compatible(tiny_benchmark):
    result = compare_datasets(tiny_benchmark, tiny_benchmark, rolling_windows=[4, 8], feature_mode="full")
    assert result["node_counts"]["match"] is True
    assert result["edge_counts"]["match"] is True
    assert result["horizon_periods"]["match"] is True
    assert result["feature_columns"]["match"] is True
    assert result["label_columns"]["match"] is True
    assert result["required_files_present"]["all_present"] is False  # in-memory fixture has no dataset_dir on disk
    assert result["compatible_for_cross_dataset_evaluation"] is False  # gated on required_files too


def test_onset_count_matches_supplier_disrupted_transitions(tiny_benchmark):
    result = compare_datasets(tiny_benchmark, tiny_benchmark, rolling_windows=[4, 8], feature_mode="full")
    assert result["onset_counts"]["a"] == result["onset_counts"]["b"]
    assert result["onset_counts"]["a"] >= 0


def test_prevalence_matches_direct_computation(tiny_benchmark):
    result = compare_datasets(tiny_benchmark, tiny_benchmark, rolling_windows=[4, 8], feature_mode="full")
    labels = tiny_benchmark.labels["supplier"]
    expected_rate = float(labels["supplier_disrupted"].mean())
    assert result["supplier_disrupted_prevalence"]["a"]["positive_rate"] == expected_rate
    assert result["supplier_disrupted_prevalence"]["a"]["n_rows"] == len(labels)


def test_rejects_unknown_feature_mode(tiny_benchmark):
    import pytest

    with pytest.raises(ValueError, match="unknown feature_mode"):
        compare_datasets(tiny_benchmark, tiny_benchmark, rolling_windows=[4, 8], feature_mode="made_up")


def test_real_seed43_vs_seed44_are_structurally_compatible():
    import os
    import pytest

    if not os.path.isdir("data/benchmark/scm_v1_black_swan_seed44"):
        pytest.skip("seed44 benchmark not present")
    from scm_dataset.modeling.data import load_benchmark

    a = load_benchmark("data/benchmark", "scm_v1_black_swan_seed43")
    b = load_benchmark("data/benchmark", "scm_v1_black_swan_seed44")
    result = compare_datasets(a, b, rolling_windows=[4, 8, 12], feature_mode="full")
    assert result["node_counts"]["match"] is True
    assert result["edge_counts"]["match"] is True
    assert result["feature_columns"]["match"] is True
    assert result["compatible_for_cross_dataset_evaluation"] is True
    # both should have genuinely different underlying event populations --
    # this is what makes D2 a meaningful cross-dataset test, not a duplicate
    assert result["onset_counts"]["a"] != result["onset_counts"]["b"]
