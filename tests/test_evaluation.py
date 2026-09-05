"""Tests for modeling/metrics.py, modeling/calibration.py, and
modeling/evaluate.py (plan §31-40/§47/§66 Check 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scm_dataset.modeling.calibration import compute_calibration, expected_calibration_error
from scm_dataset.modeling.evaluate import (
    build_risk_ranking,
    compute_warning_times,
    disruption_onset_breakdown,
    evaluate_experiment,
    generate_predictions,
    temporal_variation_summary,
)
from scm_dataset.modeling.graphsage import build_model
from scm_dataset.modeling.metrics import compute_classification_metrics, select_threshold

# ---- metrics.py ----


def test_confusion_matrix_and_precision_recall_f1_hand_computed():
    # 2 TP, 1 FP, 1 FN, 2 TN by construction
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.3])
    threshold = 0.5
    metrics = compute_classification_metrics(y_true, y_prob, threshold)

    cm = metrics["confusion_matrix"]
    assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (1, 1, 2, 2)
    precision = 1 / (1 + 1)
    recall = 1 / (1 + 2)
    f1 = 2 * precision * recall / (precision + recall)
    assert metrics["precision"] == pytest.approx(precision)
    assert metrics["recall"] == pytest.approx(recall)
    assert metrics["f1"] == pytest.approx(f1)


def test_pr_auc_and_roc_auc_use_continuous_probabilities_not_binary_predictions():
    # Perfect ranking (all positives score higher than all negatives) but a
    # threshold of 0.5 would misclassify everyone -- PR-AUC/ROC-AUC must
    # still reflect the perfect ranking, proving they're computed from
    # y_prob directly and not from the thresholded y_pred.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.61, 0.62, 0.63, 0.64, 0.65, 0.66])  # all >= 0.5 -> y_pred all 1 at threshold 0.5
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["precision"] < 1.0  # thresholded predictions are all-positive, so precision != the ranking quality


def test_roc_auc_is_safely_undefined_for_single_class_split():
    y_true = np.zeros(10)
    y_prob = np.random.RandomState(0).rand(10)
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert "undefined" in metrics["roc_auc_note"]


def test_select_threshold_fixed_ignores_data():
    assert select_threshold(np.array([1, 0]), np.array([0.9, 0.1]), policy="fixed", value=0.42) == 0.42


def test_select_threshold_f1_optimal_beats_default_on_separable_data():
    y_true = np.array([0] * 8 + [1] * 2)
    y_prob = np.array([0.05] * 8 + [0.4, 0.45])  # positives score higher but all < 0.5
    threshold = select_threshold(y_true, y_prob, policy="f1_optimal")
    f1_at_selected = compute_classification_metrics(y_true, y_prob, threshold)["f1"]
    f1_at_default = compute_classification_metrics(y_true, y_prob, 0.5)["f1"]
    assert f1_at_selected >= f1_at_default


def test_select_threshold_recall_constrained_meets_target_recall():
    rng = np.random.RandomState(0)
    y_true = np.array([1] * 20 + [0] * 80)
    y_prob = np.clip(y_true * 0.5 + rng.rand(100) * 0.5, 0, 1)
    threshold = select_threshold(y_true, y_prob, policy="recall_constrained", target_value=0.8)
    recall = compute_classification_metrics(y_true, y_prob, threshold)["recall"]
    assert recall >= 0.8 - 1e-9


def test_select_threshold_falls_back_to_fixed_value_when_unreachable():
    y_true = np.array([1, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = select_threshold(y_true, y_prob, policy="precision_constrained", value=0.5, target_value=0.99)
    assert threshold == 0.5  # no threshold reaches 99% precision here -> fallback


def test_select_threshold_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown threshold policy"):
        select_threshold(np.array([1, 0]), np.array([0.9, 0.1]), policy="magic")


# ---- calibration.py ----


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_predictions():
    # 10 groups of 10: group i predicts prob i/10 and has exactly i
    # positives out of 10 -- observed frequency equals the predicted
    # probability exactly, by construction, in every bin.
    y_true, y_prob = [], []
    for i in range(10):
        p = i / 10
        y_true += [1] * i + [0] * (10 - i)
        y_prob += [p] * 10
    ece = expected_calibration_error(np.array(y_true, dtype=float), np.array(y_prob, dtype=float), n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_is_large_for_badly_calibrated_predictions():
    y_true = np.array([0] * 50)
    y_prob = np.array([0.95] * 50)  # always confident and always wrong
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.95, abs=0.05)


def test_compute_calibration_handles_empty_input():
    result = compute_calibration(np.array([]), np.array([]))
    assert result["brier_score"] is None


# ---- evaluate.py (uses tiny_prepared_data fixture) ----


def _tiny_model(prepared):
    in_dims = {nt.value: dim for nt, dim in prepared.snapshot_builder.feature_dims().items()}
    edge_types = list(prepared.snapshot_builder.topology.edge_index_dict.keys())
    return build_model(in_dims=in_dims, edge_types=edge_types, hidden_dim=16, num_layers=2, dropout=0.0)


def test_generate_predictions_covers_every_example_with_probabilities_in_unit_range(tiny_prepared_data):
    model = _tiny_model(tiny_prepared_data)
    predictions = generate_predictions(model, tiny_prepared_data)
    assert len(predictions) == len(tiny_prepared_data.examples)
    assert predictions["risk_probability"].between(0.0, 1.0).all()
    assert set(predictions["actual_disruption"].unique()) <= {0, 1}


def test_evaluate_experiment_selects_threshold_from_validation_only(tiny_prepared_data, monkeypatch):
    model = _tiny_model(tiny_prepared_data)

    seen_split_args = []
    from scm_dataset.modeling import evaluate as evaluate_module

    original_select_threshold = evaluate_module.select_threshold

    def spy_select_threshold(y_true, y_prob, **kwargs):
        seen_split_args.append((y_true.copy(), y_prob.copy()))
        return original_select_threshold(y_true, y_prob, **kwargs)

    monkeypatch.setattr(evaluate_module, "select_threshold", spy_select_threshold)
    result = evaluate_experiment(model, tiny_prepared_data, tiny_prepared_data.config.threshold)

    predictions = result.predictions
    val = predictions[predictions["split"] == "validation"]
    assert len(seen_split_args) == 1
    seen_y_true, _ = seen_split_args[0]
    assert len(seen_y_true) == len(val)  # threshold selection was called with exactly the validation rows


def test_evaluate_experiment_produces_metrics_for_every_nonempty_split(tiny_prepared_data):
    model = _tiny_model(tiny_prepared_data)
    result = evaluate_experiment(model, tiny_prepared_data, tiny_prepared_data.config.threshold)
    for split in ("train", "validation", "test"):
        assert split in result.metrics_by_split
        assert split in result.calibration_by_split


def test_risk_ranking_is_sorted_descending_and_ranked_from_one(tiny_prepared_data):
    model = _tiny_model(tiny_prepared_data)
    predictions = generate_predictions(model, tiny_prepared_data)
    predictions["predicted_disruption"] = (predictions["risk_probability"] >= 0.5).astype(int)
    from scm_dataset.modeling.evaluate import build_supplier_context

    context = build_supplier_context(tiny_prepared_data)
    ranking = build_risk_ranking(predictions, threshold=0.5, supplier_context=context)

    assert list(ranking["rank"]) == list(range(1, len(ranking) + 1))
    assert (ranking["risk_probability"].diff().dropna() <= 1e-9).all()  # non-increasing
    assert ranking["time"].nunique() == 1  # one snapshot, not every (supplier, time) pair


def test_warning_times_never_uses_a_prediction_at_or_after_onset():
    predictions = pd.DataFrame(
        {
            "supplier_id": ["s0", "s0", "s0", "s0"],
            "time": [10, 11, 12, 13],
            "predicted_disruption": [0, 1, 1, 1],
        }
    )
    raw_labels = pd.DataFrame(
        {
            "supplier_id": ["s0"] * 20,
            "time": range(20),
            "supplier_disrupted": [0] * 13 + [1] * 7,  # onset at t=13
        }
    )
    warnings = compute_warning_times(predictions, raw_labels, horizon_periods=4)
    row = warnings[warnings.supplier_id == "s0"].iloc[0]
    assert row.onset_time == 13
    assert row.first_warning_time == 11  # first predicted_disruption==1 strictly before onset
    assert row.warning_periods == 2


def test_warning_times_reports_none_when_never_warned():
    predictions = pd.DataFrame({"supplier_id": ["s0"], "time": [0], "predicted_disruption": [0]})
    raw_labels = pd.DataFrame({"supplier_id": ["s0"] * 5, "time": range(5), "supplier_disrupted": [0, 0, 0, 1, 1]})
    warnings = compute_warning_times(predictions, raw_labels, horizon_periods=4)
    row = warnings.iloc[0]
    assert row.first_warning_time is None or (isinstance(row.first_warning_time, float) and np.isnan(row.first_warning_time))


# ---- disruption_onset_breakdown ----
# Added after discovering, on the real severity-generalization run, that
# recall was bit-identical across 5 independent seeds -- it turned out
# every seed caught the exact same already-ongoing-disruption examples and
# missed every genuinely fresh onset. This distinguishes "detecting a
# disruption already in progress" from "warning before one starts."


def test_onset_breakdown_separates_fresh_onset_from_already_ongoing_recall():
    predictions = pd.DataFrame(
        {
            "supplier_id": ["s0", "s1", "s2", "s3"],
            "time": [10, 10, 10, 10],
            "actual_disruption": [1, 1, 1, 0],
            "predicted_disruption": [1, 0, 1, 0],
            "risk_probability": [0.9, 0.2, 0.05, 0.5],
            "split": ["test"] * 4,
        }
    )
    # s0: disrupted already at t=10 and caught. s1: disrupted already at
    # t=10 but missed. s2: NOT yet disrupted at t=10 (fresh onset ahead),
    # flagged positive but with a low probability that ranks below the
    # negative s3 -- poor ranking quality despite the raw recall hit.
    # s3: negative example.
    raw_labels = pd.DataFrame(
        {
            "supplier_id": ["s0", "s1", "s2", "s3"],
            "time": [10, 10, 10, 10],
            "supplier_disrupted": [1, 1, 0, 0],
        }
    )
    breakdown = disruption_onset_breakdown(predictions, raw_labels)
    test = breakdown["test"]
    assert test["n_positive"] == 3
    assert test["n_already_ongoing"] == 2
    assert test["n_fresh_onset"] == 1
    assert test["recall_already_ongoing"] == pytest.approx(0.5)  # s0 caught, s1 missed
    assert test["recall_fresh_onset"] == pytest.approx(1.0)  # s2 caught

    # Ranking quality (GRAPH_SAGE_IMPROVEMENT_PLAN.md §5/§14): s2's fresh-onset
    # probability (0.05) ranks BELOW the negative s3 (0.5) -- worst possible
    # ranking for a single-pair comparison. The ongoing pair {s0=0.9, s1=0.2}
    # against the same negative gets one pair right and one wrong (chance).
    assert test["roc_auc_fresh_onset"] == pytest.approx(0.0)
    assert test["roc_auc_already_ongoing"] == pytest.approx(0.5)
    assert test["pr_auc_fresh_onset"] is not None
    assert test["pr_auc_already_ongoing"] is not None
    assert test["roc_auc_fresh_onset"] < test["roc_auc_already_ongoing"]


def test_onset_breakdown_omits_splits_with_no_positives():
    predictions = pd.DataFrame(
        {"supplier_id": ["s0"], "time": [0], "actual_disruption": [0], "predicted_disruption": [0], "split": ["train"]}
    )
    raw_labels = pd.DataFrame({"supplier_id": ["s0"], "time": [0], "supplier_disrupted": [0]})
    assert disruption_onset_breakdown(predictions, raw_labels) == {}


def test_evaluate_experiment_includes_onset_breakdown(tiny_prepared_data):
    model = _tiny_model(tiny_prepared_data)
    result = evaluate_experiment(model, tiny_prepared_data, tiny_prepared_data.config.threshold)
    assert isinstance(result.onset_breakdown, dict)
    for split, stats in result.onset_breakdown.items():
        assert stats["n_fresh_onset"] + stats["n_already_ongoing"] == stats["n_positive"]


# ---- temporal_variation_summary ----
# Added after discovering that GRAPH_SAGE_IMPROVEMENT_PLAN.md's
# "static_plus_graph" feature mode strips every trace of time from the
# graph, causing every supplier to get one fixed risk_probability for all
# of its prediction times -- which then scored higher than the full model,
# with zero cross-seed variance, because it happened to match a test
# window dominated by one long-running event. This check makes that
# degenerate case detectable automatically.


def test_temporal_variation_detects_a_fully_time_invariant_supplier():
    predictions = pd.DataFrame(
        {
            "supplier_id": ["s0", "s0", "s0", "s1", "s1", "s1"],
            "time": [10, 11, 12, 10, 11, 12],
            "risk_probability": [0.5, 0.5, 0.5, 0.1, 0.4, 0.9],  # s0 constant, s1 varies
            "split": ["test"] * 6,
        }
    )
    summary = temporal_variation_summary(predictions)
    assert summary["test"]["n_suppliers"] == 2
    assert summary["test"]["n_suppliers_with_time_variation"] == 1  # only s1
    assert summary["test"]["fraction_time_varying"] == pytest.approx(0.5)


def test_temporal_variation_all_varying_gives_fraction_one():
    predictions = pd.DataFrame(
        {
            "supplier_id": ["s0", "s0", "s1", "s1"],
            "time": [10, 11, 10, 11],
            "risk_probability": [0.2, 0.3, 0.6, 0.7],
            "split": ["test"] * 4,
        }
    )
    summary = temporal_variation_summary(predictions)
    assert summary["test"]["fraction_time_varying"] == pytest.approx(1.0)


def test_temporal_variation_omits_empty_splits():
    predictions = pd.DataFrame({"supplier_id": ["s0"], "time": [0], "risk_probability": [0.5], "split": ["train"]})
    summary = temporal_variation_summary(predictions)
    assert "test" not in summary
    assert "validation" not in summary


def test_evaluate_experiment_includes_temporal_variation(tiny_prepared_data):
    model = _tiny_model(tiny_prepared_data)
    result = evaluate_experiment(model, tiny_prepared_data, tiny_prepared_data.config.threshold)
    assert isinstance(result.temporal_variation, dict)
    for split, stats in result.temporal_variation.items():
        assert 0.0 <= stats["fraction_time_varying"] <= 1.0
        assert stats["n_suppliers_with_time_variation"] <= stats["n_suppliers"]
