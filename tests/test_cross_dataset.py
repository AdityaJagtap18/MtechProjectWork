"""Tests for D2 true cross-dataset evaluation
(GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md): training on one
dataset, evaluating on a completely different one, with the threshold and
preprocessing coming exclusively from the source."""

from __future__ import annotations

import numpy as np
import pytest

from scm_dataset.modeling.baselines import (
    run_logistic_regression_baseline_cross_dataset,
    run_majority_baseline_cross_dataset,
)
from scm_dataset.modeling.evaluate import evaluate_experiment, evaluate_on_target_dataset, generate_predictions
from scm_dataset.modeling.graphsage import build_model
from scm_dataset.modeling.pipeline import prepare_for_cross_dataset_eval, prepare_from_benchmark
from scm_dataset.modeling.train import train_graphsage

from conftest import make_tiny_config


def _tiny_model(prepared):
    in_dims = {nt.value: dim for nt, dim in prepared.snapshot_builder.feature_dims().items()}
    edge_types = list(prepared.snapshot_builder.topology.edge_index_dict.keys())
    return build_model(in_dims=in_dims, edge_types=edge_types, hidden_dim=16, num_layers=2, dropout=0.0)


def test_prepare_for_cross_dataset_eval_marks_every_example_as_test(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)

    assert (prepared_target.examples["split"] == "test").all()
    assert len(prepared_target.examples) > 0
    assert "validation" not in prepared_target.examples["split"].unique()
    assert "train" not in prepared_target.examples["split"].unique()


def test_prepare_for_cross_dataset_eval_never_refits_preprocessor(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)

    # the exact same object, not a re-fit copy with the same values
    assert prepared_target.preprocessor is prepared_source.preprocessor


def test_prepare_for_cross_dataset_eval_uses_target_graph_not_source(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)

    assert prepared_target.snapshot_builder.supplier_id_order() is not None
    # target's benchmark/graph is genuinely tiny_benchmark_b's, not source's
    assert prepared_target.benchmark.dataset_id == tiny_benchmark_b.dataset_id
    assert prepared_target.benchmark is tiny_benchmark_b


def test_evaluate_on_target_dataset_uses_the_supplied_threshold_verbatim(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)
    model = _tiny_model(prepared_source)

    result_low = evaluate_on_target_dataset(model, prepared_target, threshold=0.0)
    result_high = evaluate_on_target_dataset(model, prepared_target, threshold=1.0)

    assert result_low.threshold == 0.0
    assert result_high.threshold == 1.0
    # threshold=0.0 -> everyone flagged positive; threshold=1.0 -> no probability (in (0,1)) reaches it
    assert (result_low.predictions["predicted_disruption"] == 1).all()
    assert (result_high.predictions["predicted_disruption"] == 0).all()
    assert "external" in result_low.threshold_policy


def test_evaluate_on_target_dataset_metrics_only_cover_test_split(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)
    model = _tiny_model(prepared_source)

    result = evaluate_on_target_dataset(model, prepared_target, threshold=0.5)
    assert set(result.metrics_by_split.keys()) == {"test"}
    assert result.metrics_by_split["test"]["n_examples"] == len(prepared_target.examples)


def test_full_cross_dataset_pipeline_runs_end_to_end(tiny_benchmark, tiny_benchmark_b):
    """Mirrors the real D2 script's shape: train on source, evaluate
    within-seed on source, then cross-dataset on target using source's
    threshold -- with a tiny epoch budget, purely a wiring/shape check."""
    config = make_tiny_config()
    config.training.epochs = 2
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    train_result = train_graphsage(prepared_source, seed=42, verbose=False)

    source_eval = evaluate_experiment(train_result.model, prepared_source, config.threshold)
    assert "test" in source_eval.metrics_by_split

    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)
    cross_eval = evaluate_on_target_dataset(train_result.model, prepared_target, threshold=source_eval.threshold)

    assert cross_eval.predictions["risk_probability"].between(0.0, 1.0).all()
    assert set(cross_eval.predictions["actual_disruption"].unique()) <= {0, 1}
    assert cross_eval.onset_breakdown  # at least the "test" key should be present given target has some positives or not
    assert isinstance(cross_eval.temporal_variation, dict)


# ---- cross-dataset baselines ----


def test_majority_cross_dataset_uses_source_train_rate_and_source_threshold(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)

    source_train_rate = float(prepared_source.examples.loc[prepared_source.examples["split"] == "train", "target"].mean())
    result = run_majority_baseline_cross_dataset(prepared_source, prepared_target, config.threshold)

    assert np.allclose(result.predictions["risk_probability"].values, source_train_rate)
    assert set(result.predictions["split"].unique()) == {"test"}


def test_logistic_regression_cross_dataset_fits_only_on_source_train(tiny_benchmark, tiny_benchmark_b):
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)

    result = run_logistic_regression_baseline_cross_dataset(prepared_source, prepared_target, config.threshold, seed=42)
    assert set(result.predictions["split"].unique()) == {"test"}
    assert result.predictions["risk_probability"].between(0.0, 1.0).all()
    assert len(result.predictions) == len(prepared_target.examples)


def test_logistic_regression_cross_dataset_never_touches_target_labels_for_fitting(tiny_benchmark, tiny_benchmark_b, monkeypatch):
    # Corrupt the target's own label table entirely (nonsense values) and
    # confirm the fitted coefficients / threshold are unaffected -- only
    # the final scoring step should ever read prepared_target at all, and
    # even then only its features, never labels, until metrics are computed.
    config = make_tiny_config()
    prepared_source = prepare_from_benchmark(config, tiny_benchmark)
    prepared_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)

    baseline = run_logistic_regression_baseline_cross_dataset(prepared_source, prepared_target, config.threshold, seed=42)

    tampered_target = prepare_for_cross_dataset_eval(config, tiny_benchmark_b, prepared_source.preprocessor)
    tampered_target.examples = tampered_target.examples.copy()
    tampered_target.examples["target"] = 1  # every example now "positive" -- must not affect the fitted model/threshold

    tampered_result = run_logistic_regression_baseline_cross_dataset(prepared_source, tampered_target, config.threshold, seed=42)
    assert baseline.threshold == tampered_result.threshold
    # risk_probability depends only on features (unchanged), not the tampered target column
    assert (baseline.predictions["risk_probability"].values == tampered_result.predictions["risk_probability"].values).all()
