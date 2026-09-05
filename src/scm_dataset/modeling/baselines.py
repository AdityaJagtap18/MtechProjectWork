"""Non-graph baselines (plan §53-57): Majority classifier and Logistic
Regression, run over the *exact same* target, horizon, split, allowed
features, and evaluation metrics as GraphSAGE -- the only thing that
changes is the model. Logistic Regression uses the preprocessed SUPPLIER
node feature table only (static + rolling-window dynamic features, no
graph neighbor information at all): the comparison against GraphSAGE is
what answers plan §54's "does graph structure provide additional
predictive information?", and it's also literally Ablation A (plan §55,
"remove graph -- supplier-level tabular model").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..schema.nodes import NodeType
from .calibration import compute_calibration
from .config import ThresholdConfig
from .evaluate import disruption_onset_breakdown
from .metrics import compute_classification_metrics, select_threshold
from .pipeline import PreparedData

SPLITS = ("train", "validation", "test")


@dataclass
class BaselineResult:
    name: str
    predictions: pd.DataFrame
    threshold: float
    metrics_by_split: dict
    calibration_by_split: dict
    onset_breakdown: dict


def _finalize(
    name: str, predictions: pd.DataFrame, threshold_cfg: ThresholdConfig, raw_supplier_labels: pd.DataFrame,
    threshold: float | None = None,
) -> BaselineResult:
    """If `threshold` is None, selects it from `predictions`' own
    validation split (the normal, within-dataset path). If a threshold is
    passed in, it is used as-is and `predictions`' own labels never
    participate in choosing it -- the cross-dataset baselines below always
    pass in a threshold already selected on the SOURCE dataset."""
    if threshold is None:
        val = predictions[predictions["split"] == "validation"]
        threshold = select_threshold(
            val["actual_disruption"].values, val["risk_probability"].values,
            policy=threshold_cfg.policy, value=threshold_cfg.value, target_value=threshold_cfg.target_value,
        )
    predictions = predictions.copy()
    predictions["predicted_disruption"] = (predictions["risk_probability"] >= threshold).astype(int)

    metrics_by_split, calibration_by_split = {}, {}
    for split in SPLITS:
        sub = predictions[predictions["split"] == split]
        if sub.empty:
            continue
        metrics_by_split[split] = compute_classification_metrics(sub["actual_disruption"].values, sub["risk_probability"].values, threshold)
        calibration_by_split[split] = compute_calibration(sub["actual_disruption"].values, sub["risk_probability"].values)

    onset_breakdown = disruption_onset_breakdown(predictions, raw_supplier_labels)
    return BaselineResult(
        name=name, predictions=predictions, threshold=threshold, metrics_by_split=metrics_by_split,
        calibration_by_split=calibration_by_split, onset_breakdown=onset_breakdown,
    )


def run_majority_baseline(prepared: PreparedData, threshold_cfg: ThresholdConfig) -> BaselineResult:
    """plan §56: predicts the train-split positive rate for every example,
    unconditionally. Demonstrates how misleading accuracy is on a rare-event
    task (plan §66) -- it is not expected to score well on PR-AUC/F1/recall."""
    examples = prepared.examples
    train_rate = float(examples.loc[examples["split"] == "train", "target"].mean())
    predictions = examples.rename(columns={"target": "actual_disruption"}).copy()
    predictions["risk_probability"] = train_rate
    predictions = predictions[["supplier_id", "time", "risk_probability", "actual_disruption", "split"]]
    return _finalize("majority", predictions, threshold_cfg, prepared.benchmark.labels["supplier"])


def run_logistic_regression_baseline(prepared: PreparedData, threshold_cfg: ThresholdConfig, seed: int = 42) -> BaselineResult:
    """plan §57. Uses the same preprocessed supplier feature table
    GraphSAGE's supplier input projection sees (before message passing),
    fit on TRAIN examples only, `class_weight="balanced"` mirroring
    GraphSAGE's `pos_weight` (plan §34's "start with weighted BCE" applied
    to the tabular baseline's analogous imbalance handling)."""
    table = prepared.frames.frames[NodeType.SUPPLIER]
    transformed = prepared.preprocessor.transform(NodeType.SUPPLIER, table)
    feature_cols = list(transformed.columns)
    transformed = transformed.reset_index()  # -> columns: supplier_id, time, <features...>

    examples = prepared.examples.merge(transformed, on=["supplier_id", "time"], how="left")
    if examples[feature_cols].isna().any().any():
        raise ValueError("logistic regression feature matrix contains NaN after preprocessing -- unexpected")

    train = examples[examples["split"] == "train"]
    X_train = train[feature_cols].values
    y_train = train["target"].values.astype(int)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)

    X_all = examples[feature_cols].values
    probs = model.predict_proba(X_all)[:, 1]

    predictions = examples[["supplier_id", "time", "split"]].copy()
    predictions["risk_probability"] = probs
    predictions["actual_disruption"] = examples["target"].values
    return _finalize("logistic_regression", predictions, threshold_cfg, prepared.benchmark.labels["supplier"])


def run_majority_baseline_cross_dataset(prepared_source: PreparedData, prepared_target: PreparedData, threshold_cfg: ThresholdConfig) -> BaselineResult:
    """D2 cross-dataset majority baseline (GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md):
    the constant probability comes from `prepared_source`'s TRAIN split;
    the threshold comes from `run_majority_baseline` applied to the source
    (i.e. selected on the source's own validation split); both are then
    applied unchanged to every example in `prepared_target` (all
    `split="test"`, built via `pipeline.prepare_for_cross_dataset_eval`)."""
    train_rate = float(prepared_source.examples.loc[prepared_source.examples["split"] == "train", "target"].mean())
    source_result = run_majority_baseline(prepared_source, threshold_cfg)

    predictions = prepared_target.examples.rename(columns={"target": "actual_disruption"}).copy()
    predictions["risk_probability"] = train_rate
    predictions = predictions[["supplier_id", "time", "risk_probability", "actual_disruption", "split"]]
    return _finalize(
        "majority_cross_dataset", predictions, threshold_cfg, prepared_target.benchmark.labels["supplier"],
        threshold=source_result.threshold,
    )


def run_logistic_regression_baseline_cross_dataset(
    prepared_source: PreparedData, prepared_target: PreparedData, threshold_cfg: ThresholdConfig, seed: int = 42
) -> BaselineResult:
    """D2 cross-dataset Logistic Regression baseline: fit exclusively on
    `prepared_source`'s TRAIN split, threshold selected from that SAME
    fitted model's predictions on the source's own validation split (not
    a separately-refit model -- avoids any inconsistency between the
    model that sets the threshold and the model that scores the target),
    then scored on every example in `prepared_target` using the source's
    preprocessor (already baked into `prepared_target` by
    `pipeline.prepare_for_cross_dataset_eval` -- never refit on target data)."""
    table_source = prepared_source.frames.frames[NodeType.SUPPLIER]
    transformed_source = prepared_source.preprocessor.transform(NodeType.SUPPLIER, table_source)
    feature_cols = list(transformed_source.columns)
    transformed_source = transformed_source.reset_index()

    examples_source = prepared_source.examples.merge(transformed_source, on=["supplier_id", "time"], how="left")
    if examples_source[feature_cols].isna().any().any():
        raise ValueError("cross-dataset logistic regression source feature matrix contains NaN after preprocessing -- unexpected")

    train = examples_source[examples_source["split"] == "train"]
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(train[feature_cols].values, train["target"].values.astype(int))

    val = examples_source[examples_source["split"] == "validation"]
    val_probs = model.predict_proba(val[feature_cols].values)[:, 1]
    threshold = select_threshold(
        val["target"].values, val_probs, policy=threshold_cfg.policy, value=threshold_cfg.value, target_value=threshold_cfg.target_value,
    )

    table_target = prepared_target.frames.frames[NodeType.SUPPLIER]
    transformed_target = prepared_target.preprocessor.transform(NodeType.SUPPLIER, table_target).reset_index()
    examples_target = prepared_target.examples.merge(transformed_target, on=["supplier_id", "time"], how="left")
    if examples_target[feature_cols].isna().any().any():
        raise ValueError("cross-dataset logistic regression target feature matrix contains NaN after preprocessing -- unexpected")

    probs_target = model.predict_proba(examples_target[feature_cols].values)[:, 1]
    predictions = examples_target[["supplier_id", "time", "split"]].copy()
    predictions["risk_probability"] = probs_target
    predictions["actual_disruption"] = examples_target["target"].values
    return _finalize(
        "logistic_regression_cross_dataset", predictions, threshold_cfg, prepared_target.benchmark.labels["supplier"],
        threshold=threshold,
    )
