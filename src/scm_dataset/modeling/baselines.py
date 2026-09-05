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


def _finalize(name: str, predictions: pd.DataFrame, threshold_cfg: ThresholdConfig, raw_supplier_labels: pd.DataFrame) -> BaselineResult:
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
