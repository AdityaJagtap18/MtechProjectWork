"""Classification metrics and threshold selection (plan §31-38/§47).

`compute_classification_metrics` always takes continuous probabilities
plus a threshold -- PR-AUC/ROC-AUC are computed from the probabilities
directly (plan §36/§37: "do not calculate PR-AUC only from binary
predictions"), never from the thresholded predictions.

`select_threshold` must only ever be called with validation-split arrays
(plan §39/§79: "never tune the threshold on test data") -- like
`losses.compute_pos_weight`, this module enforces nothing itself; it's
`train.py`/`evaluate.py` that only ever have validation tensors in scope
when they call it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        roc_auc = None
        pr_auc = None
        roc_auc_note = f"undefined: split contains only class {int(y_true[0]) if len(y_true) else 'N/A'}"
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc_note = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "n_examples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else None,
        "roc_auc": roc_auc,
        "roc_auc_note": roc_auc_note,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_true, y_prob)) if n_classes >= 1 else None,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, policy: str, value: float = 0.5, target_value: float = 0.8) -> float:
    """plan §39/§47. `policy`:
    - "fixed": returns `value` unchanged (default 0.5).
    - "f1_optimal": threshold maximizing F1 on the given (validation) set.
    - "recall_constrained": highest-precision threshold with recall >= target_value.
    - "precision_constrained": highest-recall threshold with precision >= target_value.
    """
    if policy == "fixed":
        return float(value)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return float(value)  # can't optimize a threshold against a single-class set; fall back

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    precision, recall = precision[:-1], recall[:-1]

    if policy == "f1_optimal":
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
        if not len(f1):
            return float(value)
        return float(thresholds[int(np.argmax(f1))])
    if policy == "recall_constrained":
        mask = recall >= target_value
        if not mask.any():
            return float(value)
        candidates = np.where(mask, precision, -1)
        return float(thresholds[int(np.argmax(candidates))])
    if policy == "precision_constrained":
        mask = precision >= target_value
        if not mask.any():
            return float(value)
        candidates = np.where(mask, recall, -1)
        return float(thresholds[int(np.argmax(candidates))])
    raise ValueError(f"unknown threshold policy={policy!r}")
