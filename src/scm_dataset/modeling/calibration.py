"""Probability calibration (plan §38/§46).

Risk probabilities should be meaningful, not just a ranking score -- this
module measures how close they come. Never used to rescale predictions in
the first-run baseline (plan doesn't ask for a calibration *correction*,
only for it to be evaluated and reported).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    mean_predicted, observed_frequency, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        bin_counts.append(int(mask.sum()))
        mean_predicted.append(float(y_prob[mask].mean()) if mask.any() else None)
        observed_frequency.append(float(y_true[mask].mean()) if mask.any() else None)

    return {
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "mean_predicted_probability": mean_predicted,
        "observed_frequency": observed_frequency,
        "bin_counts": bin_counts,
    }


def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    result = {
        "brier_score": float(brier_score_loss(y_true, y_prob)) if len(y_true) else None,
        "expected_calibration_error": expected_calibration_error(y_true, y_prob, n_bins) if len(y_true) else None,
        "curve": calibration_curve_data(y_true, y_prob, n_bins) if len(y_true) else None,
    }
    return result
