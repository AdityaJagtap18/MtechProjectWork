"""Precision@K% / Recall@K% risk-ranking metrics (QGNN-v3.1 confirmation
benchmark). Purely additive: does not modify `metrics.py` or
`evaluate.py` -- no such ranking metric existed in the repository before
this, so there is nothing to reuse for its core computation, but it is
deliberately generic and model-agnostic (operates on any
`risk_probability`/`actual_disruption` prediction table) so it applies
identically to variant A and variant D, or to any other model in this
project, rather than being invented specifically for D.

"Top K%" ranks all examples in the given predictions by predicted risk
probability, descending, and takes the top `k_percent` of rows -- the
same "prioritize the highest-risk suppliers" framing as the project's
existing `evaluate.build_risk_ranking`, just summarized as a single
precision/recall figure instead of a full ranked table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_recall_at_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float) -> dict:
    """`y_true`: 0/1 array. `y_score`: predicted risk probability, same
    length. Ranks descending by `y_score`, takes the top `k_percent`
    of rows (at least 1), reports how many of the true positives in the
    whole set were caught in that top slice."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return {"k_percent": k_percent, "k_count": 0, "precision_at_k": None, "recall_at_k": None, "n_total": 0, "n_positive": 0}

    k = max(1, int(round(n * k_percent / 100.0)))
    order = np.argsort(-y_score, kind="stable")
    top_k_idx = order[:k]
    true_positives_in_top_k = int(y_true[top_k_idx].sum())
    total_positives = int(y_true.sum())

    return {
        "k_percent": k_percent,
        "k_count": k,
        "precision_at_k": true_positives_in_top_k / k,
        "recall_at_k": (true_positives_in_top_k / total_positives) if total_positives > 0 else None,
        "n_total": n,
        "n_positive": total_positives,
    }


def risk_ranking_summary(predictions: pd.DataFrame, split: str = "test", k_percents: tuple[float, ...] = (5.0, 10.0)) -> dict:
    """`predictions` must have `risk_probability`, `actual_disruption`,
    `split` columns -- the same schema every evaluation function in this
    project already produces (`evaluate.generate_predictions`,
    `qgnn_v2.generate_predictions_v2`, ...). Returns one
    `precision_recall_at_k_percent` result per requested `k_percent`."""
    sub = predictions[predictions["split"] == split]
    return {
        f"k={k}%": precision_recall_at_k_percent(sub["actual_disruption"].values, sub["risk_probability"].values, k)
        for k in k_percents
    }
