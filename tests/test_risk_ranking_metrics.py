"""Tests for the new, model-agnostic Precision@K%/Recall@K% metric
(QGNN-v3.1 confirmation benchmark)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scm_dataset.modeling.risk_ranking_metrics import precision_recall_at_k_percent, risk_ranking_summary


def test_precision_recall_at_k_percent_hand_computed_example():
    # 10 examples, 3 positives, scores chosen so the top-20% (2 rows) are
    # both positive -- precision@20%=1.0, recall@20%=2/3.
    y_true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.05, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35])
    result = precision_recall_at_k_percent(y_true, y_score, k_percent=20.0)
    assert result["k_count"] == 2
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == pytest.approx(2 / 3)


def test_precision_recall_at_k_percent_rounds_up_to_at_least_one_row():
    y_true = np.array([0, 0, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9])
    result = precision_recall_at_k_percent(y_true, y_score, k_percent=5.0)  # 5% of 4 rounds to 0 -> floored to 1
    assert result["k_count"] == 1
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0


def test_precision_recall_at_k_percent_handles_zero_positives():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.9])
    result = precision_recall_at_k_percent(y_true, y_score, k_percent=25.0)
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] is None  # undefined, not silently zero


def test_risk_ranking_summary_filters_by_split_and_reports_both_k():
    predictions = pd.DataFrame({
        "risk_probability": [0.9, 0.8, 0.1, 0.05, 0.99],
        "actual_disruption": [1, 1, 0, 0, 1],
        "split": ["test", "test", "test", "test", "train"],  # last row excluded
    })
    summary = risk_ranking_summary(predictions, split="test", k_percents=(25.0, 50.0))
    assert set(summary.keys()) == {"k=25.0%", "k=50.0%"}
    assert summary["k=25.0%"]["n_total"] == 4  # train row correctly excluded
