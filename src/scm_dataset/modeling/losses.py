"""Weighted BCE loss (plan §21/§34).

`compute_pos_weight` must only ever be called with TRAIN-split targets --
callers in `train.py` enforce this by construction (the training loop only
ever has train-split tensors in scope when it builds the loss).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(train_targets: np.ndarray) -> float:
    """pos_weight = negative_training_examples / positive_training_examples
    (plan §21). Returns 1.0 (no reweighting) if the training split happens
    to contain zero positives -- a degenerate-split guard, not a modeling
    choice; `train.py` also prints the raw counts so this is never silent."""
    positives = float(np.sum(train_targets == 1))
    negatives = float(np.sum(train_targets == 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def build_loss(class_weighting: str, train_targets: np.ndarray | None = None) -> nn.BCEWithLogitsLoss:
    if class_weighting == "none":
        return nn.BCEWithLogitsLoss()
    if class_weighting != "balanced":
        raise ValueError(f"unknown class_weighting={class_weighting!r}, expected 'balanced' or 'none'")
    if train_targets is None:
        raise ValueError("class_weighting='balanced' requires train_targets to compute pos_weight")
    pos_weight = compute_pos_weight(train_targets)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
