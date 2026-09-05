"""Tests for modeling/train.py and modeling/losses.py (plan §21/§23/§24/§34/§36)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scm_dataset.modeling.losses import build_loss, compute_pos_weight
from scm_dataset.modeling.train import class_balance_summary, train_graphsage


def test_pos_weight_is_negative_over_positive_count():
    targets = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 1 positive, 9 negative
    assert compute_pos_weight(targets) == pytest.approx(9.0)


def test_pos_weight_falls_back_to_one_with_zero_positives():
    targets = np.zeros(10)
    assert compute_pos_weight(targets) == 1.0


def test_build_loss_balanced_uses_pos_weight_from_given_targets_only():
    train_targets = np.array([1, 0, 0, 0])  # pos_weight should be 3.0
    loss_fn = build_loss("balanced", train_targets)
    assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
    assert loss_fn.pos_weight.item() == pytest.approx(3.0)


def test_build_loss_none_ignores_targets():
    loss_fn = build_loss("none")
    assert loss_fn.pos_weight is None


def test_build_loss_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown class_weighting"):
        build_loss("oversample")


def test_class_balance_summary_matches_manual_counts():
    examples = pd.DataFrame(
        {
            "supplier_id": ["s0"] * 6,
            "time": range(6),
            "target": [1, 0, 0, 1, 0, 0],
            "split": ["train", "train", "train", "validation", "validation", "test"],
        }
    )
    summary = class_balance_summary(examples)
    assert summary["train"] == {"n_examples": 3, "n_positive": 1, "n_negative": 2, "positive_rate": pytest.approx(1 / 3)}
    assert summary["validation"] == {"n_examples": 2, "n_positive": 1, "n_negative": 1, "positive_rate": 0.5}
    assert summary["test"] == {"n_examples": 1, "n_positive": 0, "n_negative": 1, "positive_rate": 0.0}


def test_train_graphsage_never_builds_a_test_split_snapshot(tiny_prepared_data, monkeypatch):
    # Directly observes which prediction times the training loop asks the
    # snapshot builder to build, rather than inferring it from a side
    # effect on the loss -- a poisoned test-split target wouldn't
    # necessarily produce a non-finite loss (BCE tolerates out-of-{0,1}
    # targets numerically), so this is the actual invariant to check.
    examples = tiny_prepared_data.examples
    test_times = set(examples.loc[examples["split"] == "test", "time"].unique())
    assert test_times, "fixture must have a non-empty test split for this test to mean anything"

    requested_times = []
    original_build = tiny_prepared_data.snapshot_builder.build

    def recording_build(t):
        requested_times.append(t)
        return original_build(t)

    monkeypatch.setattr(tiny_prepared_data.snapshot_builder, "build", recording_build)
    tiny_prepared_data.config.training.epochs = 2
    train_graphsage(tiny_prepared_data, seed=42, verbose=False)

    assert set(requested_times) & test_times == set()


def test_early_stopping_selects_best_val_pr_auc_epoch_not_last_epoch(tiny_prepared_data, monkeypatch):
    """Forces validation PR-AUC to go up then down across epochs and checks
    the checkpoint restored at the end is the peak epoch's, not the last
    trained epoch's (plan §36's "do not select the model using test/last
    performance")."""
    import scm_dataset.modeling.train as train_module

    scripted_pr_aucs = iter([0.9, 0.95, 0.3, 0.2, 0.1])  # peak at epoch 2

    def fake_metrics(y_true, y_prob, threshold):
        return {"pr_auc": next(scripted_pr_aucs), "f1": 0.5, "roc_auc": 0.5}

    monkeypatch.setattr(train_module, "compute_classification_metrics", fake_metrics)
    tiny_prepared_data.config.training.epochs = 5
    tiny_prepared_data.config.training.early_stopping_patience = 5

    result = train_graphsage(tiny_prepared_data, seed=42, verbose=False)
    assert result.best_epoch == 2
    assert result.best_val_pr_auc == pytest.approx(0.95)


def test_early_stopping_patience_actually_stops_training_early(tiny_prepared_data, monkeypatch):
    import scm_dataset.modeling.train as train_module

    scripted_pr_aucs = iter([0.9] + [0.1] * 20)  # never improves again after epoch 1

    def fake_metrics(y_true, y_prob, threshold):
        return {"pr_auc": next(scripted_pr_aucs), "f1": 0.5, "roc_auc": 0.5}

    monkeypatch.setattr(train_module, "compute_classification_metrics", fake_metrics)
    tiny_prepared_data.config.training.epochs = 100
    tiny_prepared_data.config.training.early_stopping_patience = 3

    result = train_graphsage(tiny_prepared_data, seed=42, verbose=False)
    assert result.stopped_early is True
    assert len(result.history) == 1 + 3  # peak epoch + `patience` non-improving epochs
    assert result.best_epoch == 1


def test_best_checkpoint_actually_restored_not_final_epoch_weights(tiny_prepared_data, monkeypatch):
    # Confirms load_state_dict(best_state) actually happened: the model
    # train_graphsage returns must carry the exact weights checkpointed at
    # the peak epoch, not whatever further (worse) epochs produced.
    #
    # Approach: `set_seed(seed)` resets every RNG at the start of every
    # `train_graphsage` call, and neither training nor eval-mode validation
    # consumes any *extra* randomness (dropout is off in eval). So training
    # for exactly 1 epoch, and separately training for 4 epochs where only
    # epoch 1 "improves" (scripted), must produce bit-identical epoch-1
    # weights in both runs -- letting us check the 4-epoch run's returned
    # model against a clean 1-epoch reference instead of reaching into
    # copy.deepcopy internals (which recurse through itself and break a
    # naive wrapper).
    import scm_dataset.modeling.train as train_module

    tiny_prepared_data.config.training.epochs = 1
    tiny_prepared_data.config.training.early_stopping_patience = 10
    reference = train_graphsage(tiny_prepared_data, seed=42, verbose=False)
    reference_state = {k: v.clone() for k, v in reference.model.state_dict().items()}

    call_count = {"n": 0}

    def fake_metrics(y_true, y_prob, threshold):
        call_count["n"] += 1
        pr_auc = 0.9 if call_count["n"] == 1 else 0.1  # only epoch 1 ever improves
        return {"pr_auc": pr_auc, "f1": 0.5, "roc_auc": 0.5}

    monkeypatch.setattr(train_module, "compute_classification_metrics", fake_metrics)
    tiny_prepared_data.config.training.epochs = 4

    result = train_graphsage(tiny_prepared_data, seed=42, verbose=False)
    assert result.best_epoch == 1
    for key, value in reference_state.items():
        assert torch.equal(value, result.model.state_dict()[key])
