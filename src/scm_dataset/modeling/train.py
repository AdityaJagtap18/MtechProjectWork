"""GraphSAGE training loop (plan §23/§24/§35/§36).

Each distinct prediction time `t` in the train/validation split is one
forward pass: the model computes embeddings for *every* node in that
time's graph snapshot in one call, and the supplier logits it returns
already cover every supplier the label table has an example for at that
`t` (`features.build_prediction_examples` enumerates all suppliers per
`t`) -- so "one epoch" is a loop over train-split time snapshots, not over
individual (supplier, time) rows.

Early stopping watches validation PR-AUC (plan §36), never test data. The
test split is never touched anywhere in this module.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .graphsage import HeteroGraphSAGE, build_model
from .losses import build_loss, compute_pos_weight
from .metrics import compute_classification_metrics
from .pipeline import PreparedData


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_balance_summary(examples: pd.DataFrame) -> dict:
    """plan §22/§34/§67: positive/negative counts and rates per split,
    printed before training ever starts."""
    summary = {}
    for split in ("train", "validation", "test"):
        sub = examples[examples["split"] == split]
        n = len(sub)
        n_pos = int(sub["target"].sum())
        summary[split] = {
            "n_examples": n,
            "n_positive": n_pos,
            "n_negative": n - n_pos,
            "positive_rate": (n_pos / n) if n else None,
        }
    return summary


def print_class_balance(examples: pd.DataFrame) -> dict:
    summary = class_balance_summary(examples)
    print("Class balance (prediction examples, supplier x prediction_time):")
    for split, stats in summary.items():
        rate = f"{stats['positive_rate']:.4f}" if stats["positive_rate"] is not None else "n/a"
        print(f"  {split:>10}: n={stats['n_examples']:>6}  positive={stats['n_positive']:>5}  negative={stats['n_negative']:>5}  positive_rate={rate}")
    return summary


def _target_array(split_df: pd.DataFrame, t: int, supplier_order: list[str]) -> np.ndarray:
    sub = split_df[split_df["time"] == t].set_index("supplier_id")["target"]
    return sub.reindex(supplier_order).values.astype(np.float32)


@dataclass
class TrainResult:
    model: HeteroGraphSAGE
    history: pd.DataFrame
    best_epoch: int
    best_val_pr_auc: float
    stopped_early: bool
    training_duration_seconds: float
    pos_weight: float
    class_balance: dict = field(default_factory=dict)


def train_graphsage(prepared: PreparedData, seed: int, verbose: bool = True) -> TrainResult:
    set_seed(seed)
    cfg = prepared.config
    examples = prepared.examples
    class_balance = class_balance_summary(examples)
    if verbose:
        print_class_balance(examples)

    train_ex = examples[examples["split"] == "train"]
    val_ex = examples[examples["split"] == "validation"]
    if train_ex.empty or val_ex.empty:
        raise ValueError(f"empty train or validation split (train={len(train_ex)}, validation={len(val_ex)}) -- check split.strategy/config")

    train_times = sorted(train_ex["time"].unique())
    val_times = sorted(val_ex["time"].unique())

    supplier_order = prepared.snapshot_builder.supplier_id_order()
    snapshots = {t: prepared.snapshot_builder.build(int(t)) for t in sorted(set(train_times) | set(val_times))}

    feature_dims = prepared.snapshot_builder.feature_dims()
    in_dims = {nt.value: dim for nt, dim in feature_dims.items()}
    edge_types = list(prepared.snapshot_builder.topology.edge_index_dict.keys())

    model = build_model(
        in_dims=in_dims, edge_types=edge_types, hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout, aggregation=cfg.model.aggregation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    train_targets_all = train_ex["target"].values
    pos_weight = compute_pos_weight(train_targets_all)
    loss_fn = build_loss(cfg.training.class_weighting, train_targets_all)

    best_val_pr_auc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    stopped_early = False
    history_rows = []

    rng = random.Random(seed)
    start = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_order = train_times[:]
        rng.shuffle(epoch_order)
        epoch_loss_sum, epoch_n = 0.0, 0
        for t in epoch_order:
            data = snapshots[t]
            target = torch.from_numpy(_target_array(train_ex, t, supplier_order))
            optimizer.zero_grad()
            logits = model(data.x_dict, data.edge_index_dict)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(target)
            epoch_n += len(target)
        train_loss = epoch_loss_sum / epoch_n

        model.eval()
        with torch.no_grad():
            val_logit_chunks, val_target_chunks = [], []
            val_loss_sum, val_n = 0.0, 0
            for t in val_times:
                data = snapshots[t]
                target = torch.from_numpy(_target_array(val_ex, t, supplier_order))
                logits = model(data.x_dict, data.edge_index_dict)
                loss = loss_fn(logits, target)
                val_loss_sum += loss.item() * len(target)
                val_n += len(target)
                val_logit_chunks.append(logits)
                val_target_chunks.append(target.numpy())
            val_loss = val_loss_sum / val_n
            # torch.sigmoid (not a manual 1/(1+exp(-x))) avoids overflow
            # warnings/instability for large-magnitude logits.
            val_probs = torch.sigmoid(torch.cat(val_logit_chunks)).numpy()
            val_targets = np.concatenate(val_target_chunks)
            val_metrics = compute_classification_metrics(val_targets, val_probs, threshold=0.5)

        val_pr_auc = val_metrics["pr_auc"] if val_metrics["pr_auc"] is not None else float("nan")
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_pr_auc": val_pr_auc,
                "validation_f1": val_metrics["f1"],
                "validation_roc_auc": val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else float("nan"),
            }
        )
        if verbose:
            print(f"epoch {epoch:>3}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_pr_auc={val_pr_auc:.4f}  val_f1={val_metrics['f1']:.4f}")

        current = val_pr_auc if not np.isnan(val_pr_auc) else -1.0
        if current > best_val_pr_auc:
            best_val_pr_auc = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                stopped_early = True
                break

    duration = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        history=pd.DataFrame(history_rows),
        best_epoch=best_epoch,
        best_val_pr_auc=best_val_pr_auc,
        stopped_early=stopped_early,
        training_duration_seconds=duration,
        pos_weight=pos_weight,
        class_balance=class_balance,
    )
