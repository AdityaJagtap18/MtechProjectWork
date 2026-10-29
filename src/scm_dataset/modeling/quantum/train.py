"""Instrumented training loop for the QGNN-v4 stability investigation
(Phase 1): a copy of `qgnn_v2.train_v2_head`'s loop, extended to log two
things it doesn't -- per-epoch TRAIN PR-AUC (not just loss) and, for
`HybridQuantumHead` specifically, the L2 norm of the quantum circuit's own
parameter gradients each epoch. This is the direct, numeric barren-plateau
guard the original plan (PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md section 6/9)
called for and the initial v4 implementation didn't yet capture.

A separate function rather than an edit to `qgnn_v2.py`, per this phase's
explicit requirement to leave `qgnn.py`/`qgnn_v2.py`/`qgnn_v2_reupload.py`/
`qgnn_v3.py` unmodified -- v4 stays purely additive. Duplicates
`train_v2_head`'s optimizer/loss/early-stopping setup exactly (same Adam
config, same class-weighted BCEWithLogitsLoss, same validation-PR-AUC
early stopping) so diagnostic runs are directly comparable to the
existing (non-instrumented) v4 results already on disk -- only the
logging changes, not the optimization itself.

No forward-pass/architecture change: this module only adds observation,
never alters what `HybridQuantumHead`/`MatchedCapacityClassicalHead`
compute or how they're optimized.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..losses import build_loss, compute_pos_weight
from ..metrics import compute_classification_metrics
from ..qgnn_v2 import V2PreparedData, _flatten, set_seed
from ..train import class_balance_summary, print_class_balance


@dataclass
class V4DiagnosticTrainResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_pr_auc: float
    stopped_early: bool
    training_duration_seconds: float
    pos_weight: float
    class_balance: dict = field(default_factory=dict)


def _quantum_grad_norm(model: nn.Module) -> float | None:
    """L2 norm of the quantum circuit's own parameter gradients, pooled
    across every quantum parameter tensor -- `None` for a head with no
    `.quantum` submodule (`MatchedCapacityClassicalHead`), so the same
    instrumented loop works for both arms without the caller needing to
    special-case which model it's training."""
    quantum = getattr(model, "quantum", None)
    if quantum is None:
        return None
    total_sq = 0.0
    any_grad = False
    for p in quantum.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
            any_grad = True
    return total_sq**0.5 if any_grad else None


def _reduce_grad_norm(model: nn.Module) -> float | None:
    """L2 norm of the `reduce` (Linear(in_dim, n_qubits)) layer's
    gradients -- present on both head types, lets a diagnostic report
    compare "does gradient signal reach the shared bottleneck" between
    the quantum and classical arms even though only the quantum arm has
    a `.quantum` submodule to inspect separately."""
    reduce = getattr(model, "reduce", None)
    if reduce is None:
        return None
    total_sq = 0.0
    any_grad = False
    for p in reduce.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
            any_grad = True
    return total_sq**0.5 if any_grad else None


def train_v4_head_with_diagnostics(
    v4prepared: V2PreparedData, model: nn.Module, seed: int, batch_size: int = 2048, verbose: bool = True,
) -> V4DiagnosticTrainResult:
    set_seed(seed)
    cfg = v4prepared.config

    class_balance = class_balance_summary(v4prepared.examples)
    if verbose:
        print_class_balance(v4prepared.examples)

    train_x, train_y, _ = _flatten(v4prepared, "train")
    val_x, val_y, _ = _flatten(v4prepared, "validation")
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError(f"empty train or validation split (train={train_x.shape[0]}, validation={val_x.shape[0]})")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    train_targets_all = train_y.numpy()
    pos_weight = compute_pos_weight(train_targets_all)
    loss_fn = build_loss(cfg.training.class_weighting, train_targets_all)

    best_val_pr_auc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    stopped_early = False
    history_rows = []

    rng = np.random.RandomState(seed)
    n_train = train_x.shape[0]
    start = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        epoch_loss_sum, epoch_n = 0.0, 0
        quantum_grad_norms, reduce_grad_norms = [], []
        for batch_start in range(0, n_train, batch_size):
            idx = perm[batch_start:batch_start + batch_size]
            optimizer.zero_grad()
            logits = model(train_x[idx])
            loss = loss_fn(logits, train_y[idx])
            loss.backward()
            qgn = _quantum_grad_norm(model)
            if qgn is not None:
                quantum_grad_norms.append(qgn)
            rgn = _reduce_grad_norm(model)
            if rgn is not None:
                reduce_grad_norms.append(rgn)
            optimizer.step()
            epoch_loss_sum += loss.item() * len(idx)
            epoch_n += len(idx)
        train_loss = epoch_loss_sum / epoch_n

        model.eval()
        with torch.no_grad():
            train_logits = model(train_x)
            train_probs = torch.sigmoid(train_logits).numpy()
            train_metrics = compute_classification_metrics(train_targets_all, train_probs, threshold=0.5)

            val_logit_chunks = []
            val_loss_sum, val_n = 0.0, 0
            for batch_start in range(0, val_x.shape[0], batch_size):
                x_batch = val_x[batch_start:batch_start + batch_size]
                target_batch = val_y[batch_start:batch_start + batch_size]
                logits = model(x_batch)
                loss = loss_fn(logits, target_batch)
                val_loss_sum += loss.item() * len(target_batch)
                val_n += len(target_batch)
                val_logit_chunks.append(logits)
            val_loss = val_loss_sum / val_n
            val_probs = torch.sigmoid(torch.cat(val_logit_chunks)).numpy()
            val_targets = val_y.numpy()
            val_metrics = compute_classification_metrics(val_targets, val_probs, threshold=0.5)

        val_pr_auc = val_metrics["pr_auc"] if val_metrics["pr_auc"] is not None else float("nan")
        train_pr_auc = train_metrics["pr_auc"] if train_metrics["pr_auc"] is not None else float("nan")
        history_rows.append({
            "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
            "train_pr_auc": train_pr_auc,
            "validation_pr_auc": val_pr_auc, "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else float("nan"),
            "quantum_grad_norm_mean": float(np.mean(quantum_grad_norms)) if quantum_grad_norms else None,
            "quantum_grad_norm_max": float(np.max(quantum_grad_norms)) if quantum_grad_norms else None,
            "quantum_grad_norm_min": float(np.min(quantum_grad_norms)) if quantum_grad_norms else None,
            "reduce_grad_norm_mean": float(np.mean(reduce_grad_norms)) if reduce_grad_norms else None,
        })
        if verbose:
            qgn_str = f"  q_grad_norm={history_rows[-1]['quantum_grad_norm_mean']:.6f}" if quantum_grad_norms else ""
            print(f"epoch {epoch:>3}  train_loss={train_loss:.4f}  train_pr_auc={train_pr_auc:.4f}  val_loss={val_loss:.4f}  val_pr_auc={val_pr_auc:.4f}{qgn_str}")

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

    return V4DiagnosticTrainResult(
        model=model, history=pd.DataFrame(history_rows), best_epoch=best_epoch,
        best_val_pr_auc=best_val_pr_auc, stopped_early=stopped_early,
        training_duration_seconds=duration, pos_weight=pos_weight, class_balance=class_balance,
    )
