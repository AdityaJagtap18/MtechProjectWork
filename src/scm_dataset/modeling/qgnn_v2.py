"""QGNN-v2: matched classical/quantum heads on a frozen graph-derived
representation (QGNN_V2_HYBRID_IMPLEMENTATION_AND_DEPTH_BENCHMARK.md).

    frozen GraphSAGE-Full embedding -> PCA -> 8D (graph_embedding_reduction.py)
        /                                          \
  ClassicalHybridHead (this file)          QGNN (modeling/qgnn.py, UNCHANGED --
       Linear(d,8)->ReLU->Linear(8,1)       it already only needs a
                                             [batch, n_qubits] tensor)

Neither head touches the graph at training time: embeddings are
extracted once (graph_embedding_reduction.py) and cached in
`V2PreparedData.reduced`, so training here is flat-mini-batch over
(supplier, time) rows -- mirroring `qgnn.train_qgnn`'s own flat-batch
design, generalized to work for either model type via a single shared
loop (`train_v2_head`) rather than duplicating it per model.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from .config import GraphSAGEConfig, load_config
from .data import BenchmarkData
from .losses import build_loss, compute_pos_weight
from .metrics import compute_classification_metrics


@dataclass
class QGNNv2ArchConfig:
    n_components: int = 8
    n_layers: int = 2
    mlp_hidden: int = 8
    device: str = "default.qubit"


@dataclass
class QGNNv2ExperimentConfig:
    """Wraps an ordinary, unmodified `GraphSAGEConfig` (`base`, describing
    the frozen encoder being reused) with the v2-specific `qgnn_v2`
    section `configs/qgnn.yaml`'s schema has no room for -- mirrors
    `qgnn.QGNNExperimentConfig`'s own pattern exactly."""

    base: GraphSAGEConfig = field(default_factory=GraphSAGEConfig)
    qgnn_v2: QGNNv2ArchConfig = field(default_factory=QGNNv2ArchConfig)


def load_qgnn_v2_config(path: str) -> QGNNv2ExperimentConfig:
    base = load_config(path)  # existing, unmodified function -- ignores the "qgnn_v2:" section harmlessly
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return QGNNv2ExperimentConfig(base=base, qgnn_v2=QGNNv2ArchConfig(**raw.get("qgnn_v2", {})))


class ClassicalHybridHead(nn.Module):
    """The matched classical control (plan section 5): `Linear(d, hidden)
    -> ReLU -> Linear(hidden, 1)`, no sigmoid (BCEWithLogitsLoss), same
    shape convention as QGNN's own MLP head for a fair parameter-count
    comparison (modeling/qgnn.py's QGNN.mlp)."""

    def __init__(self, d: int, hidden: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


def build_classical_head(d: int, hidden: int = 8) -> ClassicalHybridHead:
    return ClassicalHybridHead(d, hidden)


@dataclass
class V2PreparedData:
    """Flat analogue of `pipeline.PreparedData` for QGNN-v2: neither head
    needs the graph/snapshot machinery, only a (supplier_id, time) ->
    reduced-embedding lookup. Duck-types `.benchmark`/`.config` so
    `evaluate.py`'s existing `_finalize_evaluation` (via `build_supplier_context`,
    `compute_warning_times`, `disruption_onset_breakdown`, which only read
    those two attributes) works unmodified."""

    config: GraphSAGEConfig
    benchmark: BenchmarkData
    examples: pd.DataFrame  # supplier_id, time, target, split
    reduced: pd.DataFrame  # (supplier_id, time)-indexed, reduced_0..reduced_{d-1}, every split
    n_components: int


def build_v2_prepared(
    config: GraphSAGEConfig, benchmark: BenchmarkData, examples: pd.DataFrame, embedding_frame: pd.DataFrame, n_components: int,
):
    """Fits PCA on the train-split rows of `embedding_frame` only (via
    `graph_embedding_reduction.fit_embedding_pca`), transforms every row,
    and returns `(V2PreparedData, fitted_reducer)`. `examples` is the
    ORIGINAL (unreduced) `PreparedData.examples` -- identical
    supplier/time/target/split rows GraphSAGE-Full itself was scored on."""
    from .graph_embedding_reduction import fit_embedding_pca

    train_examples = examples[examples["split"] == "train"]
    reducer = fit_embedding_pca(embedding_frame, train_examples, n_components)
    reduced = reducer.transform(embedding_frame)
    reduced.columns = [f"reduced_{i}" for i in range(n_components)]
    return V2PreparedData(config=config, benchmark=benchmark, examples=examples, reduced=reduced, n_components=n_components), reducer


def apply_fitted_reducer(config: GraphSAGEConfig, benchmark: BenchmarkData, examples: pd.DataFrame, embedding_frame: pd.DataFrame, reducer) -> V2PreparedData:
    """Cross-dataset counterpart of `build_v2_prepared`: applies an
    ALREADY-fit reducer (from a different, source dataset) to this
    (target) benchmark's own embeddings -- never refits. Mirrors
    `pipeline.prepare_reduced_for_cross_dataset_eval`'s contract exactly."""
    reduced = reducer.transform(embedding_frame)
    reduced.columns = [f"reduced_{i}" for i in range(reducer.n_components)]
    examples = examples.copy()
    examples["split"] = "test"
    return V2PreparedData(config=config, benchmark=benchmark, examples=examples, reduced=reduced, n_components=reducer.n_components)


def _flatten(v2prepared: V2PreparedData, split: str) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    sub = v2prepared.examples[v2prepared.examples["split"] == split]
    joined = sub.set_index(["supplier_id", "time"]).join(v2prepared.reduced, how="left")
    if joined[v2prepared.reduced.columns].isna().any().any():
        raise ValueError(f"split={split!r}: some examples have no reduced embedding -- supplier/time mismatch")
    x = torch.tensor(joined[v2prepared.reduced.columns].values.astype(np.float32))
    y = torch.tensor(joined["target"].values.astype(np.float32))
    return x, y, joined.reset_index()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class V2TrainResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_pr_auc: float
    stopped_early: bool
    training_duration_seconds: float
    pos_weight: float
    class_balance: dict = field(default_factory=dict)


def train_v2_head(
    v2prepared: V2PreparedData, model: nn.Module, seed: int, batch_size: int = 2048, verbose: bool = True,
) -> V2TrainResult:
    """Model-agnostic training loop -- works for `ClassicalHybridHead` or
    `qgnn.QGNN` identically, since both are just `nn.Module`s mapping
    `[batch, d] -> [batch]` logits. Mirrors `qgnn.train_qgnn`'s optimizer/
    loss/early-stopping setup exactly (same Adam config, same
    class-weighted BCEWithLogitsLoss, same validation-PR-AUC early
    stopping) for a fair comparison between the two heads."""
    set_seed(seed)
    cfg = v2prepared.config
    from .train import class_balance_summary, print_class_balance

    class_balance = class_balance_summary(v2prepared.examples)
    if verbose:
        print_class_balance(v2prepared.examples)

    train_x, train_y, _ = _flatten(v2prepared, "train")
    val_x, val_y, _ = _flatten(v2prepared, "validation")
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
        for batch_start in range(0, n_train, batch_size):
            idx = perm[batch_start:batch_start + batch_size]
            optimizer.zero_grad()
            logits = model(train_x[idx])
            loss = loss_fn(logits, train_y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(idx)
            epoch_n += len(idx)
        train_loss = epoch_loss_sum / epoch_n

        model.eval()
        with torch.no_grad():
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
        history_rows.append({
            "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
            "validation_pr_auc": val_pr_auc, "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else float("nan"),
        })
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

    return V2TrainResult(
        model=model, history=pd.DataFrame(history_rows), best_epoch=best_epoch,
        best_val_pr_auc=best_val_pr_auc, stopped_early=stopped_early,
        training_duration_seconds=duration, pos_weight=pos_weight, class_balance=class_balance,
    )


def generate_predictions_v2(model: nn.Module, v2prepared: V2PreparedData) -> pd.DataFrame:
    """Mirrors `evaluate.generate_predictions`/`qgnn.generate_predictions_qgnn`:
    one forward pass per split (embeddings already cached, no snapshot
    loop needed at all)."""
    model.eval()
    frames = []
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            x, y, meta = _flatten(v2prepared, split)
            if x.shape[0] == 0:
                continue
            probs = torch.sigmoid(model(x)).numpy()
            out = meta[["supplier_id", "time"]].copy()
            out["risk_probability"] = probs
            out["actual_disruption"] = y.numpy()
            out["split"] = split
            frames.append(out)
    predictions = pd.concat(frames, ignore_index=True)
    if not predictions["risk_probability"].between(0.0, 1.0).all():
        raise ValueError("risk_probability outside [0, 1] -- sigmoid output should never happen; check model output")
    return predictions


def evaluate_v2(model: nn.Module, v2prepared: V2PreparedData, threshold_cfg):
    """Mirrors `evaluate.evaluate_experiment`, reusing its shared tail
    (`_finalize_evaluation`) directly -- unmodified, model-agnostic."""
    from .evaluate import _finalize_evaluation
    from .metrics import select_threshold

    predictions = generate_predictions_v2(model, v2prepared)
    val = predictions[predictions["split"] == "validation"]
    threshold = select_threshold(
        val["actual_disruption"].values, val["risk_probability"].values,
        policy=threshold_cfg.policy, value=threshold_cfg.value, target_value=threshold_cfg.target_value,
    )
    return _finalize_evaluation(predictions, threshold, threshold_cfg.policy, v2prepared)


def evaluate_v2_on_target(model: nn.Module, v2prepared_target: V2PreparedData, threshold: float):
    """Cross-dataset counterpart (plan section 12): scores a target
    V2PreparedData (every example already split="test", built via
    `apply_fitted_reducer`) with a source-trained model and a threshold
    already selected on the SOURCE validation split. No target label
    ever chosen the threshold."""
    from .evaluate import _finalize_evaluation

    predictions = generate_predictions_v2(model, v2prepared_target)
    return _finalize_evaluation(
        predictions, threshold, "external (selected on a different dataset's validation split)", v2prepared_target
    )
