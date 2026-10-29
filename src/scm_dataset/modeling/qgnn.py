"""Hybrid QGNN (QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md sections
1, 7-9, 12-13):

    reduced supplier representation (n_qubits-dim, already standardized
    by a `reduction.SupplierReducer`)
        -> quantum angle encoding (RY per qubit)
        -> variational circuit (trainable RY per qubit per layer + a
           linear chain of CNOTs)
        -> PauliZ expectation-value measurement per qubit
        -> small classical MLP
        -> risk logit (no sigmoid -- BCEWithLogitsLoss, same convention
           as graphsage.py)

Deliberately does NOT do heterogeneous message passing: QGNN-Reduced's
whole purpose (plan section 3C) is to test whether quantum processing of
the reduced SUPPLIER representation adds value over GraphSAGE-Reduced's
graph+MLP processing of the *exact same* reduced input, so no other node
type or edge_index is read here at all -- see reduction.py's
`ReducedGraphSnapshotBuilder`, whose `.build(t).x_dict["supplier"]` is the
only tensor this module ever consumes.

Quantum library: PennyLane, chosen because `qqcuda` (named in the plan)
does not exist as an installable package (verified: PyPI has no matching
distribution) and PennyLane has a first-class PyTorch interface, letting
this slot into the exact same nn.Module/autograd/optimizer machinery
train.py already uses. Default backend: `default.qubit` (CPU,
`diff_method="backprop"`) -- measured to be ~140x faster than
`lightning.gpu` (NVIDIA cuStateVec, `diff_method="adjoint"`) at this
qubit count for numerically identical results, because PyTorch's native
autodiff vectorizes the whole batch as tensor ops, while the
adjoint-differentiation backends appear to process a "batch" as a
serialized loop internally. `lightning.gpu` was verified working (real
GPU execution confirmed via `nvidia-smi`, correct non-zero gradients) and
remains fully supported -- pass `device_name="lightning.gpu"` explicitly
to use it -- but is not the practical default at 4-8 qubits.
`QuantumCircuitLayer` falls back to `default.qubit` if the named device
cannot be constructed at all, so this module runs correctly even on a
machine with no GPU.

Quantum encoding is deterministic and never depends on labels: for a
standardized input x, angle_i = tanh(x_i) * pi, bounding every rotation to
(-pi, pi) regardless of how large |x_i| gets after standardization.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import yaml

from .config import GraphSAGEConfig, load_config
from .losses import build_loss, compute_pos_weight
from .metrics import compute_classification_metrics
from .pipeline import PreparedData


@dataclass
class QGNNArchConfig:
    reduction_method: str = "domain_selected"  # pca | domain_selected
    n_components: int = 6
    reduction_columns: list[str] | None = None
    n_layers: int = 1
    mlp_hidden: int = 8
    device: str = "default.qubit"


@dataclass
class QGNNExperimentConfig:
    """Wraps an ordinary, unmodified `GraphSAGEConfig` (`base`) with the
    QGNN-specific fields `configs/graphsage.yaml`'s schema has no room for
    -- deliberately kept separate from `config.py` (shared infrastructure
    the plan says not to modify) rather than adding fields to
    `GraphSAGEConfig` itself. `base.model` still describes GraphSAGE-
    Reduced's architecture (it IS `HeteroGraphSAGE`); `qgnn` describes
    QGNN-Reduced's circuit."""

    base: GraphSAGEConfig = field(default_factory=GraphSAGEConfig)
    qgnn: QGNNArchConfig = field(default_factory=QGNNArchConfig)


def load_qgnn_config(path: str) -> QGNNExperimentConfig:
    base = load_config(path)  # existing, unmodified function -- ignores the "qgnn:" section harmlessly
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return QGNNExperimentConfig(base=base, qgnn=QGNNArchConfig(**raw.get("qgnn", {})))

# Measured directly (see QGNN_IMPLEMENTATION_AND_BENCHMARK.md section 10):
# at 6 qubits, "default.qubit" (CPU, diff_method="backprop", true
# PyTorch-vectorized batching) trained 5 epochs on the full seed43
# benchmark in 5.8s; "lightning.gpu" (diff_method="adjoint", which
# processes a batch as a serialized per-row loop internally for this
# device) took 13.5 minutes for numerically identical results -- ~140x
# slower. "lightning.gpu" remains fully supported and verified working
# (real GPU execution, correct gradients via nvidia-smi + gradient checks)
# and is worth revisiting at a much larger qubit count/circuit depth than
# this phase uses, but is not the practical default here.
DEFAULT_DEVICE = "default.qubit"
FALLBACK_DEVICE = "default.qubit"


def _make_qnode(n_qubits: int, n_layers: int, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)
    diff_method = "adjoint" if "lightning" in device_name else "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(angles, weights):
        qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit, dev


class QuantumCircuitLayer(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int = 1, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        try:
            self._circuit, self._device = _make_qnode(n_qubits, n_layers, device_name)
            self.device_name = device_name
        except Exception as exc:  # pragma: no cover -- exercised only on machines without the GPU backend
            print(f"QuantumCircuitLayer: device {device_name!r} unavailable ({exc!r}), falling back to {FALLBACK_DEVICE!r}")
            self._circuit, self._device = _make_qnode(n_qubits, n_layers, FALLBACK_DEVICE)
            self.device_name = FALLBACK_DEVICE
        self.weights = nn.Parameter((torch.rand(n_layers, n_qubits) * 2 - 1) * math.pi)

    def forward(self, standardized_x: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(standardized_x) * math.pi
        outputs = self._circuit(angles, self.weights)
        # `default.qubit` returns float64 expectation values regardless of
        # input dtype; `lightning.gpu` returns float32. Cast explicitly so
        # the downstream MLP (float32 weights) works with either backend.
        return torch.stack(outputs, dim=-1).to(torch.float32)


class QGNN(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int = 1, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = QuantumCircuitLayer(n_qubits, n_layers, device_name)
        self.mlp = nn.Sequential(
            nn.Linear(n_qubits, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: [batch, n_qubits] reduced+standardized supplier features.
        Returns raw logits, shape [batch] -- no sigmoid, same convention
        as `HeteroGraphSAGE.forward`."""
        q_out = self.quantum(x)
        return self.mlp(q_out).squeeze(-1)

    def quantum_resource_summary(self) -> dict:
        return {
            "qubits": self.n_qubits,
            "variational_layers": self.quantum.n_layers,
            "trainable_quantum_parameters": self.quantum.weights.numel(),
            "total_trainable_parameters": sum(p.numel() for p in self.parameters()),
            "observable": "PauliZ (one per qubit)",
            "entanglement_pattern": "linear CNOT chain (wire i -> i+1)",
            "encoding": "AngleEmbedding, rotation=Y, angle = tanh(x) * pi",
            "backend": self.quantum.device_name,
        }


def build_qgnn_model(n_qubits: int, n_layers: int = 1, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE) -> QGNN:
    return QGNN(n_qubits=n_qubits, n_layers=n_layers, mlp_hidden=mlp_hidden, device_name=device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _target_array(split_df: pd.DataFrame, t: int, supplier_order: list[str]) -> np.ndarray:
    sub = split_df[split_df["time"] == t].set_index("supplier_id")["target"]
    return sub.reindex(supplier_order).values.astype(np.float32)


@dataclass
class QGNNTrainResult:
    """Parallel to `train.TrainResult`, kept separate rather than reused
    (train.py is the frozen classical baseline's shared infrastructure --
    see the module docstring in that file)."""

    model: QGNN
    history: pd.DataFrame
    best_epoch: int
    best_val_pr_auc: float
    stopped_early: bool
    training_duration_seconds: float
    pos_weight: float
    class_balance: dict = field(default_factory=dict)


def _flatten_split(prepared: PreparedData, times: list[int], split_df, supplier_order: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for t in times:
        xs.append(prepared.snapshot_builder.build(int(t)).x_dict["supplier"])
        ys.append(torch.from_numpy(_target_array(split_df, t, supplier_order)))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_qgnn(
    prepared: PreparedData, seed: int, n_layers: int = 1, mlp_hidden: int = 8,
    device_name: str = DEFAULT_DEVICE, batch_size: int = 2048, verbose: bool = True,
) -> QGNNTrainResult:
    """Mirrors `train.train_graphsage`'s optimizer/loss/early-stopping
    setup for a fair comparison, but trains over flat mini-batches of
    (supplier, time) rows rather than looping one PennyLane circuit call
    per timestep snapshot the way `train_graphsage` must (GraphSAGE
    genuinely needs a fresh per-timestep graph for message passing;
    QGNN-Reduced reads only the reduced supplier tensor, with no
    dependency on the graph at all). Measured: each PennyLane circuit call
    carries real, largely fixed dispatch overhead independent of batch
    size (~1-2s/call on `lightning.gpu` even for a tiny 6-qubit circuit),
    so looping per timestep (62 calls/epoch on the primary seed43
    benchmark) took ~16.5 minutes for 5 epochs; flat mini-batching cuts
    that to a handful of calls per epoch. This changes nothing about the
    data, split, labels, or leakage boundary -- every (supplier, time) row
    trained on is identical either way, just shuffled and grouped
    differently (a per-row shuffle here, versus GraphSAGE's per-timestep-
    block shuffle, if anything a more standard mini-batch SGD practice,
    not a weaker one)."""
    set_seed(seed)
    cfg = prepared.config
    examples = prepared.examples
    from .train import class_balance_summary, print_class_balance  # local import: train.py is not touched, just reused for identical reporting

    class_balance = class_balance_summary(examples)
    if verbose:
        print_class_balance(examples)

    train_ex = examples[examples["split"] == "train"]
    val_ex = examples[examples["split"] == "validation"]
    if train_ex.empty or val_ex.empty:
        raise ValueError(f"empty train or validation split (train={len(train_ex)}, validation={len(val_ex)}) -- check split.strategy/config")

    train_times = sorted(train_ex["time"].unique())
    val_times = sorted(val_ex["time"].unique())

    from ..schema.nodes import NodeType

    supplier_order = prepared.snapshot_builder.supplier_id_order()
    n_qubits = prepared.snapshot_builder.feature_dims()[NodeType.SUPPLIER]
    train_x, train_y = _flatten_split(prepared, train_times, train_ex, supplier_order)
    val_x, val_y = _flatten_split(prepared, val_times, val_ex, supplier_order)

    model = build_qgnn_model(n_qubits=n_qubits, n_layers=n_layers, mlp_hidden=mlp_hidden, device_name=device_name)
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

    rng = np.random.RandomState(seed)
    n_train = train_x.shape[0]
    start = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        epoch_loss_sum, epoch_n = 0.0, 0
        for batch_start in range(0, n_train, batch_size):
            idx = perm[batch_start:batch_start + batch_size]
            x_batch = train_x[idx]
            target_batch = train_y[idx]
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(target_batch)
            epoch_n += len(target_batch)
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

    return QGNNTrainResult(
        model=model, history=pd.DataFrame(history_rows), best_epoch=best_epoch,
        best_val_pr_auc=best_val_pr_auc, stopped_early=stopped_early,
        training_duration_seconds=duration, pos_weight=pos_weight, class_balance=class_balance,
    )


def generate_predictions_qgnn(model: QGNN, prepared: PreparedData) -> pd.DataFrame:
    """Mirrors `evaluate.generate_predictions` exactly, except the model
    forward pass takes only the reduced supplier tensor for time `t`, not
    `(x_dict, edge_index_dict)`."""
    model.eval()
    supplier_order = prepared.snapshot_builder.supplier_id_order()
    all_times = sorted(prepared.examples["time"].unique())
    rows = []
    with torch.no_grad():
        for t in all_times:
            x = prepared.snapshot_builder.build(int(t)).x_dict["supplier"]
            logits = model(x)
            probs = torch.sigmoid(logits).numpy()
            for i, supplier_id in enumerate(supplier_order):
                rows.append({"supplier_id": supplier_id, "time": int(t), "risk_probability": float(probs[i])})
    pred_df = pd.DataFrame(rows)

    merged = prepared.examples.merge(pred_df, on=["supplier_id", "time"], how="left")
    merged = merged.rename(columns={"target": "actual_disruption"})
    if merged["risk_probability"].isna().any():
        raise ValueError("some prediction examples got no model prediction -- snapshot/time mismatch")
    if not merged["risk_probability"].between(0.0, 1.0).all():
        raise ValueError("risk_probability outside [0, 1] -- sigmoid output should never happen; check model output")
    return merged[["supplier_id", "time", "risk_probability", "actual_disruption", "split"]]


def evaluate_qgnn(model: QGNN, prepared: PreparedData, threshold_cfg) -> "object":
    """Mirrors `evaluate.evaluate_experiment`, reusing its shared tail
    (`_finalize_evaluation`) directly -- everything downstream of a
    predictions DataFrame (metrics, calibration, risk ranking, onset
    breakdown, temporal variation, plotting) is model-agnostic already."""
    from .evaluate import _finalize_evaluation
    from .metrics import select_threshold

    predictions = generate_predictions_qgnn(model, prepared)
    val = predictions[predictions["split"] == "validation"]
    threshold = select_threshold(
        val["actual_disruption"].values, val["risk_probability"].values,
        policy=threshold_cfg.policy, value=threshold_cfg.value, target_value=threshold_cfg.target_value,
    )
    return _finalize_evaluation(predictions, threshold, threshold_cfg.policy, prepared)


def evaluate_qgnn_on_target(model: QGNN, prepared_target: PreparedData, threshold: float) -> "object":
    """Mirrors `evaluate.evaluate_on_target_dataset` exactly (plan section
    17 / GRAPHSAGE_FINAL_GENERALIZATION_AND_IMPROVEMENT_PLAN.md's D2):
    scores `prepared_target` (built via
    `pipeline.prepare_reduced_for_cross_dataset_eval`, every example
    already `split="test"`) with a model trained on a DIFFERENT dataset,
    using a threshold already selected from that OTHER dataset's own
    validation split. No target label is ever used to choose the
    threshold, fit the reducer/preprocessing, or select anything about how
    the target is scored."""
    from .evaluate import _finalize_evaluation

    predictions = generate_predictions_qgnn(model, prepared_target)
    return _finalize_evaluation(
        predictions, threshold, "external (selected on a different dataset's validation split)", prepared_target
    )
