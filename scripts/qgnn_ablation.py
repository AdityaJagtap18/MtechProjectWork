#!/usr/bin/env python3
"""QGNN ablation suite for the DataCo real-world benchmark.

Requested scope: try different QGNN configurations (qubit count, circuit
depth), data efficiency (training-set fraction), and noise robustness
(simulated gate noise) to see whether any configuration meaningfully beats
the classical GraphSAGE/LR/RF baselines already established in
notebooks/dataco_exploration.ipynb (sections 4-8).

Loads the already-built, already-enriched DataCo graph
(data/processed/dataco_graph_enriched.pt) and a freshly-trained GraphSAGE
checkpoint (seed=2, matching the notebook's section 9 choice -- confirmed
non-collapsed by the notebook's section 8a mode-collapse check), then:

  1. qubit x layer sweep (3 seeds each) -- which QGNN capacity is best
  2. data efficiency: same best config, shrinking train-set fraction
  3. noise robustness: same best config, increasing depolarizing noise

Every run is a single seed's result; each experiment repeats several seeds
so mean/std can catch the mode-collapse failure mode the notebook's section
8 found in ~40% of GraphSAGE runs -- the same risk applies here.

Results are appended to data/processed/qgnn_ablation_results.csv one row at
a time (not held in memory until the end), so a long sweep can be
interrupted and resumed without losing completed runs -- rerunning skips
any (experiment, config, seed) combination already present in the CSV.

Usage:
    python scripts/qgnn_ablation.py
"""

from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scm_dataset.modeling.graphsage import HeteroGraphSAGE
from scm_dataset.modeling.losses import build_loss, compute_pos_weight
from scm_dataset.modeling.metrics import compute_classification_metrics
from scm_dataset.modeling.qgnn import set_seed

GRAPH_PATH = "data/processed/dataco_graph_enriched.pt"
RESULTS_PATH = "data/processed/qgnn_ablation_results.csv"
GRAPHSAGE_SEED = 2  # confirmed non-collapsed in the notebook's section 8a


# ---------------------------------------------------------------------------
# GraphSAGE (identical to notebook section 7's train_and_evaluate_graphsage,
# duplicated rather than imported since it lives in the notebook, not a
# module -- kept in sync manually, both are short and unlikely to drift)
# ---------------------------------------------------------------------------
def train_graphsage(graph, seed, hidden_dim=128, num_layers=2, dropout=0.20, epochs=100, patience=10):
    set_seed(seed)
    in_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroGraphSAGE(
        in_dims=in_dims, edge_types=graph.edge_types, hidden_dim=hidden_dim, num_layers=num_layers,
        dropout=dropout, aggregation="mean", readout_node_type="order",
    )
    train_mask, val_mask = graph["order"].train_mask, graph["order"].val_mask
    y = graph["order"].y

    # Mode-collapse fix (scripts/test_mode_collapse_fix.py): a random final-
    # layer bias can start predictions saturated toward one class, where BCE
    # gradients vanish before the model corrects course. Verified: 0/10 seeds
    # collapsed across both graphs with this fix, vs. 4/10 without it.
    train_rate = y[train_mask].numpy().mean()
    with torch.no_grad():
        model.classifier[-1].bias.fill_(math.log(train_rate / (1 - train_rate)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = build_loss("balanced", y[train_mask].numpy())

    best_val_pr_auc, best_state, patience_counter = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x_dict, graph.edge_index_dict)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(graph.x_dict, graph.edge_index_dict)
            val_probs = torch.sigmoid(logits_all[val_mask]).numpy()
        val_metrics = compute_classification_metrics(y[val_mask].numpy(), val_probs, threshold=0.5)
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_state, patience_counter = val_metrics["pr_auc"], copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def reduce_embeddings(embeddings, train_mask_np, n_qubits):
    """PCA to n_qubits dims, then standardize -- both fit on train rows only,
    frozen for val/test (same discipline as reduction.py's PCASupplierReducer).
    QuantumCircuitLayer applies tanh(x)*pi itself, so the final output here
    must already be roughly standardized, not raw PCA scores."""
    scaler1 = StandardScaler().fit(embeddings[train_mask_np])
    standardized = scaler1.transform(embeddings)
    pca = PCA(n_components=n_qubits, random_state=0).fit(standardized[train_mask_np])
    reduced = pca.transform(standardized)
    scaler2 = StandardScaler().fit(reduced[train_mask_np])
    return scaler2.transform(reduced).astype(np.float32), pca.explained_variance_ratio_.sum()


# ---------------------------------------------------------------------------
# Noisy quantum circuit -- NOT added to qgnn.py itself (that module is the
# frozen, already-validated classical-vs-QGNN comparison baseline; this is a
# new experimental extension, kept local to this script so it can't
# destabilize the existing tested code path). Depolarizing noise on
# default.mixed, applied after every gate.
# ---------------------------------------------------------------------------
def _make_noisy_qnode(n_qubits: int, n_layers: int, noise_p: float):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        for q in range(n_qubits):
            qml.RY(angles[..., q], wires=q)
            if noise_p > 0:
                qml.DepolarizingChannel(noise_p, wires=q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q], wires=q)
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=q)
                    qml.DepolarizingChannel(noise_p, wires=q + 1)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class NoisyQuantumCircuitLayer(nn.Module):
    """Same interface as qgnn.py's QuantumCircuitLayer, plus a noise_p
    parameter and default.mixed instead of default.qubit. default.mixed
    simulates a full density matrix (O(4^n) vs O(2^n) state amplitudes), so
    this is meaningfully slower per call -- expected and accepted for a
    noise-robustness study, not a bug."""

    def __init__(self, n_qubits: int, n_layers: int, noise_p: float):
        super().__init__()
        self.n_qubits, self.n_layers, self.noise_p = n_qubits, n_layers, noise_p
        import math
        self._circuit = _make_noisy_qnode(n_qubits, n_layers, noise_p)
        self.weights = nn.Parameter((torch.rand(n_layers, n_qubits) * 2 - 1) * math.pi)

    def forward(self, x):
        import math
        angles = torch.tanh(x) * math.pi
        outputs = self._circuit(angles, self.weights)
        return torch.stack(outputs, dim=-1).to(torch.float32)


class NoisyQGNN(nn.Module):
    def __init__(self, n_qubits, n_layers, mlp_hidden, noise_p):
        super().__init__()
        self.quantum = NoisyQuantumCircuitLayer(n_qubits, n_layers, noise_p)
        self.mlp = nn.Sequential(nn.Linear(n_qubits, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, x):
        return self.mlp(self.quantum(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# QGNN training (mirrors qgnn.py's train_qgnn structure: mini-batched, same
# early-stopping-on-val-PR-AUC policy as GraphSAGE for a fair comparison)
# ---------------------------------------------------------------------------
def _batched_forward_probs(model, x, batch_size):
    """Chunked inference -- required for the noisy default.mixed device,
    where a single un-batched forward over tens of thousands of rows would
    try to hold that many density-matrix copies in memory at once (O(4^n)
    each). Harmless for the fast default.qubit path too, just unnecessary
    there."""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(torch.sigmoid(model(x[start:start + batch_size])))
    return torch.cat(chunks).numpy()


def train_and_evaluate_qgnn(
    reduced_x, y_np, train_mask_np, val_mask_np, test_mask_np, seed,
    n_qubits, n_layers=1, mlp_hidden=8, noise_p=0.0, train_fraction=1.0,
    epochs=100, patience=10, batch_size=2048, eval_max_examples=None,
):
    set_seed(seed)
    from scm_dataset.modeling.qgnn import build_qgnn_model

    model = NoisyQGNN(n_qubits, n_layers, mlp_hidden, noise_p) if noise_p > 0 else build_qgnn_model(
        n_qubits=n_qubits, n_layers=n_layers, mlp_hidden=mlp_hidden, device_name="default.qubit",
    )

    # Mode-collapse fix (scripts/test_mode_collapse_fix.py / _v2.py): same
    # final-layer bias initialization as GraphSAGE, applied to the MLP head
    # both NoisyQGNN and build_qgnn_model's QGNN share the same structure for.
    # Verified: reduces (but does not eliminate) QGNN collapse -- the
    # quantum circuit's own rotation-angle init is a separate, untouched
    # source of instability this fix doesn't address.
    train_rate_for_init = float(y_np[train_mask_np].mean())
    with torch.no_grad():
        model.mlp[-1].bias.fill_(math.log(train_rate_for_init / (1 - train_rate_for_init)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    reduced_t = torch.tensor(reduced_x)
    y_t = torch.tensor(y_np.astype(np.float32))

    train_idx_full = np.where(train_mask_np)[0]
    if train_fraction < 1.0:
        rng_sub = np.random.RandomState(seed)
        n_keep = max(int(len(train_idx_full) * train_fraction), batch_size)
        train_idx = rng_sub.choice(train_idx_full, size=n_keep, replace=False)
    else:
        train_idx = train_idx_full

    val_idx_full, test_idx_full = np.where(val_mask_np)[0], np.where(test_mask_np)[0]
    if eval_max_examples is not None:
        rng_eval = np.random.RandomState(seed)
        val_idx = rng_eval.choice(val_idx_full, size=min(eval_max_examples, len(val_idx_full)), replace=False)
        test_idx = rng_eval.choice(test_idx_full, size=min(eval_max_examples, len(test_idx_full)), replace=False)
    else:
        val_idx, test_idx = val_idx_full, test_idx_full
    val_x, val_y = reduced_t[val_idx], y_t[val_idx]
    test_x, test_y = reduced_t[test_idx], y_t[test_idx]
    loss_fn = build_loss("balanced", y_np[train_idx])

    rng = np.random.RandomState(seed)
    best_val_pr_auc, best_epoch, best_state, patience_counter = -1.0, -1, None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_idx)
        for b_start in range(0, len(perm), batch_size):
            batch_idx = perm[b_start:b_start + batch_size]
            optimizer.zero_grad()
            logits = model(reduced_t[batch_idx])
            loss = loss_fn(logits, y_t[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_probs = _batched_forward_probs(model, val_x, batch_size)
        val_metrics = compute_classification_metrics(val_y.numpy(), val_probs, threshold=0.5)
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_epoch, best_state, patience_counter = val_metrics["pr_auc"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    test_probs = _batched_forward_probs(model, test_x, batch_size)
    test_metrics = compute_classification_metrics(test_y.numpy(), test_probs, threshold=0.5)
    duration = time.time() - start

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "seed": seed, "best_epoch": best_epoch, "best_val_pr_auc": best_val_pr_auc,
        "accuracy": test_metrics["accuracy"], "precision": test_metrics["precision"],
        "recall": test_metrics["recall"], "f1": test_metrics["f1"], "roc_auc": test_metrics["roc_auc"],
        "n_train_examples": len(train_idx), "n_params": n_params, "duration_seconds": duration,
        "degenerate": bool(test_metrics["recall"] > 0.85),
    }


def _already_done(results_df, **key):
    if results_df is None or results_df.empty:
        return False
    mask = pd.Series(True, index=results_df.index)
    for k, v in key.items():
        mask &= results_df[k] == v
    return mask.any()


def _append_row(row: dict):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(RESULTS_PATH)
    df_row.to_csv(RESULTS_PATH, mode="a", header=header, index=False)


def _load_existing():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None


def main():
    print(f"loading graph from {GRAPH_PATH}")
    graph = torch.load(GRAPH_PATH, weights_only=False)
    train_mask_np = graph["order"].train_mask.numpy()
    val_mask_np = graph["order"].val_mask.numpy()
    test_mask_np = graph["order"].test_mask.numpy()
    y_np = graph["order"].y.numpy()

    print(f"training frozen GraphSAGE (seed={GRAPHSAGE_SEED})")
    gs_model = train_graphsage(graph, seed=GRAPHSAGE_SEED)
    gs_model.eval()
    with torch.no_grad():
        embeddings = gs_model.encode(graph.x_dict, graph.edge_index_dict)["order"].numpy()
    print(f"embeddings: {embeddings.shape}")

    existing = _load_existing()

    # --- Experiment 1: qubit x layer sweep ---
    reduced_cache = {}
    for n_qubits in [4, 6, 8]:
        reduced, explained_var = reduce_embeddings(embeddings, train_mask_np, n_qubits)
        reduced_cache[n_qubits] = reduced
        for n_layers in [1, 2]:
            for seed in [0, 1, 2]:
                if _already_done(existing, experiment="qubit_layer_sweep", n_qubits=n_qubits, n_layers=n_layers, seed=seed):
                    print(f"skip (already done): qubit_layer_sweep n_qubits={n_qubits} n_layers={n_layers} seed={seed}")
                    continue
                print(f"running: qubit_layer_sweep n_qubits={n_qubits} n_layers={n_layers} seed={seed}")
                r = train_and_evaluate_qgnn(reduced, y_np, train_mask_np, val_mask_np, test_mask_np, seed, n_qubits, n_layers=n_layers)
                r.update(experiment="qubit_layer_sweep", n_qubits=n_qubits, n_layers=n_layers, noise_p=0.0, train_fraction=1.0, explained_variance_ratio=explained_var)
                print(f"  -> f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} degenerate={r['degenerate']} duration={r['duration_seconds']:.1f}s")
                _append_row(r)
                existing = _load_existing()

    # pick best (n_qubits, n_layers) by mean val_pr_auc among non-degenerate runs so far
    sweep_rows = existing[(existing["experiment"] == "qubit_layer_sweep") & (~existing["degenerate"])]
    if sweep_rows.empty:
        print("WARNING: every qubit/layer sweep run degenerated -- falling back to n_qubits=6, n_layers=1 for the remaining experiments")
        best_qubits, best_layers = 6, 1
    else:
        grouped = sweep_rows.groupby(["n_qubits", "n_layers"])["best_val_pr_auc"].mean()
        best_qubits, best_layers = grouped.idxmax()
    print(f"best config from sweep: n_qubits={best_qubits}, n_layers={best_layers}")
    best_reduced = reduced_cache[best_qubits]

    # --- Experiment 2: data efficiency ---
    for frac in [1.0, 0.5, 0.25, 0.1]:
        for seed in [0, 1, 2]:
            if _already_done(existing, experiment="data_efficiency", train_fraction=frac, seed=seed):
                print(f"skip (already done): data_efficiency frac={frac} seed={seed}")
                continue
            print(f"running: data_efficiency frac={frac} seed={seed}")
            r = train_and_evaluate_qgnn(best_reduced, y_np, train_mask_np, val_mask_np, test_mask_np, seed, best_qubits, n_layers=best_layers, train_fraction=frac)
            r.update(experiment="data_efficiency", n_qubits=best_qubits, n_layers=best_layers, noise_p=0.0, train_fraction=frac, explained_variance_ratio=np.nan)
            print(f"  -> f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} degenerate={r['degenerate']} n_train={r['n_train_examples']} duration={r['duration_seconds']:.1f}s")
            _append_row(r)
            existing = _load_existing()

    # --- Experiment 3: noise robustness ---
    # default.mixed (density-matrix simulation) is dramatically slower than
    # default.qubit and scales badly with both qubit count and batch size --
    # measured directly: 6 qubits at batch=256 is ~10x slower than at
    # batch=64. Fixed at 4 qubits / batch=128 / a small train subsample here
    # regardless of what the sweep picked as "best", so this experiment
    # finishes in minutes instead of hours. This trades some absolute
    # accuracy for tractability -- the goal is the noise-vs-performance
    # *trend*, not the single best possible noisy-QGNN number.
    NOISE_QUBITS, NOISE_LAYERS, NOISE_BATCH, NOISE_TRAIN_FRACTION = 4, 1, 128, 0.04
    noise_reduced = reduced_cache[NOISE_QUBITS]
    for noise_p in [0.0, 0.01, 0.05, 0.1]:
        for seed in [0, 1, 2]:
            if _already_done(existing, experiment="noise_robustness", noise_p=noise_p, seed=seed):
                print(f"skip (already done): noise_robustness noise_p={noise_p} seed={seed}")
                continue
            print(f"running: noise_robustness noise_p={noise_p} seed={seed} (n_qubits={NOISE_QUBITS}, subsampled train, default.mixed)")
            r = train_and_evaluate_qgnn(
                noise_reduced, y_np, train_mask_np, val_mask_np, test_mask_np, seed, NOISE_QUBITS,
                n_layers=NOISE_LAYERS, noise_p=noise_p, train_fraction=NOISE_TRAIN_FRACTION,
                epochs=15, patience=4, batch_size=NOISE_BATCH, eval_max_examples=2000,
            )
            r.update(experiment="noise_robustness", n_qubits=NOISE_QUBITS, n_layers=NOISE_LAYERS, noise_p=noise_p, train_fraction=NOISE_TRAIN_FRACTION, explained_variance_ratio=np.nan)
            print(f"  -> f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} degenerate={r['degenerate']} duration={r['duration_seconds']:.1f}s")
            _append_row(r)
            existing = _load_existing()

    print(f"\nall done. results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
