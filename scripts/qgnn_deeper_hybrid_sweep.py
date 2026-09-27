#!/usr/bin/env python3
"""Three genuinely untried QGNN variants, requested after the ablation +
10-seed deep dive showed no advantage at 1-2 layers via a frozen-GraphSAGE
embedding:

  1. Deeper standard QGNN: n_layers in {3,4,5} at n_qubits=6 (the ablation
     only went to 2 layers).
  2. Data re-upload QGNN (`qgnn_v2_reupload.py`, already built for the
     synthetic benchmark, unused on DataCo until now): re-encodes the input
     between variational layers, adding circuit depth/expressivity without
     more trainable quantum parameters than the reupload count implies.
  3. "Quantum-first": skips the frozen GraphSAGE embedding entirely -- PCA
     reduces the same raw leakage-audited tabular features the classical
     LR/RF baselines use directly to 6 dims, feeding the quantum circuit
     without any classical graph encoder in between. Tests whether
     GraphSAGE's embedding (whose 128 dims PCA already compressed to ~98%
     variance in just 6-8 components) is itself the bottleneck.

Same mode-collapse fix (bias init + grad clipping) applied throughout.
Results appended to data/processed/qgnn_deeper_hybrid_results.csv.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scm_dataset.modeling.graphsage import HeteroGraphSAGE
from scm_dataset.modeling.losses import build_loss
from scm_dataset.modeling.metrics import compute_classification_metrics
from scm_dataset.modeling.qgnn import build_qgnn_model, set_seed
from scm_dataset.modeling.qgnn_v2_reupload import build_qgnn_reupload_model

RESULTS_PATH = "data/processed/qgnn_deeper_hybrid_results.csv"

ORDER_CATEGORICAL = ["Type", "Market", "Shipping Mode"]
ORDER_NUMERIC = [
    "Days for shipment (scheduled)", "Order Item Discount", "Order Item Discount Rate",
    "Order Item Product Price", "Order Item Quantity", "Sales",
]
TARGET = "Late_delivery_risk"


def _batched_probs(model, x, batch_size=4096):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(torch.sigmoid(model(x[start:start + batch_size])))
    return torch.cat(chunks).numpy()


def init_bias_to_base_rate(mlp_last_linear, train_rate):
    with torch.no_grad():
        mlp_last_linear.bias.fill_(math.log(train_rate / (1 - train_rate)))


def train_qgnn_variant(model_fn, reduced_x, y_np, train_mask_np, val_mask_np, test_mask_np, seed,
                        epochs=100, patience=10, batch_size=2048):
    set_seed(seed)
    model = model_fn()
    train_rate = float(y_np[train_mask_np].mean())
    init_bias_to_base_rate(model.mlp[-1], train_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    reduced_t = torch.tensor(reduced_x)
    y_t = torch.tensor(y_np.astype(np.float32))
    train_idx = np.where(train_mask_np)[0]
    val_x, val_y = reduced_t[val_mask_np], y_t[val_mask_np]
    test_x, test_y = reduced_t[test_mask_np], y_t[test_mask_np]
    loss_fn = build_loss("balanced", y_np[train_idx])

    rng = np.random.RandomState(seed)
    best_val_pr_auc, best_epoch, best_state, patience_counter = -1.0, -1, None, 0
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
        val_probs = _batched_probs(model, val_x, batch_size)
        vm = compute_classification_metrics(val_y.numpy(), val_probs, threshold=0.5)
        if vm["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_epoch, best_state, patience_counter = vm["pr_auc"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    test_probs = _batched_probs(model, test_x, batch_size)
    m = compute_classification_metrics(test_y.numpy(), test_probs, threshold=0.5)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "seed": seed, "best_epoch": best_epoch, "f1": m["f1"], "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"], "balanced_accuracy": m["balanced_accuracy"], "brier_score": m["brier_score"],
        "precision": m["precision"], "recall": m["recall"], "n_params": n_params,
        "collapsed": bool(m["recall"] > 0.85 or m["recall"] < 0.15),
    }


def train_graphsage(graph, seed, epochs=100, patience=10):
    set_seed(seed)
    in_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroGraphSAGE(
        in_dims=in_dims, edge_types=graph.edge_types, hidden_dim=128, num_layers=2,
        dropout=0.20, aggregation="mean", readout_node_type="order",
    )
    train_mask, val_mask = graph["order"].train_mask, graph["order"].val_mask
    y = graph["order"].y
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
        vm = compute_classification_metrics(y[val_mask].numpy(), val_probs, threshold=0.5)
        if vm["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_state, patience_counter = vm["pr_auc"], copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    return model


def build_graphsage_embeddings():
    graph = torch.load("data/processed/dataco_graph_enriched.pt", weights_only=False)
    gs_model = train_graphsage(graph, seed=2)
    gs_model.eval()
    with torch.no_grad():
        embeddings = gs_model.encode(graph.x_dict, graph.edge_index_dict)["order"].numpy()
    train_mask_np = graph["order"].train_mask.numpy()
    val_mask_np = graph["order"].val_mask.numpy()
    test_mask_np = graph["order"].test_mask.numpy()
    y_np = graph["order"].y.numpy()
    return embeddings, y_np, train_mask_np, val_mask_np, test_mask_np


def reduce_to_qubits(x, train_mask_np, n_qubits):
    scaler1 = StandardScaler().fit(x[train_mask_np])
    std_x = scaler1.transform(x)
    pca = PCA(n_components=n_qubits, random_state=0).fit(std_x[train_mask_np])
    reduced = pca.transform(std_x)
    scaler2 = StandardScaler().fit(reduced[train_mask_np])
    return scaler2.transform(reduced).astype(np.float32), pca.explained_variance_ratio_.sum()


def build_raw_tabular_features():
    """The exact same feature set the classical LR/RF baselines use --
    leakage-audited, no graph, no GraphSAGE."""
    df = pd.read_csv("data/raw/dataco/DataCoSupplyChainDataset.csv", encoding="latin-1")
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
    df_sorted = df.sort_values("order date (DateOrders)").reset_index(drop=True)
    n = len(df_sorted)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "validation"
    train_mask_np = split == "train"
    val_mask_np = split == "validation"
    test_mask_np = split == "test"

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ORDER_CATEGORICAL),
        ("num", StandardScaler(), ORDER_NUMERIC),
    ])
    preprocessor.fit(df_sorted.loc[train_mask_np, ORDER_CATEGORICAL + ORDER_NUMERIC])
    X_all_raw = preprocessor.transform(df_sorted[ORDER_CATEGORICAL + ORDER_NUMERIC])
    X_all = X_all_raw.toarray() if hasattr(X_all_raw, "toarray") else X_all_raw
    y_np = df_sorted[TARGET].values.astype(np.float32)
    return X_all, y_np, train_mask_np, val_mask_np, test_mask_np


def _already_done(existing, **key):
    if existing is None or existing.empty:
        return False
    mask = pd.Series(True, index=existing.index)
    for k, v in key.items():
        mask &= existing[k] == v
    return mask.any()


def _append_row(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(RESULTS_PATH)
    df_row.to_csv(RESULTS_PATH, mode="a", header=header, index=False)


def _load_existing():
    return pd.read_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else None


def main():
    print("=== building frozen GraphSAGE embeddings (enriched graph, seed=2) ===")
    embeddings, y_np, train_mask_np, val_mask_np, test_mask_np = build_graphsage_embeddings()
    reduced_gs, var_gs = reduce_to_qubits(embeddings, train_mask_np, n_qubits=6)
    print(f"GraphSAGE-embedding PCA-6 explained variance: {var_gs:.4f}")

    existing = _load_existing()

    print("\n=== 1. Deeper standard QGNN (GraphSAGE embedding, n_qubits=6, layers 3/4/5, 3 seeds) ===")
    for n_layers in [3, 4, 5]:
        for seed in [0, 1, 2]:
            if _already_done(existing, experiment="deeper_standard", n_layers=n_layers, seed=seed):
                print(f"skip: deeper_standard n_layers={n_layers} seed={seed}")
                continue
            print(f"running: deeper_standard n_layers={n_layers} seed={seed}")
            r = train_qgnn_variant(
                lambda: build_qgnn_model(n_qubits=6, n_layers=n_layers, mlp_hidden=8, device_name="default.qubit"),
                reduced_gs, y_np, train_mask_np, val_mask_np, test_mask_np, seed,
            )
            r.update(experiment="deeper_standard", variant="standard", n_qubits=6, n_layers=n_layers, feature_source="graphsage_embedding")
            print(f"  -> f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} pr_auc={r['pr_auc']:.4f} collapsed={r['collapsed']}")
            _append_row(r)
            existing = _load_existing()

    print("\n=== 2. Data re-upload QGNN (GraphSAGE embedding, n_qubits=6, layers 3/4/5, 3 seeds) ===")
    for n_layers in [3, 4, 5]:
        for seed in [0, 1, 2]:
            if _already_done(existing, experiment="reupload", n_layers=n_layers, seed=seed):
                print(f"skip: reupload n_layers={n_layers} seed={seed}")
                continue
            print(f"running: reupload n_layers={n_layers} seed={seed}")
            r = train_qgnn_variant(
                lambda: build_qgnn_reupload_model(n_qubits=6, n_layers=n_layers, mlp_hidden=8, device_name="default.qubit"),
                reduced_gs, y_np, train_mask_np, val_mask_np, test_mask_np, seed,
            )
            r.update(experiment="reupload", variant="reupload", n_qubits=6, n_layers=n_layers, feature_source="graphsage_embedding")
            print(f"  -> f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} pr_auc={r['pr_auc']:.4f} collapsed={r['collapsed']}")
            _append_row(r)
            existing = _load_existing()

    print("\n=== 3. Quantum-first: raw tabular features -> PCA-6 -> QGNN (no GraphSAGE), standard + reupload, 3 seeds each ===")
    X_raw, y_raw, tr_raw, val_raw, test_raw = build_raw_tabular_features()
    reduced_raw, var_raw = reduce_to_qubits(X_raw, tr_raw, n_qubits=6)
    print(f"raw-tabular PCA-6 explained variance: {var_raw:.4f}")
    for variant_name, model_fn_factory, n_layers in [
        ("standard", lambda nl: (lambda: build_qgnn_model(n_qubits=6, n_layers=nl, mlp_hidden=8, device_name="default.qubit")), 1),
        ("reupload", lambda nl: (lambda: build_qgnn_reupload_model(n_qubits=6, n_layers=nl, mlp_hidden=8, device_name="default.qubit")), 3),
    ]:
        for seed in [0, 1, 2]:
            if _already_done(existing, experiment="quantum_first", variant=variant_name, seed=seed):
                print(f"skip: quantum_first variant={variant_name} seed={seed}")
                continue
            print(f"running: quantum_first variant={variant_name} n_layers={n_layers} seed={seed}")
            r = train_qgnn_variant(
                model_fn_factory(n_layers), reduced_raw, y_raw, tr_raw, val_raw, test_raw, seed,
            )
            r.update(experiment="quantum_first", variant=variant_name, n_qubits=6, n_layers=n_layers, feature_source="raw_tabular")
            print(f"  -> f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} pr_auc={r['pr_auc']:.4f} collapsed={r['collapsed']}")
            _append_row(r)
            existing = _load_existing()

    print(f"\nall done. results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
