#!/usr/bin/env python3
"""Consolidated final comparison with the full metrics set
(`compute_classification_metrics` already computes PR-AUC, balanced
accuracy, and Brier score internally -- this just surfaces them, rather than
only the F1/ROC-AUC subset used throughout the earlier notebook sections).

Also gives the 8-qubit/1-layer QGNN config (the one standout from the
qubit/layer sweep, F1=0.693 mean but only 3 seeds) a properly-powered
10-seed run, all with the mode-collapse fix
(scripts/test_mode_collapse_fix.py) applied throughout.

Usage:
    python scripts/final_comparison.py
Output:
    data/processed/final_comparison_full_metrics.csv
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scm_dataset.modeling.graphsage import HeteroGraphSAGE
from scm_dataset.modeling.losses import build_loss
from scm_dataset.modeling.metrics import compute_classification_metrics
from scm_dataset.modeling.qgnn import build_qgnn_model, set_seed

RESULTS_PATH = "data/processed/final_comparison_full_metrics.csv"

TARGET = "Late_delivery_risk"
CATEGORICAL_FEATURES = [
    "Type", "Category Id", "Customer Segment", "Customer Country", "Customer State",
    "Department Id", "Market", "Order Region", "Order Country", "Order State", "Shipping Mode",
]
NUMERIC_FEATURES = [
    "Days for shipment (scheduled)", "Order Item Discount", "Order Item Discount Rate",
    "Order Item Product Price", "Order Item Quantity", "Sales", "Product Price",
]


def flatten_metrics(name: str, m: dict, extra: dict) -> dict:
    row = {
        "model": name, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        "balanced_accuracy": m["balanced_accuracy"], "brier_score": m["brier_score"],
        "tp": m["confusion_matrix"]["tp"], "fp": m["confusion_matrix"]["fp"],
        "fn": m["confusion_matrix"]["fn"], "tn": m["confusion_matrix"]["tn"],
    }
    row.update(extra)
    return row


def run_classical():
    print("loading DataCo CSV for LR/RF")
    df = pd.read_csv("data/raw/dataco/DataCoSupplyChainDataset.csv", encoding="latin-1")
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
    df_sorted = df.sort_values("order date (DateOrders)").reset_index(drop=True)
    n = len(df_sorted)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "validation"
    df_sorted["split"] = split

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    train = df_sorted[df_sorted["split"] == "train"]
    test = df_sorted[df_sorted["split"] == "test"]
    X_train = preprocessor.fit_transform(train[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    X_test = preprocessor.transform(test[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    y_train, y_test = train[TARGET].values, test[TARGET].values

    rows = []
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(X_train, y_train)
    m = compute_classification_metrics(y_test, lr.predict_proba(X_test)[:, 1], threshold=0.5)
    rows.append(flatten_metrics("logistic_regression", m, {"seed": 42, "n_seeds": 1}))

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1).fit(X_train, y_train)
    m = compute_classification_metrics(y_test, rf.predict_proba(X_test)[:, 1], threshold=0.5)
    rows.append(flatten_metrics("random_forest", m, {"seed": 42, "n_seeds": 1}))
    return rows


def train_graphsage(graph, seed, epochs=100, patience=10):
    set_seed(seed)
    in_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroGraphSAGE(
        in_dims=in_dims, edge_types=graph.edge_types, hidden_dim=128, num_layers=2,
        dropout=0.20, aggregation="mean", readout_node_type="order",
    )
    train_mask, val_mask, test_mask = graph["order"].train_mask, graph["order"].val_mask, graph["order"].test_mask
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
    model.eval()
    with torch.no_grad():
        logits_all = model(graph.x_dict, graph.edge_index_dict)
        test_probs = torch.sigmoid(logits_all[test_mask]).numpy()
    m = compute_classification_metrics(y[test_mask].numpy(), test_probs, threshold=0.5)
    return model, m


def run_graphsage():
    rows = []
    for name, path in [("base", "data/processed/dataco_graph_base.pt"), ("enriched", "data/processed/dataco_graph_enriched.pt")]:
        print(f"GraphSAGE {name}: 5 seeds")
        graph = torch.load(path, weights_only=False)
        for seed in range(5):
            _, m = train_graphsage(graph, seed)
            rows.append(flatten_metrics(f"graphsage_{name}", m, {"seed": seed, "n_seeds": 1}))
            print(f"  seed={seed}: f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} balanced_acc={m['balanced_accuracy']:.4f} brier={m['brier_score']:.4f}")
    return rows, graph  # returns the last-loaded (enriched) graph for reuse


def _batched_probs(model, x, batch_size=4096):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(torch.sigmoid(model(x[start:start + batch_size])))
    return torch.cat(chunks).numpy()


def train_qgnn_8q1l(reduced_x, y_np, train_mask_np, val_mask_np, test_mask_np, seed, epochs=100, patience=10, batch_size=2048):
    set_seed(seed)
    model = build_qgnn_model(n_qubits=8, n_layers=1, mlp_hidden=8, device_name="default.qubit")
    train_rate = float(y_np[train_mask_np].mean())
    with torch.no_grad():
        model.mlp[-1].bias.fill_(math.log(train_rate / (1 - train_rate)))
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
    return m, best_epoch


def run_qgnn_8q1l(enriched_graph, n_seeds=10):
    print(f"training frozen GraphSAGE encoder (seed=2, enriched graph) for QGNN embeddings")
    gs_model, gs_m = train_graphsage(enriched_graph, seed=2)
    print(f"  encoder check: f1={gs_m['f1']:.4f} roc_auc={gs_m['roc_auc']:.4f}")
    gs_model.eval()
    with torch.no_grad():
        embeddings = gs_model.encode(enriched_graph.x_dict, enriched_graph.edge_index_dict)["order"].numpy()

    train_mask_np = enriched_graph["order"].train_mask.numpy()
    val_mask_np = enriched_graph["order"].val_mask.numpy()
    test_mask_np = enriched_graph["order"].test_mask.numpy()
    y_np = enriched_graph["order"].y.numpy()

    scaler1 = StandardScaler().fit(embeddings[train_mask_np])
    std_emb = scaler1.transform(embeddings)
    pca = PCA(n_components=8, random_state=0).fit(std_emb[train_mask_np])
    reduced = pca.transform(std_emb)
    scaler2 = StandardScaler().fit(reduced[train_mask_np])
    reduced_std = scaler2.transform(reduced).astype(np.float32)
    print(f"  8-qubit PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    rows = []
    print(f"QGNN 8-qubit/1-layer: {n_seeds} seeds")
    for seed in range(n_seeds):
        m, best_epoch = train_qgnn_8q1l(reduced_std, y_np, train_mask_np, val_mask_np, test_mask_np, seed)
        collapsed = m["recall"] > 0.85 or m["recall"] < 0.15
        rows.append(flatten_metrics("qgnn_8qubit_1layer", m, {"seed": seed, "n_seeds": 1, "best_epoch": best_epoch, "collapsed": collapsed}))
        print(f"  seed={seed}: f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} "
              f"balanced_acc={m['balanced_accuracy']:.4f} brier={m['brier_score']:.4f} collapsed={collapsed}")
    return rows


def main():
    all_rows = []
    all_rows.extend(run_classical())
    gs_rows, enriched_graph = run_graphsage()
    all_rows.extend(gs_rows)
    all_rows.extend(run_qgnn_8q1l(enriched_graph, n_seeds=10))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nsaved to {RESULTS_PATH}")

    print("\n=== summary (mean, collapse-excluded for qgnn) ===")
    is_collapsed = df["collapsed"].fillna(False).astype(bool) if "collapsed" in df.columns else pd.Series(False, index=df.index)
    df_clean = df[~is_collapsed]
    summary = df_clean.groupby("model")[["f1", "roc_auc", "pr_auc", "balanced_accuracy", "brier_score"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
