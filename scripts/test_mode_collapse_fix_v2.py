#!/usr/bin/env python3
"""Extends test_mode_collapse_fix.py: verify the same fix (final-layer bias
initialized to logit(train positive rate) + gradient clipping) also resolves
mode collapse on (1) the enriched graph and (2) QGNN, not just GraphSAGE on
the base graph. Same rule as before: does not modify graphsage.py or
qgnn.py, only the DataCo-specific wrapper here.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from scm_dataset.modeling.graphsage import HeteroGraphSAGE
from scm_dataset.modeling.losses import build_loss
from scm_dataset.modeling.metrics import compute_classification_metrics
from scm_dataset.modeling.qgnn import build_qgnn_model, set_seed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def train_graphsage(graph, seed, fix=True, epochs=100, patience=10, lr=0.001, grad_clip=1.0):
    set_seed(seed)
    in_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroGraphSAGE(
        in_dims=in_dims, edge_types=graph.edge_types, hidden_dim=128, num_layers=2,
        dropout=0.20, aggregation="mean", readout_node_type="order",
    )
    train_mask, val_mask = graph["order"].train_mask, graph["order"].val_mask
    y = graph["order"].y
    if fix:
        train_rate = y[train_mask].mean().item()
        with torch.no_grad():
            model.classifier[-1].bias.fill_(math.log(train_rate / (1 - train_rate)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    loss_fn = build_loss("balanced", y[train_mask].numpy())
    best_val_pr_auc, best_epoch, best_state, patience_counter = -1.0, -1, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x_dict, graph.edge_index_dict)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        if fix:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits_all = model(graph.x_dict, graph.edge_index_dict)
            val_probs = torch.sigmoid(logits_all[val_mask]).numpy()
        val_metrics = compute_classification_metrics(y[val_mask].numpy(), val_probs, threshold=0.5)
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_epoch, best_state, patience_counter = val_metrics["pr_auc"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_all = model(graph.x_dict, graph.edge_index_dict)
        test_probs = torch.sigmoid(logits_all[graph["order"].test_mask]).numpy()
    m = compute_classification_metrics(y[graph["order"].test_mask].numpy(), test_probs, threshold=0.5)
    return model, {
        "seed": seed, "best_epoch": best_epoch, "f1": m["f1"], "roc_auc": m["roc_auc"],
        "recall": m["recall"], "collapsed": bool(m["recall"] > 0.85 or m["recall"] < 0.15),
    }


def train_qgnn(reduced_x, y_np, train_mask_np, val_mask_np, test_mask_np, seed, n_qubits, fix=True,
               epochs=100, patience=10, batch_size=2048, grad_clip=1.0):
    set_seed(seed)
    model = build_qgnn_model(n_qubits=n_qubits, n_layers=1, mlp_hidden=8, device_name="default.qubit")
    if fix:
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
            if fix:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(val_x)).numpy()
        val_metrics = compute_classification_metrics(val_y.numpy(), val_probs, threshold=0.5)
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc, best_epoch, best_state, patience_counter = val_metrics["pr_auc"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(test_x)).numpy()
    m = compute_classification_metrics(test_y.numpy(), test_probs, threshold=0.5)
    return {"seed": seed, "best_epoch": best_epoch, "f1": m["f1"], "roc_auc": m["roc_auc"],
            "recall": m["recall"], "collapsed": bool(m["recall"] > 0.85 or m["recall"] < 0.15)}


def main():
    print("=== enriched graph, GraphSAGE, WITH fix, 5 seeds ===")
    enriched = torch.load("data/processed/dataco_graph_enriched.pt", weights_only=False)
    for seed in [0, 1, 2, 3, 4]:
        _, r = train_graphsage(enriched, seed, fix=True)
        print(f"  seed={seed}: f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} recall={r['recall']:.4f} "
              f"best_epoch={r['best_epoch']} collapsed={r['collapsed']}")

    print("\n=== QGNN (n_qubits=6), WITH fix, 5 seeds, using seed=2 frozen enriched-GraphSAGE embeddings ===")
    gs_model, gs_r = train_graphsage(enriched, seed=2, fix=True)
    print(f"  frozen encoder: f1={gs_r['f1']:.4f} roc_auc={gs_r['roc_auc']:.4f}")
    gs_model.eval()
    with torch.no_grad():
        embeddings = gs_model.encode(enriched.x_dict, enriched.edge_index_dict)["order"].numpy()
    train_mask_np = enriched["order"].train_mask.numpy()
    val_mask_np = enriched["order"].val_mask.numpy()
    test_mask_np = enriched["order"].test_mask.numpy()
    y_np = enriched["order"].y.numpy()

    scaler1 = StandardScaler().fit(embeddings[train_mask_np])
    std_emb = scaler1.transform(embeddings)
    pca = PCA(n_components=6, random_state=0).fit(std_emb[train_mask_np])
    reduced = pca.transform(std_emb)
    scaler2 = StandardScaler().fit(reduced[train_mask_np])
    reduced_std = scaler2.transform(reduced).astype(np.float32)

    for seed in [0, 1, 2, 3, 4]:
        r = train_qgnn(reduced_std, y_np, train_mask_np, val_mask_np, test_mask_np, seed, n_qubits=6, fix=True)
        print(f"  seed={seed}: f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} recall={r['recall']:.4f} "
              f"best_epoch={r['best_epoch']} collapsed={r['collapsed']}")


if __name__ == "__main__":
    main()
