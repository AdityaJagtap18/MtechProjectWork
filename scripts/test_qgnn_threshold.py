#!/usr/bin/env python3
"""Quick test: does F1-optimal threshold selection (on validation) rebalance
QGNN's precision/recall skew, compared to the fixed 0.5 threshold used
throughout final_comparison.py? Same 3 seeds (0,1,2), same frozen
enriched-GraphSAGE encoder (seed=2) and 8-qubit PCA reduction, same
mode-collapse fix -- only the threshold policy changes.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scm_dataset.modeling.graphsage import HeteroGraphSAGE
from scm_dataset.modeling.losses import build_loss
from scm_dataset.modeling.metrics import compute_classification_metrics, select_threshold
from scm_dataset.modeling.qgnn import build_qgnn_model, set_seed


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


def _batched_probs(model, x, batch_size=4096):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(torch.sigmoid(model(x[start:start + batch_size])))
    return torch.cat(chunks).numpy()


def train_qgnn(reduced_x, y_np, train_mask_np, val_mask_np, test_mask_np, seed, epochs=100, patience=10, batch_size=2048):
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
    best_val_pr_auc, best_state, patience_counter = -1.0, None, 0
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
            best_val_pr_auc, best_state, patience_counter = vm["pr_auc"], copy.deepcopy(model.state_dict()), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    val_probs_final = _batched_probs(model, val_x, batch_size)
    test_probs = _batched_probs(model, test_x, batch_size)

    fixed_m = compute_classification_metrics(test_y.numpy(), test_probs, threshold=0.5)
    opt_threshold = select_threshold(val_y.numpy(), val_probs_final, policy="f1_optimal")
    opt_m = compute_classification_metrics(test_y.numpy(), test_probs, threshold=opt_threshold)
    return fixed_m, opt_m, opt_threshold


def main():
    print("loading enriched graph + training frozen encoder (seed=2)")
    graph = torch.load("data/processed/dataco_graph_enriched.pt", weights_only=False)
    gs_model = train_graphsage(graph, seed=2)
    gs_model.eval()
    with torch.no_grad():
        embeddings = gs_model.encode(graph.x_dict, graph.edge_index_dict)["order"].numpy()

    train_mask_np = graph["order"].train_mask.numpy()
    val_mask_np = graph["order"].val_mask.numpy()
    test_mask_np = graph["order"].test_mask.numpy()
    y_np = graph["order"].y.numpy()

    scaler1 = StandardScaler().fit(embeddings[train_mask_np])
    std_emb = scaler1.transform(embeddings)
    pca = PCA(n_components=8, random_state=0).fit(std_emb[train_mask_np])
    reduced = pca.transform(std_emb)
    scaler2 = StandardScaler().fit(reduced[train_mask_np])
    reduced_std = scaler2.transform(reduced).astype(np.float32)

    print(f"{'seed':>4} {'thr':>6} {'fixed_f1':>9} {'opt_f1':>8} {'fixed_prec':>10} {'opt_prec':>9} {'fixed_rec':>9} {'opt_rec':>8} {'fixed_roc':>10} {'opt_roc':>8}")
    for seed in [0, 1, 2]:
        fixed_m, opt_m, thr = train_qgnn(reduced_std, y_np, train_mask_np, val_mask_np, test_mask_np, seed)
        print(f"{seed:>4} {thr:>6.3f} {fixed_m['f1']:>9.4f} {opt_m['f1']:>8.4f} {fixed_m['precision']:>10.4f} {opt_m['precision']:>9.4f} "
              f"{fixed_m['recall']:>9.4f} {opt_m['recall']:>8.4f} {fixed_m['roc_auc']:>10.4f} {opt_m['roc_auc']:>8.4f}")


if __name__ == "__main__":
    main()
