#!/usr/bin/env python3
"""Diagnose and test a fix for the mode-collapse failure mode found in
notebooks/dataco_exploration.ipynb section 8 (GraphSAGE) and
scripts/qgnn_ablation.py (QGNN): ~2-3 epochs in, some seeds converge to a
degenerate near-constant predictor (recall near 1.0 or near 0.0) and never
recover, because early stopping patience exhausts against that plateau.

Hypothesis: with random default init, the classifier's final linear layer
starts with an arbitrary output bias. If that happens to push the initial
sigmoid output strongly toward 0 or 1 for most examples, BCE gradients
saturate (sigmoid derivative -> 0 in that regime) before the model ever sees
a corrective signal -- a known failure mode in imbalanced binary
classification (the same reasoning RetinaNet's focal-loss paper uses to
justify biasing the final layer's init).

Fix tested here (does NOT modify graphsage.py or qgnn.py -- both are frozen,
shared with the synthetic benchmark; this only touches the DataCo-specific
training wrapper):
  1. Initialize the final classifier layer's bias to logit(train positive
     rate), so the model starts by predicting close to the base rate for
     every example, not an arbitrary extreme.
  2. Gradient clipping (max_norm=1.0) to prevent an early large update from
     immediately overshooting into the saturated regime anyway.

Reproduces the known base-graph result first (seeds 1, 4 should collapse,
matching the notebook's section 8 finding) as a sanity check that this
script's harness is faithful, then re-runs the same 5 seeds with the fix.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from scm_dataset.modeling.graphsage import HeteroGraphSAGE
from scm_dataset.modeling.losses import build_loss
from scm_dataset.modeling.metrics import compute_classification_metrics
from scm_dataset.modeling.qgnn import set_seed


def train_graphsage(graph, seed, fix=False, epochs=100, patience=10, lr=0.001, grad_clip=1.0):
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
        bias_init = math.log(train_rate / (1 - train_rate))
        with torch.no_grad():
            model.classifier[-1].bias.fill_(bias_init)
        print(f"    [fix] classifier final bias initialized to logit({train_rate:.4f}) = {bias_init:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    loss_fn = build_loss("balanced", y[train_mask].numpy())

    best_val_pr_auc, best_epoch, best_state, patience_counter = -1.0, -1, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x_dict, graph.edge_index_dict)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        if fix and grad_clip is not None:
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
    test_metrics = compute_classification_metrics(y[graph["order"].test_mask].numpy(), test_probs, threshold=0.5)
    return {
        "seed": seed, "best_epoch": best_epoch, "best_val_pr_auc": best_val_pr_auc,
        "f1": test_metrics["f1"], "roc_auc": test_metrics["roc_auc"],
        "precision": test_metrics["precision"], "recall": test_metrics["recall"],
        "collapsed": bool(test_metrics["recall"] > 0.85 or test_metrics["recall"] < 0.15),
    }


def main():
    print("loading base graph")
    graph = torch.load("data/processed/dataco_graph_base.pt", weights_only=False)

    print("\n=== WITHOUT fix (should reproduce: seeds 1,4 collapse) ===")
    for seed in [0, 1, 2, 3, 4]:
        r = train_graphsage(graph, seed, fix=False)
        print(f"  seed={seed}: f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} recall={r['recall']:.4f} "
              f"best_epoch={r['best_epoch']} collapsed={r['collapsed']}")

    print("\n=== WITH fix (bias init + grad clip) ===")
    for seed in [0, 1, 2, 3, 4]:
        r = train_graphsage(graph, seed, fix=True)
        print(f"  seed={seed}: f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} recall={r['recall']:.4f} "
              f"best_epoch={r['best_epoch']} collapsed={r['collapsed']}")


if __name__ == "__main__":
    main()
