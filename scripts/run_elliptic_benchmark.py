"""Elliptic++ transactions benchmark: classical GraphSAGE reference, plus
standalone QGNN and hybrid QGNN (frozen-GraphSAGE-embedding + quantum head).

This is a validation study, not part of the main SCM pipeline: it checks
whether this project's classical/quantum models behave sanely on a known,
independently-published, real-world graph benchmark (Elmougy & Liu, KDD
2023, "Demystifying Fraudulent Transactions and Illicit Nodes in the
Bitcoin Network for Financial Forensics" -- an extension of the original
Weber et al. 2019 Elliptic dataset with the same 203,769 transactions and
identical illicit/licit/unknown label counts, plus more features: 182
vs. the original 165).

Deliberately does NOT touch `pipeline.py`/`data.py`/`config.py`/`train.py`/
`evaluate.py`/`graph_embedding_reduction.py` -- those are SCM-schema-specific
(NodeType.SUPPLIER, per-period snapshot building, "supplier_id"/"time"
columns). This script reuses only the genuinely generic pieces:
`graphsage.HeteroGraphSAGE`, `qgnn.QGNN`/`build_qgnn_model`,
`quantum.heads.HybridQuantumHeadLayerNorm`, `metrics.py`, `losses.py`.

Graph: single node type ("tx"), single self-relation edge type
("tx","flow","tx"), edges symmetrized (reverse edges added) so every node
receives messages regardless of payment direction -- documented choice,
not hidden.

Split: chronological, matching the field-standard Weber/EvolveGCN
convention (train < step 35, test >= step 35), with a validation slice
carved from the tail of train (this project's own rule: never tune a
threshold or do early stopping against the test split).
    train: time_step in [1, 29]
    val:   time_step in [30, 34]
    test:  time_step in [35, 49]
Only labeled nodes (class 1=illicit, class 2=licit) enter loss/metrics;
unlabeled nodes (class 3, the majority) remain in the graph as
message-passing neighbors only.

Usage:
    python scripts/run_elliptic_benchmark.py --stage cache
    python scripts/run_elliptic_benchmark.py --stage classical --seeds 42
    python scripts/run_elliptic_benchmark.py --stage qgnn --seeds 42
    python scripts/run_elliptic_benchmark.py --stage hybrid --seeds 42
    python scripts/run_elliptic_benchmark.py --stage report
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "elliptic_pp"
OUT_DIR = REPO_ROOT / "data" / "generated" / "elliptic_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MAX_STEP = 29
VAL_MAX_STEP = 34
# test: 35-49

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from scm_dataset.modeling.graphsage import HeteroGraphSAGE  # noqa: E402
from scm_dataset.modeling.qgnn import QGNN, set_seed  # noqa: E402
from scm_dataset.modeling.quantum.heads import HybridQuantumHeadLayerNorm  # noqa: E402
from scm_dataset.modeling.losses import build_loss, compute_pos_weight  # noqa: E402
from scm_dataset.modeling.metrics import compute_classification_metrics, select_threshold  # noqa: E402

EDGE_TYPE = ("tx", "flow", "tx")


# --------------------------------------------------------------------------- #
# Stage 1: load + cache
# --------------------------------------------------------------------------- #

def build_and_cache() -> None:
    t0 = time.time()
    feat = pd.read_csv(DATA_DIR / "txs_features.csv")
    classes = pd.read_csv(DATA_DIR / "txs_classes.csv")
    edges = pd.read_csv(DATA_DIR / "txs_edgelist.csv")
    print(f"loaded raw csvs in {time.time()-t0:.1f}s: feat={feat.shape} classes={classes.shape} edges={edges.shape}")

    feat = feat.merge(classes, on="txId", how="left")
    assert feat["class"].isna().sum() == 0, "every transaction should have a class label (1/2/3)"

    tx_ids = feat["txId"].values
    id_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
    n_nodes = len(tx_ids)

    time_step = feat["Time step"].values.astype(np.int64)
    label3 = feat["class"].values.astype(np.int64)  # 1=illicit, 2=licit, 3=unknown
    labeled_mask = label3 != 3
    y = np.full(n_nodes, -1, dtype=np.int64)
    y[label3 == 1] = 1  # illicit
    y[label3 == 2] = 0  # licit

    feature_cols = [c for c in feat.columns if c not in ("txId", "Time step", "class")]
    feat_block = feat[feature_cols]
    n_missing = int(feat_block.isna().any(axis=1).sum())
    if n_missing:
        # 17 of Elliptic++'s added on-chain-lookup columns (degree/BTC/fee/
        # size/address stats) are missing for a small subset of transactions
        # -- the Elliptic++ paper notes not every tx could be matched to full
        # on-chain data. Imputed with 0 ("no such flow"), documented rather
        # than silently dropped or left as NaN.
        cols_with_nan = feat_block.columns[feat_block.isna().any()].tolist()
        print(f"imputing {n_missing} rows ({n_missing/n_nodes:.2%}) with NaN in "
              f"{len(cols_with_nan)} extended columns -> filling with 0.0: {cols_with_nan}")
        feat_block = feat_block.fillna(0.0)
    X_raw = feat_block.values.astype(np.float32)
    assert not np.isnan(X_raw).any() and not np.isinf(X_raw).any(), "unexpected NaN/Inf survived imputation"
    print(f"n_nodes={n_nodes} n_features={X_raw.shape[1]} labeled={labeled_mask.sum()} "
          f"illicit={(y==1).sum()} licit={(y==0).sum()}")

    src = edges["txId1"].map(id_to_idx)
    dst = edges["txId2"].map(id_to_idx)
    valid = src.notna() & dst.notna()
    dropped = (~valid).sum()
    if dropped:
        print(f"dropping {dropped} edges referencing txIds not present in features file")
    src, dst = src[valid].values.astype(np.int64), dst[valid].values.astype(np.int64)
    # symmetrize: add reverse edges so every node receives messages regardless
    # of payment direction (documented choice, see module docstring)
    edge_index = np.concatenate(
        [np.stack([src, dst]), np.stack([dst, src])], axis=1
    )
    print(f"edge_index shape (symmetrized) = {edge_index.shape}")

    train_mask = labeled_mask & (time_step <= TRAIN_MAX_STEP)
    val_mask = labeled_mask & (time_step > TRAIN_MAX_STEP) & (time_step <= VAL_MAX_STEP)
    test_mask = labeled_mask & (time_step > VAL_MAX_STEP)
    print(f"split sizes: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        pos = y[m & (y == 1)].shape[0]
        print(f"  {name}: n={m.sum()} illicit={pos} ({pos/max(m.sum(),1):.3%})")

    # standardize using TRAIN-split statistics only (leakage guard, matches
    # this project's own convention elsewhere: never fit on val/test)
    scaler = StandardScaler().fit(X_raw[train_mask])
    X = scaler.transform(X_raw).astype(np.float32)

    torch.save(
        {
            "X": torch.from_numpy(X),
            "edge_index": torch.from_numpy(edge_index),
            "y": torch.from_numpy(y),
            "time_step": torch.from_numpy(time_step),
            "train_mask": torch.from_numpy(train_mask),
            "val_mask": torch.from_numpy(val_mask),
            "test_mask": torch.from_numpy(test_mask),
            "feature_cols": feature_cols,
        },
        OUT_DIR / "graph.pt",
    )
    print(f"cached to {OUT_DIR/'graph.pt'} in {time.time()-t0:.1f}s total")


def load_cached() -> dict:
    return torch.load(OUT_DIR / "graph.pt", weights_only=False)


# --------------------------------------------------------------------------- #
# Stage 2: classical GraphSAGE-Full (reference + embedding source)
# --------------------------------------------------------------------------- #

def train_classical(seed: int, epochs: int = 60, patience: int = 10, hidden_dim: int = 128) -> dict:
    d = load_cached()
    X, edge_index, y = d["X"], d["edge_index"], d["y"]
    train_mask, val_mask, test_mask = d["train_mask"], d["val_mask"], d["test_mask"]

    set_seed(seed)
    model = HeteroGraphSAGE(
        in_dims={"tx": X.shape[1]}, edge_types=[EDGE_TYPE], hidden_dim=hidden_dim,
        num_layers=2, dropout=0.20, readout_node_type="tx",
    )
    x_dict = {"tx": X}
    edge_index_dict = {EDGE_TYPE: edge_index}

    train_y = y[train_mask].float()
    pos_weight = compute_pos_weight(train_y.numpy())
    loss_fn = build_loss("balanced", train_y.numpy())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_val_pr_auc, best_epoch, best_state, patience_ctr = -1.0, -1, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_dict, edge_index_dict)
        loss = loss_fn(logits[train_mask], train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(x_dict, edge_index_dict)
            val_probs = torch.sigmoid(logits_eval[val_mask]).numpy()
            val_metrics = compute_classification_metrics(y[val_mask].numpy(), val_probs, threshold=0.5)
        val_pr_auc = val_metrics["pr_auc"] or -1.0
        elapsed = time.time() - t0
        print(f"[classical seed={seed}] epoch {epoch:>3}  train_loss={loss.item():.4f}  "
              f"val_pr_auc={val_pr_auc:.4f}  val_f1={val_metrics['f1']:.4f}  elapsed={elapsed:.1f}s")
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc, best_epoch, patience_ctr = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[classical seed={seed}] early stop at epoch {epoch} (best={best_epoch})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_final = model(x_dict, edge_index_dict)
        probs_final = torch.sigmoid(logits_final).numpy()
        embeddings = model.encode(x_dict, edge_index_dict)["tx"].numpy()

    threshold = select_threshold(y[val_mask].numpy(), probs_final[val_mask.numpy()], policy="f1_optimal")
    results = {}
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        mask_np = m.numpy()
        results[name] = compute_classification_metrics(y[mask_np].numpy(), probs_final[mask_np], threshold)

    torch.save(torch.from_numpy(embeddings), OUT_DIR / f"embeddings_seed{seed}.pt")
    out = {
        "model": "classical_graphsage_full", "seed": seed, "best_epoch": best_epoch,
        "threshold": threshold, "pos_weight": pos_weight, "hidden_dim": hidden_dim,
        "training_seconds": time.time() - t0, "metrics": results,
    }
    (OUT_DIR / f"classical_seed{seed}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[classical seed={seed}] DONE test_pr_auc={results['test']['pr_auc']:.4f} "
          f"test_f1={results['test']['f1']:.4f} ({time.time()-t0:.1f}s total)")
    return out


# --------------------------------------------------------------------------- #
# Stage 3: standalone QGNN (raw features -> PCA -> circuit, no graph at all)
# --------------------------------------------------------------------------- #

def train_qgnn_standalone(seed: int, n_qubits: int = 6, n_layers: int = 1, mlp_hidden: int = 8,
                            epochs: int = 60, patience: int = 10, batch_size: int = 4096) -> dict:
    d = load_cached()
    X, y = d["X"].numpy(), d["y"]
    train_mask, val_mask, test_mask = d["train_mask"].numpy(), d["val_mask"].numpy(), d["test_mask"].numpy()

    # PCA fit on TRAIN labeled rows only (leakage guard) -> n_qubits dims
    pca = PCA(n_components=n_qubits, random_state=seed).fit(X[train_mask])
    X_reduced = pca.transform(X).astype(np.float32)
    print(f"[qgnn seed={seed}] PCA explained variance ratio sum={pca.explained_variance_ratio_.sum():.3f}")

    set_seed(seed)
    model = QGNN(n_qubits=n_qubits, n_layers=n_layers, mlp_hidden=mlp_hidden)
    x_t = torch.from_numpy(X_reduced)
    y_t = y.float()

    train_y_np = y[train_mask].numpy()
    loss_fn = build_loss("balanced", train_y_np)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    rng = np.random.RandomState(seed)

    best_val_pr_auc, best_epoch, best_state, patience_ctr = -1.0, -1, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_idx))
        epoch_loss_sum, epoch_n = 0.0, 0
        for b in range(0, len(perm), batch_size):
            idx = train_idx[perm[b:b + batch_size]]
            optimizer.zero_grad()
            logits = model(x_t[idx])
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(idx)
            epoch_n += len(idx)
        train_loss = epoch_loss_sum / epoch_n

        model.eval()
        with torch.no_grad():
            val_logits = model(x_t[val_idx])
            val_probs = torch.sigmoid(val_logits).numpy()
        val_metrics = compute_classification_metrics(y[val_mask].numpy(), val_probs, threshold=0.5)
        val_pr_auc = val_metrics["pr_auc"] or -1.0
        elapsed = time.time() - t0
        print(f"[qgnn seed={seed}] epoch {epoch:>3}  train_loss={train_loss:.4f}  "
              f"val_pr_auc={val_pr_auc:.4f}  val_f1={val_metrics['f1']:.4f}  elapsed={elapsed:.1f}s")
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc, best_epoch, patience_ctr = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[qgnn seed={seed}] early stop at epoch {epoch} (best={best_epoch})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs_final = torch.sigmoid(model(x_t)).numpy()

    threshold = select_threshold(y[val_mask].numpy(), probs_final[val_mask], policy="f1_optimal")
    results = {}
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        results[name] = compute_classification_metrics(y[m].numpy(), probs_final[m], threshold)

    out = {
        "model": "qgnn_standalone", "seed": seed, "best_epoch": best_epoch, "threshold": threshold,
        "n_qubits": n_qubits, "n_layers": n_layers, "mlp_hidden": mlp_hidden,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "training_seconds": time.time() - t0, "metrics": results,
        "quantum_resource_summary": model.quantum_resource_summary(),
    }
    (OUT_DIR / f"qgnn_seed{seed}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[qgnn seed={seed}] DONE test_pr_auc={results['test']['pr_auc']:.4f} "
          f"test_f1={results['test']['f1']:.4f} ({time.time()-t0:.1f}s total)")
    return out


# --------------------------------------------------------------------------- #
# Stage 4: hybrid QGNN (frozen GraphSAGE embedding -> quantum head)
# --------------------------------------------------------------------------- #

def train_hybrid_qgnn(seed: int, n_qubits: int = 6, n_layers: int = 2,
                        epochs: int = 60, patience: int = 10, batch_size: int = 4096) -> dict:
    """Requires `embeddings_seed{seed}.pt` from `train_classical(seed=...)` to
    already exist -- the SAME embedding the classical reference was scored
    from, so the comparison isolates "quantum head vs. MLP head" on an
    identical frozen representation, mirroring this project's own QGNN-v4
    RQ-Q3 design."""
    d = load_cached()
    y = d["y"]
    train_mask, val_mask, test_mask = d["train_mask"].numpy(), d["val_mask"].numpy(), d["test_mask"].numpy()

    emb_path = OUT_DIR / f"embeddings_seed{seed}.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing -- run train_classical(seed={seed}) first")
    H = torch.load(emb_path, weights_only=False)  # [n_nodes, hidden_dim], frozen

    set_seed(seed)
    model = HybridQuantumHeadLayerNorm(
        in_dim=H.shape[1], n_qubits=n_qubits, n_layers=n_layers, ansatz="strongly_entangling",
        elementwise_affine=False,  # matches this project's own "standing reference" config
    )
    y_t = y.float()

    train_y_np = y[train_mask].numpy()
    loss_fn = build_loss("balanced", train_y_np)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    rng = np.random.RandomState(seed)

    best_val_pr_auc, best_epoch, best_state, patience_ctr = -1.0, -1, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_idx))
        epoch_loss_sum, epoch_n = 0.0, 0
        for b in range(0, len(perm), batch_size):
            idx = train_idx[perm[b:b + batch_size]]
            optimizer.zero_grad()
            logits = model(H[idx])
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(idx)
            epoch_n += len(idx)
        train_loss = epoch_loss_sum / epoch_n

        model.eval()
        with torch.no_grad():
            val_logits = model(H[val_idx])
            val_probs = torch.sigmoid(val_logits).numpy()
        val_metrics = compute_classification_metrics(y[val_mask].numpy(), val_probs, threshold=0.5)
        val_pr_auc = val_metrics["pr_auc"] or -1.0
        elapsed = time.time() - t0
        print(f"[hybrid seed={seed}] epoch {epoch:>3}  train_loss={train_loss:.4f}  "
              f"val_pr_auc={val_pr_auc:.4f}  val_f1={val_metrics['f1']:.4f}  elapsed={elapsed:.1f}s")
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc, best_epoch, patience_ctr = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[hybrid seed={seed}] early stop at epoch {epoch} (best={best_epoch})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs_final = torch.sigmoid(model(H)).numpy()

    threshold = select_threshold(y[val_mask].numpy(), probs_final[val_mask], policy="f1_optimal")
    results = {}
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        results[name] = compute_classification_metrics(y[m].numpy(), probs_final[m], threshold)

    out = {
        "model": "hybrid_qgnn", "seed": seed, "best_epoch": best_epoch, "threshold": threshold,
        "n_qubits": n_qubits, "n_layers": n_layers, "training_seconds": time.time() - t0,
        "metrics": results, "quantum_resource_summary": model.quantum_resource_summary(),
    }
    (OUT_DIR / f"hybrid_seed{seed}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[hybrid seed={seed}] DONE test_pr_auc={results['test']['pr_auc']:.4f} "
          f"test_f1={results['test']['f1']:.4f} ({time.time()-t0:.1f}s total)")
    return out


# --------------------------------------------------------------------------- #
# Stage 5: report
# --------------------------------------------------------------------------- #

def build_report(seeds: list[int]) -> None:
    rows = []
    for model_name in ["classical", "qgnn", "hybrid"]:
        for seed in seeds:
            p = OUT_DIR / f"{model_name}_seed{seed}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for split in ("val", "test"):
                m = d["metrics"][split]
                rows.append({
                    "model": d["model"], "seed": seed, "split": split,
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"],
                    "precision": m["precision"], "recall": m["recall"],
                    "n_positive": m["n_positive"], "n_examples": m["n_examples"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results_per_seed.csv", index=False)

    summary = (
        df[df["split"] == "test"]
        .groupby("model")[["pr_auc", "roc_auc", "f1", "precision", "recall"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT_DIR / "results_summary.csv")
    print(summary)
    print(f"\nwrote {OUT_DIR/'results_per_seed.csv'} and {OUT_DIR/'results_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["cache", "classical", "qgnn", "hybrid", "report"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    if args.stage == "cache":
        build_and_cache()
    elif args.stage == "classical":
        for s in args.seeds:
            train_classical(seed=s, epochs=args.epochs)
    elif args.stage == "qgnn":
        for s in args.seeds:
            train_qgnn_standalone(seed=s, epochs=args.epochs)
    elif args.stage == "hybrid":
        for s in args.seeds:
            train_hybrid_qgnn(seed=s, epochs=args.epochs)
    elif args.stage == "report":
        build_report(args.seeds)
