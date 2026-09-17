#!/usr/bin/env python3
"""QGNN-v4 Phase 4 -- Stage 1 representation/graph/severity/fresh-onset
diagnostics (QGNN_V4_PHASE4_PLAN.md, Tracks A1-A4, H, I).

Pure diagnostics: extracts the SAME frozen-encoder (supplier_id, time) ->
128D embedding / target / split triples `build_v4_prepared` already gives
the real QGNN-v4 heads (Tracks A1-A3), reads the benchmark's own graph/
label/event CSVs directly (Track A4), and reuses SAVED prediction tables
from already-completed classical-GraphSAGE and QGNN-v4-reference runs
(Tracks H/I) -- no GraphSAGE retraining, no quantum circuit anywhere in
this script.

Track A1's diagnostic classifiers are lightweight, CPU-only sklearn
models (LogisticRegression/LinearSVC/MLPClassifier/RandomForestClassifier)
fit in seconds per seed -- categorically different from a multi-seed
QGNN/GraphSAGE training run.

Usage:
    python scripts/run_phase4_stage1_diagnostics.py
"""

from __future__ import annotations

import glob
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pointbiserialr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA as SkPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.evaluate import disruption_onset_breakdown
from scm_dataset.modeling.graph_embedding_reduction import (
    extract_supplier_embeddings,
    fit_embedding_pca,
    load_frozen_graphsage_encoder,
    prepare_frozen_encoder_input,
)
from scm_dataset.modeling.metrics import compute_classification_metrics
from scm_dataset.modeling.preprocessing import supplier_fit_mask
from scm_dataset.modeling.quantum import build_v4_prepared, load_quantum_v4_config

from _analysis_common import (
    find_latest_graphsage_full_checkpoint,
    load_period_severity as _load_period_severity,
    mcc_from_confusion,
    specificity_from_confusion,
)

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QGNN_V4_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
CLASSICAL_DIR = os.path.join(REPO_ROOT, "experiments", "classical_gnn")
OUT_DIR = os.path.join(QGNN_V4_DIR, "phase4_stage1")
BENCHMARK_ROOT = os.path.join(REPO_ROOT, "data", "benchmark")

SEEDS = [42, 43, 44, 45, 46]
SPLIT_CONFIGS = [
    ("primary", os.path.join(REPO_ROOT, "configs", "qgnn_v4.yaml"), "temporal"),
    ("severity", os.path.join(REPO_ROOT, "configs", "qgnn_v4_severity.yaml"), "severity"),
]
A3_REPRESENTATIVE_SEED = 42  # tractability: A3's per-dimension correlation/MI is computed once
# per split on one representative seed's embedding, not all 5 -- Stage 1's
# question ("is signal concentrated in a few dims") does not need
# encoder-level seed variance the way a PR-AUC comparison does. Flagged
# explicitly in the results doc, not silently narrowed.


# ---------------------------------------------------------------------------
# Shared loading (mirrors scripts/run_qgnn_v4_experiment.py exactly)
# ---------------------------------------------------------------------------


def load_v4_prepared(config_path: str, seed: int):
    cfg = load_quantum_v4_config(config_path)
    config = cfg.base
    split_suffix = config.split.strategy if config.split.strategy != "temporal" else None
    checkpoint_dir = find_latest_graphsage_full_checkpoint(CLASSICAL_DIR, seed, split_suffix=split_suffix)
    benchmark = load_benchmark(config.dataset.benchmark_path, config.dataset.dataset_id)
    prepared_full = prepare_frozen_encoder_input(config, benchmark, os.path.join(checkpoint_dir, "preprocessing"))
    encoder = load_frozen_graphsage_encoder(os.path.join(checkpoint_dir, "model.pt"), prepared_full)
    assert all(not p.requires_grad for p in encoder.parameters())
    all_times = sorted(prepared_full.examples["time"].unique())
    embedding_frame = extract_supplier_embeddings(encoder, prepared_full, all_times)
    v4prepared = build_v4_prepared(config, benchmark, prepared_full.examples, embedding_frame)
    return config, benchmark, v4prepared, embedding_frame, checkpoint_dir


def to_xy(v4prepared, split: str):
    sub = v4prepared.examples[v4prepared.examples["split"] == split]
    joined = sub.set_index(["supplier_id", "time"]).join(v4prepared.reduced, how="left")
    emb_cols = list(v4prepared.reduced.columns)
    if joined[emb_cols].isna().any().any():
        raise ValueError(f"split={split!r}: some examples have no embedding -- supplier/time mismatch")
    X = joined[emb_cols].values.astype(np.float64)
    y = joined["target"].values.astype(int)
    return X, y, joined.reset_index()


def full_metrics_row(y_true, y_prob) -> dict:
    m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    cm = m["confusion_matrix"]
    m["mcc"] = mcc_from_confusion(cm["tp"], cm["fp"], cm["fn"], cm["tn"])
    m["specificity"] = specificity_from_confusion(cm["tn"], cm["fp"])
    return m


# ---------------------------------------------------------------------------
# Track A1 -- embedding separability (diagnostic classifiers)
# ---------------------------------------------------------------------------


def build_a1_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": CalibratedClassifierCV(LinearSVC(max_iter=5000, class_weight="balanced"), method="sigmoid", cv=3),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=0, early_stopping=True),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1),
    }


def run_track_a1(split_name: str, seed: int, v4prepared) -> list[dict]:
    X_train, y_train, _ = to_xy(v4prepared, "train")
    rows = []
    for model_name, model in build_a1_models().items():
        if len(np.unique(y_train)) < 2:
            continue
        model.fit(X_train, y_train)
        for eval_split in ("validation", "test"):
            X_eval, y_eval, _ = to_xy(v4prepared, eval_split)
            if len(np.unique(y_eval)) < 2 or len(y_eval) == 0:
                continue
            prob = model.predict_proba(X_eval)[:, 1]
            m = full_metrics_row(y_eval, prob)
            rows.append({"split": split_name, "seed": seed, "model": model_name, "eval_on": eval_split, **m})
    return rows


# ---------------------------------------------------------------------------
# Track A2 -- PCA dimensionality diagnostic
# ---------------------------------------------------------------------------

PCA_COMPONENT_GRID = [4, 6, 8, 16, 32, 64, 128]


def run_track_a2(split_name: str, seed: int, v4prepared, embedding_frame) -> list[dict]:
    train_examples = v4prepared.examples[v4prepared.examples["split"] == "train"]
    rows = []
    for n_components in PCA_COMPONENT_GRID:
        reducer = fit_embedding_pca(embedding_frame, train_examples, n_components)
        explained = float(np.sum(reducer._pca.explained_variance_ratio_))
        reduced = reducer.transform(embedding_frame)
        reduced.columns = [f"pca_{i}" for i in range(n_components)]
        local = type(v4prepared)(config=v4prepared.config, benchmark=v4prepared.benchmark, examples=v4prepared.examples, reduced=reduced, n_components=n_components)

        X_train, y_train, _ = to_xy(local, "train")
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        if len(np.unique(y_train)) < 2:
            continue
        clf.fit(X_train, y_train)
        for eval_split in ("validation", "test"):
            X_eval, y_eval, _ = to_xy(local, eval_split)
            if len(np.unique(y_eval)) < 2 or len(y_eval) == 0:
                continue
            prob = clf.predict_proba(X_eval)[:, 1]
            m = full_metrics_row(y_eval, prob)
            rows.append({
                "split": split_name, "seed": seed, "n_components": n_components,
                "explained_variance_ratio_sum": explained, "eval_on": eval_split, **m,
            })
    return rows


# ---------------------------------------------------------------------------
# Track A3 -- target-aware per-dimension correlation / mutual information
# ---------------------------------------------------------------------------


def run_track_a3(split_name: str, seed: int, config, v4prepared) -> list[dict]:
    if seed != A3_REPRESENTATIVE_SEED:
        return []
    train_sub = v4prepared.examples[v4prepared.examples["split"] == "train"].set_index(["supplier_id", "time"])
    joined = train_sub.join(v4prepared.reduced, how="left")
    emb_cols = list(v4prepared.reduced.columns)
    joined = joined.dropna(subset=emb_cols)

    period_severity = _load_period_severity(REPO_ROOT, config.dataset.dataset_id, config.prediction.horizon_periods + 200)
    joined = joined.reset_index()
    joined["severity_at_t"] = joined["time"].map(period_severity).fillna(0).astype(int)

    labels_path = os.path.join(BENCHMARK_ROOT, config.dataset.dataset_id, "labels", "supplier_labels.csv")
    raw_labels = pd.read_csv(labels_path).set_index(["supplier_id", "time"])["supplier_disrupted"]
    already_disrupted = [int(raw_labels.get((sid, t), 0)) for sid, t in zip(joined["supplier_id"], joined["time"])]
    joined["fresh_onset"] = ((joined["target"] == 1) & (np.array(already_disrupted) == 0)).astype(int)

    X = joined[emb_cols].values.astype(np.float64)
    targets = {
        "target": joined["target"].values.astype(int),
        "severity_at_t": joined["severity_at_t"].values.astype(int),
        "fresh_onset": joined["fresh_onset"].values.astype(int),
    }

    rows = []
    for target_name, y in targets.items():
        if len(np.unique(y)) < 2:
            continue
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
        for j, col in enumerate(emb_cols):
            r, p = pointbiserialr(X[:, j], y) if target_name != "severity_at_t" else (np.corrcoef(X[:, j], y)[0, 1], None)
            rows.append({
                "split": split_name, "seed": seed, "target": target_name, "dimension": col,
                "correlation": float(r) if r is not None and np.isfinite(r) else None,
                "p_value": float(p) if p is not None else None,
                "mutual_information": float(mi[j]),
            })
    return rows


# ---------------------------------------------------------------------------
# Track A4 -- graph / label / event diagnostics (dataset-level, not per-seed)
# ---------------------------------------------------------------------------


def run_track_a4(dataset_id: str) -> dict:
    base = os.path.join(BENCHMARK_ROOT, dataset_id)
    nodes = pd.read_csv(os.path.join(base, "graph", "nodes.csv"))
    edges = pd.read_csv(os.path.join(base, "graph", "edges.csv"))
    events = pd.read_csv(os.path.join(base, "events", "events.csv"))
    labels = pd.read_csv(os.path.join(base, "labels", "supplier_labels.csv"))

    G = nx.Graph()
    G.add_nodes_from(nodes["node_id"] if "node_id" in nodes.columns else nodes.iloc[:, 0])
    node_id_col = "node_id" if "node_id" in nodes.columns else nodes.columns[0]
    src_col = "source_id" if "source_id" in edges.columns else edges.columns[0]
    dst_col = "target_id" if "target_id" in edges.columns else edges.columns[1]
    G.add_edges_from(zip(edges[src_col], edges[dst_col]))

    degrees = dict(G.degree())
    degree_series = pd.Series(degrees)
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_lcc = G.subgraph(largest_cc)

    avg_path_length = None
    try:
        if G_lcc.number_of_nodes() <= 3000:
            avg_path_length = float(nx.average_shortest_path_length(G_lcc))
    except Exception as exc:  # pragma: no cover -- diagnostic best-effort only
        avg_path_length = f"not computed: {exc}"

    node_type_col = "node_type" if "node_type" in nodes.columns else None
    node_type_counts = nodes[node_type_col].value_counts().to_dict() if node_type_col else {}
    edge_type_col = "edge_type" if "edge_type" in edges.columns else None
    edge_type_counts = edges[edge_type_col].value_counts().to_dict() if edge_type_col else {}

    supplier_ids = nodes.loc[nodes[node_type_col] == "supplier", node_id_col] if node_type_col else []
    supplier_disruption_rate = labels.groupby("supplier_id")["supplier_disrupted"].mean()
    supplier_degree = degree_series.reindex(supplier_ids)
    deg_label_corr = None
    if supplier_degree.notna().sum() > 2:
        aligned = pd.concat([supplier_degree, supplier_disruption_rate.reindex(supplier_degree.index)], axis=1).dropna()
        if len(aligned) > 2 and aligned.iloc[:, 1].nunique() > 1:
            deg_label_corr = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])

    horizon = int(labels["time"].max()) + 1
    period_severity = _load_period_severity(REPO_ROOT, dataset_id, horizon)
    severity_dist = pd.Series(list(period_severity.values())).value_counts().sort_index().to_dict()

    return {
        "dataset_id": dataset_id,
        "n_nodes_total": int(len(nodes)),
        "node_type_counts": {str(k): int(v) for k, v in node_type_counts.items()},
        "n_edges_total": int(len(edges)),
        "edge_type_counts": {str(k): int(v) for k, v in edge_type_counts.items()},
        "n_events_total": int(len(events)),
        "event_type_counts": events["event_type"].value_counts().to_dict() if "event_type" in events.columns else {},
        "event_severity_counts": events["severity"].value_counts().sort_index().to_dict() if "severity" in events.columns else {},
        "degree_distribution": {
            "mean": float(degree_series.mean()), "median": float(degree_series.median()),
            "min": int(degree_series.min()), "max": int(degree_series.max()),
        },
        "graph_density": float(nx.density(G)),
        "n_connected_components": len(components),
        "largest_component_size": len(largest_cc),
        "avg_path_length_largest_component": avg_path_length,
        "supplier_degree_vs_disruption_rate_pearson_r": deg_label_corr,
        "horizon_periods": horizon,
        "period_severity_distribution": {str(k): int(v) for k, v in severity_dist.items()},
        "overall_supplier_disruption_rate": float(labels["supplier_disrupted"].mean()),
    }


# ---------------------------------------------------------------------------
# Track H -- severity-level breakdown (reuses saved prediction tables)
# ---------------------------------------------------------------------------


def find_run_dir(base_dir: str, tag_suffix_pattern: str, split_strategy: str | None = None) -> list[str]:
    matches = sorted(glob.glob(os.path.join(base_dir, tag_suffix_pattern)))
    if split_strategy is None:
        return matches
    kept = []
    for d in matches:
        cfg_path = os.path.join(d, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        if cfg.get("split", {}).get("strategy") == split_strategy:
            kept.append(d)
    return kept


def severity_breakdown_for_predictions(preds: pd.DataFrame, dataset_id: str, horizon: int) -> pd.DataFrame:
    period_severity = _load_period_severity(REPO_ROOT, dataset_id, horizon)
    preds = preds.copy()
    preds["severity_at_t"] = preds["time"].map(period_severity).fillna(0).astype(int)
    rows = []
    for (split, sev), g in preds.groupby(["split", "severity_at_t"]):
        if g["actual_disruption"].nunique() < 2:
            pr_auc = roc_auc = None
        else:
            m = compute_classification_metrics(g["actual_disruption"].values, g["risk_probability"].values, threshold=0.5)
            pr_auc, roc_auc = m["pr_auc"], m["roc_auc"]
        rows.append({
            "split": split, "severity_at_t": sev, "n_examples": int(len(g)),
            "n_positive": int(g["actual_disruption"].sum()), "pr_auc": pr_auc, "roc_auc": roc_auc,
            "mean_predicted_prob": float(g["risk_probability"].mean()),
        })
    return pd.DataFrame(rows)


def classical_run_dirs(split_strategy: str) -> list[str]:
    """One directory per seed, resolved the SAME way the actual QGNN-v4 runs
    resolve their frozen encoder (`find_latest_graphsage_full_checkpoint`) --
    NOT a raw glob. `experiments/classical_gnn/` contains a third, oddly-named
    batch of severity runs (`*_hetero_graphsage_severity_seed44_seed<N>`, from
    an unrelated earlier session) that a naive `*_hetero_graphsage_severity_
    seed*` glob would also match, silently mixing 15 runs across 3 batches
    into what should be 5 -- `find_latest_graphsage_full_checkpoint`'s
    regex-anchored per-seed lookup avoids this by construction."""
    suffix = split_strategy if split_strategy != "temporal" else None
    return [find_latest_graphsage_full_checkpoint(CLASSICAL_DIR, seed, split_suffix=suffix) for seed in SEEDS]


def run_track_h() -> pd.DataFrame:
    frames = []
    for split_name, config_path, split_strategy in SPLIT_CONFIGS:
        cfg = load_quantum_v4_config(config_path).base
        for d in classical_run_dirs(split_strategy):
            preds = pd.read_csv(os.path.join(d, "predictions.csv"))
            horizon = int(preds["time"].max()) + 1
            b = severity_breakdown_for_predictions(preds, cfg.dataset.dataset_id, horizon)
            b["source"] = "classical_graphsage"
            b["config_arm"] = split_name
            b["run_dir"] = d
            frames.append(b)
        for d in find_run_dir(QGNN_V4_DIR, "*_phase2c_layernorm_noaffine_quantum_seed*", split_strategy):
            preds = pd.read_csv(os.path.join(d, "predictions.csv"))
            horizon = int(preds["time"].max()) + 1
            b = severity_breakdown_for_predictions(preds, cfg.dataset.dataset_id, horizon)
            b["source"] = "qgnn_v4_reference_6q2L_strongly_entangling_layernorm"
            b["config_arm"] = split_name
            b["run_dir"] = d
            frames.append(b)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Track I -- fresh-onset audit
# ---------------------------------------------------------------------------


def run_track_i() -> pd.DataFrame:
    rows = []
    for split_name, config_path, split_strategy in SPLIT_CONFIGS:
        cfg = load_quantum_v4_config(config_path).base
        labels_path = os.path.join(BENCHMARK_ROOT, cfg.dataset.dataset_id, "labels", "supplier_labels.csv")
        raw_labels = pd.read_csv(labels_path)

        for source_name, dirs in [
            ("classical_graphsage", classical_run_dirs(split_strategy)),
            ("qgnn_v4_reference", find_run_dir(QGNN_V4_DIR, "*_phase2c_layernorm_noaffine_quantum_seed*", split_strategy)),
        ]:
            for d in dirs:
                preds = pd.read_csv(os.path.join(d, "predictions.csv"))
                breakdown = disruption_onset_breakdown(preds, raw_labels)
                test = breakdown.get("test", {})
                rows.append({"eval_split": split_name, "source": source_name, "run_dir": d, **test})
    return pd.DataFrame(rows)


def run_track_i_diagnostic_model(diag_rows: list[dict]) -> pd.DataFrame:
    """Adds A1's best-performing diagnostic classifier's fresh-onset numbers
    to Track I's comparison (GraphSAGE embedding / classical baseline / QGNN,
    per the phase's own request)."""
    if not diag_rows:
        return pd.DataFrame()
    df = pd.DataFrame(diag_rows)
    test = df[df["eval_on"] == "test"]
    best_per_split = test.loc[test.groupby("split")["pr_auc"].idxmax()]
    return best_per_split[["split", "seed", "model", "pr_auc", "roc_auc", "recall", "precision", "n_positive"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    a1_rows, a2_rows, a3_rows = [], [], []
    dataset_id = None
    for split_name, config_path, split_strategy in SPLIT_CONFIGS:
        for seed in SEEDS:
            print(f"[A1/A2/A3] split={split_name} seed={seed} -- extracting embeddings...")
            config, benchmark, v4prepared, embedding_frame, checkpoint_dir = load_v4_prepared(config_path, seed)
            dataset_id = config.dataset.dataset_id
            a1_rows += run_track_a1(split_name, seed, v4prepared)
            a2_rows += run_track_a2(split_name, seed, v4prepared, embedding_frame)
            a3_rows += run_track_a3(split_name, seed, config, v4prepared)

    a1_df = pd.DataFrame(a1_rows)
    a2_df = pd.DataFrame(a2_rows)
    a3_df = pd.DataFrame(a3_rows)
    a1_df.to_csv(os.path.join(OUT_DIR, "representation_diagnostics.csv"), index=False)
    a2_df.to_csv(os.path.join(OUT_DIR, "pca_sweep.csv"), index=False)
    a3_df.to_csv(os.path.join(OUT_DIR, "dimension_correlations.csv"), index=False)
    print(f"A1: {len(a1_df)} rows, A2: {len(a2_df)} rows, A3: {len(a3_df)} rows")

    print("[A4] graph/label/event diagnostics...")
    a4 = run_track_a4(dataset_id)
    with open(os.path.join(OUT_DIR, "graph_diagnostics.json"), "w") as f:
        json.dump(a4, f, indent=2, default=str)
    print(json.dumps(a4, indent=2, default=str))

    print("[H] severity-level breakdown...")
    h_df = run_track_h()
    h_df.to_csv(os.path.join(OUT_DIR, "severity_level_breakdown.csv"), index=False)
    print(f"H: {len(h_df)} rows")

    print("[I] fresh-onset audit...")
    i_df = run_track_i()
    i_diag_df = run_track_i_diagnostic_model(a1_rows)
    i_df.to_csv(os.path.join(OUT_DIR, "fresh_onset_audit.csv"), index=False)
    i_diag_df.to_csv(os.path.join(OUT_DIR, "fresh_onset_audit_diagnostic_models.csv"), index=False)
    print(f"I: {len(i_df)} rows (+{len(i_diag_df)} diagnostic-model rows)")

    print(f"\nAll Stage 1 outputs written under {OUT_DIR}")


if __name__ == "__main__":
    main()
