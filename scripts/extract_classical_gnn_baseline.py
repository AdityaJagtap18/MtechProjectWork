#!/usr/bin/env python3
"""Classical GraphSAGE-Full baseline metrics, extracted from the existing
experiments/classical_gnn/ run artifacts -- no retraining
(QGNN_V4_CLASSICAL_BASELINE.md). Mirrors
scripts/analyze_qgnn_v4_extended_metrics.py's column schema exactly (same
`test_*`/`validation_*`/`train_*` field names) so classical and QGNN-v4
rows can be directly concatenated for a side-by-side comparison, without
this script or its caller needing to know which arm a row came from.

Uses `_analysis_common.find_latest_graphsage_full_checkpoint` -- the
same, already-tested checkpoint resolver QGNN-v4's own scripts use -- to
locate each seed's primary (bare `*_hetero_graphsage_seed<N>`) and
severity (`*_hetero_graphsage_severity_seed<N>`) run directory. Both use
`threshold.policy: fixed, value: 0.5` (configs/graphsage.yaml,
configs/graphsage_severity.yaml) -- identical to every QGNN-v4 config, so
the saved confusion-matrix/precision/recall/f1 values are already at the
same operating point, not recomputed at a different threshold.

Explicitly NOT available from classical's saved artifacts (stated here,
not fabricated): per-epoch train PR-AUC (train.py's history only logs
train_loss per epoch, not train_pr_auc -- only the final, best-checkpoint
train PR-AUC is available, from metrics.json); any per-layer gradient
norm (train.py's training loop never logged one, unlike QGNN-v4's
diagnostics mode) -- both fields are written as None below rather than
guessed at or derived from an unrelated quantity.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from _analysis_common import (
    find_latest_graphsage_full_checkpoint,
    load_period_severity,
    max_calibration_error,
    mcc_from_confusion,
    probability_histogram,
    specificity_from_confusion,
)
from scm_dataset.modeling.data import load_benchmark
from scm_dataset.modeling.evaluate import disruption_onset_breakdown

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSICAL_DIR = os.path.join(REPO_ROOT, "experiments", "classical_gnn")
OUT_DIR = os.path.join(CLASSICAL_DIR, "baseline_extended_metrics")
SEEDS = [42, 43, 44, 45, 46]


def load_run(d: str) -> dict:
    metrics = json.load(open(os.path.join(d, "metrics.json")))
    calib = json.load(open(os.path.join(d, "calibration.json")))
    meta = json.load(open(os.path.join(d, "run_metadata.json")))
    hist = pd.read_csv(os.path.join(d, "training_history.csv"))
    preds = pd.read_csv(os.path.join(d, "predictions.csv"))
    onset_path = os.path.join(d, "onset_breakdown.json")
    onset = json.load(open(onset_path)) if os.path.exists(onset_path) else {}
    return dict(metrics=metrics, calib=calib, meta=meta, hist=hist, preds=preds, onset=onset, dir=d)


def build_row(split: str, seed: int, run: dict, onset: dict) -> dict:
    m = run["metrics"]["by_split"]
    c = run["calib"]
    hist = run["hist"]
    meta = run["meta"]
    preds = run["preds"]
    test_split = preds[preds["split"] == "test"]

    row = {"config": "classical_gnn_full", "split": split, "seed": seed}
    for sp in ("train", "validation", "test"):
        sm = m.get(sp, {})
        cm = sm.get("confusion_matrix", {})
        prefix = sp[:5]
        row.update({
            f"{prefix}_n_examples": sm.get("n_examples"), f"{prefix}_n_positive": sm.get("n_positive"), f"{prefix}_n_negative": sm.get("n_negative"),
            f"{prefix}_pr_auc": sm.get("pr_auc"), f"{prefix}_roc_auc": sm.get("roc_auc"),
            f"{prefix}_f1": sm.get("f1"), f"{prefix}_precision": sm.get("precision"), f"{prefix}_recall": sm.get("recall"),
            f"{prefix}_balanced_accuracy": sm.get("balanced_accuracy"), f"{prefix}_accuracy": sm.get("accuracy"),
            f"{prefix}_brier_score": sm.get("brier_score"),
            f"{prefix}_tp": cm.get("tp"), f"{prefix}_fp": cm.get("fp"), f"{prefix}_fn": cm.get("fn"), f"{prefix}_tn": cm.get("tn"),
            f"{prefix}_specificity": specificity_from_confusion(cm.get("tn", 0), cm.get("fp", 0)) if cm else None,
            f"{prefix}_mcc": mcc_from_confusion(cm.get("tp", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tn", 0)) if cm else None,
            f"{prefix}_fpr": (cm["fp"] / (cm["fp"] + cm["tn"])) if cm and (cm["fp"] + cm["tn"]) > 0 else None,
            f"{prefix}_fnr": (cm["fn"] / (cm["fn"] + cm["tp"])) if cm and (cm["fn"] + cm["tp"]) > 0 else None,
            f"{prefix}_ece": c.get(sp, {}).get("expected_calibration_error"),
            f"{prefix}_mce": max_calibration_error(c[sp]["curve"]) if sp in c and c[sp].get("curve") else None,
        })

    row.update({
        "threshold": m.get("test", {}).get("threshold"),
        "prob_mean": test_split["risk_probability"].mean(), "prob_std": test_split["risk_probability"].std(),
        "prob_min": test_split["risk_probability"].min(), "prob_max": test_split["risk_probability"].max(),
        "best_epoch": meta.get("best_epoch"), "total_epochs_trained": len(hist), "stopped_early": meta.get("stopped_early"),
        "final_train_loss": hist["train_loss"].iloc[-1] if "train_loss" in hist else None,
        "final_validation_loss": hist["validation_loss"].iloc[-1] if "validation_loss" in hist else None,
        "best_validation_loss": hist["validation_loss"].min() if "validation_loss" in hist else None,
        "best_train_pr_auc": None,  # NOT AVAILABLE: train.py's history logs train_loss per epoch only, not train_pr_auc
        "best_validation_pr_auc": hist["validation_pr_auc"].max() if "validation_pr_auc" in hist else None,
        "quantum_grad_norm_mean": None,  # N/A: classical, no quantum component
        "quantum_grad_norm_min": None,
        "quantum_grad_norm_max": None,
        "quantum_grad_norm_final": None,
        "reduce_grad_norm_mean": None,  # NOT AVAILABLE: no per-layer gradient norm logged for classical GraphSAGE-Full
        "train_to_val_pr_auc_gap": (m.get("train", {}).get("pr_auc") or float("nan")) - (m.get("validation", {}).get("pr_auc") or float("nan")),
        "train_to_test_pr_auc_gap": (m.get("train", {}).get("pr_auc") or float("nan")) - (m.get("test", {}).get("pr_auc") or float("nan")),
        "val_to_test_pr_auc_gap": (m.get("validation", {}).get("pr_auc") or float("nan")) - (m.get("test", {}).get("pr_auc") or float("nan")),
        "train_to_test_roc_auc_gap": (m.get("train", {}).get("roc_auc") or float("nan")) - (m.get("test", {}).get("roc_auc") or float("nan")),
        "fresh_onset_pr_auc": onset.get("test", {}).get("pr_auc_fresh_onset"),
        "fresh_onset_recall": onset.get("test", {}).get("recall_fresh_onset"),
        "n_fresh_onset": onset.get("test", {}).get("n_fresh_onset"),
    })
    row.update({f"prob_frac_{k}": v for k, v in probability_histogram(test_split["risk_probability"]).items()})
    return row


def build_severity_level_rows(seed: int, run: dict, period_severity: dict[int, int]) -> list[dict]:
    if not period_severity:
        return []
    preds = run["preds"].copy()
    preds["severity_level"] = preds["time"].map(period_severity)
    rows = []
    for level, sub in preds.groupby("severity_level"):
        y_true, y_prob = sub["actual_disruption"].values, sub["risk_probability"].values
        n_pos = int(y_true.sum())
        pr_auc = None
        if len(set(y_true)) > 1:
            from sklearn.metrics import average_precision_score
            pr_auc = float(average_precision_score(y_true, y_prob))
        threshold = run["metrics"]["by_split"]["test"]["threshold"]
        y_pred = (y_prob >= threshold).astype(int)
        recall = float((y_pred[y_true == 1] == 1).mean()) if n_pos > 0 else None
        precision = float((y_true[y_pred == 1] == 1).mean()) if y_pred.sum() > 0 else None
        rows.append({
            "config": "classical_gnn_full", "seed": seed, "severity_level": int(level), "n_examples": len(sub),
            "n_positive": n_pos, "pr_auc": pr_auc, "recall": recall, "precision": precision,
            "prob_mean": float(y_prob.mean()), "prob_std": float(y_prob.std()),
        })
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    main_rows, severity_level_rows = [], []
    period_severity = load_period_severity(REPO_ROOT, "scm_v1_black_swan_seed43", horizon_periods=104)
    benchmark = load_benchmark("data/benchmark", "scm_v1_black_swan_seed43")

    for split, split_suffix, strategy in [("primary", None, "temporal"), ("severity", "severity", "severity")]:
        for seed in SEEDS:
            d = find_latest_graphsage_full_checkpoint(CLASSICAL_DIR, seed, split_suffix=split_suffix)
            cfg = __import__("yaml").safe_load(open(os.path.join(d, "config.yaml")))
            assert cfg["split"]["strategy"] == strategy, f"seed{seed} {split}: resolved {d} has split.strategy={cfg['split']['strategy']!r}, expected {strategy!r}"
            run = load_run(d)
            # Classical's saved onset_breakdown.json predates the
            # pr_auc_fresh_onset/roc_auc_fresh_onset fields (older schema)
            # -- recomputed here via the CURRENT, shared, already-tested
            # evaluate.disruption_onset_breakdown on the same saved
            # predictions.csv, not a different or invented definition.
            onset = disruption_onset_breakdown(run["preds"], benchmark.labels["supplier"])
            main_rows.append(build_row(split, seed, run, onset))
            print(f"{split:9s} seed{seed}: {d}")
            if split == "severity":
                severity_level_rows.extend(build_severity_level_rows(seed, run, period_severity))

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(os.path.join(OUT_DIR, "classical_gnn_all_metrics.csv"), index=False)
    print(f"\nSaved {len(main_df)} rows -> {os.path.join(OUT_DIR, 'classical_gnn_all_metrics.csv')}")

    sev_df = pd.DataFrame(severity_level_rows)
    if not sev_df.empty:
        sev_df.to_csv(os.path.join(OUT_DIR, "classical_gnn_severity_level_breakdown.csv"), index=False)
        print(f"Saved {len(sev_df)} rows -> {os.path.join(OUT_DIR, 'classical_gnn_severity_level_breakdown.csv')}")


if __name__ == "__main__":
    main()
