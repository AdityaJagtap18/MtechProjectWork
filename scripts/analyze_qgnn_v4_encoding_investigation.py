#!/usr/bin/env python3
"""Quantum Encoding Investigation (QGNN_V4_QUANTUM_ENCODING_RESULTS.md) --
pure aggregation over already-saved run artifacts, reusing
`analyze_qgnn_v4_extended_metrics.py`'s `load_run`/`build_main_row`
unchanged. E0 (the baseline pi*tanh encoding) is fully REUSED from Phase
2c/3's own reference runs (verified valid before reuse) -- only E1-E4 are
newly trained.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_qgnn_v4_extended_metrics import build_main_row, load_run  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
ENC_DIR = os.path.join(EXPERIMENTS_DIR, "encoding_investigation")
OUT_DIR = ENC_DIR

TEST_METRICS = [
    "pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "balanced_accuracy",
    "mcc", "brier_score", "ece", "n_examples", "n_positive",
]
SEEDS = [42, 43]

# (config_label, encoding_type, encoding_scale, data_reuploading, base_dir, subdir_or_None, tag)
PI = 3.141592653589793
CONFIGS = [
    ("E0_baseline", "tanh", PI, False, EXPERIMENTS_DIR, None, "phase2c_layernorm_noaffine"),
    ("E1_reduced_scale", "tanh", 0.5 * PI, False, ENC_DIR, "e1_reduced_scale", "encpilot_e1"),
    ("E2_increased_scale", "tanh", 2 * PI, False, ENC_DIR, "e2_increased_scale", "encpilot_e2"),
    ("E3_linear_clip", "clip", PI, False, ENC_DIR, "e3_linear_clip", "encpilot_e3"),
    ("E4_data_reuploading", "tanh", PI, True, ENC_DIR, "e4_data_reuploading", "encpilot_e4"),
]


def find_dir(base_dir: str, subdir: str | None, tag: str, seed: int, split_strategy: str) -> str:
    search_dir = os.path.join(base_dir, subdir) if subdir else base_dir
    pattern = os.path.join(search_dir, f"*_{tag}_quantum_seed{seed}")
    matches = sorted(glob.glob(pattern))
    kept = []
    for d in matches:
        cfg_path = os.path.join(d, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        if cfg.get("split", {}).get("strategy") == split_strategy:
            kept.append(d)
    if not kept:
        raise FileNotFoundError(f"no run directory found under {pattern!r} with split.strategy={split_strategy!r}")
    return kept[-1]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed_rows, results_rows = [], []

    for label, enc_type, enc_scale, reupload, base_dir, subdir, tag in CONFIGS:
        for split, strategy in [("primary", "temporal"), ("severity", "severity")]:
            for seed in SEEDS:
                d = find_dir(base_dir, subdir, tag, seed, strategy)
                run = load_run(d)
                row = build_main_row(label, split, seed, run)
                row["run_dir"] = d
                row["reused"] = subdir is None
                per_seed_rows.append(row)

                resource_path = os.path.join(d, "quantum_resource_summary.json")
                resource = json.load(open(resource_path)) if os.path.exists(resource_path) else {}
                results_rows.append({
                    "experiment": label, "encoding_type": enc_type, "encoding_scale": enc_scale,
                    "data_reuploading": reupload, "seed": seed, "split": split,
                    "n_samples": row.get("test_n_examples"), "n_positives": row.get("test_n_positive"),
                    "pr_auc": row.get("test_pr_auc"), "roc_auc": row.get("test_roc_auc"),
                    "f1": row.get("test_f1"), "precision": row.get("test_precision"), "recall": row.get("test_recall"),
                    "specificity": row.get("test_specificity"), "balanced_accuracy": row.get("test_balanced_accuracy"),
                    "mcc": row.get("test_mcc"), "brier": row.get("test_brier_score"), "ece": row.get("test_ece"),
                    "tp": row.get("test_tp"), "tn": row.get("test_tn"), "fp": row.get("test_fp"), "fn": row.get("test_fn"),
                    "best_epoch": row.get("best_epoch"), "total_epochs": row.get("total_epochs_trained"),
                    "val_pr_auc": row.get("valid_pr_auc"), "val_test_pr_gap": row.get("val_to_test_pr_auc_gap"),
                    "fresh_onset_pr_auc": row.get("fresh_onset_pr_auc"), "n_fresh_onset": row.get("n_fresh_onset"),
                    "quantum_parameters": resource.get("trainable_quantum_parameters"),
                    "total_trainable_parameters": resource.get("total_trainable_parameters"),
                    "n_encoding_operations": resource.get("encoding_config", {}).get("n_encoding_operations"),
                    "run_dir": d,
                })

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(os.path.join(OUT_DIR, "encoding_per_seed_results.csv"), index=False)
    print(f"encoding_per_seed_results.csv: {len(per_seed_df)} rows")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(f"results.csv: {len(results_df)} rows")

    confusion_df = per_seed_df[["config", "split", "seed", "threshold", "test_tp", "test_fp", "test_fn", "test_tn"]].copy()
    confusion_df.to_csv(os.path.join(OUT_DIR, "encoding_confusion_matrices.csv"), index=False)
    print(f"encoding_confusion_matrices.csv: {len(confusion_df)} rows")

    dynamics_cols = [
        "config", "split", "seed", "best_epoch", "total_epochs_trained", "stopped_early",
        "best_train_pr_auc", "best_validation_pr_auc", "final_train_loss", "final_validation_loss",
        "quantum_grad_norm_mean", "quantum_grad_norm_min", "quantum_grad_norm_max",
        "reduce_grad_norm_mean", "val_to_test_pr_auc_gap", "train_to_test_pr_auc_gap",
    ]
    dynamics_df = per_seed_df[[c for c in dynamics_cols if c in per_seed_df.columns]].copy()
    dynamics_df.to_csv(os.path.join(OUT_DIR, "encoding_training_dynamics.csv"), index=False)
    print(f"encoding_training_dynamics.csv: {len(dynamics_df)} rows")

    summary_rows = []
    for label, enc_type, enc_scale, reupload, base_dir, subdir, tag in CONFIGS:
        for split in ("primary", "severity"):
            sub = per_seed_df[(per_seed_df.config == label) & (per_seed_df.split == split)]
            row = {"config": label, "split": split, "encoding_type": enc_type, "encoding_scale": enc_scale, "data_reuploading": reupload, "n_seeds": len(sub)}
            for metric in TEST_METRICS:
                col = f"test_{metric}"
                vals = sub[col].dropna().values.astype(float)
                if len(vals) == 0:
                    row[f"{metric}_mean"] = row[f"{metric}_std"] = row[f"{metric}_min"] = row[f"{metric}_max"] = row[f"{metric}_range"] = None
                    continue
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
                row[f"{metric}_min"] = float(np.min(vals))
                row[f"{metric}_max"] = float(np.max(vals))
                row[f"{metric}_range"] = float(np.max(vals) - np.min(vals))
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "encoding_summary.csv"), index=False)
    print(f"encoding_summary.csv: {len(summary_df)} rows")

    pd.set_option("display.width", 240)
    headline = summary_df[["config", "split", "encoding_type", "encoding_scale", "data_reuploading", "pr_auc_mean", "pr_auc_std", "pr_auc_range", "roc_auc_mean", "mcc_mean", "mcc_std"]]
    print("\n=== Headline: test PR-AUC / ROC-AUC / MCC, mean±std (range), 2 pilot seeds ===")
    print(headline.to_string(index=False))

    configurations_json = {label: {"encoding_type": et, "encoding_scale": es, "data_reuploading": dr} for label, et, es, dr, *_ in CONFIGS}
    with open(os.path.join(OUT_DIR, "configurations.json"), "w") as f:
        json.dump(configurations_json, f, indent=2)
    print(f"\nconfigurations.json written -> {os.path.join(OUT_DIR, 'configurations.json')}")


if __name__ == "__main__":
    main()
