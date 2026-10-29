#!/usr/bin/env python3
"""QGNN-v4 Phase 4 Stage 5 (QGNN_V4_PHASE4_PLAN.md) -- pure aggregation
over already-saved run artifacts, reusing `analyze_qgnn_v4_extended_
metrics.py`'s `load_run`/`build_main_row` unchanged. A0 (StronglyEntangling)
is fully REUSED from Phase 2c/3's own reference runs (verified valid
before reuse, per this stage's own "do not recreate unnecessarily"
instruction) -- only A1 (hardware_efficient_ring) and A2
(reduced_entanglement) are newly trained this stage.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_qgnn_v4_extended_metrics import build_main_row, load_run  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
STAGE5_DIR = os.path.join(EXPERIMENTS_DIR, "phase4_stage5")
OUT_DIR = STAGE5_DIR

TEST_METRICS = [
    "pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "balanced_accuracy",
    "mcc", "brier_score", "ece", "n_examples", "n_positive",
]
SEEDS = [42, 43, 44, 45, 46]

# (config_label, description, base_dir, subdir_or_None, tag, quantum_params, head_params, entanglement)
ANSATZ_CONFIGS = [
    ("A0_strongly_entangling", "StronglyEntanglingLayers (Phase 2c/3 reference, REUSED)", EXPERIMENTS_DIR, None, "phase2c_layernorm_noaffine", 36, 817, "range-parameterized (PennyLane built-in)"),
    ("A1_hardware_ring", "Hardware-Efficient Ring: RY+RZ per qubit/layer + CNOT ring (NEW)", STAGE5_DIR, "hardware_ring", "stage5_hardware_ring", 24, 805, "CNOT ring (6 edges, wraparound)"),
    ("A2_reduced_entanglement", "Reduced Entanglement: RY only per qubit/layer + CNOT chain (NEW)", STAGE5_DIR, "reduced_entanglement", "stage5_reduced_entanglement", 12, 793, "CNOT chain (5 edges, no wraparound)"),
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
    per_seed_rows = []

    for label, desc, base_dir, subdir, tag, qparams, hparams, entangle in ANSATZ_CONFIGS:
        for split, strategy in [("primary", "temporal"), ("severity", "severity")]:
            for seed in SEEDS:
                d = find_dir(base_dir, subdir, tag, seed, strategy)
                run = load_run(d)
                row = build_main_row(label, split, seed, run)
                row["run_dir"] = d
                row["reused"] = subdir is None
                per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(os.path.join(OUT_DIR, "stage5_per_seed_results.csv"), index=False)
    print(f"stage5_per_seed_results.csv: {len(per_seed_df)} rows")

    confusion_df = per_seed_df[["config", "split", "seed", "threshold", "test_tp", "test_fp", "test_fn", "test_tn"]].copy()
    confusion_df.to_csv(os.path.join(OUT_DIR, "stage5_confusion_matrices.csv"), index=False)
    print(f"stage5_confusion_matrices.csv: {len(confusion_df)} rows")

    long_rows = []
    for _, r in per_seed_df.iterrows():
        for metric in TEST_METRICS:
            col = f"test_{metric}"
            if col in r:
                long_rows.append({"config": r["config"], "split": r["split"], "seed": r["seed"], "metric": metric, "value": r[col]})
    metrics_long_df = pd.DataFrame(long_rows)
    metrics_long_df.to_csv(os.path.join(OUT_DIR, "stage5_metrics.csv"), index=False)
    print(f"stage5_metrics.csv: {len(metrics_long_df)} rows")

    dynamics_cols = [
        "config", "split", "seed", "best_epoch", "total_epochs_trained", "stopped_early",
        "best_train_pr_auc", "best_validation_pr_auc", "final_train_loss", "final_validation_loss",
        "quantum_grad_norm_mean", "quantum_grad_norm_min", "quantum_grad_norm_max",
        "reduce_grad_norm_mean", "val_to_test_pr_auc_gap", "train_to_test_pr_auc_gap",
    ]
    dynamics_df = per_seed_df[[c for c in dynamics_cols if c in per_seed_df.columns]].copy()
    dynamics_df.to_csv(os.path.join(OUT_DIR, "stage5_training_dynamics.csv"), index=False)
    print(f"stage5_training_dynamics.csv: {len(dynamics_df)} rows")

    gap_df = per_seed_df[["config", "split", "seed", "valid_pr_auc", "test_pr_auc", "val_to_test_pr_auc_gap"]].copy()
    gap_df.to_csv(os.path.join(OUT_DIR, "stage5_validation_test_gap.csv"), index=False)
    print(f"stage5_validation_test_gap.csv: {len(gap_df)} rows")

    resources_rows = []
    for label, desc, base_dir, subdir, tag, qparams, hparams, entangle in ANSATZ_CONFIGS:
        resources_rows.append({
            "config": label, "description": desc, "qubits": 6, "layers": 2,
            "quantum_trainable_parameters": qparams, "total_trainable_head_parameters": hparams,
            "entanglement": entangle,
        })
    resources_df = pd.DataFrame(resources_rows)
    resources_df.to_csv(os.path.join(OUT_DIR, "stage5_quantum_resources.csv"), index=False)
    print(f"stage5_quantum_resources.csv: {len(resources_df)} rows")

    summary_rows = []
    for label, desc, base_dir, subdir, tag, qparams, hparams, entangle in ANSATZ_CONFIGS:
        for split in ("primary", "severity"):
            sub = per_seed_df[(per_seed_df.config == label) & (per_seed_df.split == split)]
            row = {"config": label, "split": split, "description": desc, "n_seeds": len(sub)}
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
    summary_df.to_csv(os.path.join(OUT_DIR, "stage5_summary.csv"), index=False)
    print(f"stage5_summary.csv: {len(summary_df)} rows")

    headline_rows = []
    for _, row in summary_df.iterrows():
        headline_rows.append({
            "config": row["config"], "split": row["split"], "description": row["description"],
            "pr_auc_mean": row["pr_auc_mean"], "pr_auc_std": row["pr_auc_std"], "pr_auc_range": row["pr_auc_range"],
            "roc_auc_mean": row["roc_auc_mean"], "roc_auc_std": row["roc_auc_std"],
            "mcc_mean": row["mcc_mean"], "mcc_std": row["mcc_std"],
            "n_seeds": row["n_seeds"],
        })
    results_df = pd.DataFrame(headline_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "stage5_results.csv"), index=False)
    print(f"stage5_results.csv: {len(results_df)} rows")

    pd.set_option("display.width", 240)
    print("\n=== Headline: test PR-AUC / ROC-AUC / MCC, mean±std (range), 5 seeds ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
