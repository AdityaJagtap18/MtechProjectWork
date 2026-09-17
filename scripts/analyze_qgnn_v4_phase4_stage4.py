#!/usr/bin/env python3
"""QGNN-v4 Phase 4 Stage 4 (QGNN_V4_PHASE4_PLAN.md Track F) -- pilot
analysis for the quantum-parameter-initialization strategies. Pure
aggregation over already-saved run artifacts, reusing
`analyze_qgnn_v4_extended_metrics.py`'s own `load_run`/`build_main_row`
UNCHANGED (they only take a run directory + labels, no baked-in path
assumption) -- no new metric computation, no retraining.

Stage 4 runs live one level deeper than Stage 2/Phase 3's
(`experiments/qgnn_v4/phase4_stage4/<init>/...`, via
`run_qgnn_v4_experiment.py --output-subdir`), so this script has its own
directory resolver rather than reusing `find_run_dir` (which hardcodes the
flat `experiments/qgnn_v4/` layout) -- everything downstream of "given a
run directory" is shared code.
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
STAGE4_DIR = os.path.join(EXPERIMENTS_DIR, "phase4_stage4")
OUT_DIR = STAGE4_DIR

INIT_CONFIGS = [
    ("F1_default", "default", "PennyLane TorchLayer default: uniform[0, 2*pi] -- the unchanged control"),
    ("F2_small_gaussian", "small_gaussian", "mean=0, std=0.01"),
    ("F3_identity_like", "identity_like", "every rotation parameter starts at exactly 0.0 (identity for the rotation gates only -- entangling CNOTs are unparameterized and still fire)"),
]
PILOT_SEEDS = [42, 43]
TEST_METRICS = [
    "pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "balanced_accuracy",
    "mcc", "brier_score", "ece", "n_examples", "n_positive",
]


def find_stage4_run_dir(init: str, tag: str, seed: int, split_strategy: str) -> str:
    pattern = os.path.join(STAGE4_DIR, init, f"*_{tag}_quantum_seed{seed}")
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
        raise FileNotFoundError(f"no Stage 4 run directory found under {pattern!r} with split.strategy={split_strategy!r}")
    return kept[-1]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed_rows = []
    init_stat_rows = []

    for label, init, desc in INIT_CONFIGS:
        tag = f"stage4_pilot_{init}"
        for split, strategy in [("primary", "temporal"), ("severity", "severity")]:
            for seed in PILOT_SEEDS:
                d = find_stage4_run_dir(init, tag, seed, strategy)
                run = load_run(d)
                row = build_main_row(label, split, seed, run)
                row["run_dir"] = d
                per_seed_rows.append(row)

                init_stats_path = os.path.join(d, "quantum_init_stats.json")
                if os.path.exists(init_stats_path):
                    stats = json.load(open(init_stats_path))
                    for phase in ("initial", "final"):
                        s = stats[phase]
                        init_stat_rows.append({
                            "config": label, "split": split, "seed": seed, "phase": phase,
                            "n_params": s["n_params"], "mean": s["mean"], "std": s["std"],
                            "min": s["min"], "max": s["max"], "l2_norm": s["l2_norm"],
                        })

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(os.path.join(OUT_DIR, "stage4_per_seed_results.csv"), index=False)
    print(f"stage4_per_seed_results.csv: {len(per_seed_df)} rows")

    confusion_df = per_seed_df[["config", "split", "seed", "threshold", "test_tp", "test_fp", "test_fn", "test_tn"]].copy()
    confusion_df.to_csv(os.path.join(OUT_DIR, "stage4_confusion_matrices.csv"), index=False)
    print(f"stage4_confusion_matrices.csv: {len(confusion_df)} rows")

    long_rows = []
    for _, r in per_seed_df.iterrows():
        for metric in TEST_METRICS:
            col = f"test_{metric}"
            if col in r:
                long_rows.append({"config": r["config"], "split": r["split"], "seed": r["seed"], "metric": metric, "value": r[col]})
    metrics_long_df = pd.DataFrame(long_rows)
    metrics_long_df.to_csv(os.path.join(OUT_DIR, "stage4_metrics.csv"), index=False)
    print(f"stage4_metrics.csv: {len(metrics_long_df)} rows")

    dynamics_cols = [
        "config", "split", "seed", "best_epoch", "total_epochs_trained", "stopped_early",
        "best_train_pr_auc", "best_validation_pr_auc", "final_train_loss", "final_validation_loss",
        "quantum_grad_norm_mean", "quantum_grad_norm_min", "quantum_grad_norm_max",
        "reduce_grad_norm_mean", "val_to_test_pr_auc_gap", "train_to_test_pr_auc_gap",
    ]
    dynamics_df = per_seed_df[[c for c in dynamics_cols if c in per_seed_df.columns]].copy()
    dynamics_df.to_csv(os.path.join(OUT_DIR, "stage4_training_dynamics.csv"), index=False)
    print(f"stage4_training_dynamics.csv: {len(dynamics_df)} rows")

    init_stats_df = pd.DataFrame(init_stat_rows)
    init_stats_df.to_csv(os.path.join(OUT_DIR, "stage4_initialization_stats.csv"), index=False)
    print(f"stage4_initialization_stats.csv: {len(init_stats_df)} rows")

    summary_rows = []
    for label, init, desc in INIT_CONFIGS:
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
    summary_df.to_csv(os.path.join(OUT_DIR, "stage4_summary.csv"), index=False)
    print(f"stage4_summary.csv: {len(summary_df)} rows")

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
    results_df.to_csv(os.path.join(OUT_DIR, "stage4_results.csv"), index=False)
    print(f"stage4_results.csv: {len(results_df)} rows")

    pd.set_option("display.width", 220)
    print("\n=== Headline: test PR-AUC / ROC-AUC / MCC, mean±std (range) across pilot seeds (42,43) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
