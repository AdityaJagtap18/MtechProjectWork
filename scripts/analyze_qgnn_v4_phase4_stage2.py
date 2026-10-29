#!/usr/bin/env python3
"""QGNN-v4 Phase 4 Stage 2 (QGNN_V4_PHASE4_PLAN.md Track B) -- pilot
analysis for the bottleneck/projection variants. Pure aggregation over
already-saved run artifacts, reusing `analyze_qgnn_v4_extended_metrics.py`'s
own `find_run_dir`/`load_run`/`build_main_row` UNCHANGED (they don't
hardcode Phase 2b's CONFIGS list, only take one as a parameter) -- no new
metric computation, no retraining.

B1 (the unchanged reference) is NOT rerun: it reuses the already-completed
`phase2c_layernorm_noaffine` runs for seeds 42/43 exactly as Phase 2c/3
saved them.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_qgnn_v4_extended_metrics import build_main_row, find_run_dir, load_run  # noqa: E402
from _analysis_common import load_period_severity as _load_period_severity  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
OUT_DIR = os.path.join(EXPERIMENTS_DIR, "phase4_stage2")

# (label, tag, description) -- tag_primary == tag_severity for every Stage 2
# config since each was run with one --tag across both split configs.
PILOT_CONFIGS = [
    ("B1_reference_linear", "phase2c_layernorm_noaffine", "Linear(128,6) projection -- the unchanged Phase 2c/3 reference, not rerun"),
    ("B2_nonlinear_projection", "stage2_pilot_b2_nonlinear", "Linear(128,32)->GELU->Linear(32,6)"),
    ("B3_pre_projection_norm", "stage2_pilot_b3_prenorm", "LayerNorm(128)->Linear(128,6)"),
    ("B4_pca4", "stage2_pilot_b4_pca4", "train-fit PCA(4)->Linear(4,6)"),
    ("B4_pca6", "stage2_pilot_b4_pca6", "train-fit PCA(6)->Linear(6,6)"),
    ("B4_pca8", "stage2_pilot_b4_pca8", "train-fit PCA(8)->Linear(8,6)"),
]
PILOT_SEEDS = [42, 43]

TEST_METRICS = [
    "pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "balanced_accuracy",
    "mcc", "brier_score", "ece", "n_examples", "n_positive",
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed_rows = []
    for label, tag, _desc in PILOT_CONFIGS:
        for split, strategy in [("primary", "temporal"), ("severity", "severity")]:
            for seed in PILOT_SEEDS:
                d = find_run_dir(tag, seed, strategy)
                run = load_run(d)
                row = build_main_row(label, split, seed, run)
                row["run_dir"] = d
                per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(os.path.join(OUT_DIR, "stage2_per_seed_results.csv"), index=False)
    print(f"stage2_per_seed_results.csv: {len(per_seed_df)} rows")

    confusion_df = per_seed_df[["config", "split", "seed", "threshold", "test_tp", "test_fp", "test_fn", "test_tn"]].copy()
    confusion_df.to_csv(os.path.join(OUT_DIR, "stage2_confusion_matrices.csv"), index=False)
    print(f"stage2_confusion_matrices.csv: {len(confusion_df)} rows")

    long_rows = []
    for _, r in per_seed_df.iterrows():
        for metric in TEST_METRICS:
            col = f"test_{metric}"
            if col in r:
                long_rows.append({"config": r["config"], "split": r["split"], "seed": r["seed"], "metric": metric, "value": r[col]})
    metrics_long_df = pd.DataFrame(long_rows)
    metrics_long_df.to_csv(os.path.join(OUT_DIR, "stage2_metrics.csv"), index=False)
    print(f"stage2_metrics.csv: {len(metrics_long_df)} rows")

    summary_rows = []
    for label, tag, desc in PILOT_CONFIGS:
        for split in ("primary", "severity"):
            sub = per_seed_df[(per_seed_df.config == label) & (per_seed_df.split == split)]
            row = {"config": label, "split": split, "description": desc, "n_seeds": len(sub)}
            for metric in TEST_METRICS:
                col = f"test_{metric}"
                vals = sub[col].dropna().values.astype(float)
                if len(vals) == 0:
                    row[f"{metric}_mean"] = row[f"{metric}_std"] = row[f"{metric}_min"] = row[f"{metric}_max"] = None
                    continue
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
                row[f"{metric}_min"] = float(np.min(vals))
                row[f"{metric}_max"] = float(np.max(vals))
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "stage2_summary.csv"), index=False)
    print(f"stage2_summary.csv: {len(summary_df)} rows")

    headline_rows = []
    for _, row in summary_df.iterrows():
        headline_rows.append({
            "config": row["config"], "split": row["split"], "description": row["description"],
            "pr_auc_mean": row["pr_auc_mean"], "pr_auc_std": row["pr_auc_std"],
            "roc_auc_mean": row["roc_auc_mean"], "roc_auc_std": row["roc_auc_std"],
            "mcc_mean": row["mcc_mean"], "mcc_std": row["mcc_std"],
            "n_seeds": row["n_seeds"],
        })
    results_df = pd.DataFrame(headline_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "stage2_results.csv"), index=False)
    print(f"stage2_results.csv: {len(results_df)} rows")

    pd.set_option("display.width", 200)
    print("\n=== Headline: test PR-AUC / ROC-AUC / MCC, mean±std across pilot seeds (42,43) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
