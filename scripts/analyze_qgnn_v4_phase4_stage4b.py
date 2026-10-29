#!/usr/bin/env python3
"""QGNN-v4 Phase 4 Stage 4b (QGNN_V4_PHASE4_PLAN.md, Stage 4 follow-up) --
pure aggregation over already-saved run artifacts. Reuses
`analyze_qgnn_v4_extended_metrics.py`'s `load_run`/`build_main_row`
unchanged. Mixes TWO base directories per config -- some seeds are
reused, unmodified Stage 4 artifacts (`experiments/qgnn_v4/phase4_stage4/`),
others are newly-trained Stage 4b runs
(`experiments/qgnn_v4/phase4_stage4b/`) -- exactly per this stage's own
"reuse valid existing artifacts rather than retraining" instruction.
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
STAGE4B_DIR = os.path.join(EXPERIMENTS_DIR, "phase4_stage4b")
OUT_DIR = STAGE4B_DIR

TEST_METRICS = [
    "pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "balanced_accuracy",
    "mcc", "brier_score", "ece", "n_examples", "n_positive",
]

# (config_label, description, gaussian_std_or_None, [(base_dir, subdir, tag, seed), ...])
SCALE_SWEEP_CONFIGS = [
    ("G0_default", "default: uniform[0, 2*pi] (Stage 4 F1, REUSED)", None, [
        (STAGE4_DIR, "default", "stage4_pilot_default", 42),
        (STAGE4_DIR, "default", "stage4_pilot_default", 43),
    ]),
    ("G1_gaussian_std0.001", "gaussian std=0.001 (NEW)", 0.001, [
        (STAGE4B_DIR, "gaussian_001", "stage4b_gaussian_001", 42),
        (STAGE4B_DIR, "gaussian_001", "stage4b_gaussian_001", 43),
    ]),
    ("G2_gaussian_std0.005", "gaussian std=0.005 (NEW)", 0.005, [
        (STAGE4B_DIR, "gaussian_005", "stage4b_gaussian_005", 42),
        (STAGE4B_DIR, "gaussian_005", "stage4b_gaussian_005", 43),
    ]),
    ("G3_gaussian_std0.010", "gaussian std=0.010 (= Stage 4 F2/small_gaussian, REUSED)", 0.010, [
        (STAGE4_DIR, "small_gaussian", "stage4_pilot_small_gaussian", 42),
        (STAGE4_DIR, "small_gaussian", "stage4_pilot_small_gaussian", 43),
    ]),
    ("G4_gaussian_std0.025", "gaussian std=0.025 (NEW)", 0.025, [
        (STAGE4B_DIR, "gaussian_025", "stage4b_gaussian_025", 42),
        (STAGE4B_DIR, "gaussian_025", "stage4b_gaussian_025", 43),
    ]),
    ("G5_gaussian_std0.050", "gaussian std=0.050 (NEW)", 0.050, [
        (STAGE4B_DIR, "gaussian_050", "stage4b_gaussian_050", 42),
        (STAGE4B_DIR, "gaussian_050", "stage4b_gaussian_050", 43),
    ]),
]

F3_5SEED_CONFIG = ("F3_identity_like_5seed", "identity-like, full 5-seed set (42/43 REUSED, 44/45/46 NEW)", None, [
    (STAGE4_DIR, "identity_like", "stage4_pilot_identity_like", 42),
    (STAGE4_DIR, "identity_like", "stage4_pilot_identity_like", 43),
    (STAGE4B_DIR, "identity_like", "stage4b_f3", 44),
    (STAGE4B_DIR, "identity_like", "stage4b_f3", 45),
    (STAGE4B_DIR, "identity_like", "stage4b_f3", 46),
])

ALL_CONFIGS = SCALE_SWEEP_CONFIGS + [F3_5SEED_CONFIG]


def find_dir(base_dir: str, subdir: str, tag: str, seed: int, split_strategy: str) -> str:
    pattern = os.path.join(base_dir, subdir, f"*_{tag}_quantum_seed{seed}")
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


def score_distribution_row(config: str, split: str, seed: int, preds: pd.DataFrame) -> dict:
    test = preds[preds["split"] == "test"]
    pos = test[test["actual_disruption"] == 1]["risk_probability"]
    neg = test[test["actual_disruption"] == 0]["risk_probability"]
    return {
        "config": config, "split": split, "seed": seed,
        "n_test": len(test),
        "prob_mean": float(test["risk_probability"].mean()), "prob_std": float(test["risk_probability"].std()),
        "prob_min": float(test["risk_probability"].min()), "prob_max": float(test["risk_probability"].max()),
        "positive_mean": float(pos.mean()) if len(pos) else None, "positive_std": float(pos.std()) if len(pos) else None,
        "negative_mean": float(neg.mean()) if len(neg) else None, "negative_std": float(neg.std()) if len(neg) else None,
        "pos_neg_separation": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else None,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed_rows, init_stat_rows, score_dist_rows = [], [], []

    for label, desc, std, locations in ALL_CONFIGS:
        for split, strategy in [("primary", "temporal"), ("severity", "severity")]:
            for base_dir, subdir, tag, seed in locations:
                d = find_dir(base_dir, subdir, tag, seed, strategy)
                run = load_run(d)
                row = build_main_row(label, split, seed, run)
                row["run_dir"] = d
                row["reused_from_stage4"] = base_dir == STAGE4_DIR
                per_seed_rows.append(row)

                score_dist_rows.append(score_distribution_row(label, split, seed, run["preds"]))

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
    per_seed_df.to_csv(os.path.join(OUT_DIR, "stage4b_per_seed_results.csv"), index=False)
    print(f"stage4b_per_seed_results.csv: {len(per_seed_df)} rows")

    confusion_df = per_seed_df[["config", "split", "seed", "threshold", "test_tp", "test_fp", "test_fn", "test_tn"]].copy()
    confusion_df.to_csv(os.path.join(OUT_DIR, "stage4b_confusion_matrices.csv"), index=False)
    print(f"stage4b_confusion_matrices.csv: {len(confusion_df)} rows")

    long_rows = []
    for _, r in per_seed_df.iterrows():
        for metric in TEST_METRICS:
            col = f"test_{metric}"
            if col in r:
                long_rows.append({"config": r["config"], "split": r["split"], "seed": r["seed"], "metric": metric, "value": r[col]})
    metrics_long_df = pd.DataFrame(long_rows)
    metrics_long_df.to_csv(os.path.join(OUT_DIR, "stage4b_metrics.csv"), index=False)
    print(f"stage4b_metrics.csv: {len(metrics_long_df)} rows")

    dynamics_cols = [
        "config", "split", "seed", "best_epoch", "total_epochs_trained", "stopped_early",
        "best_train_pr_auc", "best_validation_pr_auc", "final_train_loss", "final_validation_loss",
        "quantum_grad_norm_mean", "quantum_grad_norm_min", "quantum_grad_norm_max",
        "reduce_grad_norm_mean", "val_to_test_pr_auc_gap", "train_to_test_pr_auc_gap",
    ]
    dynamics_df = per_seed_df[[c for c in dynamics_cols if c in per_seed_df.columns]].copy()
    dynamics_df.to_csv(os.path.join(OUT_DIR, "stage4b_training_dynamics.csv"), index=False)
    print(f"stage4b_training_dynamics.csv: {len(dynamics_df)} rows")

    init_stats_df = pd.DataFrame(init_stat_rows)
    init_stats_df.to_csv(os.path.join(OUT_DIR, "stage4b_initialization_stats.csv"), index=False)
    print(f"stage4b_initialization_stats.csv: {len(init_stats_df)} rows")

    score_dist_df = pd.DataFrame(score_dist_rows)
    score_dist_df.to_csv(os.path.join(OUT_DIR, "stage4b_score_distributions.csv"), index=False)
    print(f"stage4b_score_distributions.csv: {len(score_dist_df)} rows")

    summary_rows = []
    for label, desc, std, locations in ALL_CONFIGS:
        for split in ("primary", "severity"):
            sub = per_seed_df[(per_seed_df.config == label) & (per_seed_df.split == split)]
            row = {"config": label, "split": split, "description": desc, "gaussian_std": std, "n_seeds": len(sub)}
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
    summary_df.to_csv(os.path.join(OUT_DIR, "stage4b_summary.csv"), index=False)
    print(f"stage4b_summary.csv: {len(summary_df)} rows")

    headline_rows = []
    for _, row in summary_df.iterrows():
        headline_rows.append({
            "config": row["config"], "split": row["split"], "description": row["description"], "gaussian_std": row["gaussian_std"],
            "pr_auc_mean": row["pr_auc_mean"], "pr_auc_std": row["pr_auc_std"], "pr_auc_range": row["pr_auc_range"],
            "roc_auc_mean": row["roc_auc_mean"], "roc_auc_std": row["roc_auc_std"],
            "mcc_mean": row["mcc_mean"], "mcc_std": row["mcc_std"],
            "n_seeds": row["n_seeds"],
        })
    results_df = pd.DataFrame(headline_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "stage4b_results.csv"), index=False)
    print(f"stage4b_results.csv: {len(results_df)} rows")

    pd.set_option("display.width", 240)
    print("\n=== Headline: test PR-AUC / ROC-AUC / MCC, mean±std (range) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
