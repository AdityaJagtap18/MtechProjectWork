#!/usr/bin/env python3
"""Extended metrics analysis for the QGNN-v4 Phase 2b configurations
(QGNN_V4_PHASE2B_EXTENDED_METRICS.md). Pure aggregation over already-saved
run artifacts -- reuses metrics.json/calibration.json/training_history.csv/
run_metadata.json/predictions.csv/onset_breakdown.json exactly as produced
by scripts/run_qgnn_v4_experiment.py, never retrains anything, matching
this project's established "aggregate, don't recompute" convention
(scripts/build_ablation_matrix.py, scripts/build_qgnn_comparison.py).

Adds only what wasn't already saved: MCC and specificity (both simple,
standard derivations from the confusion matrix every run already has),
Maximum Calibration Error (from the calibration curve every run already
has), a probability histogram, generalization gaps, cross-seed stability
statistics (mean/std/median/min/max/IQR), and per-severity-level
performance -- the last derived from the benchmark's raw events.csv
(start_time/duration/severity/recovery_delay/recovery_periods), using the
EXACT SAME max-active-severity-per-period logic
`benchmark.splits.severity_split`/`_event_windows` themselves use. This
value is not stored anywhere in the modeling pipeline's own outputs
(severity_split.csv keeps only the resulting binary train/test label, not
the period's severity value), so it's recomputed here from source, not
fabricated and not a different definition than the one that built the
split in the first place.

Learns from a bug caught during Phase 2b's own analysis: primary/severity
runs sharing an identical --tag meant a naive "most recent matching
directory" glob could return the wrong split's run. This script always
filters candidate run directories by their own saved config.yaml
split.strategy field, never by recency.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments", "qgnn_v4")
OUT_DIR = os.path.join(EXPERIMENTS_DIR, "phase2b_extended_metrics")

CONFIGS = [
    ("baseline", "diag_primary", "diag_severity"),
    ("scale", "phase2b_scale", "phase2b_scale"),
    ("scale_bias", "phase2b_scale_bias", "phase2b_scale_bias"),
    ("fixed0.5", "phase2b_fixed0.5", "phase2b_fixed0.5"),
    ("fixed2.0", "phase2b_fixed2.0", "phase2b_fixed2.0"),
]
SEEDS = [42, 43, 44, 45, 46]
STABILITY_METRICS = ["pr_auc", "roc_auc", "f1", "recall", "precision", "brier_score", "ece"]


def find_run_dir(tag: str, seed: int, split_strategy: str) -> str:
    matches = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"*_{tag}_quantum_seed{seed}")))
    for d in matches:
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        if cfg["split"]["strategy"] == split_strategy:
            return d
    raise FileNotFoundError(f"no run directory for tag={tag!r} seed={seed} split.strategy={split_strategy!r}")


def mcc_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float | None:
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(num / den) if den > 0 else None


def specificity_from_confusion(tn: int, fp: int) -> float | None:
    return float(tn / (tn + fp)) if (tn + fp) > 0 else None


def max_calibration_error(curve: dict) -> float | None:
    errs = [
        abs(acc - conf)
        for conf, acc, n in zip(curve["mean_predicted_probability"], curve["observed_frequency"], curve["bin_counts"])
        if n > 0 and conf is not None and acc is not None
    ]
    return float(max(errs)) if errs else None


def probability_histogram(probs: pd.Series) -> dict:
    bins = [(-0.001, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]
    labels = ["lt_0.1", "0.1_0.3", "0.3_0.5", "0.5_0.7", "0.7_0.9", "gt_0.9"]
    n = len(probs)
    return {lab: float(((probs > lo) & (probs <= hi)).sum() / n) if n else None for lab, (lo, hi) in zip(labels, bins)}


def load_period_severity(dataset_id: str, horizon_periods: int) -> dict[int, int]:
    """Reconstructs time -> max_active_severity exactly as
    benchmark.splits.severity_split/_event_windows compute it internally,
    from the raw events.csv the generator produced -- not stored anywhere
    in the modeling pipeline's own saved outputs."""
    events_path = os.path.join(REPO_ROOT, "data", "benchmark", dataset_id, "events", "events.csv")
    if not os.path.exists(events_path):
        return {}
    events = pd.read_csv(events_path)
    windows = []
    for _, e in events.iterrows():
        end = e["start_time"] + e["duration"] + e["recovery_delay"] + e["recovery_periods"]
        windows.append((e["start_time"], end, e["severity"]))
    result = {}
    for t in range(horizon_periods):
        active = [sev for start, end, sev in windows if start <= t < end]
        result[t] = int(max(active, default=0))
    return result


def load_run(d: str) -> dict:
    metrics = json.load(open(os.path.join(d, "metrics.json")))
    calib = json.load(open(os.path.join(d, "calibration.json")))
    meta = json.load(open(os.path.join(d, "run_metadata.json")))
    hist = pd.read_csv(os.path.join(d, "training_history.csv"))
    preds = pd.read_csv(os.path.join(d, "predictions.csv"))
    onset_path = os.path.join(d, "onset_breakdown.json")
    onset = json.load(open(onset_path)) if os.path.exists(onset_path) else {}
    return dict(metrics=metrics, calib=calib, meta=meta, hist=hist, preds=preds, onset=onset, dir=d)


def build_main_row(config: str, split: str, seed: int, run: dict) -> dict:
    m = run["metrics"]["by_split"]
    c = run["calib"]
    hist = run["hist"]
    meta = run["meta"]
    preds = run["preds"]
    test_split = preds[preds["split"] == "test"]

    row = {"config": config, "split": split, "seed": seed}
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
        "best_train_pr_auc": hist["train_pr_auc"].max() if "train_pr_auc" in hist else None,
        "best_validation_pr_auc": hist["validation_pr_auc"].max() if "validation_pr_auc" in hist else None,
        "quantum_grad_norm_mean": hist["quantum_grad_norm_mean"].mean() if "quantum_grad_norm_mean" in hist else None,
        "quantum_grad_norm_min": hist["quantum_grad_norm_mean"].min() if "quantum_grad_norm_mean" in hist else None,
        "quantum_grad_norm_max": hist["quantum_grad_norm_mean"].max() if "quantum_grad_norm_mean" in hist else None,
        "quantum_grad_norm_final": hist["quantum_grad_norm_mean"].iloc[-1] if "quantum_grad_norm_mean" in hist else None,
        "reduce_grad_norm_mean": hist["reduce_grad_norm_mean"].mean() if "reduce_grad_norm_mean" in hist else None,
        "train_to_val_pr_auc_gap": (m.get("train", {}).get("pr_auc") or float("nan")) - (m.get("validation", {}).get("pr_auc") or float("nan")),
        "train_to_test_pr_auc_gap": (m.get("train", {}).get("pr_auc") or float("nan")) - (m.get("test", {}).get("pr_auc") or float("nan")),
        "val_to_test_pr_auc_gap": (m.get("validation", {}).get("pr_auc") or float("nan")) - (m.get("test", {}).get("pr_auc") or float("nan")),
        "train_to_test_roc_auc_gap": (m.get("train", {}).get("roc_auc") or float("nan")) - (m.get("test", {}).get("roc_auc") or float("nan")),
        "fresh_onset_pr_auc": run["onset"].get("test", {}).get("pr_auc_fresh_onset"),
        "fresh_onset_recall": run["onset"].get("test", {}).get("recall_fresh_onset"),
        "n_fresh_onset": run["onset"].get("test", {}).get("n_fresh_onset"),
    })
    row.update({f"prob_frac_{k}": v for k, v in probability_histogram(test_split["risk_probability"]).items()})
    return row


def build_severity_level_rows(config: str, seed: int, run: dict, period_severity: dict[int, int]) -> list[dict]:
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
        y_pred = (y_prob >= run["metrics"]["threshold"]).astype(int)
        recall = float((y_pred[y_true == 1] == 1).mean()) if n_pos > 0 else None
        precision = float((y_true[y_pred == 1] == 1).mean()) if y_pred.sum() > 0 else None
        rows.append({
            "config": config, "seed": seed, "severity_level": int(level), "n_examples": len(sub),
            "n_positive": n_pos, "pr_auc": pr_auc, "recall": recall, "precision": precision,
            "prob_mean": float(y_prob.mean()), "prob_std": float(y_prob.std()),
        })
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    main_rows, severity_level_rows = [], []
    period_severity = load_period_severity("scm_v1_black_swan_seed43", horizon_periods=104)

    for config, primary_tag, severity_tag in CONFIGS:
        for split, tag, strategy in [("primary", primary_tag, "temporal"), ("severity", severity_tag, "severity")]:
            for seed in SEEDS:
                d = find_run_dir(tag, seed, strategy)
                run = load_run(d)
                main_rows.append(build_main_row(config, split, seed, run))
                if split == "severity":
                    severity_level_rows.extend(build_severity_level_rows(config, seed, run, period_severity))

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(os.path.join(OUT_DIR, "phase2b_all_metrics.csv"), index=False)
    print(f"Saved {len(main_df)} rows -> {os.path.join(OUT_DIR, 'phase2b_all_metrics.csv')}")

    sev_df = pd.DataFrame(severity_level_rows)
    if not sev_df.empty:
        sev_df.to_csv(os.path.join(OUT_DIR, "phase2b_severity_level_breakdown.csv"), index=False)
        print(f"Saved {len(sev_df)} rows -> {os.path.join(OUT_DIR, 'phase2b_severity_level_breakdown.csv')}")
    else:
        print("Severity-level breakdown: events.csv not found or empty -- skipped.")

    stability_rows = []
    for config, _, _ in CONFIGS:
        for split in ("primary", "severity"):
            sub = main_df[(main_df.config == config) & (main_df.split == split)]
            for metric in STABILITY_METRICS:
                col = f"test_{metric}"
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    continue
                q1, q3 = np.percentile(vals, [25, 75])
                stability_rows.append({
                    "config": config, "split": split, "metric": metric, "n": len(vals),
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)), "median": float(np.median(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)), "iqr": float(q3 - q1),
                })
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(os.path.join(OUT_DIR, "phase2b_seed_stability.csv"), index=False)
    print(f"Saved {len(stability_df)} rows -> {os.path.join(OUT_DIR, 'phase2b_seed_stability.csv')}")

    paired_rows = []
    for config, _, _ in CONFIGS:
        if config == "baseline":
            continue
        for split in ("primary", "severity"):
            base = main_df[(main_df.config == "baseline") & (main_df.split == split)].set_index("seed")
            cur = main_df[(main_df.config == config) & (main_df.split == split)].set_index("seed")
            for metric in STABILITY_METRICS:
                col = f"test_{metric}"
                if col not in base.columns:
                    continue
                diffs = (cur[col] - base[col]).dropna()
                if diffs.empty:
                    continue
                paired_rows.append({
                    "config": config, "split": split, "metric": metric,
                    "mean_diff": float(diffs.mean()), "median_diff": float(diffs.median()), "std_diff": float(diffs.std()),
                    "n_improved": int((diffs > 0).sum()), "n_worsened": int((diffs < 0).sum()), "n_unchanged": int((diffs == 0).sum()),
                    **{f"seed{s}_diff": float(v) for s, v in diffs.items()},
                })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(OUT_DIR, "phase2b_paired_vs_baseline.csv"), index=False)
    print(f"Saved {len(paired_df)} rows -> {os.path.join(OUT_DIR, 'phase2b_paired_vs_baseline.csv')}")


if __name__ == "__main__":
    main()
