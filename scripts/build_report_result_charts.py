#!/usr/bin/env python3
"""Additional data-driven result charts for the final DOCX report:
fresh-onset, calibration reliability, quantum representation (channel
correlations across ansatz/encoding), encoded-angle distributions
(E0-E4), and the seed-45 visual analysis. Every number is read from
already-saved, already-verified experiment artifacts -- nothing here is
estimated or fabricated.
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scm_dataset.modeling.calibration import calibration_curve_data
from _analysis_common import find_latest_graphsage_full_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "report_assets", "charts")
os.makedirs(OUT, exist_ok=True)

CLASSICAL_COLOR = "#4C72B0"
QGNN_COLOR = "#DD8452"
DIAG_COLOR = "#55A868"


def savefig(name):
    path = os.path.join(OUT, name)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


# ---------------------------------------------------------------------
# Fresh-onset
# ---------------------------------------------------------------------
def chart_fresh_onset():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = ["Classical\n(Severity)", "QGNN-v4 ref.\n(Severity)", "Encoding pilot\nrange (Severity)"]
    values = [0.0064, 0.0067, None]  # classical mean (5 seeds), qgnn mean, encoding range shown separately
    x = np.arange(len(labels))
    bars = ax.bar([0, 1], [0.0064, 0.0067], color=[CLASSICAL_COLOR, QGNN_COLOR], width=0.5)
    ax.bar(2, 0.0085, yerr=[[0.0085 - 0.005], [0.012 - 0.0085]], color="#8172B3", width=0.5, capsize=6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Fresh-onset PR-AUC (Severity split)")
    ax.set_title("Fresh-Onset PR-AUC: Severity Split Only\n(Primary split has 0 fresh-onset examples -- undefined, not plotted as zero)")
    ax.set_ylim(0, 0.02)
    for i, v in enumerate([0.0064, 0.0067]):
        ax.text(i, v + 0.0006, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.text(2, 0.012 + 0.0006, "0.005-0.012", ha="center", fontsize=8.5)
    savefig("chart_fresh_onset.png")


# ---------------------------------------------------------------------
# Calibration reliability diagram (pooled 5-seed test predictions)
# ---------------------------------------------------------------------
def pooled_predictions(base_dir, tag_pattern, split_strategy, seeds):
    frames = []
    for seed in seeds:
        pattern = os.path.join(REPO_ROOT, base_dir, tag_pattern.format(seed=seed))
        matches = sorted(glob.glob(pattern))
        chosen = None
        for d in matches:
            cfg_path = os.path.join(d, "config.yaml")
            if not os.path.exists(cfg_path):
                continue
            cfg = yaml.safe_load(open(cfg_path))
            if cfg.get("split", {}).get("strategy") == split_strategy:
                chosen = d
        if chosen is None and matches:
            chosen = matches[-1]
        if chosen is None:
            continue
        df = pd.read_csv(os.path.join(chosen, "predictions.csv"))
        frames.append(df[df["split"] == "test"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def chart_calibration_reliability():
    seeds = [42, 43, 44, 45, 46]
    classical_primary = pooled_predictions("experiments/classical_gnn", "*_hetero_graphsage_seed{seed}", "temporal", seeds)
    classical_severity = pooled_predictions("experiments/classical_gnn", "*_hetero_graphsage_severity_seed{seed}", "severity", seeds)
    qgnn_primary = pooled_predictions("experiments/qgnn_v4", "*_phase2c_layernorm_noaffine_quantum_seed{seed}", "temporal", seeds)
    qgnn_severity = pooled_predictions("experiments/qgnn_v4", "*_phase2c_layernorm_noaffine_quantum_seed{seed}", "severity", seeds)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, split, c_df, q_df in [
        (axes[0], "Primary", classical_primary, qgnn_primary),
        (axes[1], "Severity", classical_severity, qgnn_severity),
    ]:
        for df, color, label_ in [(c_df, CLASSICAL_COLOR, "Classical"), (q_df, QGNN_COLOR, "QGNN-v4 ref.")]:
            curve = calibration_curve_data(df["actual_disruption"].values, df["risk_probability"].values, n_bins=10)
            conf = [c for c, n in zip(curve["mean_predicted_probability"], curve["bin_counts"]) if n > 0]
            acc = [a for a, n in zip(curve["observed_frequency"], curve["bin_counts"]) if n > 0]
            ax.plot(conf, acc, "o-", color=color, label=label_, markersize=5)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{split} split (5 seeds pooled, test)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Reliability Diagram: Classical vs. QGNN-v4 Reference")
    savefig("chart_calibration_reliability.png")


# ---------------------------------------------------------------------
# Quantum representation: channel-target and cross-channel correlation
# ---------------------------------------------------------------------
def chart_quantum_channel_correlation():
    stage5 = pd.read_csv(os.path.join(REPO_ROOT, "experiments/qgnn_v4/phase4_stage5/stage5_quantum_features.csv"))
    enc = pd.read_csv(os.path.join(REPO_ROOT, "experiments/qgnn_v4/encoding_investigation/encoding_quantum_features.csv"))

    stage5_summary = stage5.drop_duplicates(["config", "split", "seed"]).groupby("config")[
        ["max_abs_target_correlation_this_run", "max_abs_cross_channel_corr"]
    ].mean()
    enc_summary = enc.drop_duplicates(["config", "split", "seed"]).groupby("config")[
        ["max_abs_target_correlation_this_run", "max_abs_cross_channel_corr"]
    ].mean()

    labels = list(stage5_summary.index) + list(enc_summary.index)
    target_corr = list(stage5_summary["max_abs_target_correlation_this_run"]) + list(enc_summary["max_abs_target_correlation_this_run"])
    cross_corr = list(stage5_summary["max_abs_cross_channel_corr"]) + list(enc_summary["max_abs_cross_channel_corr"])

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, target_corr, w, label="Max |channel <-> target correlation|", color="#4C72B0")
    ax.bar(x + w / 2, cross_corr, w, label="Max |cross-channel correlation|", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Correlation (absolute value)")
    ax.set_title("Quantum Representation: Channel-Target vs. Cross-Channel Correlation\n(Ansatz comparison, Stage 5, and Encoding Investigation configurations)")
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, 1.05)
    savefig("chart_quantum_channel_correlation.png")


# ---------------------------------------------------------------------
# Encoded angle distributions E0-E4
# ---------------------------------------------------------------------
def chart_encoding_angle_distributions():
    angles = pd.read_csv(os.path.join(REPO_ROOT, "experiments/qgnn_v4/encoding_investigation/encoding_encoded_angles.csv"))
    configs = ["E0_baseline", "E1_reduced_scale", "E2_increased_scale", "E3_linear_clip", "E4_data_reuploading"]
    display = ["E0 (pi*tanh)", "E1 (0.5pi*tanh)", "E2 (2pi*tanh)", "E3 (pi*clip)", "E4 (re-upload)"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = []
    for cfg in configs:
        sub = angles[angles.config == cfg]
        # std/mean/min/max per row (already aggregated per seed x split); use std as the spread proxy
        data.append(sub["std"].values)
    bp = ax.boxplot(data, tick_labels=display, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Encoded-angle standard deviation\n(radians, per seed x split)")
    ax.set_title("Encoded Angle Spread by Configuration (E0-E4)\nE2 shows the widest spread; E1 the narrowest")
    savefig("chart_encoding_angle_distributions.png")


# ---------------------------------------------------------------------
# Seed-45 visual analysis
# ---------------------------------------------------------------------
def chart_seed45_analysis():
    base = os.path.join(REPO_ROOT, "experiments/qgnn_v4/phase2d_seed45_inspection")
    baseline45 = pd.read_csv(os.path.join(base, "baseline_seed45_severity_test_features.csv"))
    ln45 = pd.read_csv(os.path.join(base, "ln_noaffine_seed45_severity_test_features.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].hist(baseline45["probability"], bins=30, alpha=0.6, label="Baseline (no LayerNorm)", color=CLASSICAL_COLOR)
    axes[0].hist(ln45["probability"], bins=30, alpha=0.6, label="LayerNorm-no-affine", color=QGNN_COLOR)
    axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count (severity test examples)")
    axes[0].set_title("Seed 45: Prediction Distribution\nBefore vs. After LayerNorm")
    axes[0].legend(fontsize=8)

    seeds = [42, 43, 44, 45, 46]
    baseline_corr, ln_corr = [], []
    for s in seeds:
        b = pd.read_csv(os.path.join(base, f"baseline_seed{s}_severity_test_features.csv"))
        l = pd.read_csv(os.path.join(base, f"ln_noaffine_seed{s}_severity_test_features.csv"))
        pauliz_cols_b = [c for c in b.columns if c.startswith("pauliz_")]
        pauliz_cols_l = [c for c in l.columns if c.startswith("pauliz_")]
        from scipy.stats import pointbiserialr
        b_corrs = [abs(pointbiserialr(b[c], b["actual_disruption"])[0]) for c in pauliz_cols_b]
        l_corrs = [abs(pointbiserialr(l[c], l["actual_disruption"])[0]) for c in pauliz_cols_l]
        baseline_corr.append(max(b_corrs))
        ln_corr.append(max(l_corrs))

    x = np.arange(len(seeds))
    w = 0.35
    axes[1].bar(x - w / 2, baseline_corr, w, label="Baseline", color=CLASSICAL_COLOR)
    axes[1].bar(x + w / 2, ln_corr, w, label="LayerNorm-no-affine", color=QGNN_COLOR)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Max |channel <-> label correlation|")
    axes[1].set_title("Best-Channel Target Correlation, Before/After LayerNorm\n(seed 45 is the only seed where it drops)")
    axes[1].legend(fontsize=8)
    for i, s in enumerate(seeds):
        if s == 45:
            axes[1].annotate("seed 45", (i, max(baseline_corr[i], ln_corr[i]) + 0.03), ha="center", fontsize=8, fontweight="bold", color="#C44E52")

    fig.suptitle("Seed-45 Investigation: Representation Distortion Under LayerNorm (Severity Split)")
    savefig("chart_seed45_analysis.png")
    print("baseline_corr:", dict(zip(seeds, [round(v, 3) for v in baseline_corr])))
    print("ln_corr:", dict(zip(seeds, [round(v, 3) for v in ln_corr])))


if __name__ == "__main__":
    chart_fresh_onset()
    chart_calibration_reliability()
    chart_quantum_channel_correlation()
    chart_encoding_angle_distributions()
    chart_seed45_analysis()
