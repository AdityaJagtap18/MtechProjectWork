#!/usr/bin/env python3
"""QGNN-v4 Final Research Consolidation: thesis-quality figures built
directly from QGNN_V4_MASTER_RESULTS.csv / QGNN_V4_REFERENCE_COMPARISON.csv
-- no new numbers, only re-plotting already-verified, already-cited
values. Every figure with error bars distinguishes 5-seed ("full") from
2-seed ("pilot") evidence in its own label -- never implies statistical
significance that wasn't tested.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "final_figures")
os.makedirs(OUT_DIR, exist_ok=True)

master = pd.read_csv(os.path.join(REPO_ROOT, "QGNN_V4_MASTER_RESULTS.csv"))


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


# --- Figure 1: Classical vs QGNN reference, both splits ---
ref = master[master.phase == "Reference"]
fig, ax = plt.subplots(figsize=(6, 4.5))
labels = ["Primary", "Severity"]
classical_vals = [ref[(ref.experiment == "Classical GraphSAGE-Full") & (ref.split == "primary")].pr_auc.iloc[0],
                   ref[(ref.experiment == "Classical GraphSAGE-Full") & (ref.split == "severity")].pr_auc.iloc[0]]
classical_std = [ref[(ref.experiment == "Classical GraphSAGE-Full") & (ref.split == "primary")].pr_auc_std.iloc[0],
                  ref[(ref.experiment == "Classical GraphSAGE-Full") & (ref.split == "severity")].pr_auc_std.iloc[0]]
qgnn_vals = [ref[(ref.experiment == "QGNN-v4 standing reference") & (ref.split == "primary")].pr_auc.iloc[0],
             ref[(ref.experiment == "QGNN-v4 standing reference") & (ref.split == "severity")].pr_auc.iloc[0]]
qgnn_std = [ref[(ref.experiment == "QGNN-v4 standing reference") & (ref.split == "primary")].pr_auc_std.iloc[0],
            ref[(ref.experiment == "QGNN-v4 standing reference") & (ref.split == "severity")].pr_auc_std.iloc[0]]
x = range(len(labels))
width = 0.35
ax.bar([i - width / 2 for i in x], classical_vals, width, yerr=classical_std, capsize=4, label="Classical GraphSAGE-Full", color="#4C72B0")
ax.bar([i + width / 2 for i in x], qgnn_vals, width, yerr=qgnn_std, capsize=4, label="QGNN-v4 (standing reference)", color="#DD8452")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("PR-AUC (test, mean ± std, 5 seeds)")
ax.set_title("Classical vs. QGNN-v4: Primary and Severity PR-AUC")
ax.set_ylim(0, 1.0)
ax.legend()
savefig("fig1_classical_vs_qgnn_primary_severity.png")

# --- Figure 2: Ablation summary across phases (primary PR-AUC) ---
ablation_primary = master[(master.split == "primary") & (master.phase != "Reference")].copy()
ablation_primary = ablation_primary.sort_values("pr_auc", ascending=True)
fig, ax = plt.subplots(figsize=(9, 12))
colors = ["#55A868" if st == "full" else "#C44E52" for st in ablation_primary.seed_type]
ax.barh(ablation_primary.experiment + " (" + ablation_primary.split + ")", ablation_primary.pr_auc,
        xerr=ablation_primary.pr_auc_std, capsize=2, color=colors)
ax.axvline(0.8112, color="black", linestyle="--", linewidth=1, label="QGNN-v4 reference (0.811, 5-seed)")
ax.axvline(0.8070, color="gray", linestyle=":", linewidth=1, label="Classical reference (0.807, 5-seed)")
ax.set_xlabel("Primary PR-AUC")
ax.set_title("Ablation Summary: Primary PR-AUC across all tested configurations\n(green=5-seed full evaluation, red=2-seed pilot)", fontsize=11)
ax.legend(loc="lower right", fontsize=8)
savefig("fig2_ablation_primary_pr_auc.png")

# --- Figure 3: Ablation summary across phases (severity PR-AUC) ---
ablation_sev = master[(master.split == "severity") & (master.phase != "Reference")].copy()
ablation_sev = ablation_sev.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=True)
fig, ax = plt.subplots(figsize=(9, 12))
colors = ["#55A868" if st == "full" else "#C44E52" for st in ablation_sev.seed_type]
ax.barh(ablation_sev.experiment + " (" + ablation_sev.split + ")", ablation_sev.pr_auc,
        xerr=ablation_sev.pr_auc_std, capsize=2, color=colors)
ax.axvline(0.3976, color="black", linestyle="--", linewidth=1, label="QGNN-v4 reference (0.398, 5-seed)")
ax.axvline(0.4487, color="gray", linestyle=":", linewidth=1, label="Classical reference (0.449, 5-seed)")
ax.set_xlabel("Severity PR-AUC")
ax.set_title("Ablation Summary: Severity PR-AUC across all tested configurations\n(green=5-seed full evaluation, red=2-seed pilot)", fontsize=11)
ax.legend(loc="lower right", fontsize=8)
savefig("fig3_ablation_severity_pr_auc.png")

# --- Figure 4: Primary vs Severity scatter, all configurations ---
both = master.pivot_table(index=["phase", "experiment"], columns="split", values="pr_auc", aggfunc="first").reset_index()
both = both.dropna(subset=["primary", "severity"])
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(both["primary"], both["severity"], alpha=0.7, s=40, color="#4C72B0")
ax.scatter([0.8112], [0.3976], color="black", marker="*", s=250, label="QGNN-v4 reference", zorder=5)
ax.scatter([0.8070], [0.4487], color="red", marker="*", s=250, label="Classical reference", zorder=5)
ax.set_xlabel("Primary PR-AUC")
ax.set_ylabel("Severity PR-AUC")
ax.set_title("Primary vs. Severity PR-AUC, every tested configuration")
ax.legend()
savefig("fig4_primary_vs_severity_scatter.png")

# --- Figure 5: Seed-wise PR-AUC, Classical vs QGNN reference (from QGNN_V4_PHASE2C_REPORT.md Table 4) ---
seeds = [42, 43, 44, 45, 46]
classical_primary = [0.7874, 0.6888, 0.8745, 0.8491, 0.8354]
qgnn_primary = [0.8682, 0.6807, 0.8197, 0.7951, 0.8924]
classical_severity = [0.4430, 0.4560, 0.4378, 0.4358, 0.4708]
qgnn_severity = [0.3943, 0.4496, 0.4156, 0.2900, 0.4385]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(seeds, classical_primary, "o-", label="Classical", color="#4C72B0")
axes[0].plot(seeds, qgnn_primary, "s-", label="QGNN-v4", color="#DD8452")
axes[0].set_title("Primary PR-AUC per seed")
axes[0].set_xlabel("Seed")
axes[0].set_ylabel("PR-AUC")
axes[0].legend()
axes[1].plot(seeds, classical_severity, "o-", label="Classical", color="#4C72B0")
axes[1].plot(seeds, qgnn_severity, "s-", label="QGNN-v4", color="#DD8452")
axes[1].set_title("Severity PR-AUC per seed (seed 45 = LayerNorm reversal)")
axes[1].set_xlabel("Seed")
axes[1].legend()
fig.suptitle("Seed-wise PR-AUC: Classical vs. QGNN-v4 (LayerNorm-no-affine reference)")
savefig("fig5_seedwise_pr_auc.png")

# --- Figure 6: Calibration (ECE) comparison ---
calib = master[(master.ece.notna())].copy()
calib = calib.sort_values("ece")
fig, ax = plt.subplots(figsize=(9, 10))
colors = ["#55A868" if st == "full" else "#C44E52" for st in calib.seed_type]
ax.barh(calib.experiment + " (" + calib.split + ")", calib.ece, color=colors)
ax.set_xlabel("Expected Calibration Error (lower = better)")
ax.set_title("Calibration comparison across configurations\n(green=5-seed full, red=2-seed pilot)")
savefig("fig6_calibration_ece.png")

print("\nAll figures written to", OUT_DIR)
