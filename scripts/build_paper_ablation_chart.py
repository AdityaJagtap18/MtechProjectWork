#!/usr/bin/env python3
"""Clean ablation PR-AUC bar chart matching exactly the rows of Table 2
(tab:six-factors) -- unlike fig2_ablation_primary_pr_auc.png, which plots
every row in QGNN_V4_MASTER_RESULTS.csv (dozens of extra pilot configs
not in the paper's table), making it unreadably dense once shrunk to fit
the page. This version plots only the 12 configurations the paper's own
Table 2 reports, in the same order, so the figure and the table agree
one-to-one.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_ROOT)
OUT_DIR = os.path.join(REPO_ROOT, "final_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# (label, mean, std, is_range, full_seeds)
# Values verified against Table 2 / QGNN_V4_MASTER_RESULTS.csv /
# phase4_stage5/stage5_summary.csv. Range rows (2-seed pilots across
# multiple sub-variants) use (min, max) in place of (mean, std).
ROWS = [
    ("Classical\nGraphSAGE-Full", 0.807, 0.066, False, True),
    ("QGNN standing\nreference", 0.811, 0.074, False, True),
    ("4 qubits", 0.847, 0.102, False, True),
    ("8 qubits", 0.810, 0.091, False, True),
    ("1 layer", 0.693, 0.198, False, True),
    ("3 layers", 0.804, 0.186, False, True),
    ("Hardware-efficient\nring", 0.8065, 0.147, False, True),
    ("Reduced\nentanglement", 0.8138, 0.105, False, True),
    ("Projection\n(nonlinear/pre-norm/PCA)", (0.391, 0.729), None, True, False),
    ("Initialization\n(small-Gaussian/identity)", (0.726, 0.828), None, True, False),
    ("Identity-like,\n5-seed", 0.777, 0.059, False, True),
    ("Encoding\n(scale/clip/re-upload)", (0.581, 0.803), None, True, False),
]

labels = [r[0] for r in ROWS]
means = []
err_lo = []
err_hi = []
colors = []
for label, val, std, is_range, is_full in ROWS:
    if is_range:
        lo, hi = val
        mean = (lo + hi) / 2
        means.append(mean)
        err_lo.append(mean - lo)
        err_hi.append(hi - mean)
    else:
        means.append(val)
        err_lo.append(std)
        err_hi.append(std)
    colors.append("#55A868" if is_full else "#C44E52")

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(x, means, yerr=[err_lo, err_hi], capsize=3, color=colors, width=0.62)
ax.axhline(0.811, color="black", linestyle="--", linewidth=1, label="QGNN standing reference (0.811, 5-seed)")
ax.axhline(0.807, color="gray", linestyle=":", linewidth=1, label="Classical reference (0.807, 5-seed)")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Primary PR-AUC", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8, fontweight="bold")
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")
ax.legend(loc="lower left", fontsize=8)

from matplotlib.patches import Patch
handles, _ = ax.get_legend_handles_labels()
handles += [Patch(facecolor="#55A868", label="5-seed full evaluation"),
            Patch(facecolor="#C44E52", label="2-seed pilot (range shown)")]
ax.legend(handles=handles, loc="lower left", fontsize=7.5)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_ablation_table2_configs.png")
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"saved {out_path}")
