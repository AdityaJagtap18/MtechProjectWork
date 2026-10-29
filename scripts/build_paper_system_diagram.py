#!/usr/bin/env python3
"""Paper-ready variant of Figure 1 (overall system architecture):
no title, no subtitle, no legend footer, all box text bold. Same
content/layout as report_assets/diagrams/fig01_overall_system_architecture.png
(scripts/build_report_diagrams_1.py::figure1), just trimmed for direct
inclusion in the paper where a caption already does the title's job.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_diagram_helpers import (
    COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE, COLOR_DATA, COLOR_DATA_EDGE,
    COLOR_FROZEN, COLOR_FROZEN_EDGE, COLOR_OUTPUT, COLOR_OUTPUT_EDGE,
    COLOR_QUANTUM, COLOR_QUANTUM_EDGE, arrow, box, down_arrow,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_figures")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    # Content spans roughly y=[30.5, 101.6] in the 0-100-ish coordinate
    # space the box()/arrow() helpers use; set a tight ylim directly
    # (bypassing new_fig's fixed header-band range) so there's no leftover
    # blank space where the title/subtitle/legend used to be.
    fig, ax = plt.subplots(figsize=(7.9, 6.8), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(28, 104)
    ax.axis("off")

    x, w = 20, 60
    y = 97
    box(ax, x, y, w, 4.6, "Synthetic Supply-Chain Dataset\n(104 time periods, black-swan event engine)", face=COLOR_DATA, edge=COLOR_DATA_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.5)
    y -= 8.1
    box(ax, x, y, w, 4.6, "Data Generation / Loading\n(graph, operations, events, labels)", face=COLOR_DATA, edge=COLOR_DATA_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.5)
    y -= 8.1
    box(ax, x, y, w, 4.6, "Supply-Chain Graph Construction\n2,670 nodes . 7,675 edges . 9 relation types", face=COLOR_DATA, edge=COLOR_DATA_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.5)
    y -= 8.1
    box(ax, x, y, w, 4.6, "Node / Edge Features\n(leakage-safe rolling aggregates, static attributes)", face=COLOR_DATA, edge=COLOR_DATA_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.5)
    y -= 8.1
    box(ax, x, y, w, 5.0, "Heterogeneous GraphSAGE Encoder\n2-layer mean-aggregation, 128-dimensional hidden layer", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.7)
    y -= 8.5
    box(ax, x, y, w, 4.6, "128-D Supplier Embedding  (FROZEN after this point)", face=COLOR_FROZEN, edge=COLOR_FROZEN_EDGE, fontweight="bold")

    branch_y = y
    down_arrow(ax, 32, branch_y, branch_y - 4.9)
    down_arrow(ax, 68, branch_y, branch_y - 4.9)
    y2 = branch_y - 9.8
    box(ax, 12, y2, 34, 5.6, "Classical Baseline\nMLP(128->64->1) classifier\n(full end-to-end training)", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE, fontweight="bold")
    box(ax, 54, y2, 34, 5.6, "Quantum Head (QGNN)\nLinear(128,6) -> quantum circuit\n-> LayerNorm -> Linear(6,1)", face=COLOR_QUANTUM, edge=COLOR_QUANTUM_EDGE, fontweight="bold")

    down_arrow(ax, 29, y2, y2 - 3.5)
    down_arrow(ax, 71, y2, y2 - 3.5)
    y3 = y2 - 8.0
    box(ax, 12, y3, 34, 3.9, "Risk Probability\n(Classical path)", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold")
    box(ax, 54, y3, 34, 3.9, "Risk Probability\n(Quantum path)", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold")

    arrow(ax, 29, y3, 42, y3 - 3.8, connectionstyle="arc3,rad=0.15")
    arrow(ax, 71, y3, 58, y3 - 3.8, connectionstyle="arc3,rad=-0.15")
    y4 = y3 - 7.8
    box(ax, 22, y4, 56, 4.8, "Evaluation: PR-AUC, ROC-AUC, F1, MCC, Brier, ECE\nPrimary (temporal) split and Severity (OOD) split", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold")

    out_path = os.path.join(OUT, "fig_system_architecture_paper.png")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
