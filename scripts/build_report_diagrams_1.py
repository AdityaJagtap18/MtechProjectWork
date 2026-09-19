#!/usr/bin/env python3
"""Figures 1-4: overall system architecture, data processing pipeline,
graph construction schematic, classical GraphSAGE flow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_diagram_helpers import (
    subtitle,
    COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE, COLOR_DATA, COLOR_DATA_EDGE,
    COLOR_FROZEN, COLOR_FROZEN_EDGE, COLOR_OUTPUT, COLOR_OUTPUT_EDGE,
    COLOR_QUANTUM, COLOR_QUANTUM_EDGE, arrow, box, down_arrow, label,
    legend_swatch, new_fig, save, title,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report_assets", "diagrams")
os.makedirs(OUT, exist_ok=True)


def figure1():
    fig, ax = new_fig(9, 12)
    title(ax, "Overall Research System Architecture", y=98)
    subtitle(ax, "End-to-end path from raw synthetic data to a risk prediction, showing where the\nClassical and QGNN-v4 paths share the frozen encoder and diverge afterward.")

    x, w = 20, 60
    y = 87
    box(ax, x, y, w, 4.2, "Synthetic Supply-Chain Data\n(scm_v1_black_swan_seed43)", face=COLOR_DATA, edge=COLOR_DATA_EDGE)
    down_arrow(ax, 50, y, y - 3.2)
    y -= 7.4
    box(ax, x, y, w, 4.2, "Data Generation / Loading\n(graph, operations, events, labels)", face=COLOR_DATA, edge=COLOR_DATA_EDGE)
    down_arrow(ax, 50, y, y - 3.2)
    y -= 7.4
    box(ax, x, y, w, 4.2, "Supply-Chain Graph Construction\n2,670 nodes . 7,675 edges . 9 relation types", face=COLOR_DATA, edge=COLOR_DATA_EDGE)
    down_arrow(ax, 50, y, y - 3.2)
    y -= 7.4
    box(ax, x, y, w, 4.2, "Node / Edge Features\n(leakage-safe rolling aggregates, static attributes)", face=COLOR_DATA, edge=COLOR_DATA_EDGE)
    down_arrow(ax, 50, y, y - 3.2)
    y -= 7.4
    box(ax, x, y, w, 4.6, "Classical GraphSAGE (HeteroGraphSAGE)\n2-layer mean-aggregation SAGEConv, hidden_dim=128", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.4)
    y -= 7.8
    box(ax, x, y, w, 4.2, "128-D Supplier Embedding  (FROZEN after this point)", face=COLOR_FROZEN, edge=COLOR_FROZEN_EDGE, fontweight="bold")

    branch_y = y
    down_arrow(ax, 32, branch_y, branch_y - 4.5)
    down_arrow(ax, 68, branch_y, branch_y - 4.5)
    y2 = branch_y - 9
    box(ax, 12, y2, 34, 5.2, "Classical Baseline\nMLP(128->64->1) classifier\n(full end-to-end training)", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE)
    box(ax, 54, y2, 34, 5.2, "QGNN-v4 Head\nLinear(128,6) -> quantum circuit\n-> LayerNorm -> Linear(6,1)", face=COLOR_QUANTUM, edge=COLOR_QUANTUM_EDGE)

    down_arrow(ax, 29, y2, y2 - 3.2)
    down_arrow(ax, 71, y2, y2 - 3.2)
    y3 = y2 - 7.4
    box(ax, 12, y3, 34, 3.6, "Risk Probability\n(Classical path)", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE)
    box(ax, 54, y3, 34, 3.6, "Risk Probability\n(QGNN-v4 path)", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE)

    arrow(ax, 29, y3, 42, y3 - 3.5, connectionstyle="arc3,rad=0.15")
    arrow(ax, 71, y3, 58, y3 - 3.5, connectionstyle="arc3,rad=-0.15")
    y4 = y3 - 7.2
    box(ax, 22, y4, 56, 4.4, "Evaluation: PR-AUC, ROC-AUC, F1, MCC, Brier, ECE\nPrimary (temporal) split and Severity (OOD) split", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold")

    legend_swatch(ax, 4, 4, 3, 2, COLOR_DATA, COLOR_DATA_EDGE, "Data / graph construction", fontsize=7.5)
    legend_swatch(ax, 4, 1.3, 3, 2, COLOR_FROZEN, COLOR_FROZEN_EDGE, "Frozen (no gradient beyond this point)", fontsize=7.5)
    legend_swatch(ax, 38, 4, 3, 2, COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE, "Classical component", fontsize=7.5)
    legend_swatch(ax, 38, 1.3, 3, 2, COLOR_QUANTUM, COLOR_QUANTUM_EDGE, "Quantum component", fontsize=7.5)
    legend_swatch(ax, 72, 4, 3, 2, COLOR_OUTPUT, COLOR_OUTPUT_EDGE, "Output / evaluation", fontsize=7.5)

    save(fig, os.path.join(OUT, "fig01_overall_system_architecture.png"))


def figure2():
    fig, ax = new_fig(9, 12.5)
    title(ax, "Data Processing Pipeline", y=98)
    subtitle(ax, "How raw generated data becomes labeled, split training examples.\nEvery step below is implemented in src/scm_dataset/ -- no step is invented.")

    steps = [
        ("Raw Synthetic SCM Data", "Generator output: graph/, operations/, events/, labels/ CSVs"),
        ("Entities", "Suppliers . Materials . Plants . Products . Regions . Procurement orders"),
        ("Relationships", "9 relation types: supplier-material, procurement-plant, plant-product, ..."),
        ("Graph Construction", "Static topology (PyG HeteroData), deterministic node ordering"),
        ("Node Features", "Static attributes + leakage-safe rolling aggregates (order volume,\nfulfillment ratio, inventory, production, demand -- windows 4/8/12)"),
        ("Event / Disruption Labels", "supplier_disrupted derived from simulated fulfillment shortfall,\nNOT from static risk/criticality input attributes"),
        ("Dataset Assembly", "Target Y(supplier,t) = max(supplier_disrupted[t+1..t+H]), H=4 periods"),
    ]
    x, w = 14, 72
    y = 89
    for i, (head, sub) in enumerate(steps):
        box(ax, x, y, w, 5.6, f"{head}\n{sub}", face=COLOR_DATA, edge=COLOR_DATA_EDGE, fontsize=8.7)
        if i < len(steps) - 1:
            down_arrow(ax, 50, y, y - 3.6)
        y -= 9.2

    y -= 1.0
    box(ax, x, y, w, 4.6, "Train / Validation / Test Split Assignment", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold")
    down_arrow(ax, 34, y, y - 4.2)
    down_arrow(ax, 66, y, y - 4.2)
    y -= 8.6
    box(ax, 12, y, 34, 5.6, "Primary (Temporal) Split\nTrain on early periods (70%),\ntest on later periods (15%)\n-- evaluates FUTURE generalization", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontsize=8.3)
    box(ax, 54, y, 34, 5.6, "Severity (OOD) Split\nTrain on severity <=3,\ntest on severity 4-5\n-- evaluates OOD generalization", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontsize=8.3)

    save(fig, os.path.join(OUT, "fig02_data_processing_pipeline.png"))


def figure3():
    fig, ax = new_fig(9.5, 8)
    title(ax, "Schematic Heterogeneous Supply-Chain Graph")
    subtitle(ax, "Representative node types and relations only -- NOT a rendering of all 2,670 nodes.\nActual graph: 2,670 nodes, 7,675 edges, 9 relation types (Table 3).")

    W, Hh = 14.0, 7.0
    nodes = {
        "Region": (50, 84),
        "Supplier": (11, 56),
        "Material": (37, 56),
        "Plant": (63, 56),
        "Product": (89, 56),
        "Procurement\n(order)": (37, 24),
    }
    colors = {
        "Region": ("#FBE9E7", "#C0745B"),
        "Supplier": ("#E8EEF7", COLOR_DATA_EDGE),
        "Material": ("#E9F5EC", COLOR_CLASSICAL_EDGE),
        "Plant": ("#FDF0E3", COLOR_QUANTUM_EDGE),
        "Product": ("#F5E9F5", COLOR_OUTPUT_EDGE),
        "Procurement\n(order)": ("#FFF9DB", "#B8960B"),
    }
    for name, (x, y) in nodes.items():
        face, edge = colors[name]
        box(ax, x - W / 2, y - Hh / 2, W, Hh, name, face=face, edge=edge, fontsize=9.5, fontweight="bold")

    def anchor(name, side):
        x, y = nodes[name]
        m = 1.0  # clear the rounded-box padding
        return {"L": (x - W / 2 - m, y), "R": (x + W / 2 + m, y),
                "T": (x, y + Hh / 2 + m), "B": (x, y - Hh / 2 - m)}[side]

    # (source, source side, target, target side, label, curvature)
    edges = [
        ("Supplier", "R", "Material", "L", "supplier_material", 0.0),
        ("Material", "R", "Plant", "L", "material_plant", 0.0),
        ("Plant", "R", "Product", "L", "plant_product", 0.0),
        ("Supplier", "T", "Region", "L", "supplier_region", -0.15),
        ("Plant", "T", "Region", "B", "plant_region", 0.0),
        ("Product", "T", "Region", "R", "product_region", 0.15),
        ("Supplier", "B", "Procurement\n(order)", "L", "supplier_procurement", 0.15),
        ("Procurement\n(order)", "T", "Material", "B", "procurement_material", 0.0),
        ("Procurement\n(order)", "R", "Plant", "B", "procurement_plant", -0.15),
    ]
    for a_, sa, b_, sb, lbl, rad in edges:
        x1, y1 = anchor(a_, sa)
        x2, y2 = anchor(b_, sb)
        cs = f"arc3,rad={rad}" if rad else None
        arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.3, connectionstyle=cs)
        # label sits on the edge at its midpoint, on a white patch
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if rad:
            dx, dy = x2 - x1, y2 - y1
            mx += rad * dy * 0.5
            my -= rad * dx * 0.5
        if y1 == y2:  # horizontal edge: label above the arrow so the arrowhead stays visible
            ax.text(mx, my + 2.6, lbl, ha="center", va="bottom", fontsize=7.2, color="#444444")
        else:
            ax.text(mx, my, lbl, ha="center", va="center", fontsize=7.2, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

    label(ax, 50, 8, "Reverse relations (rev_*) are added for message passing but carry no new\nsemantic information -- every reverse edge is implied by an existing forward edge.", fontsize=7.8, style="italic")

    save(fig, os.path.join(OUT, "fig03_graph_construction_schematic.png"))


def figure4():
    fig, ax = new_fig(8.5, 11)
    title(ax, "Classical GraphSAGE Processing Flow", y=97)
    subtitle(ax, "The frozen encoder shared by every experiment in this project.")

    x, w = 18, 64
    y = 87
    box(ax, x, y, w, 4.6, "Node / Edge Features\n(per node type, static + rolling dynamic)", face=COLOR_DATA, edge=COLOR_DATA_EDGE)
    down_arrow(ax, 50, y, y - 3.4)
    y -= 8.2
    box(ax, x, y, w, 4.2, "Type-specific Input Projection\nLinear per node type -> hidden_dim=128", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE)
    down_arrow(ax, 50, y, y - 3.2)
    y -= 7.6
    box(ax, x, y, w, 5.4, "HeteroGraphSAGE Layer 1\nSAGEConv (mean aggregation) per relation,\ncombined per node type via HeteroConv(aggr='mean')", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE)
    down_arrow(ax, 50, y, y - 3.6)
    y -= 9.0
    box(ax, x, y, w, 5.4, "HeteroGraphSAGE Layer 2\nSame aggregation, second hop of neighborhood\ninformation now reachable", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE)
    down_arrow(ax, 50, y, y - 3.6)
    y -= 9.0
    box(ax, x, y, w, 4.6, "Supplier-Node Readout\nh_dict['supplier'] -- one 128-D vector per supplier per period", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.4)
    y -= 8.2

    branch_y = y
    down_arrow(ax, 30, branch_y, branch_y - 4.2)
    down_arrow(ax, 70, branch_y, branch_y - 4.2)
    y2 = branch_y - 8.4
    box(ax, 10, y2, 36, 5.4, "Classical path (this project's\nend-to-end baseline):\nMLP(128->64->1) classifier,\ntrained jointly with the encoder", face=COLOR_CLASSICAL, edge=COLOR_CLASSICAL_EDGE, fontsize=8.3)
    box(ax, 52, y2, 36, 5.4, "QGNN-v4 path (this project's\nquantum experiments):\nembedding FROZEN, re-used as\ninput to every QGNN-v4 head", face=COLOR_FROZEN, edge=COLOR_FROZEN_EDGE, fontsize=8.3)

    label(ax, 50, 4, "requires_grad_(False) is asserted programmatically for every QGNN-v4 training\nrun -- verified by test, not only documented (QGNN_V4_REPRODUCIBILITY_CHECKLIST.md).", fontsize=7.6, style="italic")

    save(fig, os.path.join(OUT, "fig04_classical_graphsage_flow.png"))


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
