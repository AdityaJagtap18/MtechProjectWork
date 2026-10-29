#!/usr/bin/env python3
"""Figures 5-8 + trainable/frozen diagram."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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


def figure5():
    fig, ax = new_fig(8.5, 12.5)
    title(ax, "QGNN-v4 Detailed Architecture", y=98)
    subtitle(ax, "Internal structure of the standing QGNN-v4 reference head.\n6 qubits . 2 variational layers . 36 quantum params . 817 total head params.")

    x, w = 20, 60
    y = 90
    steps = [
        ("Frozen GraphSAGE-Full", COLOR_FROZEN, COLOR_FROZEN_EDGE),
        ("128-D Supplier Embedding", COLOR_FROZEN, COLOR_FROZEN_EDGE),
        ("Linear(128 -> 6)  [trainable]", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
        ("6 Classical Features", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
        ("pi * tanh(x)  (bounded encoding)", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
        ("RY Angle Encoding\n(AngleEmbedding)", COLOR_QUANTUM, COLOR_QUANTUM_EDGE),
    ]
    for i, (t, f, e) in enumerate(steps):
        box(ax, x, y, w, 3.6, t, face=f, edge=e)
        down_arrow(ax, 50, y, y - 2.6)
        y -= 6.2

    box(ax, x - 4, y - 15.5, w + 8, 16.2, "", face="#FDF0E3", edge=COLOR_QUANTUM_EDGE, fontsize=1)
    label(ax, 50, y - 1.3, "6-Qubit Variational Quantum Circuit", fontsize=10.5, fontweight="bold")
    label(ax, 50, y - 4.2, "Variational Layer 1  (StronglyEntanglingLayers)\n3 rotation params/qubit/layer + range-parameterized entanglement", fontsize=8.3)
    down_arrow(ax, 50, y - 6.0, y - 7.6, lw=1.3)
    label(ax, 50, y - 9.0, "Entanglement (CNOT pattern, fixed / not trainable)", fontsize=8.3)
    down_arrow(ax, 50, y - 10.4, y - 12.0, lw=1.3)
    label(ax, 50, y - 13.4, "Variational Layer 2  (StronglyEntanglingLayers)", fontsize=8.3)
    y = y - 15.5
    down_arrow(ax, 50, y, y - 2.6)
    y -= 6.2

    steps2 = [
        ("Pauli-Z Expectation Values\n(6 quantum features)", COLOR_QUANTUM, COLOR_QUANTUM_EDGE),
        ("LayerNorm(6, no affine)  [trainable: none]", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
        ("Linear(6 -> 1)  [trainable]", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
        ("Risk Probability (sigmoid of logit)", COLOR_OUTPUT, COLOR_OUTPUT_EDGE),
    ]
    for i, (t, f, e) in enumerate(steps2):
        box(ax, x, y, w, 3.8, t, face=f, edge=e, fontweight="bold" if i == len(steps2) - 1 else "normal")
        if i < len(steps2) - 1:
            down_arrow(ax, 50, y, y - 2.6)
        y -= 6.2

    label(ax, 50, 5, "Resource summary: default.qubit simulator . diff_method=\"backprop\" (exact) . no shots . no hardware . no noise", fontsize=7.8, style="italic")

    save(fig, os.path.join(OUT, "fig05_qgnn_detailed_architecture.png"))


def figure6():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    ax.text(50, 38.5, "Logical Quantum Circuit (StronglyEntanglingLayers, 6 qubits, 2 layers)", ha="center", fontsize=12.5, fontweight="bold")
    ax.text(50, 35.8, "Schematic of the actual circuit.py implementation: AngleEmbedding(RY) once, then two variational layers.\nEach layer = one arbitrary single-qubit rotation (Rot) per qubit + a range-parameterized entangling CNOT pattern.", ha="center", fontsize=8, linespacing=1.4)

    n_qubits = 6
    wire_y = [30 - 4.2 * i for i in range(n_qubits)]
    x_start, x_end = 8, 94
    for y in wire_y:
        ax.plot([x_start, x_end], [y, y], color="#333333", linewidth=1.1, zorder=1)
    for i, y in enumerate(wire_y):
        ax.text(x_start - 2.5, y, f"q{i}", ha="right", va="center", fontsize=10, fontweight="bold")

    def gate_box(x, y, text, w=4.6, h=3.0, face="#E8EEF7", edge="#4C72B0"):
        from matplotlib.patches import FancyBboxPatch
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.15,rounding_size=0.3", facecolor=face, edgecolor=edge, linewidth=1.3, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.8, zorder=4)

    x_encode = 16
    for y in wire_y:
        gate_box(x_encode, y, "RY($\\theta$)", face="#FDF0E3", edge="#DD8452")
    ax.text(x_encode, 33, "AngleEmbedding\n(input encoding)", ha="center", fontsize=7.5, style="italic")

    def entangle_ring(x, wire_y, label_text):
        pairs = [(i, i + 1) for i in range(len(wire_y) - 1)]
        for c, t in pairs:
            y1, y2 = wire_y[c], wire_y[t]
            ax.add_patch(Circle((x, y1), 0.45, color="black", zorder=3))
            ax.plot([x, x], [y1, y2], color="black", linewidth=1.3, zorder=2)
            circle = Circle((x, y2), 0.9, facecolor="white", edgecolor="black", linewidth=1.3, zorder=3)
            ax.add_patch(circle)
            ax.plot([x - 0.6, x + 0.6], [y2, y2], color="black", linewidth=1.1, zorder=4)
            ax.plot([x, x], [y2 - 0.6, y2 + 0.6], color="black", linewidth=1.1, zorder=4)
        ax.text(x, wire_y[-1] - 3.0, label_text, ha="center", fontsize=7.3, style="italic")

    for layer_i, x0 in enumerate([28, 58]):
        for y in wire_y:
            gate_box(x0, y, "Rot\n($\\phi,\\theta,\\omega$)", w=5.6, face="#E9F5EC", edge="#55A868")
        ax.text(x0, 33, f"Variational Layer {layer_i + 1}\n(trainable rotation)", ha="center", fontsize=7.5, style="italic")
        entangle_ring(x0 + 8, wire_y, "entangling CNOTs\n(range-parameterized)")

    x_meas = 86
    for y in wire_y:
        gate_box(x_meas, y, "$\\langle Z \\rangle$", w=4.6, face="#F5E9F5", edge="#8172B3")
    ax.text(x_meas, 33, "Pauli-Z\nmeasurement", ha="center", fontsize=7.5, style="italic")

    ax.text(50, 1.0, "Not a fabricated connectivity: the entangling pattern shown follows PennyLane's StronglyEntanglingLayers\n(a chain plus a wrap-around 'range' connection); exact per-layer range parameter is set by PennyLane's own template logic.", ha="center", fontsize=6.8, color="#555555", linespacing=1.4)

    save(fig, os.path.join(OUT, "fig06_quantum_circuit_diagram.png"))


def figure7():
    fig, ax = new_fig(8.5, 12)
    title(ax, "Training and Evaluation Workflow", y=98)
    subtitle(ax, "One QGNN-v4 training run, from configuration to a KEEP / INVESTIGATE / DROP decision.")

    x, w = 20, 60
    y = 90
    steps = [
        "Configuration\n(architecture, split, hyperparameters fixed by the experiment design)",
        "Seed Selection\n(42, 43, 44, 45, 46 -- or a 2-seed pilot subset, 42/43)",
        "Load Frozen GraphSAGE Checkpoint\n(per-seed, per-split -- never retrained)",
        "QGNN-v4 Forward + Backward Pass\n(Adam, lr=0.001, weight_decay=0.0001, balanced class weighting)",
        "Validation PR-AUC (every epoch)",
        "Early Stopping (patience=10)",
        "Best Checkpoint Selected\n(by validation PR-AUC only -- never by test data)",
        "Test-Set Evaluation\n(PR-AUC, ROC-AUC, F1, MCC, Brier, ECE, confusion matrix)",
        "Quantum Representation Analysis\n(Pauli-Z channel stats, where applicable)",
        "Cross-Seed Aggregation (mean +/- std across seeds)",
    ]
    for i, s in enumerate(steps):
        box(ax, x, y, w, 4.4, s, face=COLOR_DATA if i < 3 else (COLOR_QUANTUM if i in (3, 8) else COLOR_CLASSICAL), edge=COLOR_DATA_EDGE if i < 3 else (COLOR_QUANTUM_EDGE if i in (3, 8) else COLOR_CLASSICAL_EDGE), fontsize=8.5)
        down_arrow(ax, 50, y, y - 3.0)
        y -= 7.4

    box(ax, x, y, w, 4.6, "Experimental Decision", face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold")
    down_arrow(ax, 50, y, y - 3.6)
    y -= 9.6
    decisions = ["KEEP", "INVESTIGATE", "DROP"]
    dx = [18, 50, 82]
    for lbl, cx in zip(decisions, dx):
        box(ax, cx - 12, y, 24, 4.4, lbl, face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontweight="bold", fontsize=10)

    save(fig, os.path.join(OUT, "fig07_training_evaluation_workflow.png"))


def figure8():
    fig, ax = new_fig(9, 13.6)
    title(ax, "Research Experiment Roadmap", y=99)
    subtitle(ax, "Thirteen sequential investigations, each answering one controlled question before the next began.")

    stages = [
        ("Initial QGNN Benchmark", "Does a 6-qubit quantum head beat a matched-width classical control?"),
        ("Phase 1 -- Diagnostics", "Is instability caused by barren plateaus or vanishing gradients?"),
        ("Phase 2 -- Patience", "Is early stopping cutting training short?"),
        ("Phase 2b -- Output Scaling", "Does a learnable output scale fix calibration?"),
        ("Phase 2c -- LayerNorm", "Does per-example output normalization fix severity collapse?"),
        ("Phase 2d -- Seed 45", "Why does one seed reverse under LayerNorm specifically?"),
        ("Phase 3 -- Qubits / Depth", "Does more quantum capacity help?"),
        ("Stage 1 -- Representation Audit", "Does the frozen embedding itself contain enough signal?"),
        ("Stage 2 -- Projection", "Does a different classical-to-quantum bottleneck help?"),
        ("Stage 4 / 4b -- Initialization", "Does the starting point of the quantum parameters matter?"),
        ("Stage 5 -- Ansatz", "Does entanglement topology matter?"),
        ("Encoding Investigation", "Does the angle-encoding mechanism matter?"),
        ("Final Consolidation", "What do all nine factors, taken together, actually establish?"),
    ]
    x, w = 10, 80
    y = 93
    for i, (name, question) in enumerate(stages):
        is_last = i == len(stages) - 1
        box(ax, x, y, w, 4.6, name, face=COLOR_OUTPUT if is_last else COLOR_QUANTUM, edge=COLOR_OUTPUT_EDGE if is_last else COLOR_QUANTUM_EDGE, fontsize=9, fontweight="bold")
        label(ax, x + w + 1.5, y + 2.3, question, fontsize=7.6, ha="left", style="italic")
        if not is_last:
            down_arrow(ax, 15, y, y - 2.0)
        y -= 6.9

    save(fig, os.path.join(OUT, "fig08_research_roadmap.png"))


def figure_trainable():
    fig, ax = new_fig(8.5, 7.3)
    title(ax, "Frozen vs. Trainable Components (QGNN-v4 Reference)", fontsize=12)
    subtitle(ax, "Only the head on the right receives gradient updates; the encoder on the left never does.")

    # Left: frozen encoder
    box(ax, 3, 47, 33, 40, "FROZEN\n(no gradient)\n\nGraphSAGE-Full encoder\nHeteroGraphSAGE,\n2 layers, hidden_dim=128\n\n~1.2 M parameters\nrequires_grad = False",
        face=COLOR_FROZEN, edge=COLOR_FROZEN_EDGE, fontweight="bold", fontsize=9)
    label(ax, 19.5, 41, "Trained once per seed / split,\nthen frozen before any\nhead experiment begins.", fontsize=8, style="italic")

    # Arrow: embedding handed to the head
    arrow(ax, 37.5, 67, 47.5, 67)
    label(ax, 42.5, 71, "128-D\nembedding", fontsize=8)

    # Right: trainable head, stacked
    items = [
        ("Linear(128 -> 6)", "TRAINABLE  |  774 parameters", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
        ("6-qubit, 2-layer quantum circuit", "TRAINABLE  |  36 parameters", COLOR_QUANTUM, COLOR_QUANTUM_EDGE),
        ("LayerNorm(6, no affine)", "no trainable parameters  |  0", COLOR_DATA, COLOR_DATA_EDGE),
        ("Linear(6 -> 1)", "TRAINABLE  |  7 parameters", COLOR_CLASSICAL, COLOR_CLASSICAL_EDGE),
    ]
    y = 76.0
    for i, (name, sub, face, edge) in enumerate(items):
        box(ax, 48, y, 49, 10.5, f"{name}\n{sub}", face=face, edge=edge, fontsize=9, fontweight="bold")
        if i < len(items) - 1:
            down_arrow(ax, 72.5, y - 0.6, y - 3.2)
        y -= 13.5

    # Total band
    box(ax, 3, 9, 94, 16, "Trainable total (QGNN-v4 head):  774 + 36 + 0 + 7 = 817 parameters\n"
        "The ~1.2 M-parameter frozen encoder is excluded from this count\n(it is roughly 1,500x larger than the trainable head).",
        face=COLOR_OUTPUT, edge=COLOR_OUTPUT_EDGE, fontsize=9, fontweight="bold")

    label(ax, 50, 4, "Frozen-encoder isolation is asserted in the training script's run loop for every QGNN run and covered by a dedicated unit test.", fontsize=7.8, style="italic")

    save(fig, os.path.join(OUT, "fig09_trainable_vs_frozen.png"))


if __name__ == "__main__":
    figure5()
    figure6()
    figure7()
    figure8()
    figure_trainable()
