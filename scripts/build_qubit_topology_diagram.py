#!/usr/bin/env python3
"""Qubit connectivity/topology diagram for the QGNN-v4 reference circuit --
complementary to the gate-level circuit diagram (Appendix B, Figure B1),
which shows gates in time order on horizontal wires. This shows the same
circuit's actual qubit-to-qubit entangling structure as a graph: qubits as
nodes, CNOT gates as edges, which qubit reaches which within the circuit.

Edges are read directly off the real decomposed circuit tape (PennyLane's
own qml.workflow.construct_tape at device level) via
build_quantum_layer(n_qubits=6, n_layers=2, ansatz='strongly_entangling')
-- not guessed or hand-drawn. StronglyEntanglingLayers' internal "range"
parameter (which sets each layer's entangling offset) is whatever
PennyLane computed for this qubit/layer count; this script reports
whatever it finds, it does not assume range=1.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scm_dataset.modeling.quantum.circuit import build_quantum_layer  # noqa: E402

OUT_DIR = os.path.join(ROOT, "report_assets", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

N_QUBITS = 6
N_LAYERS = 2

torch.manual_seed(0)
layer = build_quantum_layer(n_qubits=N_QUBITS, n_layers=N_LAYERS, ansatz="strongly_entangling")
qnode = layer.qnode
weights = layer.weights.detach()
inputs = torch.linspace(-1.0, 1.0, N_QUBITS)

tape = qml.workflow.construct_tape(qnode, level="device")(inputs, weights)
cnots = [tuple(op.wires) for op in tape.operations if op.name == "CNOT"]

# StronglyEntanglingLayers applies n_qubits CNOTs per layer -> split the
# flat CNOT list back into its layers positionally.
per_layer = len(cnots) // N_LAYERS
layers = [cnots[i * per_layer : (i + 1) * per_layer] for i in range(N_LAYERS)]
print("CNOT pairs per layer:", layers)

fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
G = nx.MultiDiGraph()
G.add_nodes_from(range(N_QUBITS))
pos = nx.circular_layout(G)
# put qubit 0 at the top, going clockwise, matching the gate-diagram's q0..q5 order
angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, N_QUBITS, endpoint=False)
pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

nx.draw_networkx_nodes(G, pos, node_size=1600, node_color="#4C72B0", ax=ax)
nx.draw_networkx_labels(
    G, pos, labels={i: f"q{i}" for i in range(N_QUBITS)}, font_color="white", font_weight="bold", font_size=13, ax=ax
)

layer_styles = [
    {"color": "#DD8452", "style": "solid", "connectionstyle": "arc3,rad=0.08", "label": "Layer 1 CNOTs"},
    {"color": "#55A868", "style": "dashed", "connectionstyle": "arc3,rad=0.22", "label": "Layer 2 CNOTs"},
]
for layer_idx, pairs in enumerate(layers):
    style = layer_styles[layer_idx % len(layer_styles)]
    for (c, t) in pairs:
        ax.annotate(
            "",
            xy=pos[int(t)],
            xytext=pos[int(c)],
            arrowprops=dict(
                arrowstyle="-|>",
                color=style["color"],
                linestyle=style["style"],
                linewidth=2.0,
                connectionstyle=style["connectionstyle"],
                shrinkA=22,
                shrinkB=22,
                mutation_scale=18,
            ),
        )

legend_handles = [
    plt.Line2D([0], [0], color=s["color"], linestyle=s["style"], linewidth=2.5, label=s["label"])
    for s in layer_styles[:N_LAYERS]
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=10)
ax.set_title(
    "QGNN-v4 Reference Circuit -- Qubit Connectivity\n(StronglyEntanglingLayers, 6 qubits, 2 layers, control→target arrows)",
    fontsize=11.5,
    pad=28,
)
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.35, 1.45)
ax.axis("off")

out_path = os.path.join(OUT_DIR, "fig_appendixB4_qubit_topology_pennylane.png")
fig.savefig(out_path, bbox_inches="tight")
print("Wrote", out_path)
