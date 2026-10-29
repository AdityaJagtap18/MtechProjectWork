#!/usr/bin/env python3
"""Generates ground-truth quantum circuit diagrams and specs directly from
this project's own production code (`src/scm_dataset/modeling/quantum/
circuit.py::build_quantum_layer`), using PennyLane's own `qml.draw_mpl` and
`qml.specs` -- not hand-drawn approximations.

Covers exactly the circuit configurations this project actually ran:
  - the standing QGNN-v4 reference circuit (6 qubits, 2 layers,
    strongly_entangling) used for every headline Primary/Severity result
    in the report (configs/qgnn_v4.yaml, configs/qgnn_v4_severity.yaml).
  - the two alternate ansatz variants from the Phase 4 Stage 5 ablation
    (hardware_efficient_ring, reduced_entanglement), reported in
    Section 13.6 / QGNN_V4_ABLATION_TABLE.csv.

Output: PNGs in report_assets/diagrams/ and a JSON of qml.specs()-derived
details for each, consumed by scripts/append_pennylane_appendix.py.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scm_dataset.modeling.quantum.circuit import build_quantum_layer  # noqa: E402

OUT_DIR = os.path.join(ROOT, "report_assets", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

N_QUBITS = 6
N_LAYERS = 2

CONFIGS = [
    {
        "key": "reference_strongly_entangling",
        "ansatz": "strongly_entangling",
        "title": "QGNN-v4 Reference Circuit -- StronglyEntanglingLayers (6 qubits, 2 layers)",
        "fig_name": "fig_appendixB1_reference_circuit_pennylane.png",
    },
    {
        "key": "hardware_efficient_ring",
        "ansatz": "hardware_efficient_ring",
        "title": "Ablation variant -- hardware_efficient_ring (RY+RZ, CNOT ring; 6 qubits, 2 layers)",
        "fig_name": "fig_appendixB2_hardware_efficient_ring_pennylane.png",
    },
    {
        "key": "reduced_entanglement",
        "ansatz": "reduced_entanglement",
        "title": "Ablation variant -- reduced_entanglement (RY only, CNOT chain; 6 qubits, 2 layers)",
        "fig_name": "fig_appendixB3_reduced_entanglement_pennylane.png",
    },
]

details = {}

for cfg in CONFIGS:
    torch.manual_seed(0)
    layer = build_quantum_layer(n_qubits=N_QUBITS, n_layers=N_LAYERS, ansatz=cfg["ansatz"])
    qnode = layer.qnode
    weights = layer.weights.detach()
    inputs = torch.linspace(-1.0, 1.0, N_QUBITS)

    # level="device" decomposes PennyLane's own templates (AngleEmbedding,
    # StronglyEntanglingLayers, ...) down to elementary gates (RY/Rot/CNOT)
    # -- the actual gate sequence PennyLane executes, not a template box.
    fig, ax = qml.draw_mpl(qnode, decimals=2, style="default", level="device")(inputs, weights)
    ax.set_title(cfg["title"], fontsize=11, pad=14)
    fig.savefig(os.path.join(OUT_DIR, cfg["fig_name"]), dpi=200, bbox_inches="tight")
    plt.close(fig)

    specs = qml.specs(qnode, level="device")(inputs, weights)
    resources = specs.resources
    details[cfg["key"]] = {
        "ansatz": cfg["ansatz"],
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "num_wires": specs.num_device_wires,
        "num_gates": resources.num_gates,
        "depth": resources.depth,
        "gate_types": dict(resources.gate_types),
        "trainable_params": int(weights.numel()),
        "device": "default.qubit",
        "diff_method": "backprop",
        "fig_name": cfg["fig_name"],
    }
    print(f"{cfg['key']}: {details[cfg['key']]}")

with open(os.path.join(OUT_DIR, "appendix_b_circuit_specs.json"), "w") as f:
    json.dump(details, f, indent=2)

print("Wrote", os.path.join(OUT_DIR, "appendix_b_circuit_specs.json"))
