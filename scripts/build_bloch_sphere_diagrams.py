#!/usr/bin/env python3
"""Per-qubit Bloch-sphere diagrams for the QGNN-v4 reference circuit,
computed from the actual production code
(src/scm_dataset/modeling/quantum/circuit.py) rather than illustrative
example states.

Two states are shown, both read via qml.density_matrix on the real
6-qubit device:
  - "encoding": right after AngleEmbedding (RY), before any entangling
    layer -- each qubit is still in a pure state, so its Bloch vector has
    length 1 and lies in the X-Z plane (RY has no Y-component); the
    vector's angle directly reflects that qubit's classical input feature.
  - "full": after the complete 2-layer StronglyEntanglingLayers circuit --
    each qubit's REDUCED single-qubit state is generally mixed (Bloch
    vector length < 1) precisely because the entangling CNOTs correlate it
    with the other 5 qubits; shorter arrows here are a direct, real
    signature of entanglement, not a plotting artifact.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from qutip import Bloch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scm_dataset.modeling.quantum.circuit import build_quantum_layer  # noqa: E402

OUT_DIR = os.path.join(ROOT, "report_assets", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

N_QUBITS = 6
N_LAYERS = 2

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.real(np.trace(rho @ PAULI_X)),
            np.real(np.trace(rho @ PAULI_Y)),
            np.real(np.trace(rho @ PAULI_Z)),
        ]
    )


torch.manual_seed(0)
layer = build_quantum_layer(n_qubits=N_QUBITS, n_layers=N_LAYERS, ansatz="strongly_entangling")
dev = layer.qnode.device
weights = layer.weights.detach()
inputs = torch.linspace(-1.0, 1.0, N_QUBITS)


@qml.qnode(dev, interface="torch")
def encoding_state(inputs):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    return [qml.density_matrix(wires=[i]) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def full_circuit_state(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.density_matrix(wires=[i]) for i in range(N_QUBITS)]


def render_panel(rhos, title, fig_name):
    fig = plt.figure(figsize=(12, 8), dpi=200)
    for i, rho_t in enumerate(rhos):
        rho = rho_t.detach().numpy()
        vec = bloch_vector(rho)
        purity = np.linalg.norm(vec)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        b = Bloch(fig=fig, axes=ax)
        b.vector_color = ["#d6336c"]
        b.vector_width = 4
        b.point_color = ["#d6336c"]
        b.add_vectors(vec)
        b.render()
        ax.set_title(f"qubit {i}  (|r| = {purity:.2f})", fontsize=11, y=1.12)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(hspace=0.45, top=0.86)
    out_path = os.path.join(OUT_DIR, fig_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)


rho_encoding = encoding_state(inputs)
render_panel(
    rho_encoding,
    "QGNN-v4 -- Per-Qubit Bloch Vectors After RY Angle Encoding (pure states, before entanglement)",
    "fig_appendixB5_bloch_encoding_pennylane.png",
)

rho_full = full_circuit_state(inputs, weights)
render_panel(
    rho_full,
    "QGNN-v4 -- Per-Qubit Bloch Vectors After the Full 2-Layer Circuit (reduced states; |r|<1 signals entanglement)",
    "fig_appendixB6_bloch_full_circuit_pennylane.png",
)
