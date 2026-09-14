"""Quantum circuit construction for QGNN-v4
(PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md sections 5-7): `AngleEmbedding`
(RY) followed by a `StronglyEntanglingLayers`/`BasicEntanglerLayers`
ansatz, wrapped as a plain `nn.Module` via `qml.qnn.TorchLayer` so it
composes with ordinary PyTorch submodules exactly like any other layer in
this repo's models.

Deliberately distinct from `qgnn.py`'s v1-v3 circuit (a hand-rolled RY +
linear/ring/all-to-all CNOT chain, called directly as a QNode with a
manually-created `nn.Parameter` for its weights): this module uses
PennyLane's own standard, citable "hardware-efficient ansatz" building
blocks instead, per the plan's explicit choice. Both are legitimate,
independently useful architecture choices -- this module does not replace
or modify `qgnn.py`.

`diff_method="backprop"` is the default because, on a classical statevector
simulator, it differentiates the whole circuit analytically through the
same autograd graph PyTorch already uses -- exact, not an approximation,
and far cheaper per step than `"parameter-shift"`. Parameter-shift is
necessary only once this circuit runs on real hardware or a shot-based
(non-analytic) simulator (plan section 7/15's Phase Q7).
"""

from __future__ import annotations

import pennylane as qml
import torch.nn as nn

ANSATZ_BUILDERS = {
    "strongly_entangling": qml.StronglyEntanglingLayers,
    "basic_entangler": qml.BasicEntanglerLayers,
}


def build_quantum_layer(
    n_qubits: int,
    n_layers: int,
    ansatz: str = "strongly_entangling",
    diff_method: str = "backprop",
    device_name: str = "default.qubit",
) -> nn.Module:
    """Returns a plain `nn.Module` (`qml.qnn.TorchLayer`) mapping
    `[batch, n_qubits]` rotation angles to `[batch, n_qubits]` PauliZ
    expectation values, with its own trainable ansatz weights registered
    as ordinary PyTorch parameters (`.parameters()` picks them up, so an
    outer `torch.optim.Adam(model.parameters(), ...)` needs no changes to
    also train the circuit)."""
    if ansatz not in ANSATZ_BUILDERS:
        raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {sorted(ANSATZ_BUILDERS)}")
    ansatz_layer = ANSATZ_BUILDERS[ansatz]
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        ansatz_layer(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": ansatz_layer.shape(n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)
