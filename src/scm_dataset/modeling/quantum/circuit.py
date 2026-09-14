"""Quantum circuit construction for QGNN-v4
(PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md sections 5-7, Phase 3 ansatz
ablation): `AngleEmbedding` (RY) followed by one of four ansatz choices,
wrapped as a plain `nn.Module` via `qml.qnn.TorchLayer` so it composes
with ordinary PyTorch submodules exactly like any other layer in this
repo's models.

    strongly_entangling      PennyLane's StronglyEntanglingLayers (Phase
                              1-2d reference: 3 rotation params/qubit/layer
                              + a range-parameterized entangling pattern)
    basic_entangler          PennyLane's BasicEntanglerLayers (1 rotation
                              param/qubit/layer, default RX + ring CNOT --
                              already supported, not part of Phase 3's
                              planned matrix)
    hardware_efficient_ring  Phase 3 Ansatz 2: RY then RZ per qubit per
                              layer (2 params/qubit/layer) + a CNOT RING
                              (wraparound included)
    reduced_entanglement     Phase 3 Ansatz 3: RY only per qubit per layer
                              (1 param/qubit/layer) + a CNOT CHAIN (linear,
                              no wraparound) -- identical in structure to
                              qgnn.py's own v1-v3 ansatz, though qgnn.py
                              never modified here and this is a separate
                              circuit built on the frozen-embedding-direct
                              pipeline, not a reuse of that module's code.

Deliberately distinct from `qgnn.py`'s v1-v3 circuit (a hand-rolled RY +
linear/ring/all-to-all CNOT chain, called directly as a QNode with a
manually-created `nn.Parameter` for its weights): this module uses
PennyLane's own standard, citable "hardware-efficient ansatz" building
blocks for the first two, and its own hand-built (but PennyLane-gate-based)
circuits for the Phase 3 additions. Both are legitimate, independently
useful architecture choices -- this module does not replace or modify
`qgnn.py`.

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
CUSTOM_ANSATZE = {"hardware_efficient_ring", "reduced_entanglement"}
ALL_ANSATZE = sorted(set(ANSATZ_BUILDERS) | CUSTOM_ANSATZE)


def _ring_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Linear chain plus one wrap-around edge: q0->q1, ..., q_{n-2}->q_{n-1}, q_{n-1}->q0."""
    pairs = [(q, q + 1) for q in range(n_qubits - 1)]
    pairs.append((n_qubits - 1, 0))
    return pairs


def _chain_pairs(n_qubits: int) -> list[tuple[int, int]]:
    return [(q, q + 1) for q in range(n_qubits - 1)]


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
    if ansatz not in ALL_ANSATZE:
        raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {ALL_ANSATZE}")
    dev = qml.device(device_name, wires=n_qubits)

    if ansatz in ANSATZ_BUILDERS:
        ansatz_layer = ANSATZ_BUILDERS[ansatz]

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            ansatz_layer(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": ansatz_layer.shape(n_layers, n_qubits)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    if ansatz == "hardware_efficient_ring":
        pairs = _ring_pairs(n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    if ansatz == "reduced_entanglement":
        pairs = _chain_pairs(n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(weights[layer, q], wires=q)
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    raise AssertionError(f"unreachable: {ansatz!r} in ALL_ANSATZE but not handled")
