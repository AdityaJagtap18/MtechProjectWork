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
import torch
import torch.nn as nn

ANSATZ_BUILDERS = {
    "strongly_entangling": qml.StronglyEntanglingLayers,
    "basic_entangler": qml.BasicEntanglerLayers,
}
CUSTOM_ANSATZE = {"hardware_efficient_ring", "reduced_entanglement"}
ALL_ANSATZE = sorted(set(ANSATZ_BUILDERS) | CUSTOM_ANSATZE)


def _small_gaussian_init(tensor: torch.Tensor) -> torch.Tensor:
    """Phase 4 Stage 4 F2: mean=0, std=0.01 -- starts every trainable
    rotation parameter close to (but not exactly) zero. Kept as its own
    fixed-std strategy (rather than folded into `_gaussian_init` below) so
    every already-saved Stage 4 F2 run stays reproducible byte-for-byte
    under the exact name that produced it -- `"gaussian"` with
    `gaussian_std=0.01` (Stage 4b's G3) is numerically identical to this,
    just reached via the newer, configurable-std path."""
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.01)


def _gaussian_init(std: float):
    """Phase 4 Stage 4b Part B (QGNN_V4_PHASE4_PLAN.md): the small-Gaussian
    idea generalized to a configurable std, for the scale sweep (0.001,
    0.005, 0.010, 0.025, 0.050) motivated by Stage 4's own finding that F1
    starts at a large parameter norm (~20.8) while F2/F3 start near zero --
    this tests whether it's specifically the INITIAL SCALE, not "Gaussian
    vs. uniform" as a family, that drove Stage 4's Primary/Severity
    trade-off. Returns a callable (not applied immediately) so
    `resolve_quantum_init` can build it lazily per requested std."""
    def init(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init


def _identity_like_init(tensor: torch.Tensor) -> torch.Tensor:
    """Phase 4 Stage 4 F3: every trainable rotation parameter starts at
    exactly 0.0. For every ansatz `build_quantum_layer` supports, the
    ONLY thing `weights` parameterizes is a per-qubit-per-layer rotation
    gate (`Rot`/`RY`/`RZ`/`RX`, depending on ansatz) -- `Rot(0,0,0)`,
    `RY(0)`, `RZ(0)`, and `RX(0)` are each EXACTLY the single-qubit
    identity operator, so this is a real (not approximate) identity
    initialization of every rotation gate.

    IMPORTANT LIMITATION, stated explicitly per this phase's own
    instruction not to overclaim: the entangling CNOT pattern in every
    ansatz here (ring, chain, or StronglyEntanglingLayers'/
    BasicEntanglerLayers' own built-in pattern) is a FIXED, unparameterized
    part of the circuit -- it is applied unconditionally regardless of
    `weights`' value, and CNOT is not the identity. So this initialization
    makes the ROTATION gates exactly identity at step 0, but the full
    circuit (rotations + entanglement) is NOT the identity transformation
    even at initialization -- entanglement between qubits still occurs on
    the very first forward pass. "identity-like" describes the trainable
    part of the circuit only, not the whole quantum layer."""
    return torch.nn.init.zeros_(tensor)


QUANTUM_INIT_STRATEGIES = {
    # None -> PennyLane's own TorchLayer default when init_method isn't
    # given: torch.nn.init.uniform_(tensor, a=0, b=2*pi). Not reimplemented
    # here -- passing None through to TorchLayer IS the F1 control, so it
    # is guaranteed to be byte-for-byte what every prior phase already ran.
    "default": None,
    "small_gaussian": _small_gaussian_init,
    "identity_like": _identity_like_init,
}


def resolve_quantum_init(quantum_init: str, gaussian_std: float | None = None):
    """`quantum_init="gaussian"` is the one strategy that takes a
    parameter (`gaussian_std`, required in that case, ignored/must be None
    otherwise) -- every other strategy name is a fixed, parameterless
    lookup in `QUANTUM_INIT_STRATEGIES`."""
    if quantum_init == "gaussian":
        if gaussian_std is None:
            raise ValueError("quantum_init='gaussian' requires gaussian_std to be set")
        return _gaussian_init(gaussian_std)
    if gaussian_std is not None:
        raise ValueError(f"gaussian_std is only meaningful for quantum_init='gaussian', got quantum_init={quantum_init!r}")
    if quantum_init not in QUANTUM_INIT_STRATEGIES:
        raise ValueError(f"unknown quantum_init={quantum_init!r}, expected 'gaussian' or one of {sorted(QUANTUM_INIT_STRATEGIES)}")
    return QUANTUM_INIT_STRATEGIES[quantum_init]


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
    quantum_init: str = "default",
    gaussian_std: float | None = None,
    data_reuploading: bool = False,
) -> nn.Module:
    """Returns a plain `nn.Module` (`qml.qnn.TorchLayer`) mapping
    `[batch, n_qubits]` rotation angles to `[batch, n_qubits]` PauliZ
    expectation values, with its own trainable ansatz weights registered
    as ordinary PyTorch parameters (`.parameters()` picks them up, so an
    outer `torch.optim.Adam(model.parameters(), ...)` needs no changes to
    also train the circuit).

    `quantum_init` (Phase 4 Stage 4, QGNN_V4_PHASE4_PLAN.md): "default"
    (F1, the control -- passes `init_method=None` through to `TorchLayer`
    unchanged, so every prior phase's runs are reproduced exactly),
    "small_gaussian" (F2, fixed std=0.01), "identity_like" (F3), or
    "gaussian" (Stage 4b's configurable-std sweep -- requires
    `gaussian_std`). Only ever changes the INITIAL VALUES of the existing
    `weights` parameter -- same shape, same `.parameters()` count, same
    circuit -- see `QUANTUM_INIT_STRATEGIES`/`resolve_quantum_init`.

    `data_reuploading` (Quantum Encoding Investigation, E4,
    QGNN_V4_QUANTUM_ENCODING_RESULTS.md): `False` (default, E0-E3) keeps
    the existing single `AngleEmbedding` before the full multi-layer
    ansatz. `True` re-encodes the SAME `inputs` angles before EACH
    variational layer instead -- `weights`' shape is completely
    unchanged (still `(n_layers, n_qubits, ...)`), only sliced one layer
    at a time per iteration, so this adds ZERO trainable parameters
    despite repeating the (parameter-free) AngleEmbedding operation
    `n_layers` times instead of once."""
    if ansatz not in ALL_ANSATZE:
        raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {ALL_ANSATZE}")
    init_method = resolve_quantum_init(quantum_init, gaussian_std)
    dev = qml.device(device_name, wires=n_qubits)

    if ansatz in ANSATZ_BUILDERS:
        ansatz_layer = ANSATZ_BUILDERS[ansatz]

        if data_reuploading:
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def circuit(inputs, weights):
                for layer in range(n_layers):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    ansatz_layer(weights[layer : layer + 1], wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        else:
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                ansatz_layer(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": ansatz_layer.shape(n_layers, n_qubits)}
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    if ansatz == "hardware_efficient_ring":
        pairs = _ring_pairs(n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            if not data_reuploading:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                if data_reuploading:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for q in range(n_qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    if ansatz == "reduced_entanglement":
        pairs = _chain_pairs(n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            if not data_reuploading:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                if data_reuploading:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for q in range(n_qubits):
                    qml.RY(weights[layer, q], wires=q)
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    raise AssertionError(f"unreachable: {ansatz!r} in ALL_ANSATZE but not handled")
