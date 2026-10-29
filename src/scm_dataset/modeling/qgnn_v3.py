"""QGNN-v3 controlled architecture study -- entanglement topology and
input-encoding variants of the depth-3 QGNN selected in
`QGNN_V2_DEPTH_BENCHMARK.md`. Purely additive: does not modify `qgnn.py`,
`qgnn_v2.py`, or `qgnn_v2_reupload.py`. Variant A (baseline) is not
reimplemented here at all -- it is `qgnn.build_qgnn_model` run unchanged,
directly, exactly as the plan requires ("Run the current QGNN-v2 depth-3
configuration unchanged"). Variants B/C reuse a small shared circuit
builder parameterized only by which CNOT pairs it entangles, verified by
test to reduce EXACTLY to `qgnn.py`'s own circuit when given the linear
pairing. Variant D reuses `qgnn._make_qnode` (the baseline circuit
builder itself, imported directly, completely unmodified) and changes
only the classical angle computation upstream of it.

    B (ring entanglement):      q0-q1-...-q7-q0 per layer (linear + one wrap-around edge)
    C (all-to-all entanglement): every qubit pair connected per layer
    D (trainable input scaling): angle = tanh(a*x + b) * pi, a/b trainable,
                                  initialized a=1/b=0 (identical to the
                                  fixed baseline transform at init)

All three keep the exact same MLP head shape as `qgnn.QGNN`
(`Linear(n_qubits, mlp_hidden) -> ReLU -> Linear(mlp_hidden, 1)`, no
sigmoid) and the exact same `[batch, n_qubits] -> [batch]` forward
convention, so every existing `qgnn_v2.py` utility
(`train_v2_head`/`evaluate_v2`/`generate_predictions_v2`) works with them
unchanged.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn

from .qgnn import DEFAULT_DEVICE, FALLBACK_DEVICE, _make_qnode


def linear_entangler_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """The existing baseline's own entanglement pattern (`qgnn.py`'s
    `for q in range(n_qubits - 1): CNOT(q, q+1)`), exposed here only so
    tests can prove variant B/C's shared builder reduces to the baseline
    exactly when given this pairing -- not used to build variant A itself,
    which reuses `qgnn.build_qgnn_model` directly."""
    return [(q, q + 1) for q in range(n_qubits - 1)]


def ring_entangler_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Linear chain plus one wrap-around edge: q0->q1, ..., q_{n-2}->q_{n-1}, q_{n-1}->q0."""
    pairs = linear_entangler_pairs(n_qubits)
    pairs.append((n_qubits - 1, 0))
    return pairs


def all_to_all_entangler_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Every qubit pair, one CNOT each, consistent ascending-index
    convention (source always the lower index): C(n_qubits, 2) pairs."""
    return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def _make_qnode_with_entangler(n_qubits: int, n_layers: int, device_name: str, entangler_pairs: list[tuple[int, int]]):
    dev = qml.device(device_name, wires=n_qubits)
    diff_method = "adjoint" if "lightning" in device_name else "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(angles, weights):
        qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q], wires=q)
            for c, t in entangler_pairs:
                qml.CNOT(wires=[c, t])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit, dev


class _EntanglementVariantCircuit(nn.Module):
    """Shared implementation for variants B and C -- identical to
    `qgnn.QuantumCircuitLayer` except the CNOT pairing is a parameter
    instead of the hardcoded linear chain."""

    def __init__(self, n_qubits: int, n_layers: int, device_name: str, entangler_pairs: list[tuple[int, int]], topology_name: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler_pairs = entangler_pairs
        self.topology_name = topology_name
        try:
            self._circuit, self._device = _make_qnode_with_entangler(n_qubits, n_layers, device_name, entangler_pairs)
            self.device_name = device_name
        except Exception as exc:  # pragma: no cover
            print(f"_EntanglementVariantCircuit ({topology_name}): device {device_name!r} unavailable ({exc!r}), falling back to {FALLBACK_DEVICE!r}")
            self._circuit, self._device = _make_qnode_with_entangler(n_qubits, n_layers, FALLBACK_DEVICE, entangler_pairs)
            self.device_name = FALLBACK_DEVICE
        self.weights = nn.Parameter((torch.rand(n_layers, n_qubits) * 2 - 1) * math.pi)

    def forward(self, standardized_x: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(standardized_x) * math.pi
        outputs = self._circuit(angles, self.weights)
        return torch.stack(outputs, dim=-1).to(torch.float32)


def _resource_summary(quantum, total_params: int, extra: dict) -> dict:
    return {
        "qubits": quantum.n_qubits,
        "variational_layers": quantum.n_layers,
        "trainable_quantum_parameters": quantum.weights.numel(),
        "total_trainable_parameters": total_params,
        "cnot_count_per_layer": len(quantum.entangler_pairs) if hasattr(quantum, "entangler_pairs") else 7,
        "cnot_count_total": (len(quantum.entangler_pairs) if hasattr(quantum, "entangler_pairs") else 7) * quantum.n_layers,
        "observable": "PauliZ (one per qubit)",
        "backend": quantum.device_name,
        **extra,
    }


class QGNNRingEntanglement(nn.Module):
    """Variant B: entanglement changed to a ring (linear chain + one
    wrap-around edge); everything else identical to the baseline."""

    def __init__(self, n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = _EntanglementVariantCircuit(n_qubits, n_layers, device_name, ring_entangler_pairs(n_qubits), "ring")
        self.mlp = nn.Sequential(nn.Linear(n_qubits, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.quantum(x)).squeeze(-1)

    def quantum_resource_summary(self) -> dict:
        return _resource_summary(self.quantum, sum(p.numel() for p in self.parameters()), {
            "entanglement_pattern": "ring (linear chain + wrap-around edge q_{n-1} -> q_0)",
            "encoding": "AngleEmbedding, rotation=Y, angle = tanh(x) * pi (unchanged from baseline)",
        })


class QGNNAllToAllEntanglement(nn.Module):
    """Variant C: entanglement changed to all-to-all (every qubit pair,
    one CNOT each); everything else identical to the baseline."""

    def __init__(self, n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = _EntanglementVariantCircuit(n_qubits, n_layers, device_name, all_to_all_entangler_pairs(n_qubits), "all_to_all")
        self.mlp = nn.Sequential(nn.Linear(n_qubits, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.quantum(x)).squeeze(-1)

    def quantum_resource_summary(self) -> dict:
        return _resource_summary(self.quantum, sum(p.numel() for p in self.parameters()), {
            "entanglement_pattern": "all-to-all (every qubit pair, ascending-index CNOT convention)",
            "encoding": "AngleEmbedding, rotation=Y, angle = tanh(x) * pi (unchanged from baseline)",
        })


class QuantumCircuitLayerTrainableScaling(nn.Module):
    """Variant D: entanglement/circuit are `qgnn._make_qnode` itself,
    imported and reused completely unmodified -- the ONLY difference from
    baseline is the classical angle computation upstream of the circuit
    (trainable per-feature scale `a` and shift `b`, initialized to
    a=1/b=0 so this is numerically IDENTICAL to the baseline's fixed
    `tanh(x) * pi` at initialization -- verified by test)."""

    def __init__(self, n_qubits: int, n_layers: int = 3, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        try:
            self._circuit, self._device = _make_qnode(n_qubits, n_layers, device_name)
            self.device_name = device_name
        except Exception as exc:  # pragma: no cover
            print(f"QuantumCircuitLayerTrainableScaling: device {device_name!r} unavailable ({exc!r}), falling back to {FALLBACK_DEVICE!r}")
            self._circuit, self._device = _make_qnode(n_qubits, n_layers, FALLBACK_DEVICE)
            self.device_name = FALLBACK_DEVICE
        self.weights = nn.Parameter((torch.rand(n_layers, n_qubits) * 2 - 1) * math.pi)
        self.scale_a = nn.Parameter(torch.ones(n_qubits))
        self.scale_b = nn.Parameter(torch.zeros(n_qubits))

    def forward(self, standardized_x: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(self.scale_a * standardized_x + self.scale_b) * math.pi
        outputs = self._circuit(angles, self.weights)
        return torch.stack(outputs, dim=-1).to(torch.float32)


class QGNNTrainableScaling(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = QuantumCircuitLayerTrainableScaling(n_qubits, n_layers, device_name)
        self.mlp = nn.Sequential(nn.Linear(n_qubits, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.quantum(x)).squeeze(-1)

    def quantum_resource_summary(self) -> dict:
        added_scaling_params = self.quantum.scale_a.numel() + self.quantum.scale_b.numel()
        return {
            "qubits": self.n_qubits,
            "variational_layers": self.quantum.n_layers,
            "trainable_quantum_parameters": self.quantum.weights.numel(),
            "added_scaling_parameters": added_scaling_params,
            "total_trainable_parameters": sum(p.numel() for p in self.parameters()),
            "cnot_count_per_layer": self.n_qubits - 1,
            "cnot_count_total": (self.n_qubits - 1) * self.quantum.n_layers,
            "observable": "PauliZ (one per qubit)",
            "entanglement_pattern": "linear CNOT chain (wire i -> i+1) -- unchanged from baseline",
            "encoding": "AngleEmbedding, rotation=Y, angle = tanh(a*x + b) * pi, a/b trainable per-feature (init a=1, b=0)",
            "backend": self.quantum.device_name,
        }


def build_qgnn_ring_model(n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE) -> QGNNRingEntanglement:
    return QGNNRingEntanglement(n_qubits, n_layers, mlp_hidden, device_name)


def build_qgnn_all_to_all_model(n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE) -> QGNNAllToAllEntanglement:
    return QGNNAllToAllEntanglement(n_qubits, n_layers, mlp_hidden, device_name)


def build_qgnn_trainable_scaling_model(n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE) -> QGNNTrainableScaling:
    return QGNNTrainableScaling(n_qubits, n_layers, mlp_hidden, device_name)
