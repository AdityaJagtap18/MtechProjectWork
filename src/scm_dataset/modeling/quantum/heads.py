"""Quantum and matched-capacity-classical heads for QGNN-v4
(PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md sections 5, 7, 10): both consume the
SAME frozen GraphSAGE-Full supplier embedding (raw `hidden_dim`-dim,
extracted via `graph_embedding_reduction.extract_supplier_embeddings`,
reused unchanged -- no PCA) and reduce it with their OWN trainable
bottleneck of identical width, so the RQ-Q3 comparison isolates "what
happens inside the bottleneck" (a quantum circuit vs. a plain ReLU) rather
than differing input representations.

    HybridQuantumHead:            Linear(d, n_qubits) -> pi*tanh -> quantum circuit -> Linear(n_qubits, 1)
    MatchedCapacityClassicalHead: Linear(d, n_qubits) -> ReLU     -> Linear(n_qubits, 1)   (RQ-Q3 control)

No sigmoid in either -- raw logits, same `BCEWithLogitsLoss` convention as
every other model in this repo (`graphsage.py`, `qgnn.py`, `qgnn_v2.py`).
`pi * tanh(...)` bounds the encoding to a valid `[-pi, pi]` rotation range
regardless of how large the frozen embedding's magnitude gets (it is not
naturally bounded), mirroring `qgnn.py`'s own `tanh(x) * pi` convention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .circuit import build_quantum_layer


class HybridQuantumHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 6,
        n_layers: int = 2,
        ansatz: str = "strongly_entangling",
        diff_method: str = "backprop",
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.device_name = device_name
        self.reduce = nn.Linear(in_dim, n_qubits)
        self.quantum = build_quantum_layer(n_qubits, n_layers, ansatz=ansatz, diff_method=diff_method, device_name=device_name)
        self.out = nn.Linear(n_qubits, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """`h`: [batch, in_dim] frozen supplier embedding. Returns raw
        logits, shape [batch]."""
        angles = math.pi * torch.tanh(self.reduce(h))
        q_out = self.quantum(angles).to(torch.float32)
        return self.out(q_out).squeeze(-1)

    def quantum_resource_summary(self) -> dict:
        return {
            "qubits": self.n_qubits,
            "variational_layers": self.n_layers,
            "ansatz": self.ansatz,
            "trainable_quantum_parameters": sum(p.numel() for p in self.quantum.parameters()),
            "total_trainable_parameters": sum(p.numel() for p in self.parameters()),
            "observable": "PauliZ (one per qubit)",
            "encoding": "AngleEmbedding, rotation=Y, angle = pi * tanh(Linear(h))",
            "backend": self.device_name,
        }


class MatchedCapacityClassicalHead(nn.Module):
    """RQ-Q3 fairness control (plan section 10): identical bottleneck
    width and forward shape to `HybridQuantumHead`, with the quantum
    circuit replaced by a plain `ReLU` -- isolates "does bottlenecking to
    n_qubits dims help/hurt" from "is the thing inside the bottleneck
    specifically quantum"."""

    def __init__(self, in_dim: int, n_qubits: int = 6):
        super().__init__()
        self.n_qubits = n_qubits
        self.reduce = nn.Linear(in_dim, n_qubits)
        self.activation = nn.ReLU()
        self.out = nn.Linear(n_qubits, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation(self.reduce(h))).squeeze(-1)
