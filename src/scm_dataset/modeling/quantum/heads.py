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


class HybridQuantumHeadOutputScale(nn.Module):
    """Phase 2b of the QGNN-v4 stability investigation
    (QGNN_V4_PHASE1_DIAGNOSTICS.md, QGNN_V4_PHASE2_REPORT.md): tests
    whether the instability/calibration gap found in Phase 1 is an
    output-scale / optimization-dynamics issue rather than an
    expressiveness one, by inserting a scale (and optionally bias)
    transform on the raw PauliZ output, before the SAME `Linear(n_qubits, 1)`
    output layer `HybridQuantumHead` already has:

        HybridQuantumHead:            ... -> pauliz -> Linear(n_qubits,1) -> logit
        HybridQuantumHeadOutputScale: ... -> pauliz -> alpha*pauliz[+beta] -> Linear(n_qubits,1) -> logit

    Methodological note, stated explicitly because it matters for how any
    result here should be read: `alpha`/`beta` are mathematically
    redundant with `Linear(n_qubits,1)`'s own weights/bias in the
    infinite-training-time function-class sense -- `alpha*Linear(x)+beta`
    is itself exactly representable as a DIFFERENT `Linear(n_qubits,1)`.
    This is deliberately NOT an expressiveness change (the investigation
    explicitly rules out architecture changes at this phase). What it CAN
    change is the optimization trajectory -- a different effective
    initial scale and gradient magnitude on the pre-`Linear`
    representation -- which is the actual hypothesis under test, and is
    the same reason `LayerNorm`/`BatchNorm`'s learnable affine parameters
    matter in practice despite being technically absorbable into a
    following linear layer. Any PR-AUC/calibration difference found here
    should be attributed to optimization dynamics, not added capacity.

    `trainable_scale=True` makes `alpha` (and `beta`, if `use_bias=True`)
    an `nn.Parameter`. `trainable_scale=False` fixes `alpha` as a
    non-trainable constant (Phase 2b's "fixed output scale" experiment --
    the value must be chosen without looking at test performance, per the
    investigation's own rule; this class only exposes the mechanism, the
    caller picks the value)."""

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 6,
        n_layers: int = 2,
        ansatz: str = "strongly_entangling",
        diff_method: str = "backprop",
        device_name: str = "default.qubit",
        alpha_init: float = 1.0,
        use_bias: bool = False,
        trainable_scale: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.device_name = device_name
        self.trainable_scale = trainable_scale
        self.use_bias = use_bias
        self.alpha_init = alpha_init
        self.reduce = nn.Linear(in_dim, n_qubits)
        self.quantum = build_quantum_layer(n_qubits, n_layers, ansatz=ansatz, diff_method=diff_method, device_name=device_name)
        self.out = nn.Linear(n_qubits, 1)
        if trainable_scale:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(0.0)) if use_bias else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        angles = math.pi * torch.tanh(self.reduce(h))
        q_out = self.quantum(angles).to(torch.float32)
        scaled = self.alpha * q_out
        if self.beta is not None:
            scaled = scaled + self.beta
        return self.out(scaled).squeeze(-1)

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
            "output_scale": {"trainable": self.trainable_scale, "alpha_init": self.alpha_init, "use_bias": self.use_bias},
        }


class HybridQuantumHeadLayerNorm(nn.Module):
    """Phase 2c of the QGNN-v4 stability investigation
    (QGNN_V4_PHASE2B_REPORT.md's recommended next step,
    QGNN_V4_PHASE2C_REPORT.md): replaces Phase 2b's scalar
    `alpha`/`beta` reparameterization with a genuinely data-dependent
    normalization of the raw PauliZ output, before the SAME
    `Linear(n_qubits, 1)` output layer `HybridQuantumHead` already has:

        HybridQuantumHead:              ... -> pauliz -> Linear(n_qubits,1) -> logit
        HybridQuantumHeadLayerNorm:     ... -> pauliz -> LayerNorm(n_qubits) -> Linear(n_qubits,1) -> logit

    Unlike Phase 2b's `alpha*pauliz+beta` (mathematically absorbable into
    `Linear(n_qubits,1)`'s own weights -- see `HybridQuantumHeadOutputScale`'s
    docstring), `LayerNorm` computes its normalizing statistics (mean,
    variance) from each example's own 6 PauliZ values at that forward
    pass -- input-dependent, not a fixed or globally-learned affine map.
    This IS a genuine, non-redundant change to what the head computes,
    not just a reparameterization of an already-reachable function.

    `elementwise_affine=False` (variant A) applies pure normalization,
    no learnable scale/shift. `elementwise_affine=True` (variant B) adds
    a learnable per-qubit `weight`/`bias` on top (PyTorch's default
    `LayerNorm` init: weight=1, bias=0 -- so at initialization, variant B
    starts identical to variant A before any training)."""

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 6,
        n_layers: int = 2,
        ansatz: str = "strongly_entangling",
        diff_method: str = "backprop",
        device_name: str = "default.qubit",
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.device_name = device_name
        self.elementwise_affine = elementwise_affine
        self.reduce = nn.Linear(in_dim, n_qubits)
        self.quantum = build_quantum_layer(n_qubits, n_layers, ansatz=ansatz, diff_method=diff_method, device_name=device_name)
        self.norm = nn.LayerNorm(n_qubits, elementwise_affine=elementwise_affine)
        self.out = nn.Linear(n_qubits, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        angles = math.pi * torch.tanh(self.reduce(h))
        q_out = self.quantum(angles).to(torch.float32)
        normed = self.norm(q_out)
        return self.out(normed).squeeze(-1)

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
            "normalization": {"type": "LayerNorm", "elementwise_affine": self.elementwise_affine},
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
