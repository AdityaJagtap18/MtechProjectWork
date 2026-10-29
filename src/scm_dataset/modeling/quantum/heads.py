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
    starts identical to variant A before any training).

    Phase 4 Stage 2 (QGNN_V4_PHASE4_PLAN.md Track B) adds three OPTIONAL,
    backward-compatible knobs on the INPUT side only -- the post-quantum
    LayerNorm/output stage above is untouched by all of them, per Stage
    2's own "keep the existing post-quantum LayerNorm behavior unchanged"
    rule:

        pre_projection_norm=False, projection_type="linear" (both
        defaults): IDENTICAL to every prior phase's head -- this is
        Stage 2's B1 control, not a new code path.

        pre_projection_norm=True (B3): LayerNorm(in_dim) applied to `h`
        itself, before `reduce`.

        projection_type="nonlinear" (B2): replaces the single
        `Linear(in_dim, n_qubits)` bottleneck with `Linear(in_dim,
        projection_hidden_dim) -> GELU -> Linear(projection_hidden_dim,
        n_qubits)` -- a small, justified nonlinear projection, not an
        arbitrarily deep MLP.

    B4 (PCA-informed projection) needs no new code here: it is this SAME
    class called with `in_dim=pca_components` on an already PCA-reduced
    (train-fit-only, via the existing `qgnn_v2.build_v2_prepared`/
    `graph_embedding_reduction.fit_embedding_pca`) embedding frame --
    the reduction happens upstream of the head, not inside it.

    Phase 4 Stage 4 (QGNN_V4_PHASE4_PLAN.md Track F) adds one more
    OPTIONAL, backward-compatible knob: `quantum_init` controls only the
    INITIAL VALUES of the quantum circuit's own trainable parameters
    (`build_quantum_layer`/`circuit.QUANTUM_INIT_STRATEGIES`) -- same
    shape, same parameter count, same circuit either way.
    `quantum_init="default"` (the default) is byte-for-byte what every
    prior phase already ran.

    Phase 4 Stage 4b adds `gaussian_std`: only meaningful (and required)
    when `quantum_init="gaussian"`, for the initialization-scale sweep
    that generalizes Stage 4's fixed-std F2 ("small_gaussian", std=0.01)
    to any std -- see `circuit._gaussian_init`.

    Quantum Encoding Investigation (QGNN_V4_QUANTUM_ENCODING_RESULTS.md)
    adds three more OPTIONAL, backward-compatible knobs, all on the
    classical-to-quantum ENCODING step specifically (the "angle = pi *
    tanh(reduce(h))" line), not the projection or circuit topology:

        encoding_type="tanh" (default, E0): the existing bounded
        nonlinear mapping. encoding_type="clip": angle = encoding_scale *
        clamp(reduce(h), -1, 1) -- a bounded LINEAR mapping (E3), isolating
        the effect of tanh's nonlinear compression from the angular scale
        itself.

        encoding_scale=math.pi (default, E0): the existing pi multiplier.
        E1 uses 0.5*pi, E2 uses 2*pi -- everything else about the mapping
        (tanh vs. clip) stays whatever encoding_type already set.

        data_reuploading=False (default, E0): the existing single
        AngleEmbedding before the full variational circuit. True (E4)
        re-encodes the SAME angles before EACH variational layer instead
        (`circuit.build_quantum_layer`'s `data_reuploading` flag) --
        purely a circuit-construction change; AngleEmbedding itself has no
        trainable parameters, so this adds zero trainable parameters
        despite repeating the encoding operation `n_layers` times."""

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 6,
        n_layers: int = 2,
        ansatz: str = "strongly_entangling",
        diff_method: str = "backprop",
        device_name: str = "default.qubit",
        elementwise_affine: bool = False,
        projection_type: str = "linear",
        projection_hidden_dim: int = 32,
        pre_projection_norm: bool = False,
        quantum_init: str = "default",
        gaussian_std: float | None = None,
        encoding_type: str = "tanh",
        encoding_scale: float = math.pi,
        data_reuploading: bool = False,
    ):
        super().__init__()
        if projection_type not in ("linear", "nonlinear"):
            raise ValueError(f"unknown projection_type={projection_type!r}, expected 'linear' or 'nonlinear'")
        if encoding_type not in ("tanh", "clip"):
            raise ValueError(f"unknown encoding_type={encoding_type!r}, expected 'tanh' or 'clip'")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.device_name = device_name
        self.elementwise_affine = elementwise_affine
        self.projection_type = projection_type
        self.projection_hidden_dim = projection_hidden_dim
        self.pre_projection_norm = pre_projection_norm
        self.quantum_init = quantum_init
        self.gaussian_std = gaussian_std
        self.encoding_type = encoding_type
        self.encoding_scale = encoding_scale
        self.data_reuploading = data_reuploading

        self.pre_norm = nn.LayerNorm(in_dim) if pre_projection_norm else None
        if projection_type == "linear":
            self.reduce = nn.Linear(in_dim, n_qubits)
        else:
            self.reduce = nn.Sequential(
                nn.Linear(in_dim, projection_hidden_dim), nn.GELU(), nn.Linear(projection_hidden_dim, n_qubits)
            )
        self.quantum = build_quantum_layer(n_qubits, n_layers, ansatz=ansatz, diff_method=diff_method, device_name=device_name, quantum_init=quantum_init, gaussian_std=gaussian_std, data_reuploading=data_reuploading)
        self.norm = nn.LayerNorm(n_qubits, elementwise_affine=elementwise_affine)
        self.out = nn.Linear(n_qubits, 1)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """The classical-to-quantum encoding step in isolation (Quantum
        Encoding Investigation Section 10's "encoded angle distribution
        analysis" calls this directly, without running the rest of
        forward(), to inspect what the circuit actually receives)."""
        z = self.reduce(h)
        bounded = torch.tanh(z) if self.encoding_type == "tanh" else torch.clamp(z, -1.0, 1.0)
        return self.encoding_scale * bounded

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is not None:
            h = self.pre_norm(h)
        angles = self.encode(h)
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
            "encoding": f"AngleEmbedding, rotation=Y, angle = {self.encoding_scale:.6f} * {self.encoding_type}(projection(h))",
            "backend": self.device_name,
            "normalization": {"type": "LayerNorm", "elementwise_affine": self.elementwise_affine},
            "projection": {
                "type": self.projection_type,
                "hidden_dim": self.projection_hidden_dim if self.projection_type == "nonlinear" else None,
                "pre_projection_norm": self.pre_projection_norm,
                "reduce_parameters": sum(p.numel() for p in self.reduce.parameters()),
            },
            "quantum_init": self.quantum_init,
            "gaussian_std": self.gaussian_std,
            "encoding_config": {
                "encoding_type": self.encoding_type,
                "encoding_scale": self.encoding_scale,
                "data_reuploading": self.data_reuploading,
                "n_encoding_operations": self.n_layers if self.data_reuploading else 1,
            },
        }

    def quantum_parameter_stats(self) -> dict:
        """Phase 4 Stage 4 §16's initialization audit: mean/std/min/max/L2
        norm of the quantum circuit's own trainable parameters AT WHATEVER
        POINT this is called -- the caller decides whether that's "right
        after construction" (initial distribution) or "after training"
        (final distribution) by choosing when to call it. Pooled across
        every quantum weight tensor (currently always a single `weights`
        tensor, but this doesn't assume that)."""
        values = torch.cat([p.detach().flatten() for p in self.quantum.parameters()])
        return {
            "quantum_init": self.quantum_init,
            "n_params": int(values.numel()),
            "mean": float(values.mean().item()),
            "std": float(values.std().item()) if values.numel() > 1 else 0.0,
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "l2_norm": float(values.norm(p=2).item()),
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
