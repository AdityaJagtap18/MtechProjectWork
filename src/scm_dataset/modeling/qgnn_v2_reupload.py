"""QGNN-v2 depth-3 + data re-uploading -- controlled follow-up experiment
(QGNN-v2 Data Re-Uploading plan). Purely additive: does not modify
`qgnn.py` or `qgnn_v2.py`. Reuses `qgnn_v2.py`'s `V2PreparedData` /
`build_v2_prepared` / `train_v2_head` / `evaluate_v2` /
`generate_predictions_v2` unchanged (all already model-agnostic --
nothing about them assumes a particular circuit), and reuses
`graph_embedding_reduction.py` unchanged for the frozen encoder + PCA.
The only new code is the circuit itself and its thin `nn.Module` wrapper,
mirroring `qgnn.QuantumCircuitLayer`/`qgnn.QGNN` as closely as possible so
the only intentional architectural difference is the repeated encoding:

    input x
        -> angle encoding                  (initial AngleEmbedding)
        -> variational layer 1
        -> data re-upload of x             (AngleEmbedding again)
        -> variational layer 2
        -> data re-upload of x
        -> variational layer 3
        -> measurement
        -> same MLP head shape as qgnn.QGNN

Re-uploading repeats the SAME fixed, non-trainable encoding gates between
layers -- it adds circuit depth/gate count, not trainable parameters:
this model has exactly the same 24 trainable quantum parameters (3 layers
x 8 qubits) as the baseline depth-3 QGNN; only the gate sequence differs.
Reported explicitly in `quantum_resource_summary()` rather than left
implicit.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn

from .qgnn import DEFAULT_DEVICE, FALLBACK_DEVICE


def _make_qnode_reupload(n_qubits: int, n_layers: int, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)
    diff_method = "adjoint" if "lightning" in device_name else "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(angles, weights):
        qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if layer < n_layers - 1:  # re-upload BETWEEN layers only -- not before layer 0 (already encoded), not after the last layer (goes straight to measurement)
                qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit, dev


class QuantumCircuitLayerReupload(nn.Module):
    """Identical to `qgnn.QuantumCircuitLayer` except for the re-upload
    gates inserted between variational layers. Same parameter count,
    same fallback-device behavior, same float32 cast for cross-backend
    dtype consistency."""

    def __init__(self, n_qubits: int, n_layers: int = 3, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        try:
            self._circuit, self._device = _make_qnode_reupload(n_qubits, n_layers, device_name)
            self.device_name = device_name
        except Exception as exc:  # pragma: no cover -- exercised only on machines without the GPU backend
            print(f"QuantumCircuitLayerReupload: device {device_name!r} unavailable ({exc!r}), falling back to {FALLBACK_DEVICE!r}")
            self._circuit, self._device = _make_qnode_reupload(n_qubits, n_layers, FALLBACK_DEVICE)
            self.device_name = FALLBACK_DEVICE
        self.weights = nn.Parameter((torch.rand(n_layers, n_qubits) * 2 - 1) * math.pi)

    def forward(self, standardized_x: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(standardized_x) * math.pi
        outputs = self._circuit(angles, self.weights)
        return torch.stack(outputs, dim=-1).to(torch.float32)


class QGNNReupload(nn.Module):
    """Same MLP head shape as `qgnn.QGNN` (`Linear(n_qubits, mlp_hidden)
    -> ReLU -> Linear(mlp_hidden, 1)`, no sigmoid) -- the only intentional
    architectural difference from the baseline depth-3 QGNN is the
    repeated data encoding inside `QuantumCircuitLayerReupload`."""

    def __init__(self, n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = QuantumCircuitLayerReupload(n_qubits, n_layers, device_name)
        self.mlp = nn.Sequential(
            nn.Linear(n_qubits, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: [batch, n_qubits] reduced graph embedding. Returns raw
        logits, shape [batch] -- identical convention to `qgnn.QGNN.forward`."""
        q_out = self.quantum(x)
        return self.mlp(q_out).squeeze(-1)

    def quantum_resource_summary(self) -> dict:
        n_reuploads = self.quantum.n_layers - 1
        return {
            "qubits": self.n_qubits,
            "variational_layers": self.quantum.n_layers,
            "trainable_quantum_parameters": self.quantum.weights.numel(),
            "total_trainable_parameters": sum(p.numel() for p in self.parameters()),
            "data_reuploads": n_reuploads,
            "total_angle_embedding_calls": 1 + n_reuploads,
            "observable": "PauliZ (one per qubit)",
            "entanglement_pattern": "linear CNOT chain (wire i -> i+1)",
            "encoding": "AngleEmbedding, rotation=Y, angle = tanh(x) * pi, re-uploaded between each pair of consecutive variational layers",
            "backend": self.quantum.device_name,
        }


def build_qgnn_reupload_model(n_qubits: int, n_layers: int = 3, mlp_hidden: int = 8, device_name: str = DEFAULT_DEVICE) -> QGNNReupload:
    return QGNNReupload(n_qubits=n_qubits, n_layers=n_layers, mlp_hidden=mlp_hidden, device_name=device_name)
