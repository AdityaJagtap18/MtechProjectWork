"""QGNN-v4: PennyLane `StronglyEntanglingLayers` hybrid head on a frozen
GraphSAGE-Full embedding (PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md).

Purely additive: does not modify `qgnn.py`, `qgnn_v2.py`, `qgnn_v3.py`, or
`qgnn_v2_reupload.py`. Reuses their frozen-encoder pipeline
(`graph_embedding_reduction.py`) and flat-mini-batch training/eval harness
(`qgnn_v2.train_v2_head` / `evaluate_v2` / `generate_predictions_v2`,
already model-agnostic) unchanged. What's new here is the architecture:
a trainable `Linear(in_dim, n_qubits)` bottleneck (not PCA) feeding a
`StronglyEntanglingLayers`/`BasicEntanglerLayers` ansatz built via
`qml.qnn.TorchLayer` (not the hand-rolled RY+CNOT-chain circuit in
`qgnn.py`), plus a matched-capacity classical control for the RQ-Q3
fairness ablation.
"""

from __future__ import annotations

from .circuit import build_quantum_layer
from .heads import HybridQuantumHead, HybridQuantumHeadOutputScale, MatchedCapacityClassicalHead
from .model import (
    QuantumV4ArchConfig,
    QuantumV4ExperimentConfig,
    build_v4_prepared,
    load_quantum_v4_config,
)
from .train import V4DiagnosticTrainResult, train_v4_head_with_diagnostics

__all__ = [
    "build_quantum_layer",
    "HybridQuantumHead",
    "HybridQuantumHeadOutputScale",
    "MatchedCapacityClassicalHead",
    "QuantumV4ArchConfig",
    "QuantumV4ExperimentConfig",
    "build_v4_prepared",
    "load_quantum_v4_config",
    "V4DiagnosticTrainResult",
    "train_v4_head_with_diagnostics",
]
