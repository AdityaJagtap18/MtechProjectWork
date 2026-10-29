"""QGNN-v4 experiment wiring (PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md
sections 4, 8): reuses the existing frozen-encoder extraction pipeline
(`graph_embedding_reduction.py`) and the existing flat-mini-batch
training/eval harness (`qgnn_v2.py`'s `train_v2_head` / `evaluate_v2` /
`generate_predictions_v2`, already model-agnostic) unchanged. The only new
piece here is how the frozen embedding reaches the head: RAW `hidden_dim`
embedding, not PCA'd, since the trainable `Linear(in_dim, n_qubits)` inside
each v4 head IS the dimensionality reduction, learned jointly with the
rest of the head via the training-split loss -- exactly like any other
supervised layer (no separate leakage-sensitive "reducer" artifact to
fit/save/load, unlike `qgnn_v2.py`'s PCA step).

Mirrors `qgnn.QGNNExperimentConfig` / `qgnn_v2.QGNNv2ExperimentConfig`'s
own pattern of wrapping an unmodified `GraphSAGEConfig` rather than adding
fields to `config.py` (shared infrastructure this repo's other QGNN
variants also deliberately leave untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yaml

from ..config import GraphSAGEConfig, load_config
from ..data import BenchmarkData
from ..qgnn_v2 import V2PreparedData


@dataclass
class QuantumV4ArchConfig:
    n_qubits: int = 6
    n_layers: int = 2
    ansatz: str = "strongly_entangling"  # | "basic_entangler"
    diff_method: str = "backprop"  # | "parameter-shift"
    device: str = "default.qubit"  # | "lightning.qubit"


@dataclass
class QuantumV4ExperimentConfig:
    """Wraps an ordinary, unmodified `GraphSAGEConfig` (`base`, describing
    the frozen encoder being reused) with the v4-specific `quantum_v4`
    section `configs/qgnn_v4.yaml`'s schema has no room for."""

    base: GraphSAGEConfig = field(default_factory=GraphSAGEConfig)
    quantum_v4: QuantumV4ArchConfig = field(default_factory=QuantumV4ArchConfig)


def load_quantum_v4_config(path: str) -> QuantumV4ExperimentConfig:
    base = load_config(path)  # existing, unmodified function -- ignores the "quantum_v4:" section harmlessly
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return QuantumV4ExperimentConfig(base=base, quantum_v4=QuantumV4ArchConfig(**raw.get("quantum_v4", {})))


def build_v4_prepared(
    config: GraphSAGEConfig, benchmark: BenchmarkData, examples: pd.DataFrame, embedding_frame: pd.DataFrame,
) -> V2PreparedData:
    """Wraps the RAW (unreduced) frozen-encoder embedding frame as a
    `V2PreparedData` directly -- no PCA fit, so there is no separate
    reducer artifact to save/load/version. `V2PreparedData` only ever
    duck-types `.reduced` as "a (supplier_id, time)-indexed DataFrame of
    input columns for the head," which the raw embedding frame already is,
    so `qgnn_v2.train_v2_head` / `evaluate_v2` / `generate_predictions_v2`
    work on it completely unmodified."""
    return V2PreparedData(config=config, benchmark=benchmark, examples=examples, reduced=embedding_frame, n_components=embedding_frame.shape[1])
