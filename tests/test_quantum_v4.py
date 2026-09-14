"""Tests for QGNN-v4 (PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md): the
`StronglyEntanglingLayers` hybrid head and its matched-capacity classical
control (RQ-Q3), both on a raw frozen-encoder embedding. Mirrors
`test_qgnn_v2.py`'s pattern: uses `tiny_benchmark` and a
freshly-trained-then-frozen tiny GraphSAGE, not the real primary
benchmark, so these run fast and don't depend on experiments/classical_gnn/
existing on disk."""

from __future__ import annotations

import math

import torch

from scm_dataset.modeling.graph_embedding_reduction import extract_supplier_embeddings
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.modeling.qgnn_v2 import evaluate_v2, generate_predictions_v2, train_v2_head
from scm_dataset.modeling.quantum import HybridQuantumHead, MatchedCapacityClassicalHead, build_v4_prepared
from scm_dataset.modeling.quantum.circuit import build_quantum_layer
from scm_dataset.modeling.train import train_graphsage

from conftest import make_tiny_config


def _tiny_frozen_encoder(tiny_benchmark):
    """Trains a tiny GraphSAGE briefly (reusing the existing, unmodified
    train_graphsage), then freezes it -- standing in for "an existing
    GraphSAGE-Full checkpoint" without needing one saved on disk."""
    config = make_tiny_config()
    config.training.epochs = 3
    prepared = prepare_from_benchmark(config, tiny_benchmark)
    result = train_graphsage(prepared, seed=42, verbose=False)
    model = result.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return model, prepared


def _tiny_v4_prepared(tiny_benchmark):
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)
    v4prepared = build_v4_prepared(prepared.config, prepared.benchmark, prepared.examples, emb)
    return v4prepared, model


def test_circuit_output_shape():
    layer = build_quantum_layer(n_qubits=4, n_layers=2)
    x = torch.zeros(5, 4)
    out = layer(x)
    assert out.shape == (5, 4)
    assert torch.isfinite(out).all()


def test_circuit_basic_entangler_ansatz_also_builds_and_runs():
    layer = build_quantum_layer(n_qubits=3, n_layers=1, ansatz="basic_entangler")
    out = layer(torch.randn(4, 3))
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_unknown_ansatz_rejected():
    try:
        build_quantum_layer(n_qubits=3, n_layers=1, ansatz="not_a_real_ansatz")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_encoding_bounded_for_extreme_input_magnitudes():
    head = HybridQuantumHead(in_dim=6, n_qubits=4, n_layers=1)
    extreme = torch.tensor([[1e6, -1e6, 1e-6, 0.0, 5e3, -5e3]] * 3)
    angles = math.pi * torch.tanh(head.reduce(extreme))
    assert (angles <= math.pi).all() and (angles >= -math.pi).all()


def test_hybrid_quantum_head_forward_shape_no_nans():
    head = HybridQuantumHead(in_dim=10, n_qubits=4, n_layers=2)
    logits = head(torch.randn(7, 10))
    assert logits.shape == (7,)
    assert torch.isfinite(logits).all()


def test_gradient_flows_to_quantum_params():
    head = HybridQuantumHead(in_dim=6, n_qubits=4, n_layers=1)
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
    loss.backward()
    quantum_params = list(head.quantum.parameters())
    assert quantum_params, "quantum TorchLayer must expose trainable parameters"
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in quantum_params)


def test_backbone_frozen_after_v4_training(tiny_benchmark):
    v4prepared, model = _tiny_v4_prepared(tiny_benchmark)
    v4prepared.config.training.epochs = 2
    head = HybridQuantumHead(in_dim=v4prepared.n_components, n_qubits=4, n_layers=1)
    train_v2_head(v4prepared, head, seed=42, verbose=False)
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())


def test_matched_capacity_control_shares_bottleneck_width_with_quantum_head():
    in_dim, n_qubits = 128, 6
    quantum = HybridQuantumHead(in_dim, n_qubits=n_qubits, n_layers=2)
    classical = MatchedCapacityClassicalHead(in_dim, n_qubits=n_qubits)

    assert quantum.reduce.weight.shape == classical.reduce.weight.shape == (n_qubits, in_dim)
    assert quantum.out.weight.shape == classical.out.weight.shape == (1, n_qubits)

    bottleneck_params = sum(p.numel() for p in quantum.reduce.parameters()) + sum(p.numel() for p in quantum.out.parameters())
    classical_params = sum(p.numel() for p in classical.parameters())
    assert classical_params == bottleneck_params  # classical head IS exactly reduce+out, no extra parameters

    quantum_only_params = sum(p.numel() for p in quantum.parameters()) - bottleneck_params
    assert quantum_only_params > 0  # the circuit contributes parameters beyond the shared bottleneck


def test_deterministic_given_seed():
    torch.manual_seed(0)
    head1 = HybridQuantumHead(in_dim=6, n_qubits=4, n_layers=1)
    torch.manual_seed(0)
    head2 = HybridQuantumHead(in_dim=6, n_qubits=4, n_layers=1)
    x = torch.randn(4, 6)
    out1 = head1(x)
    out2 = head2(x)
    assert torch.allclose(out1, out2)


def test_train_v2_head_end_to_end_quantum_and_matched_classical(tiny_benchmark):
    v4prepared, _ = _tiny_v4_prepared(tiny_benchmark)
    v4prepared.config.training.epochs = 3
    v4prepared.config.training.early_stopping_patience = 10
    in_dim = v4prepared.n_components

    classical = MatchedCapacityClassicalHead(in_dim, n_qubits=4)
    classical_result = train_v2_head(v4prepared, classical, seed=42, verbose=False)
    preds = generate_predictions_v2(classical_result.model, v4prepared)
    assert preds["risk_probability"].between(0.0, 1.0).all()
    eval_result = evaluate_v2(classical_result.model, v4prepared, v4prepared.config.threshold)
    assert "test" in eval_result.metrics_by_split

    quantum = HybridQuantumHead(in_dim, n_qubits=4, n_layers=1)
    initial_weights = [p.detach().clone() for p in quantum.quantum.parameters()]
    quantum_result = train_v2_head(v4prepared, quantum, seed=42, verbose=False)
    moved_weights = list(quantum_result.model.quantum.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(initial_weights, moved_weights))  # gradients actually reached the quantum circuit
    quantum_eval = evaluate_v2(quantum_result.model, v4prepared, v4prepared.config.threshold)
    assert "test" in quantum_eval.metrics_by_split
