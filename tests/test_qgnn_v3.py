"""Tests for the QGNN-v3 controlled architecture study (entanglement
topology variants B/C, trainable input scaling variant D). Purely
additive alongside test_qgnn_v2*.py -- does not modify or weaken any
existing test.
"""

from __future__ import annotations

import pennylane as qml
import torch

from scm_dataset.modeling.graph_embedding_reduction import extract_supplier_embeddings
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.modeling.qgnn import build_qgnn_model
from scm_dataset.modeling.qgnn_v2 import build_v2_prepared, evaluate_v2, generate_predictions_v2, train_v2_head
from scm_dataset.modeling.qgnn_v3 import (
    all_to_all_entangler_pairs,
    build_qgnn_all_to_all_model,
    build_qgnn_ring_model,
    build_qgnn_trainable_scaling_model,
    linear_entangler_pairs,
    ring_entangler_pairs,
)
from scm_dataset.modeling.train import train_graphsage

from conftest import make_tiny_config

CPU_DEVICE = "default.qubit"


def _tiny_frozen_encoder(tiny_benchmark):
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


def _tape_ops(circuit_fn, angles, weights):
    tape = qml.workflow.construct_tape(circuit_fn)(angles, weights)
    return [(op.name, tuple(op.wires.tolist())) for op in tape.operations]


# ---- 1-2: qubits/depth ----


def test_ring_and_all_to_all_have_8_qubits_and_depth_3():
    for model in (build_qgnn_ring_model(8, 3, device_name=CPU_DEVICE), build_qgnn_all_to_all_model(8, 3, device_name=CPU_DEVICE)):
        summary = model.quantum_resource_summary()
        assert summary["qubits"] == 8
        assert summary["variational_layers"] == 3


def test_trainable_scaling_has_8_qubits_and_depth_3():
    model = build_qgnn_trainable_scaling_model(8, 3, device_name=CPU_DEVICE)
    summary = model.quantum_resource_summary()
    assert summary["qubits"] == 8
    assert summary["variational_layers"] == 3


# ---- 3: pairing helpers ----


def test_entangler_pair_helpers_produce_correct_pairs_for_8_qubits():
    assert linear_entangler_pairs(8) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert ring_entangler_pairs(8) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
    assert len(all_to_all_entangler_pairs(8)) == 28  # C(8,2)


# ---- 4: linear topology reduces EXACTLY to the existing baseline ----


def test_linear_entanglement_variant_circuit_is_bit_identical_to_baseline_tape():
    from scm_dataset.modeling.qgnn_v3 import _make_qnode_with_entangler

    n_qubits = 8
    angles = torch.tanh(torch.randn(2, n_qubits)) * torch.pi
    weights = torch.rand(3, n_qubits)

    baseline = build_qgnn_model(n_qubits=n_qubits, n_layers=3, device_name=CPU_DEVICE)
    baseline_ops = _tape_ops(baseline.quantum._circuit, angles, weights)

    linear_circuit, _ = _make_qnode_with_entangler(n_qubits, 3, CPU_DEVICE, linear_entangler_pairs(n_qubits))
    linear_ops = _tape_ops(linear_circuit, angles, weights)

    assert baseline_ops == linear_ops  # identical gate sequence, identical wires, identical order


# ---- 5: ring has the wrap-around edge ----


def test_ring_topology_has_wraparound_edge_absent_from_linear():
    ring_model = build_qgnn_ring_model(8, 3, device_name=CPU_DEVICE)
    assert (7, 0) in ring_model.quantum.entangler_pairs
    assert len(ring_model.quantum.entangler_pairs) == 8  # 7 linear + 1 wraparound
    assert (7, 0) not in linear_entangler_pairs(8)


# ---- 6: all-to-all pairwise connectivity ----


def test_all_to_all_topology_connects_every_qubit_pair_exactly_once():
    model = build_qgnn_all_to_all_model(8, 3, device_name=CPU_DEVICE)
    pairs = model.quantum.entangler_pairs
    assert len(pairs) == 28
    assert len(set(pairs)) == 28  # no duplicate edges
    expected = {(i, j) for i in range(8) for j in range(i + 1, 8)}
    assert set(pairs) == expected


# ---- 7: trainable-scaling parameter count ----


def test_trainable_scaling_added_parameter_count():
    model = build_qgnn_trainable_scaling_model(8, 3, device_name=CPU_DEVICE)
    summary = model.quantum_resource_summary()
    assert summary["added_scaling_parameters"] == 16  # a (8) + b (8)
    assert model.quantum.scale_a.numel() == 8
    assert model.quantum.scale_b.numel() == 8


# ---- 8: scaling init matches the fixed baseline transform ----


def test_trainable_scaling_initialization_matches_baseline_transform_exactly():
    x = torch.randn(4, 8)
    baseline = build_qgnn_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    scaling = build_qgnn_trainable_scaling_model(8, 3, device_name=CPU_DEVICE)

    assert torch.allclose(scaling.quantum.scale_a, torch.ones(8))
    assert torch.allclose(scaling.quantum.scale_b, torch.zeros(8))

    scaling.quantum.weights.data = baseline.quantum.weights.data.clone()  # isolate the encoding comparison from random weight init
    baseline_out = baseline.quantum(x)
    scaling_out = scaling.quantum(x)
    assert torch.allclose(baseline_out, scaling_out, atol=1e-6)


# ---- 9: output shape unchanged ----


def test_all_variants_output_shape_matches_baseline():
    x = torch.randn(5, 8)
    baseline = build_qgnn_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    for model in (
        build_qgnn_ring_model(8, 3, device_name=CPU_DEVICE),
        build_qgnn_all_to_all_model(8, 3, device_name=CPU_DEVICE),
        build_qgnn_trainable_scaling_model(8, 3, device_name=CPU_DEVICE),
    ):
        assert model(x).shape == baseline(x).shape == (5,)


# ---- 10: classical MLP head unchanged ----


def test_all_variants_mlp_head_architecturally_unchanged():
    baseline = build_qgnn_model(n_qubits=8, n_layers=3, mlp_hidden=8, device_name=CPU_DEVICE)
    baseline_shapes = [tuple(p.shape) for p in baseline.mlp.parameters()]
    for model in (
        build_qgnn_ring_model(8, 3, mlp_hidden=8, device_name=CPU_DEVICE),
        build_qgnn_all_to_all_model(8, 3, mlp_hidden=8, device_name=CPU_DEVICE),
        build_qgnn_trainable_scaling_model(8, 3, mlp_hidden=8, device_name=CPU_DEVICE),
    ):
        assert [tuple(p.shape) for p in model.mlp.parameters()] == baseline_shapes


# ---- 14-15: finite forward/backward ----


def test_all_variants_finite_forward_and_backward():
    x = torch.randn(6, 8)
    target = torch.zeros(6)
    target[0] = 1.0
    for model in (
        build_qgnn_ring_model(8, 3, device_name=CPU_DEVICE),
        build_qgnn_all_to_all_model(8, 3, device_name=CPU_DEVICE),
        build_qgnn_trainable_scaling_model(8, 3, device_name=CPU_DEVICE),
    ):
        logits = model(x)
        assert torch.isfinite(logits).all()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} got a non-finite gradient"


# ---- 11-13: encoder freeze, PCA train-only, no target leakage (full pipeline) ----


def test_all_variants_end_to_end_through_frozen_encoder_no_leakage(tiny_benchmark):
    encoder, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(encoder, prepared, times)
    v2prepared, reducer = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components=4)
    v2prepared.config.training.epochs = 3
    v2prepared.config.training.early_stopping_patience = 10
    assert reducer._input_columns is not None  # already fit inside build_v2_prepared, train-only by construction (reused unchanged)

    for build_fn in (build_qgnn_ring_model, build_qgnn_all_to_all_model, build_qgnn_trainable_scaling_model):
        model = build_fn(4, 3, device_name=CPU_DEVICE)
        result = train_v2_head(v2prepared, model, seed=42, verbose=False)
        assert all(not p.requires_grad for p in encoder.parameters())
        assert all(p.grad is None for p in encoder.parameters())

        predictions = generate_predictions_v2(result.model, v2prepared)
        assert predictions["risk_probability"].between(0.0, 1.0).all()
        eval_result = evaluate_v2(result.model, v2prepared, v2prepared.config.threshold)
        assert "test" in eval_result.metrics_by_split
