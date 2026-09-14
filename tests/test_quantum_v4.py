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
from scm_dataset.modeling.quantum import HybridQuantumHead, HybridQuantumHeadOutputScale, MatchedCapacityClassicalHead, build_v4_prepared
from scm_dataset.modeling.quantum.circuit import build_quantum_layer
from scm_dataset.modeling.quantum.train import train_v4_head_with_diagnostics
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


def test_diagnostic_training_logs_quantum_grad_norm_for_quantum_head_only(tiny_benchmark):
    """The instrumented loop (Phase 1 of the stability investigation) must
    log a real, non-null quantum_grad_norm_mean for HybridQuantumHead
    every epoch, and None for MatchedCapacityClassicalHead (no .quantum
    submodule to inspect) -- and must produce the SAME optimizer/loss/
    early-stopping behavior as train_v2_head, just with extra columns."""
    v4prepared, _ = _tiny_v4_prepared(tiny_benchmark)
    v4prepared.config.training.epochs = 4
    v4prepared.config.training.early_stopping_patience = 10
    in_dim = v4prepared.n_components

    quantum = HybridQuantumHead(in_dim, n_qubits=4, n_layers=1)
    result = train_v4_head_with_diagnostics(v4prepared, quantum, seed=42, verbose=False)
    assert "quantum_grad_norm_mean" in result.history.columns
    assert "train_pr_auc" in result.history.columns
    assert result.history["quantum_grad_norm_mean"].notna().all()
    assert (result.history["quantum_grad_norm_mean"] > 0).all()
    assert result.history["train_pr_auc"].between(0.0, 1.0).all()

    classical = MatchedCapacityClassicalHead(in_dim, n_qubits=4)
    classical_result = train_v4_head_with_diagnostics(v4prepared, classical, seed=42, verbose=False)
    assert classical_result.history["quantum_grad_norm_mean"].isna().all()
    assert classical_result.history["reduce_grad_norm_mean"].notna().all()


def test_diagnostic_training_matches_train_v2_head_optimization_behavior(tiny_benchmark):
    """Same seed, same model config -> the instrumented loop must reach
    the same best_val_pr_auc as train_v2_head (only logging differs, not
    the optimizer/loss/early-stopping mechanics)."""
    v4prepared, _ = _tiny_v4_prepared(tiny_benchmark)
    v4prepared.config.training.epochs = 4
    v4prepared.config.training.early_stopping_patience = 10
    in_dim = v4prepared.n_components

    from scm_dataset.modeling.qgnn_v2 import set_seed as v4_set_seed

    v4_set_seed(42)  # must precede model construction so weight init is identical, not just training-time shuffling
    plain_result = train_v2_head(v4prepared, MatchedCapacityClassicalHead(in_dim, n_qubits=4), seed=42, verbose=False)
    v4_set_seed(42)
    diag_result = train_v4_head_with_diagnostics(v4prepared, MatchedCapacityClassicalHead(in_dim, n_qubits=4), seed=42, verbose=False)
    assert plain_result.best_val_pr_auc == diag_result.best_val_pr_auc
    assert plain_result.best_epoch == diag_result.best_epoch


def test_output_scale_head_at_alpha_1_matches_baseline_head_exactly():
    """Phase 2b's core methodological claim (heads.py docstring): alpha=1,
    no bias, is mathematically a no-op -- HybridQuantumHeadOutputScale must
    reduce to byte-identical output to plain HybridQuantumHead when given
    the same weights. Verifies the wiring, not just the math on paper."""
    torch.manual_seed(0)
    baseline = HybridQuantumHead(in_dim=10, n_qubits=4, n_layers=1)
    torch.manual_seed(0)
    scaled = HybridQuantumHeadOutputScale(in_dim=10, n_qubits=4, n_layers=1, alpha_init=1.0, use_bias=False, trainable_scale=True)
    x = torch.randn(6, 10)
    assert torch.allclose(baseline(x), scaled(x))


def test_output_scale_trainable_alpha_receives_gradient():
    head = HybridQuantumHeadOutputScale(in_dim=6, n_qubits=4, n_layers=1, alpha_init=1.0, use_bias=True, trainable_scale=True)
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
    loss.backward()
    assert head.alpha.grad is not None and head.alpha.grad.item() != 0
    assert head.beta.grad is not None


def test_fixed_scale_alpha_is_not_a_trainable_parameter():
    head = HybridQuantumHeadOutputScale(in_dim=6, n_qubits=4, n_layers=1, alpha_init=0.5, use_bias=False, trainable_scale=False)
    param_names = [n for n, _ in head.named_parameters()]
    assert "alpha" not in param_names  # registered as a buffer, not a Parameter
    assert head.alpha.item() == 0.5
    x = torch.randn(5, 6)
    y = torch.randint(0, 2, (5,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
    loss.backward()  # must not raise even though alpha has no grad
    assert head.reduce.weight.grad is not None  # the rest of the head still trains normally


def test_fixed_scale_alpha_actually_scales_the_quantum_output():
    """alpha != 1 must change the forward pass, not be silently ignored."""
    torch.manual_seed(0)
    unscaled = HybridQuantumHeadOutputScale(in_dim=6, n_qubits=4, n_layers=1, alpha_init=1.0, use_bias=False, trainable_scale=False)
    torch.manual_seed(0)
    doubled = HybridQuantumHeadOutputScale(in_dim=6, n_qubits=4, n_layers=1, alpha_init=2.0, use_bias=False, trainable_scale=False)
    x = torch.randn(5, 6)
    assert not torch.allclose(unscaled(x), doubled(x))
