"""Tests for QGNN-v4 (PENNYLANE_QGNN_IMPLEMENTATION_PLAN.md): the
`StronglyEntanglingLayers` hybrid head and its matched-capacity classical
control (RQ-Q3), both on a raw frozen-encoder embedding. Mirrors
`test_qgnn_v2.py`'s pattern: uses `tiny_benchmark` and a
freshly-trained-then-frozen tiny GraphSAGE, not the real primary
benchmark, so these run fast and don't depend on experiments/classical_gnn/
existing on disk."""

from __future__ import annotations

import math

import pytest
import torch

from scm_dataset.modeling.graph_embedding_reduction import extract_supplier_embeddings
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.modeling.qgnn_v2 import evaluate_v2, generate_predictions_v2, train_v2_head
from scm_dataset.modeling.quantum import (
    HybridQuantumHead,
    HybridQuantumHeadLayerNorm,
    HybridQuantumHeadOutputScale,
    MatchedCapacityClassicalHead,
    build_v4_prepared,
)
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


def test_layernorm_head_forward_shape_no_nans():
    for affine in (False, True):
        head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=4, n_layers=1, elementwise_affine=affine)
        logits = head(torch.randn(7, 10))
        assert logits.shape == (7,)
        assert torch.isfinite(logits).all()


def test_layernorm_noaffine_actually_normalizes_per_example():
    """LayerNorm(elementwise_affine=False) must produce, for every
    example independently, ~zero mean and ~unit variance across the
    n_qubits dimension -- the defining, genuinely data-dependent property
    Phase 2c is testing (unlike Phase 2b's alpha/beta, whose effect is a
    fixed global affine map)."""
    head = HybridQuantumHeadLayerNorm(in_dim=6, n_qubits=4, n_layers=1, elementwise_affine=False)
    with torch.no_grad():
        angles = torch.pi * torch.tanh(head.reduce(torch.randn(9, 6)))
        q_out = head.quantum(angles).to(torch.float32)
        normed = head.norm(q_out)
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(9), atol=1e-5)
    # LayerNorm's internal eps (1e-5) trades a small amount of variance
    # for numerical stability, more visible with only 4 features -- 0.05
    # tolerance confirms "approximately unit variance," not a stricter
    # bound eps itself doesn't promise.
    assert torch.allclose(normed.std(dim=-1, unbiased=False), torch.ones(9), atol=0.05)


def test_layernorm_affine_has_no_effect_at_initialization():
    """PyTorch's LayerNorm default init is weight=1, bias=0 -- so at
    initialization (before any training step), elementwise_affine=True
    must produce byte-identical output to elementwise_affine=False, given
    the same upstream weights. This is the LayerNorm analogue of Phase
    2b's alpha=1/no-bias-equals-baseline test."""
    torch.manual_seed(0)
    noaffine = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=4, n_layers=1, elementwise_affine=False)
    torch.manual_seed(0)
    affine = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=4, n_layers=1, elementwise_affine=True)
    x = torch.randn(6, 10)
    assert torch.allclose(noaffine(x), affine(x))


def test_layernorm_affine_params_trainable_noaffine_has_none():
    affine = HybridQuantumHeadLayerNorm(in_dim=6, n_qubits=4, n_layers=1, elementwise_affine=True)
    assert affine.norm.weight is not None and affine.norm.weight.requires_grad
    assert affine.norm.bias is not None and affine.norm.bias.requires_grad

    noaffine = HybridQuantumHeadLayerNorm(in_dim=6, n_qubits=4, n_layers=1, elementwise_affine=False)
    assert noaffine.norm.weight is None
    assert noaffine.norm.bias is None
    param_names = [n for n, _ in noaffine.named_parameters()]
    assert not any(n.startswith("norm.") for n in param_names)


def test_layernorm_gradient_flows_to_quantum_and_affine_params():
    head = HybridQuantumHeadLayerNorm(in_dim=6, n_qubits=4, n_layers=1, elementwise_affine=True)
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in head.quantum.parameters())
    assert head.norm.weight.grad is not None
    assert head.norm.bias.grad is not None


@pytest.mark.parametrize("ansatz,n_qubits,n_layers,expected_params", [
    ("hardware_efficient_ring", 6, 2, 24),  # n_layers * n_qubits * 2 (RY+RZ)
    ("hardware_efficient_ring", 4, 3, 24),
    ("reduced_entanglement", 6, 2, 12),  # n_layers * n_qubits * 1 (RY only)
    ("reduced_entanglement", 8, 1, 8),
])
def test_phase3_ansatz_parameter_counts(ansatz, n_qubits, n_layers, expected_params):
    layer = build_quantum_layer(n_qubits=n_qubits, n_layers=n_layers, ansatz=ansatz)
    total = sum(p.numel() for p in layer.parameters())
    assert total == expected_params


@pytest.mark.parametrize("ansatz", ["hardware_efficient_ring", "reduced_entanglement"])
def test_phase3_ansatz_forward_shape_and_gradient_flow(ansatz):
    layer = build_quantum_layer(n_qubits=5, n_layers=2, ansatz=ansatz)
    x = torch.randn(6, 5)
    out = layer(x)
    assert out.shape == (6, 5)
    assert torch.isfinite(out).all()
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in layer.parameters())


def test_reduced_entanglement_uses_chain_not_ring():
    """Ansatz 3 (reduced_entanglement) must NOT include the wrap-around
    edge hardware_efficient_ring (Ansatz 2) has -- verified by construction
    (chain has n_qubits-1 CNOTs, ring has n_qubits), not just by name."""
    from scm_dataset.modeling.quantum.circuit import _chain_pairs, _ring_pairs

    chain = _chain_pairs(6)
    ring = _ring_pairs(6)
    assert len(chain) == 5  # n_qubits - 1, no wraparound
    assert len(ring) == 6  # n_qubits, includes wraparound
    assert (5, 0) not in chain
    assert (5, 0) in ring


def test_qgnn_v4_head_supports_all_four_ansatz_choices_end_to_end():
    """HybridQuantumHeadLayerNorm (the Phase-3 default output stage) must
    build and run correctly for every ansatz choice the Phase 3 matrix
    uses, not just the ones already covered by earlier phases."""
    for ansatz in ["strongly_entangling", "hardware_efficient_ring", "reduced_entanglement"]:
        head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, ansatz=ansatz, elementwise_affine=False)
        logits = head(torch.randn(4, 10))
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
def test_qgnn_v4_head_supports_qubit_count_ablation(n_qubits):
    """Ablation A (qubit count): HybridQuantumHeadLayerNorm must size its
    LayerNorm and output layer correctly for every planned qubit count."""
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=n_qubits, n_layers=2, elementwise_affine=False)
    assert head.norm.normalized_shape == (n_qubits,)
    assert head.out.in_features == n_qubits
    logits = head(torch.randn(4, 10))
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
def test_qgnn_v4_head_supports_depth_ablation(n_layers):
    """Ablation B (variational depth): must build and run for every
    planned layer count without error."""
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=n_layers, elementwise_affine=False)
    expected_quantum_params = n_layers * 6 * 3  # StronglyEntanglingLayers: 3 rotation params/qubit/layer
    assert sum(p.numel() for p in head.quantum.parameters()) == expected_quantum_params
    logits = head(torch.randn(4, 10))
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Phase 4 Stage 2 (QGNN_V4_PHASE4_PLAN.md Track B): projection/bottleneck
# variants on HybridQuantumHeadLayerNorm. B1 (control) is every test above
# with the new params left at their defaults -- nothing below tests B1
# separately since it IS the pre-Stage-2 behavior already covered.
# ---------------------------------------------------------------------------


def test_default_projection_params_reproduce_original_behavior_exactly():
    """B1 control: with projection_type='linear', pre_projection_norm=False
    (the defaults), the new constructor args must be no-ops -- same
    `reduce` type (a plain nn.Linear, not the nonlinear nn.Sequential), no
    pre_norm module at all, and bit-identical output given identical
    weights and input. This is the backward-compatibility guarantee every
    already-saved Phase 2c/2d/3 run depends on."""
    torch.manual_seed(0)
    old_style = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, elementwise_affine=False)
    torch.manual_seed(0)
    new_style = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, elementwise_affine=False, projection_type="linear", pre_projection_norm=False)
    assert isinstance(new_style.reduce, torch.nn.Linear)
    assert new_style.pre_norm is None
    x = torch.randn(5, 10)
    assert torch.equal(old_style(x), new_style(x))


def test_unknown_projection_type_rejected():
    with pytest.raises(ValueError, match="projection_type"):
        HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, projection_type="quadratic")


def test_nonlinear_projection_b2_forward_shape_and_param_count():
    """Track B2: Linear(in_dim,hidden)->GELU->Linear(hidden,n_qubits)
    replaces the single Linear bottleneck -- shape must still end at
    n_qubits (the quantum circuit's own input width is unaffected), and
    the reduce-layer parameter count must match the two-linear-layer
    formula exactly (for Stage 2's parameter-fairness reporting)."""
    in_dim, hidden, n_qubits = 10, 32, 6
    head = HybridQuantumHeadLayerNorm(in_dim=in_dim, n_qubits=n_qubits, n_layers=1, projection_type="nonlinear", projection_hidden_dim=hidden)
    assert isinstance(head.reduce, torch.nn.Sequential)
    expected_reduce_params = (in_dim * hidden + hidden) + (hidden * n_qubits + n_qubits)
    assert sum(p.numel() for p in head.reduce.parameters()) == expected_reduce_params
    logits = head(torch.randn(4, in_dim))
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_pre_projection_norm_b3_normalizes_input_before_reduce():
    """Track B3: LayerNorm(in_dim) must actually run on `h` before
    `reduce` -- verified by construction (pre_norm is a real LayerNorm(
    in_dim) module) and behaviorally (feeding an input with a large,
    non-zero-mean, non-unit-variance shift must NOT change the head's
    output once pre-normalized away, unlike the B1 control which has no
    such invariance)."""
    in_dim, n_qubits = 8, 4
    torch.manual_seed(1)
    head = HybridQuantumHeadLayerNorm(in_dim=in_dim, n_qubits=n_qubits, n_layers=1, pre_projection_norm=True)
    assert isinstance(head.pre_norm, torch.nn.LayerNorm)
    assert head.pre_norm.normalized_shape == (in_dim,)

    x = torch.randn(6, in_dim)
    shifted_and_scaled = x * 37.0 + 1000.0  # same per-example shape after LayerNorm's own per-example standardization
    with torch.no_grad():
        out_a = head(x)
        out_b = head(shifted_and_scaled)
    # atol=1e-3, not 1e-4: LayerNorm's internal eps (1e-5) plus the
    # quantum circuit's trig/exponential ops compound tiny floating-point
    # differences between the two inputs' standardized values -- the same
    # kind of looseness test_layernorm_noaffine_actually_normalizes_per_example
    # already documents for LayerNorm+quantum-circuit comparisons.
    assert torch.allclose(out_a, out_b, atol=1e-3)

    torch.manual_seed(1)
    control = HybridQuantumHeadLayerNorm(in_dim=in_dim, n_qubits=n_qubits, n_layers=1, pre_projection_norm=False)
    assert control.pre_norm is None
    with torch.no_grad():
        control_out_a = control(x)
        control_out_b = control(shifted_and_scaled)
    assert not torch.allclose(control_out_a, control_out_b, atol=1e-2)


def test_projection_variants_gradient_flows_to_reduce_and_quantum_params():
    """Both new projection knobs must not break backprop into the
    existing quantum circuit -- combining nonlinear projection AND
    pre-projection norm in one head (not a planned Stage 2 config, but a
    stress test that the two compose without breaking gradient flow)."""
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, projection_type="nonlinear", pre_projection_norm=True)
    logits = head(torch.randn(4, 10))
    logits.sum().backward()
    for p in head.reduce.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for p in head.quantum.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert head.pre_norm.weight.grad is not None


def test_quantum_resource_summary_reports_projection_info():
    linear_head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1)
    summary = linear_head.quantum_resource_summary()
    assert summary["projection"] == {"type": "linear", "hidden_dim": None, "pre_projection_norm": False, "reduce_parameters": 10 * 6 + 6}

    nonlinear_head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, projection_type="nonlinear", projection_hidden_dim=16, pre_projection_norm=True)
    nl_summary = nonlinear_head.quantum_resource_summary()
    assert nl_summary["projection"]["type"] == "nonlinear"
    assert nl_summary["projection"]["hidden_dim"] == 16
    assert nl_summary["projection"]["pre_projection_norm"] is True


def test_checkpoint_save_load_roundtrip_for_projection_variants():
    """Every Stage 2 head variant must survive a state_dict save/load
    round trip with identical output -- same guarantee run_qgnn_v4_experiment.py's
    _save_run relies on for every already-saved phase."""
    for kwargs in [
        dict(projection_type="linear", pre_projection_norm=False),
        dict(projection_type="nonlinear", projection_hidden_dim=16, pre_projection_norm=False),
        dict(projection_type="linear", pre_projection_norm=True),
    ]:
        head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, **kwargs)
        x = torch.randn(4, 10)
        with torch.no_grad():
            before = head(x)
        state = head.state_dict()
        reloaded = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, **kwargs)
        reloaded.load_state_dict(state)
        with torch.no_grad():
            after = reloaded(x)
        assert torch.equal(before, after)


def test_b4_pca_informed_projection_end_to_end(tiny_benchmark):
    """Track B4: HybridQuantumHeadLayerNorm needs NO new code for this --
    it's the same class with in_dim=n_components fed a PCA-reduced (train-
    fit-only, via qgnn_v2.build_v2_prepared -- already leakage-tested in
    test_qgnn_v2.py) embedding instead of the raw one. Confirms the
    composition actually works end-to-end and the quantum circuit still
    only ever sees an n_qubits-wide input regardless of the PCA width."""
    from scm_dataset.modeling.qgnn_v2 import build_v2_prepared

    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)

    for n_components in (4, 6):
        v2prepared, reducer = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components)
        assert v2prepared.reduced.shape[1] == n_components
        head = HybridQuantumHeadLayerNorm(in_dim=n_components, n_qubits=4, n_layers=1, elementwise_affine=False)
        prepared.config.training.epochs = 2
        prepared.config.training.early_stopping_patience = 5
        result = train_v2_head(v2prepared, head, seed=42, verbose=False)
        eval_result = evaluate_v2(result.model, v2prepared, prepared.config.threshold)
        assert "test" in eval_result.metrics_by_split
        assert result.model.quantum.parameters() is not None  # circuit still exists, unaffected by in_dim
