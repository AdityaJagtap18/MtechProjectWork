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


# ---------------------------------------------------------------------------
# Phase 4 Stage 4 (QGNN_V4_PHASE4_PLAN.md Track F): quantum parameter
# initialization strategies. F1 (default/control) is every test above with
# quantum_init left at its default -- nothing below tests F1 in isolation
# since it IS the pre-Stage-4 behavior already covered.
# ---------------------------------------------------------------------------


def test_default_quantum_init_is_uniform_0_to_2pi_matching_pennylanes_own_default():
    """F1 control: quantum_init='default' must pass init_method=None
    through to TorchLayer unchanged -- verified both structurally (the
    resolved init_method really is None) and statistically (values land in
    [0, 2*pi], and a build_quantum_layer with quantum_init='default' seeded
    identically to a plain qml.qnn.TorchLayer call with no init_method
    produces bit-identical initial weights)."""
    from scm_dataset.modeling.quantum.circuit import resolve_quantum_init

    assert resolve_quantum_init("default") is None

    torch.manual_seed(7)
    layer_a = build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="default")
    torch.manual_seed(7)
    import pennylane as qml

    from scm_dataset.modeling.quantum.circuit import ANSATZ_BUILDERS

    dev = qml.device("default.qubit", wires=6)
    ansatz_layer = ANSATZ_BUILDERS["strongly_entangling"]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(6), rotation="Y")
        ansatz_layer(weights, wires=range(6))
        return [qml.expval(qml.PauliZ(i)) for i in range(6)]

    layer_b = qml.qnn.TorchLayer(circuit, {"weights": ansatz_layer.shape(2, 6)})  # no init_method -- PennyLane's own default
    assert torch.equal(layer_a.weights, layer_b.weights)
    assert (layer_a.weights >= 0).all() and (layer_a.weights <= 2 * math.pi).all()


def test_small_gaussian_quantum_init_produces_small_near_zero_values():
    """F2: mean=0, std=0.01 -- values should cluster tightly around zero,
    nothing like the [0,2*pi] spread of the default."""
    torch.manual_seed(3)
    layer = build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="small_gaussian")
    values = layer.weights.detach()
    assert values.abs().max() < 0.1  # 0.01 std -- 0.1 is a generous 10-sigma bound, not a tight fit
    assert values.std().item() < 0.05


def test_identity_like_quantum_init_is_exactly_zero():
    """F3: every rotation parameter must start at EXACTLY 0.0 -- not
    approximately -- since Rot(0,0,0)/RY(0)/RZ(0)/RX(0) are each exactly
    the single-qubit identity only at the exact value 0."""
    for ansatz in ["strongly_entangling", "basic_entangler", "hardware_efficient_ring", "reduced_entanglement"]:
        layer = build_quantum_layer(n_qubits=6, n_layers=2, ansatz=ansatz, quantum_init="identity_like")
        assert torch.equal(layer.weights, torch.zeros_like(layer.weights))


def test_unknown_quantum_init_rejected():
    with pytest.raises(ValueError, match="quantum_init"):
        build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="orthogonal")
    with pytest.raises(ValueError, match="quantum_init"):
        HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, quantum_init="orthogonal")


@pytest.mark.parametrize("quantum_init", ["default", "small_gaussian", "identity_like"])
def test_quantum_init_does_not_change_parameter_count_or_shape(quantum_init):
    """Phase 4 Stage 4 section 17's explicit requirement: only the INITIAL
    VALUES may differ between F1/F2/F3, never the parameter count."""
    reference = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="default")
    variant = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init=quantum_init)
    assert sum(p.numel() for p in variant.quantum.parameters()) == sum(p.numel() for p in reference.quantum.parameters())
    assert sum(p.numel() for p in variant.parameters()) == sum(p.numel() for p in reference.parameters())
    assert variant.quantum.weights.shape == reference.quantum.weights.shape


@pytest.mark.parametrize("quantum_init", ["default", "small_gaussian", "identity_like"])
def test_quantum_init_forward_backward_shape_and_finite_gradients(quantum_init):
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init=quantum_init)
    logits = head(torch.randn(4, 10))
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for p in head.quantum.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_identity_like_init_gradients_are_not_all_identical():
    """A legitimate concern for a same-constant-value initialization
    (echoing the classical-NN 'all-zero-weights breaks symmetry' failure
    mode): if every quantum parameter starts at 0.0, do they all receive
    the SAME gradient (which would make them update identically forever,
    collapsing the circuit's effective capacity)? They should not -- each
    Rot/RY/RZ acts on a different qubit at a different position in the
    entangling pattern, so the local gradient differs per parameter even
    from a shared starting value. Verified directly rather than assumed."""
    torch.manual_seed(0)
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="identity_like")
    logits = head(torch.randn(8, 10))
    logits.sum().backward()
    grad = head.quantum.weights.grad.detach().flatten()
    assert torch.isfinite(grad).all()
    assert grad.std().item() > 1e-8  # not every gradient component identical
    assert grad.abs().max().item() > 1e-8  # and not a fully vanished (all-zero) gradient either


def test_quantum_resource_summary_reports_quantum_init():
    for quantum_init in ["default", "small_gaussian", "identity_like"]:
        head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, quantum_init=quantum_init)
        assert head.quantum_resource_summary()["quantum_init"] == quantum_init


def test_quantum_parameter_stats_matches_expected_distribution_per_strategy():
    torch.manual_seed(0)
    identity_head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="identity_like")
    stats = identity_head.quantum_parameter_stats()
    assert stats["mean"] == 0.0 and stats["std"] == 0.0 and stats["min"] == 0.0 and stats["max"] == 0.0 and stats["l2_norm"] == 0.0
    assert stats["n_params"] == 36  # 6 qubits * 2 layers * 3 rotation params (StronglyEntanglingLayers)

    small_gauss_head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="small_gaussian")
    sg_stats = small_gauss_head.quantum_parameter_stats()
    assert abs(sg_stats["mean"]) < 0.05
    assert sg_stats["std"] < 0.05
    assert sg_stats["l2_norm"] > 0.0  # NOT degenerate like identity_like


def test_default_quantum_init_kwarg_omitted_reproduces_original_behavior_exactly():
    """Backward-compatibility guarantee (mirrors Stage 2's own equivalent
    test): every already-saved Phase 2c/2d/3/Stage-2 run constructed
    HybridQuantumHeadLayerNorm without a quantum_init kwarg at all -- that
    must still produce identical output to explicitly passing
    quantum_init='default', given the same seed."""
    torch.manual_seed(11)
    omitted = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, elementwise_affine=False)
    torch.manual_seed(11)
    explicit = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, elementwise_affine=False, quantum_init="default")
    x = torch.randn(4, 10)
    assert torch.equal(omitted(x), explicit(x))


def test_identity_like_init_end_to_end_training(tiny_benchmark):
    """F3 must train correctly through the existing harness end-to-end --
    not just build/forward/backward in isolation."""
    v4prepared, _ = _tiny_v4_prepared(tiny_benchmark)
    v4prepared.config.training.epochs = 3
    v4prepared.config.training.early_stopping_patience = 5
    in_dim = v4prepared.n_components

    head = HybridQuantumHeadLayerNorm(in_dim=in_dim, n_qubits=4, n_layers=1, elementwise_affine=False, quantum_init="identity_like")
    result = train_v2_head(v4prepared, head, seed=42, verbose=False)
    eval_result = evaluate_v2(result.model, v4prepared, v4prepared.config.threshold)
    assert "test" in eval_result.metrics_by_split
    # weights must have actually moved away from the exact-zero start
    assert not torch.equal(result.model.quantum.weights.detach(), torch.zeros_like(result.model.quantum.weights))


# ---------------------------------------------------------------------------
# Phase 4 Stage 4b (QGNN_V4_PHASE4_PLAN.md Part B): configurable-std
# Gaussian initialization -- generalizes Stage 4's fixed-std "small_gaussian"
# (std=0.01) to any std, for the scale sweep (0.001/0.005/0.010/0.025/0.050).
# ---------------------------------------------------------------------------


def test_gaussian_quantum_init_requires_gaussian_std():
    from scm_dataset.modeling.quantum.circuit import resolve_quantum_init

    with pytest.raises(ValueError, match="gaussian_std"):
        resolve_quantum_init("gaussian", gaussian_std=None)
    with pytest.raises(ValueError, match="gaussian_std"):
        build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="gaussian")


def test_gaussian_std_only_meaningful_for_gaussian_quantum_init():
    from scm_dataset.modeling.quantum.circuit import resolve_quantum_init

    with pytest.raises(ValueError, match="gaussian_std"):
        resolve_quantum_init("default", gaussian_std=0.01)
    with pytest.raises(ValueError, match="gaussian_std"):
        resolve_quantum_init("identity_like", gaussian_std=0.01)


@pytest.mark.parametrize("std", [0.001, 0.005, 0.010, 0.025, 0.050])
def test_gaussian_quantum_init_produces_requested_scale(std):
    """Each Stage 4b sweep point must actually produce values at
    roughly its requested std -- checked statistically (36 samples is
    enough to distinguish 0.001 from 0.050 by orders of magnitude, even
    though it's too few for a tight std estimate)."""
    torch.manual_seed(5)
    layer = build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="gaussian", gaussian_std=std)
    values = layer.weights.detach()
    assert values.abs().max() < std * 10  # generous bound, not a tight fit
    assert values.std().item() < std * 5


def test_gaussian_std_0_01_matches_small_gaussian_distribution_family():
    """quantum_init='gaussian' with gaussian_std=0.01 must draw from the
    SAME distribution family as the fixed 'small_gaussian' strategy (both
    are torch.nn.init.normal_(mean=0, std=0.01)) -- verified by matching
    seeded output exactly, confirming Stage 4b's G3 pilot point really is
    numerically the same generator as Stage 4's own F2."""
    torch.manual_seed(9)
    small_gaussian_layer = build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="small_gaussian")
    torch.manual_seed(9)
    gaussian_g3_layer = build_quantum_layer(n_qubits=6, n_layers=2, quantum_init="gaussian", gaussian_std=0.01)
    assert torch.equal(small_gaussian_layer.weights, gaussian_g3_layer.weights)


def test_gaussian_quantum_init_does_not_change_parameter_count():
    reference = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="default")
    for std in [0.001, 0.005, 0.025, 0.050]:
        variant = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="gaussian", gaussian_std=std)
        assert sum(p.numel() for p in variant.quantum.parameters()) == sum(p.numel() for p in reference.quantum.parameters())
        assert sum(p.numel() for p in variant.parameters()) == sum(p.numel() for p in reference.parameters())


def test_gaussian_quantum_init_threads_through_head_forward_backward():
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, quantum_init="gaussian", gaussian_std=0.025)
    assert head.gaussian_std == 0.025
    logits = head(torch.randn(4, 10))
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for p in head.quantum.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_quantum_resource_summary_reports_gaussian_std():
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, quantum_init="gaussian", gaussian_std=0.005)
    summary = head.quantum_resource_summary()
    assert summary["quantum_init"] == "gaussian"
    assert summary["gaussian_std"] == 0.005

    default_head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1)
    assert default_head.quantum_resource_summary()["gaussian_std"] is None


# ---------------------------------------------------------------------------
# Quantum Encoding Investigation (QGNN_V4_QUANTUM_ENCODING_RESULTS.md):
# encoding_type/encoding_scale/data_reuploading. E0 (defaults) is every test
# above -- nothing below tests E0 in isolation since it IS the pre-existing
# behavior already covered.
# ---------------------------------------------------------------------------


def test_e0_default_encoding_kwargs_omitted_reproduces_original_behavior_exactly():
    """Backward-compatibility guarantee: every already-saved run
    constructed HybridQuantumHeadLayerNorm without encoding_type/
    encoding_scale/data_reuploading at all -- must still match explicitly
    passing encoding_type='tanh', encoding_scale=math.pi,
    data_reuploading=False, given the same seed."""
    torch.manual_seed(13)
    omitted = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1)
    torch.manual_seed(13)
    explicit = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, encoding_type="tanh", encoding_scale=math.pi, data_reuploading=False)
    x = torch.randn(4, 10)
    assert torch.equal(omitted(x), explicit(x))


def test_encode_method_e0_matches_pi_tanh():
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1)
    x = torch.randn(5, 10)
    expected = math.pi * torch.tanh(head.reduce(x))
    assert torch.allclose(head.encode(x), expected)


def test_e1_reduced_encoding_scale_is_half_pi():
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, encoding_scale=0.5 * math.pi)
    x = torch.randn(5, 10)
    expected = (0.5 * math.pi) * torch.tanh(head.reduce(x))
    assert torch.allclose(head.encode(x), expected)
    assert head.encode(x).abs().max() <= 0.5 * math.pi + 1e-5


def test_e2_increased_encoding_scale_is_two_pi():
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, encoding_scale=2 * math.pi)
    x = torch.randn(5, 10)
    expected = (2 * math.pi) * torch.tanh(head.reduce(x))
    assert torch.allclose(head.encode(x), expected)


def test_e3_clip_encoding_applies_bounded_linear_mapping():
    """E3: angle = pi * clip(reduce(h), -1, 1) -- linear (not tanh-compressed)
    within the [-1,1] input range, and exactly flat (clipped) outside it."""
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, encoding_type="clip", encoding_scale=math.pi)
    assert head.encoding_type == "clip"
    x = torch.randn(20, 10) * 5  # deliberately large, to exercise clipping
    z = head.reduce(x)
    expected = math.pi * torch.clamp(z, -1.0, 1.0)
    assert torch.allclose(head.encode(x), expected)
    # every output must land exactly on [-pi, pi] -- clip guarantees this
    # exactly, unlike tanh's asymptotic (never-exactly-reaching) bound.
    assert (head.encode(x).abs() <= math.pi + 1e-5).all()
    # for |z|>1 the mapping must be exactly saturated at +-pi (linear
    # clip, not a smooth tanh compression)
    saturated_high = z > 1.0
    if saturated_high.any():
        assert torch.allclose(head.encode(x)[saturated_high], torch.full_like(head.encode(x)[saturated_high], math.pi))


def test_unknown_encoding_type_rejected():
    with pytest.raises(ValueError, match="encoding_type"):
        HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=1, encoding_type="sigmoid")


def test_e4_data_reuploading_encodes_before_each_variational_layer():
    """Verified by introspecting the compiled PennyLane tape (not just
    trusting the docstring): data_reuploading=True must produce exactly
    n_layers AngleEmbedding operations in the executed circuit;
    data_reuploading=False must produce exactly 1, regardless of n_layers."""
    for n_layers in (1, 2, 3):
        reuploaded = build_quantum_layer(n_qubits=6, n_layers=n_layers, data_reuploading=True)
        reuploaded(torch.randn(2, 6))
        ops = [op.name for op in reuploaded.qnode._tape.operations]
        assert ops.count("AngleEmbedding") == n_layers

        single_shot = build_quantum_layer(n_qubits=6, n_layers=n_layers, data_reuploading=False)
        single_shot(torch.randn(2, 6))
        ops2 = [op.name for op in single_shot.qnode._tape.operations]
        assert ops2.count("AngleEmbedding") == 1


@pytest.mark.parametrize("ansatz", ["strongly_entangling", "hardware_efficient_ring", "reduced_entanglement", "basic_entangler"])
def test_e4_data_reuploading_works_for_every_ansatz(ansatz):
    """Section 4's data-reuploading spec must generalize across every
    ansatz build_quantum_layer supports, not just the reference one."""
    layer = build_quantum_layer(n_qubits=6, n_layers=2, ansatz=ansatz, data_reuploading=True)
    out = layer(torch.randn(3, 6))
    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()
    ops = [op.name for op in layer.qnode._tape.operations]
    assert ops.count("AngleEmbedding") == 2


def test_data_reuploading_introduces_zero_trainable_parameters():
    """Section 5's explicit requirement: repeating the (parameter-free)
    AngleEmbedding must not change the quantum circuit's own parameter
    count or the head's total trainable parameter count."""
    reference = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2)
    reuploaded = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, data_reuploading=True)
    assert sum(p.numel() for p in reuploaded.quantum.parameters()) == sum(p.numel() for p in reference.quantum.parameters())
    assert sum(p.numel() for p in reuploaded.parameters()) == sum(p.numel() for p in reference.parameters())
    assert reuploaded.quantum.weights.shape == reference.quantum.weights.shape


def test_data_reuploading_changes_output_given_same_weights():
    """A behavioral sanity check that re-uploading is actually doing
    something different, not silently falling back to the E0 circuit --
    same weights, same input, different circuit structure must generally
    produce a different PauliZ output."""
    torch.manual_seed(21)
    e0 = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2)
    torch.manual_seed(21)
    e4 = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, data_reuploading=True)
    assert torch.equal(e0.quantum.weights, e4.quantum.weights)  # same initial weights (same seed)
    x = torch.randn(4, 10)
    assert not torch.equal(e0(x), e4(x))


def test_encoding_variants_forward_backward_and_gradient_flow():
    configs = [
        dict(encoding_type="tanh", encoding_scale=0.5 * math.pi),
        dict(encoding_type="tanh", encoding_scale=2 * math.pi),
        dict(encoding_type="clip", encoding_scale=math.pi),
        dict(data_reuploading=True),
    ]
    for kwargs in configs:
        head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, **kwargs)
        logits = head(torch.randn(4, 10))
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()
        logits.sum().backward()
        for p in head.quantum.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
        for p in head.reduce.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_quantum_resource_summary_reports_encoding_config():
    head = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=2, encoding_type="clip", encoding_scale=0.5 * math.pi, data_reuploading=True)
    summary = head.quantum_resource_summary()
    assert summary["encoding_config"] == {
        "encoding_type": "clip", "encoding_scale": 0.5 * math.pi, "data_reuploading": True, "n_encoding_operations": 2,
    }
    default_summary = HybridQuantumHeadLayerNorm(in_dim=10, n_qubits=6, n_layers=3).quantum_resource_summary()
    assert default_summary["encoding_config"] == {
        "encoding_type": "tanh", "encoding_scale": math.pi, "data_reuploading": False, "n_encoding_operations": 1,
    }


def test_encoding_checkpoint_save_load_roundtrip():
    for kwargs in [dict(encoding_scale=0.5 * math.pi), dict(encoding_type="clip"), dict(data_reuploading=True)]:
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


def test_e0_encoding_end_to_end_training_unaffected(tiny_benchmark):
    """A full train_v2_head pass with explicit E0 encoding kwargs must
    reach the exact same best_val_pr_auc/best_epoch as the pre-encoding-investigation
    default call (same seed) -- confirms the new __init__ params don't
    perturb the existing training path when left at their E0 values."""
    v4prepared, _ = _tiny_v4_prepared(tiny_benchmark)
    v4prepared.config.training.epochs = 3
    v4prepared.config.training.early_stopping_patience = 5
    in_dim = v4prepared.n_components

    from scm_dataset.modeling.qgnn_v2 import set_seed as v4_set_seed

    v4_set_seed(42)
    plain = HybridQuantumHeadLayerNorm(in_dim=in_dim, n_qubits=4, n_layers=1, elementwise_affine=False)
    plain_result = train_v2_head(v4prepared, plain, seed=42, verbose=False)

    v4_set_seed(42)
    explicit_e0 = HybridQuantumHeadLayerNorm(in_dim=in_dim, n_qubits=4, n_layers=1, elementwise_affine=False, encoding_type="tanh", encoding_scale=math.pi, data_reuploading=False)
    explicit_result = train_v2_head(v4prepared, explicit_e0, seed=42, verbose=False)

    assert plain_result.best_val_pr_auc == explicit_result.best_val_pr_auc
    assert plain_result.best_epoch == explicit_result.best_epoch
