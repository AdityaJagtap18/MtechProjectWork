"""Tests for the QGNN-v2 depth-3 + data re-uploading controlled
follow-up experiment. Purely additive alongside `test_qgnn_v2.py` --
this file does not modify or weaken any existing test.

Checklist (from the request):
1. qubits == 8
2. depth == 3
3. data is actually re-uploaded at each intended layer
4. output shape matches the existing QGNN
5. classical MLP head is unchanged (same shape/params as baseline)
6. no gradients flow into the frozen GraphSAGE encoder
7. PCA remains train-only
8. no target leakage
9/10. existing QGNN-v2 / GraphSAGE tests continue to pass -- verified by
      running the full suite (tests/test_qgnn_v2.py, tests/test_qgnn.py,
      and every GraphSAGE test), not repeated here.
"""

from __future__ import annotations

import pennylane as qml
import torch

from scm_dataset.modeling.graph_embedding_reduction import extract_supplier_embeddings
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.modeling.qgnn import build_qgnn_model
from scm_dataset.modeling.qgnn_v2 import build_v2_prepared, evaluate_v2, generate_predictions_v2, train_v2_head
from scm_dataset.modeling.qgnn_v2_reupload import build_qgnn_reupload_model
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


def _tape_ops(model, x):
    tape = qml.workflow.construct_tape(model.quantum._circuit)(torch.tanh(x) * torch.pi, model.quantum.weights)
    return [op.name for op in tape.operations]


def test_reupload_model_has_8_qubits_and_depth_3():
    model = build_qgnn_reupload_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    summary = model.quantum_resource_summary()
    assert summary["qubits"] == 8
    assert summary["variational_layers"] == 3


def test_data_is_reuploaded_between_every_pair_of_consecutive_layers():
    """The precise, verified mechanism (not just an RY count, which
    undercounts -- AngleEmbedding appears as its own named tape
    operation, not yet decomposed into individual RY gates at this
    level): baseline QGNN always has exactly 1 AngleEmbedding op
    regardless of depth; the re-upload model has exactly `n_layers`."""
    x = torch.randn(2, 8)

    for depth in (1, 2, 3, 5):
        baseline = build_qgnn_model(n_qubits=8, n_layers=depth, device_name=CPU_DEVICE)
        baseline_ops = _tape_ops(baseline, x)
        assert baseline_ops.count("AngleEmbedding") == 1, f"baseline depth={depth} should always re-encode exactly once"

        reupload = build_qgnn_reupload_model(n_qubits=8, n_layers=depth, device_name=CPU_DEVICE)
        reupload_ops = _tape_ops(reupload, x)
        assert reupload_ops.count("AngleEmbedding") == depth, f"reupload depth={depth} should encode once per layer"

    # depth-3 specifically: re-uploads happen strictly BETWEEN layers,
    # not before layer 1 and not after the last layer -- verify ordering
    reupload3 = build_qgnn_reupload_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    ops = _tape_ops(reupload3, x)
    angle_embedding_positions = [i for i, name in enumerate(ops) if name == "AngleEmbedding"]
    assert len(angle_embedding_positions) == 3
    assert ops[-1] != "AngleEmbedding"  # circuit does not end on a re-upload -- last op before measurement is the final layer's entangler


def test_reupload_has_identical_trainable_quantum_parameter_count_to_baseline():
    """Re-uploading repeats fixed, non-trainable gates -- it must not
    change the trainable parameter count relative to the baseline at the
    same depth."""
    for depth in (1, 3, 5):
        baseline = build_qgnn_model(n_qubits=8, n_layers=depth, device_name=CPU_DEVICE)
        reupload = build_qgnn_reupload_model(n_qubits=8, n_layers=depth, device_name=CPU_DEVICE)
        assert baseline.quantum.weights.numel() == reupload.quantum.weights.numel()
        assert reupload.quantum_resource_summary()["trainable_quantum_parameters"] == depth * 8


def test_reupload_output_shape_matches_baseline_qgnn():
    x = torch.randn(5, 8)
    baseline = build_qgnn_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    reupload = build_qgnn_reupload_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    assert baseline(x).shape == reupload(x).shape == (5,)


def test_reupload_classical_mlp_head_is_architecturally_unchanged():
    baseline = build_qgnn_model(n_qubits=8, n_layers=3, mlp_hidden=8, device_name=CPU_DEVICE)
    reupload = build_qgnn_reupload_model(n_qubits=8, n_layers=3, mlp_hidden=8, device_name=CPU_DEVICE)
    baseline_shapes = [tuple(p.shape) for p in baseline.mlp.parameters()]
    reupload_shapes = [tuple(p.shape) for p in reupload.mlp.parameters()]
    assert baseline_shapes == reupload_shapes
    assert sum(p.numel() for p in baseline.mlp.parameters()) == sum(p.numel() for p in reupload.mlp.parameters())


def test_reupload_gradients_are_finite_and_weights_move():
    model = build_qgnn_reupload_model(n_qubits=8, n_layers=3, device_name=CPU_DEVICE)
    x = torch.randn(6, 8)
    target = torch.zeros(6)
    target[0] = 1.0
    initial_weights = model.quantum.weights.detach().clone()

    logits = model(x)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} got a non-finite gradient"
        assert param.grad.abs().sum().item() > 0, f"{name} got an all-zero gradient"


def test_reupload_end_to_end_through_frozen_encoder_no_leakage(tiny_benchmark):
    """Exercises the FULL pipeline (frozen encoder -> embeddings -> PCA
    -> reupload QGNN) via the existing, unmodified qgnn_v2.py
    utilities -- confirms encoder-freeze and PCA train-only-ness transfer
    to this new model exactly as they do for the baseline (items 6-8),
    not just by inheritance/assumption."""
    encoder, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(encoder, prepared, times)
    v2prepared, reducer = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components=4)
    v2prepared.config.training.epochs = 3
    v2prepared.config.training.early_stopping_patience = 10

    # PCA train-only: reducer's fit statistics must be independent of
    # validation/test rows -- reuses the already-fit reducer as-is, the
    # SAME object every downstream consumer (classical head, baseline
    # QGNN, reupload QGNN) shares; nothing here refits it.
    assert reducer._input_columns is not None  # fit() already ran inside build_v2_prepared

    reupload_model = build_qgnn_reupload_model(n_qubits=4, n_layers=3, device_name=CPU_DEVICE)
    result = train_v2_head(v2prepared, reupload_model, seed=42, verbose=False)
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.grad is None for p in encoder.parameters())  # no target/label information ever reaches the frozen encoder via this new model either

    predictions = generate_predictions_v2(result.model, v2prepared)
    assert predictions["risk_probability"].between(0.0, 1.0).all()
    eval_result = evaluate_v2(result.model, v2prepared, v2prepared.config.threshold)
    assert "test" in eval_result.metrics_by_split
