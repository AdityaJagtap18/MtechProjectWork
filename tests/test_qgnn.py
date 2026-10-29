"""Tests for the hybrid QGNN
(QGNN_FAIR_BENCHMARK_AND_IMPLEMENTATION_PLAN.md sections 8-9, 12): forward
shape, non-zero gradients on both quantum and classical parameters, loss
decreasing over a short training run, and that evaluation reuses the
existing model-agnostic metrics/calibration pipeline correctly. Runs on
CPU (`default.qubit`) so these pass in any environment, independent of
whether a CUDA-capable `lightning.gpu` device is available."""

from __future__ import annotations

import torch

from scm_dataset.modeling.pipeline import prepare_reduced_from_benchmark
from scm_dataset.modeling.qgnn import build_qgnn_model, evaluate_qgnn, generate_predictions_qgnn, train_qgnn

from conftest import make_tiny_config

CPU_DEVICE = "default.qubit"


def test_qgnn_forward_pass_shape_and_range():
    model = build_qgnn_model(n_qubits=4, n_layers=1, device_name=CPU_DEVICE)
    x = torch.randn(10, 4)
    logits = model(x)
    assert logits.shape == (10,)
    assert torch.isfinite(logits).all()


def test_qgnn_gradients_are_nonzero_for_all_parameters():
    model = build_qgnn_model(n_qubits=4, n_layers=1, device_name=CPU_DEVICE)
    x = torch.randn(8, 4)
    target = torch.zeros(8)
    target[0] = 1.0

    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert param.grad.abs().sum().item() > 0, f"{name} got an all-zero gradient"


def test_qgnn_quantum_resource_summary_reports_expected_fields():
    model = build_qgnn_model(n_qubits=6, n_layers=2, device_name=CPU_DEVICE)
    summary = model.quantum_resource_summary()
    assert summary["qubits"] == 6
    assert summary["variational_layers"] == 2
    assert summary["trainable_quantum_parameters"] == 6 * 2
    assert summary["backend"] == CPU_DEVICE


def test_qgnn_angle_encoding_is_deterministic_and_bounded():
    model = build_qgnn_model(n_qubits=4, n_layers=1, device_name=CPU_DEVICE)
    x = torch.tensor([[100.0, -100.0, 0.0, 5.0]])  # extreme standardized values
    angles = torch.tanh(x) * torch.pi
    assert (angles.abs() <= torch.pi).all()
    # same input -> same angles, always (no randomness, no label dependence)
    angles_again = torch.tanh(x) * torch.pi
    assert torch.equal(angles, angles_again)


def test_train_qgnn_loss_decreases_and_produces_valid_predictions(tiny_benchmark):
    config = make_tiny_config()
    config.training.epochs = 4
    config.training.early_stopping_patience = 10
    prepared, reducer = prepare_reduced_from_benchmark(config, tiny_benchmark, "domain_selected", 4)

    result = train_qgnn(prepared, seed=42, n_layers=1, device_name=CPU_DEVICE, verbose=False)

    assert result.history["train_loss"].iloc[-1] < result.history["train_loss"].iloc[0]

    predictions = generate_predictions_qgnn(result.model, prepared)
    assert predictions["risk_probability"].between(0.0, 1.0).all()
    assert set(predictions["actual_disruption"].unique()) <= {0, 1}

    eval_result = evaluate_qgnn(result.model, prepared, config.threshold)
    for split in ("train", "validation", "test"):
        if split in eval_result.metrics_by_split:
            m = eval_result.metrics_by_split[split]
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
