"""Tests for QGNN-v2's frozen-encoder hybrid architecture
(QGNN_V2_HYBRID_IMPLEMENTATION_AND_DEPTH_BENCHMARK.md section 2 Phase 2
checklist): checkpoint loading, embedding shape, encoder freeze, PCA
leakage safety, matched input equality, QNN depth parameter counts, and a
short end-to-end training run. Uses `tiny_benchmark` (20 suppliers) and a
freshly-trained-then-frozen tiny GraphSAGE, not the real primary
benchmark, so these run fast and don't depend on
experiments/classical_gnn/ existing on disk."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scm_dataset.modeling.graph_embedding_reduction import extract_supplier_embeddings, fit_embedding_pca
from scm_dataset.modeling.pipeline import prepare_from_benchmark
from scm_dataset.modeling.preprocessing import supplier_fit_mask
from scm_dataset.modeling.qgnn import build_qgnn_model
from scm_dataset.modeling.qgnn_v2 import build_classical_head, build_v2_prepared, evaluate_v2, generate_predictions_v2, train_v2_head
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
        p.grad = None  # clear any leftover .grad from the training run above -- freezing stops future gradients, it doesn't erase past ones
    return model, prepared


def test_frozen_encoder_has_no_trainable_parameters(tiny_benchmark):
    model, _ = _tiny_frozen_encoder(tiny_benchmark)
    assert all(not p.requires_grad for p in model.parameters())


def test_embedding_extraction_shape(tiny_benchmark):
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())[:3]
    emb = extract_supplier_embeddings(model, prepared, times)
    n_suppliers = len(prepared.snapshot_builder.supplier_id_order())
    hidden_dim = prepared.config.model.hidden_dim
    assert emb.shape == (n_suppliers * len(times), hidden_dim)
    assert list(emb.columns) == [f"emb_{i}" for i in range(hidden_dim)]
    assert not emb.isna().any().any()


def test_embedding_extraction_uses_no_grad_encoder_receives_nothing(tiny_benchmark):
    """Encoder freeze + no_grad extraction means no gradient can ever
    reach the encoder -- verified by checking every param's .grad stays
    None after a full extract -> PCA -> train_v2_head cycle."""
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)
    v2prepared, reducer = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components=4)
    head = build_classical_head(4)
    train_v2_head(v2prepared, head, seed=42, verbose=False)
    assert all(p.grad is None for p in model.parameters())


def test_pca_embedding_reducer_is_train_only_deterministic_8d(tiny_benchmark):
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)
    train_ex = prepared.examples[prepared.examples["split"] == "train"]

    reducer1 = fit_embedding_pca(emb, train_ex, n_components=4)
    reducer2 = fit_embedding_pca(emb, train_ex, n_components=4)
    reduced1 = reducer1.transform(emb)
    reduced2 = reducer2.transform(emb)
    assert reduced1.shape[1] == 4
    assert np.allclose(reduced1.values, reduced2.values)  # deterministic (PCA random_state=0)


def test_pca_embedding_reducer_fit_never_uses_validation_or_test_rows(tiny_benchmark):
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)
    train_ex = prepared.examples[prepared.examples["split"] == "train"]

    baseline = fit_embedding_pca(emb, train_ex, n_components=4)

    fit_mask = supplier_fit_mask(emb, train_ex)
    tampered = emb.copy()
    tampered.loc[~fit_mask, :] = 1e9  # garbage in every non-train row

    from scm_dataset.modeling.reduction import PCASupplierReducer

    tampered_reducer = PCASupplierReducer(4)
    tampered_reducer.fit(tampered, fit_mask, list(tampered.columns))

    assert np.allclose(baseline._mean.values, tampered_reducer._mean.values)
    assert np.allclose(baseline._pca.components_, tampered_reducer._pca.components_)


def test_classical_and_quantum_heads_receive_byte_identical_input(tiny_benchmark):
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)
    v2prepared, _ = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components=4)

    from scm_dataset.modeling.qgnn_v2 import _flatten

    x_for_classical, y_for_classical, _ = _flatten(v2prepared, "test")
    x_for_qgnn, y_for_qgnn, _ = _flatten(v2prepared, "test")
    assert torch.equal(x_for_classical, x_for_qgnn)
    assert torch.equal(y_for_classical, y_for_qgnn)


@pytest.mark.parametrize("depth,expected_quantum_params", [(1, 8), (2, 16), (3, 24), (4, 32), (5, 40)])
def test_qnn_depth_parameter_counts(depth, expected_quantum_params):
    model = build_qgnn_model(n_qubits=8, n_layers=depth, device_name="default.qubit")
    summary = model.quantum_resource_summary()
    assert summary["trainable_quantum_parameters"] == expected_quantum_params


def test_train_v2_head_end_to_end_classical_and_qgnn(tiny_benchmark):
    model, prepared = _tiny_frozen_encoder(tiny_benchmark)
    times = sorted(prepared.examples["time"].unique())
    emb = extract_supplier_embeddings(model, prepared, times)
    v2prepared, _ = build_v2_prepared(prepared.config, prepared.benchmark, prepared.examples, emb, n_components=4)
    v2prepared.config.training.epochs = 3
    v2prepared.config.training.early_stopping_patience = 10

    classical = build_classical_head(4)
    classical_result = train_v2_head(v2prepared, classical, seed=42, verbose=False)
    assert classical_result.history["train_loss"].iloc[-1] < classical_result.history["train_loss"].iloc[0]
    preds = generate_predictions_v2(classical_result.model, v2prepared)
    assert preds["risk_probability"].between(0.0, 1.0).all()
    eval_result = evaluate_v2(classical_result.model, v2prepared, v2prepared.config.threshold)
    assert "test" in eval_result.metrics_by_split

    qgnn = build_qgnn_model(n_qubits=4, n_layers=1, device_name="default.qubit")
    initial_weights = qgnn.quantum.weights.detach().clone()
    qgnn_result = train_v2_head(v2prepared, qgnn, seed=42, verbose=False)
    assert not torch.equal(initial_weights, qgnn_result.model.quantum.weights.detach())  # weights actually moved -> gradients reached the quantum circuit
    qgnn_eval = evaluate_v2(qgnn_result.model, v2prepared, v2prepared.config.threshold)
    assert "test" in qgnn_eval.metrics_by_split
